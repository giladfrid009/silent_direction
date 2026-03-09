import torch
from tqdm.auto import tqdm

from src.model import TargetedModel
from src.data import TableLoader, TableIterator
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.metrics import Metrics
from src.losses import Loss
from src.functional import project, compute_targets_mask
from src.norm.utils import probe_layer_dim
from src.config import StopCriteria
from src.utils.logging import create_logger


logger = create_logger(__name__)


def score_dual(kl_div: float, target_kl: float, proj_norm: float) -> float:
    """
    Compute the scalar score used for model selection under a hard feasibility check.

    The score equals ``proj_norm`` only when the KL-divergence constraint is satisfied;
    otherwise it is set to zero. This is used to track the best feasible direction found
    during optimization.

    Args:
        kl_div: Measured KL divergence between the baseline and modified model outputs.
        target_kl: Maximum allowed KL divergence.
        proj_norm: Projection-based objective value for the current direction.

    Returns:
        The feasible score, equal to ``proj_norm`` if ``kl_div <= target_kl``,
        and ``0.0`` otherwise.
    """
    return proj_norm if kl_div <= target_kl else 0.0


def train_norm_dual(
    targeted_model: TargetedModel,
    layer: str,
    dl_train: TableLoader,
    stop_criteria: StopCriteria,
    target_kl: float,
    learning_rate: float = 0.1,
    dual_learning_rate: float = 0.1,
    penalty_coef: float = 1.0,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """
    Learn a single direction in activation space that maximizes relative projection norm
    while constraining the output shift induced by removing that direction.

    The method optimizes a vector ``w`` at a chosen model layer. For each batch, the code:
    1. Runs the model normally to collect baseline logits and layer activations.
    2. Normalizes the activations and measures how strongly they project onto the current
       direction.
    3. Runs the model again while subtracting the projection onto that direction from the
       layer activations.
    4. Measures the KL divergence between baseline and modified logits.
    5. Updates the direction using a dual-style objective that maximizes projection norm
       subject to a KL constraint.

    The best returned direction is the best *feasible* one, meaning the one with the
    highest projection score among iterations satisfying ``kl_div <= target_kl``.

    Args:
        targeted_model: Wrapper around the model and tokenizer, providing tokenization,
            forward pass, device, and dtype utilities.
        layer: Name of the model layer from which activations are extracted and manipulated.
        dl_train: Training data loader containing prompt batches.
        stop_criteria: Early stopping controller. It is reset at the beginning of training
            and updated once per optimization step using the feasibility-based score.
        target_kl: Maximum allowed KL divergence between the baseline and modified outputs.
        learning_rate: Learning rate for the optimizer over the direction vector.
        dual_learning_rate: Step size used to update the dual coefficient associated with
            the KL constraint.
        penalty_coef: Coefficient of the quadratic penalty applied when the KL constraint
            is violated.

    Returns:
        A tuple ``(best_direction, history)`` where:

        - ``best_direction`` is the best feasible unit direction found during training,
          represented as a 1D tensor of shape ``[layer_dim]``.
        - ``history`` is a list of per-step metric dictionaries, including optimization
          loss, KL divergence, projection score, agreement metrics, dual variables, and
          feasibility indicators.

    Notes:
        - The optimized parameter is ``w``, while the actual direction used in the
          objective is its normalized version ``v``.
        - The manipulation subtracts the projection onto ``w`` from the layer activations,
          thereby testing the functional importance of that direction.
        - The objective uses both a dual term and a quadratic penalty:
              loss = -proj_l2_rel + kl_coef * (kl_div - target_kl)
                     + penalty_coef * relu(kl_div - target_kl)^2
        - The returned best direction is selected using ``score_dual()``, not by lowest
          loss.
    """

    # dual ascent variables
    kl_coef = 0.0

    stop_criteria.reset()
    layer_dim = probe_layer_dim(targeted_model, layer)

    w = torch.randn(layer_dim, device=targeted_model.device, dtype=targeted_model.dtype, requires_grad=True)
    optim = torch.optim.Adam([w], lr=learning_rate)

    best_score = -float("inf")
    best_direction = w.clone().detach()

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=w, normalize=True)
        return activations - projection

    extractor = ActivationExtractor(targeted_model.model, layer)
    manipulator = ActivationManipulator(targeted_model.model, layer, manipulation_fn=subtract_projection)

    history = []
    num_steps = min(stop_criteria.max_steps, len(dl_train))
    pbar = tqdm(TableIterator(dl_train, num_batches=num_steps), desc="Training", leave=False)

    for batch in pbar:
        if stop_criteria.should_stop():
            break

        conversations = batch["prompt"]
        encodings = targeted_model.tokenize(conversations)
        targets_mask = compute_targets_mask(encodings)

        optim.zero_grad()
        v = torch.nn.functional.normalize(w, dim=0)

        with torch.inference_mode(), extractor.capture():
            baseline_logits = targeted_model.forward(encodings).logits
            activations = extractor.get_activations()[layer]
            activations_normalized = torch.nn.functional.normalize(activations, dim=-1)
            logger.debug(f"activations.shape = {activations.shape}")

        with manipulator.capture():
            modified_logits = targeted_model.forward(encodings).logits

        kl_div = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=None,
        )

        proj_l2_rel = Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=v,
            targets_mask=targets_mask,
            squared=True,
        )

        top10_agr = Metrics.topk_agreement(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
            top_k=10,
        )

        top1_acc = Metrics.top1_accuracy(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
        )

        score = score_dual(
            kl_div=kl_div.item(),
            target_kl=target_kl,
            proj_norm=proj_l2_rel.item(),
        )

        if score > best_score:
            best_score = score
            best_direction = v.detach().clone()

        # dual ascent: maximize proj_l2_rel subject to kl_div <= target_kl
        constraint = kl_div - target_kl  # positive if constraint is violated, negative if satisfied
        penalty = penalty_coef * torch.relu(constraint).square()  # quadratic penalty for constraint violation
        loss = -proj_l2_rel + kl_coef * constraint + penalty

        # main step
        loss.backward()
        optim.step()

        # coef step; if constraint is violated, increase kl_coef to penalize more
        kl_coef = max(0.0, kl_coef + dual_learning_rate * constraint.item())

        stop_criteria.update(value=score)

        METRICS = {
            "loss": loss.item(),
            "kl_div": kl_div.item(),
            "proj_l2_rel": proj_l2_rel.item(),
            "top1_acc": top1_acc.item(),
            "top10_agr": top10_agr.item(),
            "score": score,
            "best_score": best_score,
            # additional dual metrics
            "constraint": constraint.item(),
            "kl_coef": kl_coef,
            "penalty": penalty.item(),
            "feasible": float(kl_div.item() <= target_kl),
        }

        # update progress
        history.append(METRICS)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in METRICS.items()})

    pbar.close()

    return best_direction, history
