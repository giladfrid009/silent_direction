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


def score_hinge(kl_div: float, target_kl: float, proj_norm: float, tol_factor: float = 2.0) -> float:
    """
    Compute the scalar score used for model selection under a hard feasibility check.

    The score equals ``proj_norm`` only when the KL-divergence constraint is satisfied;
    otherwise it is set to zero. This is used to track the best feasible direction found
    during optimization.

    Args:
        kl_div: Measured KL divergence between the baseline and modified model outputs.
        target_kl: Maximum allowed KL divergence.
        proj_norm: Projection-based objective value for the current direction.
        tol_factor: Tolerance factor for the KL constraint. The constraint is considered satisfied
            if ``kl_div <= tol_factor * target_kl``. This allows for some slack in the feasibility check,
            which can be helpful in practice to account for noise in KL estimation.

    Returns:
        The feasible score, equal to ``proj_norm`` if ``kl_div <= target_kl * tol_factor``,
        and ``0.0`` otherwise.
    """
    return proj_norm if kl_div <= target_kl * tol_factor else 0.0


def train_norm_hinge(
    targeted_model: TargetedModel,
    layer: str,
    dl_train: TableLoader,
    stop_criteria: StopCriteria,
    target_kl: float,
    learning_rate: float = 0.1,
    hinge_coef: float = 1.0,
    tol_factor: float = 2.0,
    reduction: str = "none",
) -> tuple[torch.Tensor, list[dict[str, float]]]:

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

        with manipulator.capture():
            modified_logits = targeted_model.forward(encodings).logits

        kl_div = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=None,
            reduction="batchmean" if reduction == "mean" else reduction,
        )

        proj_l2_rel = Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=v,
            targets_mask=targets_mask,
            squared=True,
            reduction=reduction,
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

        kl_div_value = kl_div.mean().item()
        proj_l2_rel_value = proj_l2_rel.mean().item()

        score = score_hinge(
            kl_div=kl_div_value,
            target_kl=target_kl,
            proj_norm=proj_l2_rel_value,
            tol_factor=tol_factor,
        )

        if score > best_score:
            best_score = score
            best_direction = v.detach().clone()

        # hinge loss: maximize proj_l2_rel while penalizing KL divergence above the target_kl threshold
        constraint = kl_div - target_kl  # positive if constraint is violated, negative if satisfied
        penalty = hinge_coef * torch.relu(constraint)  # linear penalty for constraint violation
        loss = (-proj_l2_rel + penalty).mean()

        loss.backward()
        optim.step()

        stop_criteria.update(value=score)

        METRICS = {
            "loss": loss.item(),
            "kl_div": kl_div_value,
            "proj_l2_rel": proj_l2_rel_value,
            "top1_acc": top1_acc.item(),
            "top10_agr": top10_agr.item(),
            "score": score,
            "best_score": best_score,
            # additional metrics
            "constraint": constraint.mean().item(),
            "penalty": penalty.mean().item(),
            "feasible": (kl_div <= target_kl * tol_factor).mean().item(),
        }

        # update progress
        history.append(METRICS)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in METRICS.items()})

    pbar.close()

    return best_direction, history
