import torch
from tqdm import tqdm

from src.model import TargetedModel
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.metrics import Metrics
from src.losses import Loss
from src.data import TableLoader, TableIterator
from src.functional import project, compute_targets_mask
from src.principal.utils import compute_empirical_mean
from src.config import StopCriteria


@torch.inference_mode()
def evaluate(
    targeted_model: TargetedModel,
    layer: str,
    dl_eval: TableLoader,
    direction: torch.Tensor,
    stop_criteria: StopCriteria,
) -> dict[str, float]:

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=direction, normalize=True)
        return activations - projection

    extractor = ActivationExtractor(targeted_model.model, layer)
    manipulator = ActivationManipulator(targeted_model.model, layer, manipulation_fn=subtract_projection)

    METRICS = {
        "top1_acc": 0.0,
        "top10_agr": 0.0,
        "kl_div": 0.0,
        "proj_l2_raw": 0.0,
        "proj_l2_rel": 0.0,
        "proj_var_raw": 0.0,
        "proj_var_rel": 0.0,
        "full_l2_raw": 0.0,
        "full_var_raw": 0.0,
        "full_var_rel": 0.0,
    }

    mean_activ, mean_activ_norm = compute_empirical_mean(
        targeted_model=targeted_model,
        layer=layer,
        dl=dl_eval,
        iterations=100, # TODO: currently hard-coded
    )

    current_step = 0
    iterations = min(stop_criteria.max_steps, len(dl_eval))
    pbar = tqdm(TableIterator(dl_eval, num_batches=iterations), desc="Evaluating", leave=False)

    for batch in pbar:
        if stop_criteria.should_stop():
            break

        conversations = batch["prompt"]
        encodings = targeted_model.tokenize(conversations)
        targets_mask = compute_targets_mask(encodings)

        with extractor.capture():
            baseline_logits = targeted_model.forward(encodings).logits
            activations = extractor.get_activations()[layer]
            activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        with manipulator.capture():
            modified_logits = targeted_model.forward(encodings).logits

        METRICS["proj_l2_raw"] += Loss.projection_l2_norm(
            activations=activations,
            direction=direction,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()

        METRICS["proj_l2_rel"] += Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()

        METRICS["proj_var_raw"] += Loss.projection_total_variance(
            activations=activations,
            direction=direction,
            targets_mask=targets_mask,
            mean_activation=mean_activ,
        ).item()

        METRICS["proj_var_rel"] += Loss.projection_total_variance(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
            mean_activation=mean_activ_norm,
        ).item()

        METRICS["full_l2_raw"] += Loss.l2_norm(
            activations=activations,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()

        METRICS["full_var_raw"] += Loss.total_variance(
            activations=activations,
            targets_mask=targets_mask,
            mean_activation=mean_activ,
            reduction="mean",
        ).item()

        METRICS["full_var_rel"] += Loss.total_variance(
            activations=activations_normalized,
            targets_mask=targets_mask,
            mean_activation=mean_activ_norm,
            reduction="mean",
        ).item()

        METRICS["kl_div"] += Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
        ).item()

        METRICS["top1_acc"] += Metrics.top1_accuracy(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
        )

        METRICS["top10_agr"] += Metrics.topk_agreement(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=10,
        )

        # update progress bar
        current_step += 1
        pbar.set_postfix({k: f"{v / current_step:.4f}" for k, v in METRICS.items()})
        stop_criteria.update()

    METRICS = {k: v / current_step for k, v in METRICS.items()}

    pbar.close()

    return METRICS
