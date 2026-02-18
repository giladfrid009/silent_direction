import torch
from tqdm.auto import tqdm

from src.model import TargetedModel
from src.data import TableLoader, TableIterator
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.metrics import Metrics
from src.losses import Loss
from src.functional import project, compute_targets_mask
from src.config import StopCriteria


@torch.inference_mode()
def evaluate_norm(
    targeted_model: TargetedModel,
    layer: str,
    dl_eval: TableLoader,
    direction: torch.Tensor,
    stop_criteria: StopCriteria,
) -> dict[str, float]:

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=direction, normalize=True)
        return activations - projection

    manipulator = ActivationManipulator(targeted_model.model, layer, manipulation_fn=subtract_projection)
    extractor = ActivationExtractor(targeted_model.model, layer)

    METRICS = {
        "top1_acc": 0.0,
        "top10_agr": 0.0,
        "kl_div": 0.0,
        "proj_l2_raw": 0.0,
        "proj_l2_norm": 0.0,
    }

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

        METRICS["proj_l2_norm"] += Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
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

        # update progress
        step_num = pbar.n + 1
        pbar.set_postfix({k: f"{v / step_num:.4f}" for k, v in METRICS.items()})
        stop_criteria.update()

    pbar.close()

    METRICS = {k: v / iterations for k, v in METRICS.items()}

    METRICS["redundancy_score"] = Metrics.redundancy_score(
        proj_norm=METRICS["proj_l2_norm"],
        top1_acc=METRICS["top1_acc"],
        top10_agr=METRICS["top10_agr"],
    )

    return METRICS
