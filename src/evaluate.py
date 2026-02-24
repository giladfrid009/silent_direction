import torch
from tqdm import tqdm
import pandas as pd
from collections import defaultdict

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
) -> tuple[dict[str, float], pd.DataFrame]:

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=direction, normalize=True)
        return activations - projection

    stop_criteria.reset()
    extractor = ActivationExtractor(targeted_model.model, layer)
    manipulator = ActivationManipulator(targeted_model.model, layer, manipulation_fn=subtract_projection)

    dataset_metrics = defaultdict(lambda: 0.0)
    sample_metrics = defaultdict(list)

    mean_activation, mean_activation_normalized = compute_empirical_mean(
        targeted_model=targeted_model,
        layer=layer,
        dl=dl_eval,
        iterations=100,  # TODO: currently hard-coded
    )

    iterations = min(stop_criteria.max_steps, len(dl_eval))
    for batch in tqdm(TableIterator(dl_eval, num_batches=iterations), desc="Evaluating", leave=False):
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

        batch_metrics: dict[str, torch.Tensor] = {}

        batch_metrics["proj_l2_raw"] = Loss.projection_l2_norm(
            activations=activations,
            direction=direction,
            targets_mask=targets_mask,
            reduction="samplesum",
        )

        batch_metrics["proj_l2_rel"] = Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
            reduction="samplesum",
        )

        batch_metrics["proj_var_raw"] = Loss.projection_total_variance(
            activations=activations,
            direction=direction,
            targets_mask=targets_mask,
            mean_activation=mean_activation,
            reduction="samplesum",
        )

        batch_metrics["proj_var_rel"] = Loss.projection_total_variance(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
            mean_activation=mean_activation_normalized,
            reduction="samplesum",
        )

        batch_metrics["full_l2_raw"] = Loss.l2_norm(
            activations=activations,
            targets_mask=targets_mask,
            reduction="samplesum",
        )

        batch_metrics["full_var_raw"] = Loss.total_variance(
            activations=activations,
            targets_mask=targets_mask,
            mean_activation=mean_activation,
            reduction="samplesum",
        )

        batch_metrics["full_var_rel"] = Loss.total_variance(
            activations=activations_normalized,
            targets_mask=targets_mask,
            mean_activation=mean_activation_normalized,
            reduction="samplesum",
        )

        batch_metrics["kl_div"] = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            reduction="samplesum",
        )

        batch_metrics["top1_acc"] = Metrics.top1_accuracy(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            reduction="samplesum",
        )

        batch_metrics["top10_agr"] = Metrics.topk_agreement(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=10,
            reduction="samplesum",
        )

        # update per-sample metrics
        sample_elements = targets_mask.sum(dim=1).clamp(min=1)  # shape: (batch_size,)
        sample_metrics["prompt"].extend(conversations)
        for metric_name, metric_value in batch_metrics.items():
            sample_metrics[metric_name].extend((metric_value / sample_elements).tolist())

        # update global metrics
        batch_elements = targets_mask.sum().item()
        dataset_metrics["total_elements"] += batch_elements
        for metric_name, metric_value in batch_metrics.items():
            dataset_metrics[metric_name] += metric_value.sum().item()

        stop_criteria.update()

    total_elements = max(dataset_metrics.pop("total_elements"), 1)
    dataset_metrics = {k: v / total_elements for k, v in dataset_metrics.items()}
    sample_metrics = pd.DataFrame(sample_metrics)

    return dataset_metrics, sample_metrics
