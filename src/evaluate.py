import torch
from tqdm import tqdm
import pandas as pd

from src.model import TargetedModel
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.metrics import Metrics
from src.losses import Loss
from src.data import TableLoader, TableIterator
from src.functional import project, compute_targets_mask
from src.principal.utils import compute_empirical_mean
from src.config import StopCriteria


# TODO: fix all places where we call evaluate, to expect that we also return pd.DataFrame
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

    global_metrics = {
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

    mean_activation, mean_activation_normalized = compute_empirical_mean(
        targeted_model=targeted_model,
        layer=layer,
        dl=dl_eval,
        iterations=100,  # TODO: currently hard-coded
    )

    current_step = 0
    iterations = min(stop_criteria.max_steps, len(dl_eval))
    pbar = tqdm(TableIterator(dl_eval, num_batches=iterations), desc="Evaluating", leave=False)

    all_results = []

    for i, batch in enumerate(pbar):
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

        batch_metrics = {}

        batch_metrics["proj_l2_raw"] = Loss.projection_l2_norm(
            activations=activations,
            direction=direction,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()

        batch_metrics["proj_l2_rel"] = Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()

        batch_metrics["proj_var_raw"] = Loss.projection_total_variance(
            activations=activations,
            direction=direction,
            targets_mask=targets_mask,
            mean_activation=mean_activation,
            reduction="mean",
        ).item()

        batch_metrics["proj_var_rel"] = Loss.projection_total_variance(
            activations=activations_normalized,
            direction=direction,
            targets_mask=targets_mask,
            mean_activation=mean_activation_normalized,
            reduction="mean",
        ).item()

        batch_metrics["full_l2_raw"] = Loss.l2_norm(
            activations=activations,
            targets_mask=targets_mask,
            reduction="mean",
        ).item()

        batch_metrics["full_var_raw"] = Loss.total_variance(
            activations=activations,
            targets_mask=targets_mask,
            mean_activation=mean_activation,
            reduction="mean",
        ).item()

        batch_metrics["full_var_rel"] = Loss.total_variance(
            activations=activations_normalized,
            targets_mask=targets_mask,
            mean_activation=mean_activation_normalized,
            reduction="mean",
        ).item()

        batch_metrics["kl_div"] = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            reduction="batchmean",
        ).item()

        # TODO: implement reduction
        batch_metrics["top1_acc"] = Metrics.top1_accuracy(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
        )

        # TODO: implement reduction
        batch_metrics["top10_agr"] = Metrics.topk_agreement(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=10,
        )
        
        # update global metrics
        global_metrics = {k: global_metrics[k] + batch_metrics[k] for k in global_metrics.keys()}

        # update progress bar
        current_step += 1
        pbar.set_postfix({k: f"{v / current_step:.4f}" for k, v in batch_metrics.items()})
        stop_criteria.update()

        for conv in conversations:
            all_results.append({"prompt": conv, "batch_index": i, **batch_metrics})

    global_metrics = {k: v / current_step for k, v in global_metrics.items()}
    all_results = pd.DataFrame(all_results)
    
    pbar.close()

    return global_metrics, all_results
