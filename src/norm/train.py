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


def train_norm(
    targeted_model: TargetedModel,
    layer: str,
    dl_train: TableLoader,
    stop_criteria: StopCriteria,
    learning_rate: float = 0.01,
    proj_weight: float = 0.1,
) -> tuple[torch.Tensor, list[dict[str, float]]]:

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

        with torch.no_grad(), extractor.capture():
            baseline_logits = targeted_model.forward(encodings).logits.detach()
            activations = extractor.get_activations()[layer].detach()
            activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        with manipulator.capture():
            modified_logits = targeted_model.forward(encodings).logits

        kl_div = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=None,
        )

        proj_norm = Loss.projection_l2_norm(
            activations=activations_normalized,
            direction=v,
            targets_mask=targets_mask,
            reduction="mean",
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

        score_val = Metrics.redundancy_score(
            proj_norm=proj_norm.item(),
            top1_acc=top1_acc,
            top10_agr=top10_agr,
        )

        if score_val > best_score:
            best_score = score_val
            best_direction = v.detach().clone()

        # compute loss and step
        loss = kl_div - proj_weight * proj_norm
        loss.backward()
        optim.step()

        METRICS = {
            "loss": loss.item(),
            "kl_div": kl_div.item(),
            "proj_norm": proj_norm.item(),
            "top1_acc": top1_acc,
            "top10_agr": top10_agr,
            "score": score_val,
        }

        # update progress
        history.append(METRICS)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in METRICS.items()})
        stop_criteria.update(value=score_val)

    pbar.close()

    return best_direction, history
