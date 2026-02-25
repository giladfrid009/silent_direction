import torch
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.model import TargetedModel
from src.data import TableLoader, TableIterator
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.metrics import Metrics
from src.losses import Loss
from src.functional import project, compute_targets_mask
from src.norm.utils import probe_layer_dim, redundancy_score_norm
from src.config import StopCriteria


def train_norm(
    targeted_model: TargetedModel,
    layer: str,
    dl_train: TableLoader,
    stop_criteria: StopCriteria,
    learning_rate: float = 0.01,
    proj_weight: float = 0.1,
    kl_weight: float = 1.0,
) -> tuple[torch.Tensor, list[dict[str, float]]]:

    stop_criteria.reset()
    layer_dim = probe_layer_dim(targeted_model, layer)

    w = torch.randn(layer_dim, device=targeted_model.device, dtype=targeted_model.dtype, requires_grad=True)
    optim = torch.optim.Adam([w], lr=learning_rate)
    sched = None

    if stop_criteria.patience is not None:
        sched = ReduceLROnPlateau(
            optim,
            mode="max",
            factor=0.1,
            patience=int(stop_criteria.patience // 2.5),
            threshold_mode="abs",
            threshold=stop_criteria.patience_delta,
        )

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

        score = redundancy_score_norm(
            proj_norm=proj_l2_rel.item(),
            top1_acc=top1_acc.item(),
            top10_agr=top10_agr.item(),
        )

        if score > best_score:
            best_score = score
            best_direction = v.detach().clone()

        # compute loss and update
        loss = kl_weight * kl_div - proj_weight * proj_l2_rel
        loss.backward()
        optim.step()

        stop_criteria.update(value=score)
        if sched is not None:
            sched.step(score)

        METRICS = {
            "loss": loss.item(),
            "kl_div": kl_div.item(),
            "proj_l2_rel": proj_l2_rel.item(),
            "top1_acc": top1_acc.item(),
            "top10_agr": top10_agr.item(),
            "score": score,
            "learning_rate": sched.get_last_lr()[0] if sched is not None else learning_rate,
            "best_score": best_score,
        }

        # update progress
        history.append(METRICS)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in METRICS.items()})

    pbar.close()

    return best_direction, history
