import torch
from tqdm.auto import tqdm

from src.model import TargetedModel
from src.data import TableLoader, TableIterator
from src.activation_extractor import ActivationExtractor, ActivationManipulator
from src.metrics import Metrics
from src.losses import Loss
from src.functional import project, compute_targets_mask
from src.principal.utils import probe_layer_dim, compute_empirical_mean


def find_redundant_principle_subspace(
    targeted_model: TargetedModel,
    layer: str,
    dl_train: TableLoader,
    num_steps: int = 200,
    learning_rate: float = 0.01,
    proj_weight: float = 0.1,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """
    Returns:
        (torch.Tensor): The redundant direction found (unit norm).
    """
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

    _, mean_activ = compute_empirical_mean(
        model=targeted_model,
        layer=layer,
        dl=dl_train,
        iterations=100,
    )
    
    num_steps = min(num_steps, len(dl_train))
    pbar = tqdm(TableIterator(dl_train, num_batches=num_steps), desc="Training", leave=False)
    history = []

    for batch in pbar:
        conversations = batch["prompt"]
        encodings = targeted_model.tokenize(conversations)
        targets_mask = compute_targets_mask(encodings)

        optim.zero_grad()
        v = torch.nn.functional.normalize(w, dim=0)

        with torch.no_grad(), extractor.capture():
            baseline_logits = targeted_model.forward(encodings).logits.detach()
            activations = extractor.get_activations()[layer].detach()
            activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        # manipulator.set_manipulation(subtract_projection)
        with manipulator.capture():
            modified_logits = targeted_model.forward(encodings).logits

        kl_div = Loss.kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=None,
        )

        proj_var = Loss.projection_total_variance(
            activations=activations_normalized,
            direction=v,
            targets_mask=targets_mask,
            mean_activation=mean_activ,
        )

        top10_agreement = Metrics.topk_agreement(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
            top_k=10,
        )

        top1_accuracy = Metrics.top1_accuracy(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
        )

        score_val = Metrics.redundancy_score(
            proj_norm=proj_var.item(),
            top1_acc=top1_accuracy,
            top10_agr=top10_agreement,
        )

        if score_val > best_score:
            best_score = score_val
            best_direction = v.detach().clone()

        loss = kl_div - proj_weight * proj_var
        loss.backward()
        optim.step()
        
        METRICS = {
            "loss": loss.item(),
            "kl_div": kl_div.item(),
            "proj_var": proj_var.item(),
            "top1_acc": top1_accuracy,
            "top10_agr": top10_agreement,
            "redundancy_score": score_val,
        }

        history.append(METRICS)
        pbar.set_postfix({k: f"{v:.4f}" for k, v in METRICS.items()})

    pbar.close()

    if best_direction is None:
        best_direction = v.detach().clone()

    return best_direction, history
