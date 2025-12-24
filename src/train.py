import torch
import time
from transformers import PreTrainedModel, PreTrainedTokenizer
from math import floor
from tqdm import tqdm

from src.utils.torch import extract_device
from src.activ_extractor import ActivationExtractor
from src.activ_manipulator import ActivationManipulator
from src.model_helpers import tokenize
from src.aliases import Conv

from src.metrics import (
    compute_top_k_token_agreement,
    compute_top_1_accuracy,
    compute_redundancy_score,
)

from src.losses import (
    compute_kl_divergence,
    compute_targets_mask,
    compute_projected_l2_norm,
)

from src.model_helpers import project


def find_redundant_1d_subspace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    conversations: list,
    num_iterations: int = 200,
    lr: float = 0.01,
    projection_weight: float = 0.1,
    batch_size: int = 32,
) -> torch.Tensor:
    """
    Find a redundant 1D subspace: a direction where activations have large projections,
    but removing this projection doesn't significantly change the output tokens.

    This identifies linear subspaces that the model "uses" but that are somewhat redundant
    for the final prediction.

    T is a projection operator: T = v v^T where v is a unit vector (1D projection)
    For activation e, we subtract T(e) = (v^T e) v from e

    We want to find v such that:
    - Top-1 token predictions remain unchanged (high agreement) - PRIMARY OBJECTIVE
    - ||T(e)|| is large for NORMALIZED activations (proportional importance) - SECONDARY

    CRITICAL: v is constrained to be a unit vector (||v|| = 1) throughout optimization.
    This prevents trivially scaling up the projection by multiplying v by a large constant.
    We achieve this using reparametrization: optimize unconstrained w, then use v = w/||w||

    Args:
        model: The language model
        tokenizer: Tokenizer for the model
        layer_name: Name of layer to manipulate
        conversations: List of ALL conversations (will be batched internally)
        num_iterations: Number of optimization steps
        lr: Learning rate
        projection_weight: Weight for projection term (default 0.1, lower = more emphasis on distribution preservation)
        batch_size: Number of conversations per batch

    Returns:
        Optimized direction vector v (unit vector)
    """
    device = extract_device(model)

    manipulator = ActivationManipulator(model, layer_name)
    activ_extractor = ActivationExtractor(model, layer_name)

    with torch.no_grad():
        dummy_input = tokenize([{"role": "user", "content": "Hello"}], tokenizer).to(device)
        with activ_extractor.capture():
            _ = model(**dummy_input)
        activations = activ_extractor.get_activations()[layer_name]
        layer_dim = activations.size(-1)

    if layer_dim != model.config.hidden_size:
        print(f"Warning: Layer {layer_name} dimension {layer_dim} does not match model hidden size {model.config.hidden_size}")

    w = torch.randn(layer_dim, device=device, dtype=model.dtype, requires_grad=True)
    optim = torch.optim.Adam([w], lr=lr)

    num_conversations = len(conversations)
    num_batches = (num_conversations + batch_size - 1) // batch_size

    print(f"Searching for redundant 1D subspace over {num_iterations} iterations...")
    print(f"Using {num_conversations} conversations in {num_batches} batches of size {batch_size}")
    print(f"Loss balance: KL-div weight=1.0, projection weight={projection_weight}")

    best_score = -float("inf")
    best_direction = None

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=w, normalize=True)
        return activations - projection

    for iteration in range(num_iterations):
        batch_indices = torch.randperm(num_conversations)[:batch_size].tolist()
        batch_convs = [conversations[i] for i in batch_indices]
        encodings = tokenize(batch_convs, tokenizer).to(device)

        optim.zero_grad()
        v = torch.nn.functional.normalize(w, dim=0)

        with torch.no_grad(), activ_extractor.capture():
            baseline_logits = model(**encodings).logits.detach()
            activations = activ_extractor.get_activations()[layer_name].detach()

        manipulator.set_manipulation(subtract_projection)
        with manipulator.capture():
            modified_logits = model(**encodings).logits

        targets_mask = compute_targets_mask(encodings)

        # TODO: we might want to compute KL-div over top-k tokens, and not all.
        kl_div = compute_kl_divergence(
            baseline_logits=baseline_logits,
            modified_logits=modified_logits,
            targets_mask=targets_mask,
            top_k=None,
        )

        activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        projection_norm = compute_projected_l2_norm(
            activations=activations_normalized,
            direction=v,
            targets_mask=targets_mask,
            normalize=False,
        )

        loss = kl_div - projection_weight * projection_norm
        loss.backward()
        optim.step()

        top10_agreement = compute_top_k_token_agreement(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
            top_k=10,
        )

        top1_accuracy = compute_top_1_accuracy(
            baseline_logits=baseline_logits.detach(),
            modified_logits=modified_logits.detach(),
            targets_mask=targets_mask,
        )

        score_val = compute_redundancy_score(projection_norm.item(), top1_accuracy.item(), top10_agreement.item())

        if score_val > best_score:
            best_score = score_val
            best_direction = v.detach().clone()

        if iteration % 20 == 0:
            print(
                f"Iter {iteration:3d}: Top-1-Acc={top1_accuracy.item():.4f}, "
                f"Top-10-Agr={top10_agreement.item():.4f}, "
                f"KL-Div={kl_div.item():.6f}, "
                f"Proj-Norm={projection_norm.item():.4f}, "
                f"Score={score_val:.6f}, "
                f"Loss={loss.item():.6f}"
            )

    if best_direction is None:
        best_direction = v.detach().clone()

    print(f"\n✓ Best score: {best_score:.4f}")
    print(f"✓ Final direction norm: {best_direction.norm().item():.6f}")

    return best_direction


@torch.inference_mode()
def validate_redundant_subspace(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    layer_name: str,
    convs: list[Conv],
    redundant_dir: torch.Tensor,
    batch_size: int = 8,
) -> dict[str, float]:
    device = extract_device(model)
    num_samples = len(convs)
    num_batches = (num_samples + batch_size - 1) // batch_size

    all_top1_accuracies = []
    all_top10_agreements = []
    all_kl_divs = []

    proj_raw_sum = 0.0
    proj_norm_sum = 0.0
    total_tokens = 0

    print(f"Validating over {num_samples} samples in {num_batches} batches...")

    manipulator = ActivationManipulator(model, layer_name)
    activ_extractor = ActivationExtractor(model, layer_name)

    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=redundant_dir, normalize=True)
        return activations - projection

    for batch_idx in tqdm(range(num_batches), leave=False, desc="Validation"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)

        batch_convs = convs[start_idx:end_idx]
        encodings = tokenize(batch_convs, tokenizer).to(device)

        with activ_extractor.capture():
            baseline_logits = model(**encodings).logits
            activations = activ_extractor.get_activations()[layer_name]

        manipulator.set_manipulation(subtract_projection)
        with manipulator.capture():
            modified_logits = model(**encodings).logits

        targets_mask = compute_targets_mask(encodings)

        top1_accuracy = compute_top_1_accuracy(baseline_logits, modified_logits, targets_mask)
        top10_agreement = compute_top_k_token_agreement(baseline_logits, modified_logits, targets_mask, top_k=10)
        kl_div = compute_kl_divergence(baseline_logits, modified_logits, targets_mask)

        all_top1_accuracies.append(top1_accuracy.item())
        all_top10_agreements.append(top10_agreement.item())
        all_kl_divs.append(kl_div.item())

        proj_scalars_raw = compute_projected_l2_norm(
            activations=activations,
            direction=redundant_dir,
            targets_mask=targets_mask,
            normalize=True,
            reduction="sum",
        )

        activations_normalized = torch.nn.functional.normalize(activations, dim=-1)

        proj_scalars_normalized = compute_projected_l2_norm(
            activations=activations_normalized,
            direction=redundant_dir,
            targets_mask=targets_mask,
            normalize=True,
            reduction="sum",
        )

        proj_raw_sum += proj_scalars_raw.float().sum().item()
        proj_norm_sum += proj_scalars_normalized.float().sum().item()
        total_tokens += targets_mask.float().sum().item()

    avg_top1_accuracy = sum(all_top1_accuracies) / len(all_top1_accuracies)
    avg_top10_agreement = sum(all_top10_agreements) / len(all_top10_agreements)
    avg_kl_div = sum(all_kl_divs) / len(all_kl_divs)

    projection_mean_raw = proj_raw_sum / total_tokens
    projection_mean_normalized = proj_norm_sum / total_tokens

    return {
        "top1_accuracy": avg_top1_accuracy,
        "top10_agreement": avg_top10_agreement,
        "kl_divergence": avg_kl_div,
        "projection_mean_raw": projection_mean_raw,
        "projection_mean_normalized": projection_mean_normalized,
        "redundancy_score": compute_redundancy_score(projection_mean_normalized, avg_top1_accuracy, avg_top10_agreement),
    }
