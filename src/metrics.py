import math
import torch


def compute_redundancy_score(proj_norm: float, top1_accuracy: float, top10_agreement: float) -> float:
    """
    Compute redundancy score for a direction.

    This is the canonical metric used everywhere to evaluate how good a redundant direction is.
    Higher score = better redundant direction (high projection, high preservation of outputs).

    Args:
        projection_norm: Mean normalized projection magnitude (0-1 range)
        top1_accuracy: Fraction of positions where top-1 token matches (0-1 range)
        top10_agreement: Fraction of top-10 tokens that overlap (0-1 range)

    Returns:
        Redundancy score (higher is better)
    """
    return proj_norm * top1_accuracy * top10_agreement


def project(activations: torch.Tensor, direction: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Args:
        activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
        direction: direction vector to project onto, shape (hidden_size,)
        normalize: whether to normalize the direction vector
    """
    if normalize:
        direction = torch.nn.functional.normalize(direction, dim=-1)

    coeffs = activations @ direction
    projection = coeffs.unsqueeze(-1).expand_as(activations) * direction 
    return projection


def compute_projected_l2_norm(
    activations: torch.Tensor,
    direction: torch.Tensor,
    targets_mask: torch.Tensor,
    normalize: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute the L2 norm of the projection of activations onto a given direction.

    Args:
        activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
        targets_mask: mask indicating target positions, shape (batch_size, seq_len)
        direction: direction vector to project onto, shape (hidden_size,)
        reduce: reduction method, either "mean", "sum" or "none"

    Returns:
        L2 norm of the projected activations. 
        - If reduce is "none", returns tensor of norms for each position, of shape (num_target_positions,)
        - If reduce is "mean" or "sum", returns a single scalar tensor.
    """
    assert reduction in {"mean", "sum", "none"}, f"Invalid reduce method: {reduction}"
    
    if normalize:
        direction = torch.nn.functional.normalize(direction, dim=-1)

    targets_mask = targets_mask.bool()
    activations = activations[targets_mask].view(-1, activations.size(-1))

    norms = torch.abs(activations @ direction)

    if reduction == "mean":
        return norms.mean()
    if reduction == "sum":
        return norms.sum()
    if reduction == "none":
        return norms

    raise ValueError(f"Invalid reduce method: {reduction}")


def compute_projected_variance(
    activations: torch.Tensor,
    direction: torch.Tensor,
    targets_mask: torch.Tensor,
    normalize: bool = False,
) -> torch.Tensor:
    """
    Compute the variance of the projection of activations onto a given direction.
    Variance computed along the

    Args:
        activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
        direction: direction vector to project onto, shape (hidden_size,)

    Returns:
        variance of the projected activations
    """    
    if normalize:
        direction = torch.nn.functional.normalize(direction, dim=-1)

    targets_mask = targets_mask.bool()
    activations = activations[targets_mask].view(-1, activations.size(-1))

    return torch.var(activations @ direction, unbiased=True)


@torch.inference_mode()
def compute_top_k_token_agreement(
    baseline_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    targets_mask: torch.Tensor,
    top_k: int = 10,
) -> torch.Tensor:
    """
    Compute top-k token agreement between two sets of logits, over all non-padding tokens.

    This measures how often the top-k predicted tokens are the same between two distributions.
    A better metric than KL-div for our purpose since we care about which tokens are predicted,
    not the exact probability distribution.

    Returns:
        Average top-k agreement score (1.0 = perfect agreement, 0.0 = no overlap)
    """
    # Extract only non-padding targets
    targets_mask = targets_mask.bool()
    baseline_logits = baseline_logits[targets_mask].view(-1, baseline_logits.size(-1))
    modified_logits = modified_logits[targets_mask].view(-1, modified_logits.size(-1))

    # Get top-k tokens for each position
    topk_baseline = torch.topk(baseline_logits, top_k, dim=-1).indices  # (num_tokens, k)
    topk_modified = torch.topk(modified_logits, top_k, dim=-1).indices  # (num_tokens, k)

    # Compute overlap: how many of top-k tokens are the same
    # For each position, count how many tokens appear in both top-k lists
    agreements = []
    for i in range(topk_baseline.shape[0]):
        overlap = len(set(topk_baseline[i].tolist()) & set(topk_modified[i].tolist()))
        agreements.append(overlap / top_k)

    return torch.tensor(agreements).mean()


@torch.inference_mode()
def compute_top_1_accuracy(
    baseline_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    targets_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute top-1 token accuracy: how often is the most likely token the same?
    Over all non-padding tokens.

    Returns:
        Fraction of positions where top-1 token matches
    """
    # Extract only non-padding tokens
    targets_mask = targets_mask.bool()
    baseline_logits = baseline_logits[targets_mask].view(-1, baseline_logits.size(-1))
    modified_logits = modified_logits[targets_mask].view(-1, modified_logits.size(-1))

    # Get top-1 tokens
    top1_baseline = torch.argmax(baseline_logits, dim=-1)
    top1_modified = torch.argmax(modified_logits, dim=-1)

    # Compute accuracy
    accuracy = (top1_baseline == top1_modified).float().mean()

    return accuracy


# TODO: implement top_k here.
def compute_kl_divergence(
    baseline_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    targets_mask: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Compute KL divergence between two logit distributions over all non-padding tokens.

    Returns:
        Mean KL divergence across all targets positions
    """
    if top_k is not None:
        # select only top_k logits according to the baseline_logits and compute over that
        topk_values, topk_indices = torch.topk(baseline_logits, top_k, dim=-1)
        baseline_logits = topk_values
        modified_logits = torch.gather(modified_logits, -1, topk_indices)

    # Extract only non-padding tokens
    targets_mask = targets_mask.bool()
    baseline_logits = baseline_logits[targets_mask].view(-1, baseline_logits.size(-1))
    modified_logits = modified_logits[targets_mask].view(-1, modified_logits.size(-1))

    # Convert to log probabilities
    log_baselines = torch.nn.functional.log_softmax(baseline_logits, dim=-1)
    log_modified = torch.nn.functional.log_softmax(modified_logits, dim=-1)

    kl_div = torch.nn.functional.kl_div(log_modified, log_baselines, reduction="batchmean", log_target=True)

    return kl_div


def compute_js_divergence(
    baseline_logits: torch.Tensor,
    modified_logits: torch.Tensor,
    targets_mask: torch.Tensor,
    top_k: int | None = None,
) -> torch.Tensor:
    """
    Compute Jensen-Shannon divergence between two logit distributions over all non-padding tokens.
    Symmetric and bounded between 0 and 1.

    Returns:
        Mean JS divergence across all targets positions
    """
    if top_k is not None:
        # select only top_k logits according to the baseline_logits and compute over that
        topk_values, topk_indices = torch.topk(baseline_logits, top_k, dim=-1)
        baseline_logits = topk_values
        modified_logits = torch.gather(modified_logits, -1, topk_indices)

    # Extract only non-padding tokens
    targets_mask = targets_mask.bool()
    baseline_logits = baseline_logits[targets_mask].view(-1, baseline_logits.size(-1))
    modified_logits = modified_logits[targets_mask].view(-1, modified_logits.size(-1))

    # Convert to probabilities
    p = torch.nn.functional.softmax(baseline_logits, dim=-1)
    q = torch.nn.functional.softmax(modified_logits, dim=-1)
    m = 0.5 * (p + q)

    kl_pm = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(baseline_logits, dim=-1), m, reduction="batchmean", log_target=False)
    kl_qm = torch.nn.functional.kl_div(torch.nn.functional.log_softmax(modified_logits, dim=-1), m, reduction="batchmean", log_target=False)

    js_div = 0.5 * (kl_pm + kl_qm) / math.log(2)

    return js_div
