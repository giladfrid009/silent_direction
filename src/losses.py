import math
import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding


def compute_targets_mask(encodings: BatchEncoding) -> torch.Tensor:
    """
    Computes the mask over input tokens that should be considered for loss computation.
    Outputs that correspond to these tokens are considered; others are ignored.
    """
    return encodings.attention_mask


class Loss:
    @staticmethod
    def projection_l2_norm(
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
            direction = F.normalize(direction, dim=-1)

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

    @staticmethod
    def projection_variance(
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
            direction = F.normalize(direction, dim=-1)

        targets_mask = targets_mask.bool()
        activations = activations[targets_mask].view(-1, activations.size(-1))

        return torch.var(activations @ direction, unbiased=True)

    @staticmethod
    def kl_divergence(
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
        log_baselines = F.log_softmax(baseline_logits, dim=-1)
        log_modified = F.log_softmax(modified_logits, dim=-1)

        return F.kl_div(log_modified, log_baselines, reduction="batchmean", log_target=True)

    @staticmethod
    def js_divergence(
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
        p = F.softmax(baseline_logits, dim=-1)
        q = F.softmax(modified_logits, dim=-1)
        m = 0.5 * (p + q)

        kl_pm = F.kl_div(F.log_softmax(baseline_logits, dim=-1), m, reduction="batchmean", log_target=False)
        kl_qm = F.kl_div(F.log_softmax(modified_logits, dim=-1), m, reduction="batchmean", log_target=False)

        js_div = 0.5 * (kl_pm + kl_qm) / math.log(2)

        return js_div
