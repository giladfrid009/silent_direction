import math
import torch
import torch.nn.functional as F


class Loss:
    @staticmethod
    def l2_norm(
        activations: torch.Tensor,
        targets_mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the L2 norm of activations at target positions.

        Args:
            activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
            targets_mask: mask indicating target positions, shape (batch_size, seq_len)
            reduce: reduction method, either "mean", "sum" or "none"

        Returns:
            L2 norm of the activations.
            - If reduce is "none", returns tensor of norms for each position, of shape (num_target_positions,)
            - If reduce is "mean" or "sum", returns a single scalar tensor.
        """
        assert reduction in {"mean", "sum", "none"}, f"Invalid reduce method: {reduction}"

        activations = activations[targets_mask].view(-1, activations.size(-1))
        norms = torch.norm(activations, dim=-1)

        if reduction == "mean":
            return norms.mean()
        if reduction == "sum":
            return norms.sum()
        if reduction == "none":
            return norms

        raise ValueError(f"Invalid reduce method: {reduction}")

    @staticmethod
    def projection_l2_norm(
        activations: torch.Tensor,
        direction: torch.Tensor,
        targets_mask: torch.Tensor,
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

        direction = F.normalize(direction, dim=-1)
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
    def total_variance(
        activations: torch.Tensor,
        targets_mask: torch.Tensor,
        mean_activation: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the total variance of activations around a given mean activation.
        Variance computed along the hidden_size dimension.

        Args:
            activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
            targets_mask: mask indicating target positions, shape (batch_size, seq_len)
            mean_activation: mean activation vector, shape (hidden_size,)

        Returns:
            variance of the activations
        """
        assert reduction in {"mean", "sum", "none"}, f"Invalid reduce method: {reduction}"

        activations = activations[targets_mask].view(-1, activations.size(-1))
        diffs = activations - mean_activation.unsqueeze(0)

        # total variance is mean squared distance from the mean activation
        var = torch.norm(diffs, dim=-1) ** 2

        if reduction == "mean":
            return var.mean()
        if reduction == "sum":
            return var.sum()
        if reduction == "none":
            return var

        raise ValueError(f"Invalid reduce method: {reduction}")

    @staticmethod
    def projection_total_variance(
        activations: torch.Tensor,
        direction: torch.Tensor,
        targets_mask: torch.Tensor,
        mean_activation: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute the total variance of the projections of activations onto a given direction.

        Args:
            activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
            targets_mask: mask indicating target positions, shape (batch_size, seq_len)
            direction: direction vector to project onto, shape (hidden_size,)
            mean_activation: mean activation vector, shape (hidden_size,)

        Returns:
            variance of the projected activations
        """

        assert reduction in {"mean", "sum", "none"}, f"Invalid reduce method: {reduction}"

        direction = F.normalize(direction, dim=-1)

        # project both activations and mean_activation
        activations = activations[targets_mask].view(-1, activations.size(-1))
        projected_activations = (activations @ direction).unsqueeze(-1) * direction
        projected_mean = (mean_activation @ direction).unsqueeze(-1) * direction

        var = torch.norm(projected_activations - projected_mean, dim=-1) ** 2

        if reduction == "mean":
            return var.mean()
        if reduction == "sum":
            return var.sum()
        if reduction == "none":
            return var

        raise ValueError(f"Invalid reduce method: {reduction}")

    @staticmethod
    def kl_divergence(
        baseline_logits: torch.Tensor,
        modified_logits: torch.Tensor,
        targets_mask: torch.Tensor,
        top_k: int | None = None,
        reduction: str = "batchmean",
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

        return F.kl_div(log_modified, log_baselines, reduction=reduction, log_target=True)

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
