import torch


class RunningAverage:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
        self.sum = 0.0

    def update(self, value: float) -> float:
        self.values.append(value)
        self.sum += value
        if len(self.values) > self.window_size:
            removed = self.values.pop(0)
            self.sum -= removed
            
        return self.average()

    def average(self) -> float:
        if not self.values:
            raise ValueError("No values to average")
        return self.sum / len(self.values)


class Metrics:
    @staticmethod
    @torch.inference_mode()
    def topk_agreement(
        baseline_logits: torch.Tensor,
        modified_logits: torch.Tensor,
        targets_mask: torch.Tensor,
        top_k: int = 10,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute top-k token agreement between two sets of logits, over all non-padding tokens.

        This measures how often the top-k predicted tokens are the same between two distributions.
        A better metric than KL-div for our purpose since we care about which tokens are predicted,
        not the exact probability distribution.

        Args:
            baseline_logits: (batch_size, seq_len, vocab_size) logits from the original model
            modified_logits: (batch_size, seq_len, vocab_size) logits from the modified model
            targets_mask: (batch_size, seq_len) boolean mask indicating which positions are targets (e.g. non-padding tokens)
            top_k: how many top tokens to consider for agreement
            reduction: how to reduce the agreement scores across positions ("mean", "sum", "none", "samplemean", "samplesum")

        Returns:
            Top-k agreement score (1.0 = perfect agreement, 0.0 = no overlap)
            - If reduction is "none", returns tensor of agreement scores for each position, of shape (num_target_positions,)
            - If reduction is "mean" or "sum", returns a single scalar tensor.
            - If reduction is "samplemean" or "samplesum", returns mean or sum agreement score per sample (averaged over target positions), of shape (batch_size,)
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

        agreements_tensor = torch.tensor(agreements, device=targets_mask.device)

        if reduction == "mean":
            return agreements_tensor.mean()

        if reduction == "sum":
            return agreements_tensor.sum()

        if reduction == "none":
            return agreements_tensor

        if reduction == "samplemean":
            result = torch.zeros_like(targets_mask, dtype=agreements_tensor.dtype)
            result[targets_mask] = agreements_tensor
            return result.sum(dim=-1) / targets_mask.sum(dim=-1).clamp(min=1)

        if reduction == "samplesum":
            result = torch.zeros_like(targets_mask, dtype=agreements_tensor.dtype)
            result[targets_mask] = agreements_tensor
            return result.sum(dim=-1)

        raise ValueError(f"Invalid reduction method: {reduction}")

    @staticmethod
    @torch.inference_mode()
    def top1_accuracy(
        baseline_logits: torch.Tensor,
        modified_logits: torch.Tensor,
        targets_mask: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """
        Compute top-1 token accuracy: how often is the most likely token the same?
        Over all non-padding tokens.

        Args:
            baseline_logits: (batch_size, seq_len, vocab_size) logits from the original model
            modified_logits: (batch_size, seq_len, vocab_size) logits from the modified model
            targets_mask: (batch_size, seq_len) boolean mask indicating which positions are targets (e.g. non-padding tokens)
            reduction: how to reduce the accuracy scores across positions ("mean", "sum", "none", "samplemean", "samplesum")

        Returns:
            Top-1 accuracy score between the two logits.
            - If reduction is "none", returns tensor of accuracy scores for each position, of shape (num_target_positions,)
            - If reduction is "mean" or "sum", returns a single scalar tensor.
            - If reduction is "samplemean" or "samplesum", returns mean or sum accuracy score per sample (averaged over target positions), of shape (batch_size,)
        """
        # Extract only non-padding tokens
        targets_mask = targets_mask.bool()
        baseline_logits = baseline_logits[targets_mask].view(-1, baseline_logits.size(-1))
        modified_logits = modified_logits[targets_mask].view(-1, modified_logits.size(-1))

        # Get top-1 tokens
        top1_baseline = torch.argmax(baseline_logits, dim=-1)
        top1_modified = torch.argmax(modified_logits, dim=-1)

        # Compute accuracy
        accuracy = (top1_baseline == top1_modified).float()

        if reduction == "mean":
            return accuracy.mean()

        if reduction == "sum":
            return accuracy.sum()

        if reduction == "none":
            return accuracy

        if reduction == "samplemean":
            result = torch.zeros_like(targets_mask, dtype=accuracy.dtype)
            result[targets_mask] = accuracy
            return result.sum(dim=-1) / targets_mask.sum(dim=-1).clamp(min=1)

        if reduction == "samplesum":
            result = torch.zeros_like(targets_mask, dtype=accuracy.dtype)
            result[targets_mask] = accuracy
            return result.sum(dim=-1)

        raise ValueError(f"Invalid reduction method: {reduction}")
