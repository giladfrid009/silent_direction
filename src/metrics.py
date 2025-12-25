import torch


class Metrics:
    @staticmethod
    def redundancy_score(proj_norm: float, top1_accuracy: float, top10_agreement: float) -> float:
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

    @staticmethod
    @torch.inference_mode()
    def topk_agreement(
        baseline_logits: torch.Tensor,
        modified_logits: torch.Tensor,
        targets_mask: torch.Tensor,
        top_k: int = 10,
    ) -> float:
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

        return torch.tensor(agreements).mean().item()

    @staticmethod
    @torch.inference_mode()
    def top1_accuracy(
        baseline_logits: torch.Tensor,
        modified_logits: torch.Tensor,
        targets_mask: torch.Tensor,
    ) -> float:
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

        return accuracy.item()
