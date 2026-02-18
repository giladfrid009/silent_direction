import torch
import torch.nn.functional as F
from transformers.tokenization_utils_base import BatchEncoding


def project(activations: torch.Tensor, direction: torch.Tensor, normalize: bool = False) -> torch.Tensor:
    """
    Args:
        activations: tensor of activations, shape (batch_size, seq_len, hidden_size)
        direction: direction vector to project onto, shape (hidden_size,)
        normalize: whether to normalize the direction vector
    """
    if normalize:
        direction = F.normalize(direction, dim=-1)

    coeffs = activations @ direction
    projection = coeffs.unsqueeze(-1).expand_as(activations) * direction
    return projection


def compute_targets_mask(encodings: BatchEncoding) -> torch.Tensor:
    """
    Computes the mask over input tokens that should be considered for loss computation.
    Outputs that correspond to these tokens are considered; others are ignored.
    """
    return encodings.attention_mask.bool()

