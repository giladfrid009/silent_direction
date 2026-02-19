import torch
from tqdm.auto import tqdm

from src.model import TargetedModel
from src.data import TableLoader, TableIterator
from src.activation_extractor import ActivationExtractor
from src.functional import compute_targets_mask
from src.utils.logging import create_logger


logger = create_logger(__name__)


def redundancy_score_principal(proj_var: float, top1_acc: float, top10_agr: float) -> float:
    return proj_var * top1_acc * top10_agr


@torch.no_grad()
def probe_layer_dim(targeted_model: TargetedModel, layer: str) -> int:

    extractor = ActivationExtractor(targeted_model.model, layer)
    input = [[{"role": "user", "content": "Hello, how are you?"}]] if targeted_model.is_chat else ["Hello, how are you?"]
    encodings = targeted_model.tokenize(input)

    with extractor.capture():
        targeted_model.forward(encodings)
        activations = extractor.get_activations()[layer]
        layer_dim = activations.size(-1)

    model_dim = targeted_model.model.config.hidden_size
    if layer_dim != model_dim:
        logger.warning(f"Layer {layer} dimension {layer_dim} does not match model hidden size {model_dim}")
        logger.info(f"Activations shape: {activations.shape}")

    return layer_dim


@torch.no_grad()
def compute_empirical_mean(
    targeted_model: TargetedModel,
    layer: str,
    dl: TableLoader,
    iterations: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the empirical mean of activations at the specified layer over the given conversations.
    The mean is computed over all token positions and all conversations.

    Returns: 
        - mean_activation (torch.Tensor): The empirical mean activation vector (shape: [hidden_size])
        - mean_activation_normalized (torch.Tensor): The empirical mean of the normalized activations (shape: [hidden_size])
    """
    extractor = ActivationExtractor(targeted_model.model, layer)
    mean_act: torch.Tensor = None  # type: ignore
    mean_act_norm: torch.Tensor = None  # type: ignore

    count_total = 0
    iterations = min(iterations, len(dl))
    pbar = tqdm(TableIterator(dl, num_batches=iterations), desc="Computing Empirical Mean", leave=False)

    for batch in pbar:
        conversations = batch["prompt"]
        encodings = targeted_model.tokenize(conversations)
        targets_mask = compute_targets_mask(encodings)

        with extractor.capture():
            _ = targeted_model.forward(encodings)
            acts = extractor.get_activations()[layer]  # Shape: (batch_size, seq_len, hidden_size)
            acts = acts[targets_mask]  # Shape: (num_valid_tokens, hidden_size)
            acts_norm = torch.nn.functional.normalize(acts, dim=-1)

        count_batch = acts.size(0)
        acts_sum = acts.sum(dim=0)  # Shape: (hidden_size,)
        acts_sum_norm = acts_norm.sum(dim=0)  # Shape: (hidden_size,)

        if mean_act is None:
            mean_act = torch.zeros_like(acts_sum)

        if mean_act_norm is None:
            mean_act_norm = torch.zeros_like(acts_sum)

        # update running mean
        # TODO: SHOULD WE JUST REPLACE IT WITH REGULAR SUMMATION AND COMPUTE MEAN AT THE END?
        mean_act = mean_act * (count_total / (count_batch + count_total)) + acts_sum / (count_batch + count_total)
        mean_act_norm = mean_act_norm * (count_total / (count_batch + count_total)) + acts_sum_norm / (count_batch + count_total)
        count_total += count_batch

    return mean_act, mean_act_norm
