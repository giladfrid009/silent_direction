from src.activation_extractor import ActivationExtractor
from src.model import TargetedModel
from src.utils.logging import create_logger
import torch


logger = create_logger(__name__)


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
