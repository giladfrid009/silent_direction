from src.utils.huggingface import load_hf_model
from transformers import PreTrainedModel, PreTrainedTokenizer  # pyright: ignore[reportPrivateImportUsage]
import torch
from src.utils.logging import create_logger


logger = create_logger(__name__)


SUPPORTED_MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-7b",
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B-Instruct",
    "meta-llama/Llama-3.1-70B",
    "google/gemma-2b-it",
    "google/gemma-2b",
    "google/gemma-7b-it",
    "google/gemma-7b",
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
    "google/gemma-2-9b-it",
    "google/gemma-2-9b",
    "google/gemma-2-27b-it",
    "google/gemma-2-27b",
    "google/gemma-3-270m-it",
    "google/gemma-3-270m",
    "google/gemma-3-1b-it",
    "google/gemma-3-1b",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen1.5-7B",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-72B-Instruct",
    "Qwen/Qwen2.5-72B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-8B-Base",
    "microsoft/phi-1_5",
    "microsoft/Phi-3-mini-4k-instruct",
    "microsoft/Phi-3-medium-4k-instruct",
    "microsoft/Phi-4-mini-instruct",
]


def load_model(
    model_name: str,
    dtype: torch.dtype | str = "auto",
    device_map: str = "cuda:0",
    hf_token: str | None = None,
    **kwargs,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    if kwargs is None:
        kwargs = {}

    config = kwargs.copy()

    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' is not supported. Supported models are: {SUPPORTED_MODELS}")

    # Model-Dependent configurations

    config.setdefault("model_name", model_name)
    config.setdefault("device_map", device_map)
    config.setdefault("dtype", dtype)
    config.setdefault("hf_token", hf_token)

    if model_name == "meta-llama/Llama-2-7b":
        config["dtype"] = torch.bfloat16  # doesn't train otherwise at all

    if model_name == "meta-llama/Llama-2-7b-chat-hf":
        config["dtype"] = torch.bfloat16  # doesn't train otherwise at all

    if model_name == "microsoft/phi-1_5":
        config["dtype"] = torch.bfloat16  # TODO: test.
        pass

    model, tokenizer = load_hf_model(**config)

    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    logger.info("Disabled gradients for all model parameters and set model to eval mode.")

    return model, tokenizer
