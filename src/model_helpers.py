from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.aliases import Conv
import torch
import torch.nn.functional as F

from src.activation_extractor import ActivationManipulator
from src.utils.torch import extract_device


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


def tokenize(convs: list[Conv], tokenizer: PreTrainedTokenizer, max_length: int = 512, **kwargs) -> BatchEncoding:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"
    return tokenizer.apply_chat_template(
        convs,
        add_generation_prompt=True,
        padding=True,
        padding_side="left",
        return_dict=True,
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
        enable_thinking=False,
        **kwargs,
    )  # type: ignore


@torch.inference_mode()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[Conv],
    max_new_tokens: int = 100,
    **kwargs,
) -> list[str]:
    """
    Generate text from prompts using the model and tokenizer.

    Args:
        model: Language model
        tokenizer: Tokenizer
        prompts: List of prompt strings
        max_new_tokens: Maximum tokens to generate
    Returns:
        List of generated texts
    """
    device = extract_device(model)

    # Generate
    generated_texts = []

    for conv in prompts:
        # Tokenize prompt only (no target)
        tokens = tokenize([conv], tokenizer).to(device)
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask

        # Generate with or without ablation
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            **kwargs,
        )

        # Decode (skip prompt tokens)
        generated_ids = outputs[0][input_ids.shape[1] :]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text)

    return generated_texts


@torch.inference_mode()
def generate_ablated(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    prompts: list[Conv],
    layer_name: str,
    direction: torch.Tensor,
    max_new_tokens: int = 100,
    **kwargs,
) -> list[str]:
    def subtract_projection(activations: torch.Tensor) -> torch.Tensor:
        projection = project(activations, direction=direction, normalize=True)
        return activations - projection

    manipulator = ActivationManipulator(model, layer_name, manipulation_fn=subtract_projection)

    with manipulator.capture():
        return generate(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
