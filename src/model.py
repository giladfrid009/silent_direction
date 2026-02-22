from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedModel, PreTrainedTokenizer
from src.aliases import Conv
from src.utils.torch import extract_device
from src.utils.logging import create_logger
import torch


logger = create_logger(__name__)


class TargetedModel:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        is_chat: bool = True,
    ):
        if is_chat and (not hasattr(tokenizer, "chat_template")) or tokenizer.chat_template is None:
            raise ValueError("Tokenizer does not have a chat template, but is_chat=True")

        elif not is_chat and hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            logger.warning("Tokenizer has a chat template, but is_chat=False.")

        self.model = model
        self.tokenizer = tokenizer
        self.is_chat = is_chat
        self.device = extract_device(model)
        self.dtype = self.model.dtype

        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "right"

        # disable gradients for all model parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad_(False)

    def get_hparams(self) -> dict:
        return {
            "model_name": self.model.name_or_path,
            "tokenizer_name": self.tokenizer.name_or_path,
            "is_chat": self.is_chat,
        }

    def tokenize(
        self,
        convs: list[Conv] | list[str],
        max_length: int | None = 512,
        **kwargs,
    ) -> BatchEncoding:

        if len(convs) == 0:
            raise ValueError("No conversations provided for tokenization")

        if self.is_chat and (not isinstance(convs[0], list) or not all(isinstance(c, dict) for c in convs[0])):
            raise ValueError("Convs should be a list of Conv (dict with 'role' and 'content') when is_chat=True")

        if not self.is_chat and not isinstance(convs[0], str):
            raise ValueError("Convs should be a list of strings when is_chat=False")

        if self.is_chat:
            encodings = self.tokenizer.apply_chat_template(
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
        else:
            encodings = self.tokenizer(
                convs,
                padding=True,
                padding_side="left",
                return_dict=True,
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
                **kwargs,
            )

        return encodings.to(self.device)

    def forward(self, encodings: BatchEncoding, **kwargs):
        return self.model(**encodings, **kwargs)

    @torch.inference_mode()
    def generate(
        self,
        prompts: list[Conv] | list[str],
        max_new_tokens: int = 100,
        **kwargs,
    ) -> list[str]:
        """
        Args:
            prompts: List of prompt objects/strings
            max_new_tokens: Maximum tokens to generate per prompt
            **kwargs: Additional generation kwargs to pass to model.generate()
        Returns:
            List of generated texts (one per prompt)
        """
        if len(prompts) == 0:
            logger.warning("No prompts provided for generation")
            return []

        encodings = self.tokenize(prompts, max_length=None)
        input_ids: torch.Tensor = encodings.input_ids  # [B, L]
        attention_mask: torch.Tensor = encodings.attention_mask  # [B, L]

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            **kwargs,
        )  # type: ignore

        return self.tokenizer.batch_decode(
            outputs[:, input_ids.shape[1] :],
            skip_special_tokens=True,
        )
