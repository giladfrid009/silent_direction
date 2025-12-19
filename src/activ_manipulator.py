import torch
from torch import nn, Tensor
from typing import Optional, Callable
from src.activ_extractor import ActivationExtractor
from src.utils.logging import create_logger


logger = create_logger(__name__)


class ActivationManipulator(ActivationExtractor):
    """
    Extends ActivationExtractor to manipulate activations during forward pass.
    Manipulates the OUTPUT of the target layer (same as ActivationExtractor captures outputs).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        *layer_specs: str | type,
        exact_match: bool = True,
        manipulation_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        Args:
            model: PyTorch model
            layer_name: Name of the layer to manipulate
            manipulation_fn: Function that takes activations and returns modified activations
        """
        # Force capture_output=True to ensure we're working with outputs
        super().__init__(model, *layer_specs, exact_match=exact_match, capture_output=True)
        self.manipulation_fn = manipulation_fn

    def set_manipulation(self, manipulation_fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
        """Set the manipulation function to apply to activations."""
        self.manipulation_fn = manipulation_fn

    def _create_output_hook(self, layer_name: str):
        """Create a forward hook to capture layer output."""

        def hook_fn(module: nn.Module, args: tuple[Tensor, ...], output: tuple[Tensor, ...] | Tensor):
            if isinstance(output, tuple):
                logger.debug(f"[Layer {layer_name}]: Output is a tuple; using the first element.")
                new_output = output[0]

                if not isinstance(new_output, torch.Tensor):
                    raise ValueError(f"[Layer {layer_name}]: Expected output to be a Tensor, got {type(output)}")

                if self.manipulation_fn is not None:
                    new_output = self.manipulation_fn(new_output)

                self._activations[layer_name] = new_output

                return (new_output,) + output[1:]

            else:
                if not isinstance(output, torch.Tensor):
                    raise ValueError(f"[Layer {layer_name}]: Expected output to be a Tensor, got {type(output)}")

                if self.manipulation_fn is not None:
                    output = self.manipulation_fn(output)

                self._activations[layer_name] = output
                return output

        return hook_fn

    def _create_input_hook(self, layer_name: str):
        """Not supported."""
        raise NotImplementedError("ActivationManipulator only supports output manipulation.")
