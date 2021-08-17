from typing import Optional

from torch import Tensor
from torch.nn import Module


def save_prev_output(
    module: Module, input: Optional[Tensor], output: Optional[Tensor]
) -> None:
    """A hook to save previous output as a buffer named ``prev_output``.

    Examples:

        >>> import torch
        >>> _ = torch.manual_seed(42)
        >>> m = torch.nn.Linear(3, 2)
        >>> hook = m.register_forward_hook(save_prev_output)
        >>> input = torch.randn(1, 3)
        >>> m(input)
        tensor([[-1.1647,  0.0244]], grad_fn=<AddmmBackward>)
        >>> m.prev_output
        tensor([[-1.1647,  0.0244]], grad_fn=<AddmmBackward>)
    """
    module.register_buffer("prev_output", output, persistent=False)
