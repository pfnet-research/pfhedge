from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module


def prev_hedge(i: Optional[int], derivative, hedger: Module) -> Tensor:
    if i is None:
        raise ValueError("i must be provided for prev_hedge")
    if hasattr(hedger, "prev_output"):
        return hedger.get_buffer("prev_output")
    else:
        # spot: shape (N, T)
        return torch.zeros_like(derivative.ul().spot[:, :1])


def barrier(
    i: int, derivative, hedger=None, threshold: float = 1.0, up: bool = True
) -> Tensor:
    if up:
        # shape: (N, i)
        touch_threshold = derivative.ul().spot[..., : i + 1] >= threshold
    else:
        touch_threshold = derivative.ul().spot[..., : i + 1] <= threshold
    return touch_threshold.any(dim=-1, keepdim=True).to(derivative.ul().spot)
