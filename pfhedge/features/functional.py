from typing import Optional

from torch import Tensor


def barrier(
    i: Optional[int], derivative, hedger=None, threshold: float = 1.0, up: bool = True
) -> Tensor:
    if i is None:
        raise ValueError("not supported")
    if up:
        # shape: (N, i)
        touch_threshold = derivative.ul().spot[..., : i + 1] >= threshold
    else:
        touch_threshold = derivative.ul().spot[..., : i + 1] <= threshold
    return touch_threshold.any(dim=-1, keepdim=True).to(derivative.ul().spot)
