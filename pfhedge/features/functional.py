import torch
from torch import Tensor


def volatility(i: int, derivative, hedger=None) -> Tensor:
    value = derivative.ul().volatility
    if isinstance(value, torch.Tensor):
        if not derivative.ul().spot.size() == value.size():
            raise ValueError("spot tensor and volatility tensor are not the same size")
        return value[:, i : i + 1]
    else:
        return torch.full_like(derivative.ul().spot[:, :1], value)


def prev_hedge(i: int, derivative, hedger) -> Tensor:
    if hasattr(hedger, "prev_output"):
        return hedger.prev_output
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


def zeros(i: int, derivative, hedger=None) -> Tensor:
    return torch.zeros_like(derivative.ul().spot[:, :1])


def empty(i: int, derivative, hedger=None) -> Tensor:
    return torch.empty_like(derivative.ul().spot[:, :1])


def max_moneyness(i: int, derivative, hedger=None) -> Tensor:
    m = derivative.ul().spot[..., : i + 1].max(dim=1, keepdim=True).values
    k = derivative.strike
    return m / k


def max_log_moneyness(i: int, derivative, hedger=None) -> Tensor:
    return max_moneyness(i, derivative=derivative).log()
