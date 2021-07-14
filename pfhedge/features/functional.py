import torch
from torch import Tensor
from torch.nn import Module


def volatility(i, derivative, hedger=None) -> Tensor:
    value = derivative.underlier.volatility
    return torch.full_like(derivative.underlier.spot[:, :1], value)


def prev_hedge(i, derivative, hedger) -> Tensor:
    if hasattr(hedger, "prev_output"):
        return hedger.prev_output
    else:
        # spot: shape (N, T)
        return torch.zeros_like(derivative.underlier.spot[:, :1])


def barrier(
    i, derivative, hedger=None, threshold: float = 1.0, up: bool = True
) -> Tensor:
    if up:
        # shape: (N, i)
        touch_threshold = derivative.underlier.spot[..., : i + 1] >= threshold
    else:
        touch_threshold = derivative.underlier.spot[..., : i + 1] <= threshold
    return touch_threshold.any(dim=-1, keepdim=True).to(derivative.underlier.spot)


def zeros(i, derivative, hedger=None) -> Tensor:
    return torch.zeros_like(derivative.underlier.spot[:, :1])


def empty(i, derivative, hedger=None) -> Tensor:
    return torch.empty_like(derivative.underlier.spot[:, :1])


def max_moneyness(i, derivative, hedger=None) -> Tensor:
    m = derivative.underlier.spot[..., : i + 1].max(dim=1, keepdim=True).values
    k = derivative.strike
    return m / k


def max_log_moneyness(i, derivative, hedger=None) -> Tensor:
    return max_moneyness(i, derivative=derivative).log()
