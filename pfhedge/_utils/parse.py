from typing import Optional

from torch import Tensor


def parse_spot(
    *,
    spot: Optional[Tensor] = None,
    strike: Optional[Tensor] = None,
    moneyness: Optional[Tensor] = None,
    log_moneyness: Optional[Tensor] = None,
    **kwargs
) -> Tensor:
    if spot is not None:
        return spot
    elif moneyness is not None and strike is not None:
        return moneyness * strike
    elif log_moneyness is not None and strike is not None:
        return log_moneyness.exp() * strike
    else:
        raise ValueError("Insufficient parameters to parse spot")


def parse_volatility(
    *, volatility: Optional[Tensor] = None, variance: Optional[Tensor] = None, **kwargs
) -> Tensor:
    if volatility is not None:
        return volatility
    elif variance is not None:
        return variance.clamp(min=0.0).sqrt()
    else:
        raise ValueError("Insufficient parameters to parse volatility")
