from numbers import Real
from typing import Any
from typing import Optional
from typing import Union

import torch
from torch import Tensor


def _as_optional_tensor(input: Optional[Union[Tensor, Real]]) -> Optional[Tensor]:
    return torch.as_tensor(input) if input is not None else input


def parse_spot(
    *,
    spot: Optional[Tensor] = None,
    strike: Optional[Tensor] = None,
    moneyness: Optional[Tensor] = None,
    log_moneyness: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    spot = _as_optional_tensor(spot)
    strike = _as_optional_tensor(strike)
    moneyness = _as_optional_tensor(moneyness)
    log_moneyness = _as_optional_tensor(log_moneyness)

    if spot is not None:
        return spot
    elif moneyness is not None and strike is not None:
        return moneyness * strike
    elif log_moneyness is not None and strike is not None:
        return log_moneyness.exp() * strike
    else:
        raise ValueError("Insufficient parameters to parse spot")


def parse_volatility(
    *,
    volatility: Optional[Tensor] = None,
    variance: Optional[Tensor] = None,
    **kwargs: Any,
) -> Tensor:
    if volatility is not None:
        return volatility
    elif variance is not None:
        return variance.clamp(min=0.0).sqrt()
    else:
        raise ValueError("Insufficient parameters to parse volatility")


def parse_time_to_maturity(
    *, time_to_maturity: Optional[Tensor] = None, **kwargs: Any
) -> Tensor:
    if time_to_maturity is not None:
        return time_to_maturity
    else:
        raise ValueError("Insufficient parameters to parse time_to_maturity")
