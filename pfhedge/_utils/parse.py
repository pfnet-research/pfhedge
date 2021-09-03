from numbers import Real
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
    **kwargs
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
        raise ValueError("Insufficient parameters to parse `spot`")


def parse_moneyness(
    *,
    spot: Optional[Tensor] = None,
    strike: Optional[Tensor] = None,
    moneyness: Optional[Tensor] = None,
    log_moneyness: Optional[Tensor] = None,
    **kwargs
) -> Tensor:
    spot = _as_optional_tensor(spot)
    strike = _as_optional_tensor(strike)
    moneyness = _as_optional_tensor(moneyness)
    log_moneyness = _as_optional_tensor(log_moneyness)

    if moneyness is not None:
        return moneyness
    elif log_moneyness is not None:
        return log_moneyness.exp()
    elif spot is not None and strike is not None:
        return spot / strike
    else:
        raise ValueError("Insufficient parameters to parse `moneyness`")
