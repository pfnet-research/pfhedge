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
        raise ValueError("Insufficient parameters to parse `spot`")
