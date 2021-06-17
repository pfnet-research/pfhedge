from torch import Tensor


def parse_spot(
    *,
    spot: Tensor = None,
    strike: Tensor = None,
    moneyness: Tensor = None,
    log_moneyness: Tensor = None,
    **kwargs
):
    if spot is not None:
        return spot
    elif moneyness is not None:
        return moneyness * strike
    elif log_moneyness is not None and strike is not None:
        return log_moneyness.exp() * strike
    else:
        raise ValueError("Insufficient parameters")
