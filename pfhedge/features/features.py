from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float
from pfhedge.instruments.derivative.base import BaseOption

from ._base import Feature
from ._base import StateIndependentFeature


class Moneyness(StateIndependentFeature):
    """Moneyness of the underlying instrument of the derivative.

    Args:
        log (bool, default=False): If ``True``, represents log moneyness.
    """

    derivative: BaseOption

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "log_moneyness" if self.log else "moneyness"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.moneyness(time_step, log=self.log).unsqueeze(-1)


class LogMoneyness(Moneyness):
    """Log moneyness of the underlying instrument of the derivative."""

    derivative: BaseOption

    def __init__(self) -> None:
        super().__init__(log=True)


class TimeToMaturity(StateIndependentFeature):
    """Remaining time to the maturity of the derivative."""

    derivative: BaseOption

    def __str__(self) -> str:
        return "time_to_maturity"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.time_to_maturity(time_step).unsqueeze(-1)


class ExpiryTime(TimeToMaturity):
    """Alias for ``TimeToMaturity``."""

    def __str__(self) -> str:
        return "expiry_time"


class UnderlierSpot(StateIndependentFeature):
    """Spot of the underlier of the derivative."""

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "underlier_log_spot" if self.log else "underlier_spot"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        output = self.derivative.ul().spot[:, index].unsqueeze(-1)
        if self.log:
            output.log_()
        return output


class Spot(StateIndependentFeature):
    """Spot of the derivative."""

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "log_spot" if self.log else "spot"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        output = self.derivative.spot[:, index].unsqueeze(-1)
        if self.log:
            output.log_()
        return output


class Volatility(StateIndependentFeature):
    """Volatility of the underlier of the derivative."""

    def __str__(self) -> str:
        return "volatility"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().volatility[:, index].unsqueeze(-1)


class Variance(StateIndependentFeature):
    """Variance of the underlier of the derivative."""

    def __str__(self) -> str:
        return "variance"

    def get(self, time_step: Optional[int]) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().variance[:, index].unsqueeze(-1)


class PrevHedge(Feature):
    """Previous holding of underlier."""

    hedger: Module

    def __str__(self) -> str:
        return "prev_hedge"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        if time_step is None:
            raise ValueError("time_step for prev_output should be specified")
        return self.hedger.get_buffer("prev_output")


class Barrier(StateIndependentFeature):
    """A feature which signifies whether the price of the underlier have reached
    the barrier. The output 1.0 means that the price have touched the barrier,
    and 0.0 otherwise.

    Args:
        threshold (float): The price level of the barrier.
        up (bool, default True): If ``True``, signifies whether the price has exceeded
            the barrier upward.
            If ``False``, signifies whether the price has exceeded the barrier downward.
    """

    def __init__(self, threshold: float, up: bool = True) -> None:
        super().__init__()
        self.threshold = threshold
        self.up = up

    def __repr__(self) -> str:
        params = [_format_float(self.threshold), "up=" + str(self.up)]
        return self._get_name() + "(" + ", ".join(params) + ")"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        spot = self.derivative.ul().spot
        if time_step is None:
            if self.up:
                max = spot.cummax(-1).values
                output = (max >= self.threshold).to(spot.dtype)
            else:
                min = spot.cummin(-1).values
                output = (min <= self.threshold).to(spot.dtype)
        else:
            if self.up:
                max = spot[..., : time_step + 1].max(-1, keepdim=True).values
                output = (max >= self.threshold).to(spot.dtype)
            else:
                min = spot[..., : time_step + 1].min(-1, keepdim=True).values
                output = (min <= self.threshold).to(self.derivative.ul().spot.dtype)
        return output.unsqueeze(-1)


class Zeros(StateIndependentFeature):
    """A feature of which value is always zero."""

    def __str__(self) -> str:
        return "zeros"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.zeros_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class Empty(StateIndependentFeature):
    """A feature of which value is always empty."""

    def __str__(self) -> str:
        return "empty"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.empty_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class MaxMoneyness(StateIndependentFeature):
    """Cumulative maximum of moneyness.

    Args:
        log (bool, default=False): If ``True``, represents log moneyness.
    """

    derivative: BaseOption

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "max_log_moneyness" if self.log else "max_moneyness"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.max_moneyness(time_step, log=self.log).unsqueeze(-1)


class MaxLogMoneyness(MaxMoneyness):
    """Cumulative maximum of log Moneyness."""

    derivative: BaseOption

    def __init__(self) -> None:
        super().__init__(log=True)
