from typing import List
from typing import Optional
from typing import Type

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float
from pfhedge.instruments.derivative.base import BaseDerivative
from pfhedge.instruments.derivative.base import OptionMixin

from ._base import Feature
from ._base import StateIndependentFeature
from ._getter import FeatureFactory


# for mypy only
class OptionType(BaseDerivative, OptionMixin):
    pass


class Moneyness(StateIndependentFeature):
    """Moneyness of the derivative.

    Moneyness reads :math:`S / K` where
    :math:`S` is the spot price of the underlying instrument and
    :math:`K` is the strike of the derivative.

    Name:
        ``'moneyness'``

    Examples:
        >>> from pfhedge.features import Moneyness
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906]])
        >>> f = Moneyness().of(derivative)
        >>> f.get()
        tensor([[[0.5000],
                 [0.5008],
                 [0.5022],
                 [0.5036],
                 [0.4965],
                 [0.4953]]])
    """

    derivative: OptionType

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "log_moneyness" if self.log else "moneyness"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.moneyness(time_step, log=self.log).unsqueeze(-1)


class LogMoneyness(Moneyness):
    r"""Log-moneyness of the derivative.

    Log-moneyness reads :math:`\log(S / K)` where
    :math:`S` is the spot price of the underlying instrument and
    :math:`K` is the strike of the derivative.

    Name:
        ``'log_moneyness'``
    """

    derivative: OptionType

    def __init__(self) -> None:
        super().__init__(log=True)


class TimeToMaturity(StateIndependentFeature):
    """Remaining time to the maturity of the derivative.

    Name:
        ``'time_to_maturity'``

    Examples:
        >>> from pfhedge.features import Moneyness
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> f = TimeToMaturity().of(derivative)
        >>> f.get()
        tensor([[[0.0200],
                 [0.0160],
                 [0.0120],
                 [0.0080],
                 [0.0040],
                 [0.0000]]])
    """

    derivative: OptionType
    name = "time_to_maturity"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.time_to_maturity(time_step).unsqueeze(-1)


class ExpiryTime(TimeToMaturity):
    """Alias for ``TimeToMaturity``."""

    def __str__(self) -> str:
        return "expiry_time"


class UnderlierSpot(StateIndependentFeature):
    """Spot price of the underlier of the derivative.

    Name:
        ``'underlier_spot'``
    """

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


class UnderlierLogSpot(UnderlierSpot):
    """Logarithm of the spot price of the underlier of the derivative.

    Name:
        ``'underlier_log_spot'``
    """

    def __init__(self):
        super().__init__(log=True)


class Spot(StateIndependentFeature):
    """Spot price of the derivative.

    Name:
        ``'spot'``
    """

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
    """Volatility of the underlier of the derivative.

    Name:
        ``'volatility'``

    Examples:
    """

    name = "volatility"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().volatility[:, index].unsqueeze(-1)


class Variance(StateIndependentFeature):
    """Variance of the underlier of the derivative.

    Name:
        ``'variance'``
    """

    name = "variance"

    def get(self, time_step: Optional[int]) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().variance[:, index].unsqueeze(-1)


class PrevHedge(Feature):
    """Previous holding of underlier.

    Name:
        ``'prev_hedge'``
    """

    hedger: Module
    name = "prev_hedge"

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

    Examples:
        >>> from pfhedge.features import Barrier
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906]])
        >>> f = Barrier(threshold=1.004, up=True).of(derivative)
        >>> f.get()
        tensor([[[0.],
                 [0.],
                 [1.],
                 [1.],
                 [1.],
                 [1.]]])
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
    """A feature filled with the scalar value 0.

    Name:
        ``'zeros'``

    Examples:
        >>> from pfhedge.features import Zeros
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> f = Zeros().of(derivative)
        >>> f.get()
        tensor([[[0.],
                 [0.],
                 [0.],
                 [0.],
                 [0.],
                 [0.]]])
    """

    name = "zeros"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.zeros_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class Ones(StateIndependentFeature):
    """A feature filled with the scalar value 1.

    Name:
        ``'ones'``

    Examples:
        >>> from pfhedge.features import Ones
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> f = Ones().of(derivative)
        >>> f.get()
        tensor([[[1.],
                 [1.],
                 [1.],
                 [1.],
                 [1.],
                 [1.]]])
    """

    name = "ones"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.ones_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class Empty(StateIndependentFeature):
    """A feature filled with uninitialized data.

    Name:
        ``'empty'``

    Examples:
        >>> from pfhedge.features import Empty
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> f = Empty().of(derivative)
        >>> f.get()
        tensor([[[...],
                 [...],
                 [...],
                 [...],
                 [...],
                 [...]]])
    """

    name = "empty"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.empty_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class MaxMoneyness(StateIndependentFeature):
    """Cumulative maximum of moneyness.

    Name:
        ``'max_moneyness'``

    Examples:
        >>> from pfhedge.features import MaxMoneyness
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        ...
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250, strike=2.0)
        >>> derivative.simulate()
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906]])
        >>> f = MaxMoneyness().of(derivative)
        >>> f.get()
        tensor([[[0.5000],
                 [0.5008],
                 [0.5022],
                 [0.5036],
                 [0.5036],
                 [0.5036]]])
    """

    derivative: OptionType

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "max_log_moneyness" if self.log else "max_moneyness"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        return self.derivative.max_moneyness(time_step, log=self.log).unsqueeze(-1)


class MaxLogMoneyness(MaxMoneyness):
    """Cumulative maximum of log Moneyness.

    Name:
        ``'max_log_moneyness'``
    """

    derivative: OptionType

    def __init__(self) -> None:
        super().__init__(log=True)


FEATURES: List[Type[Feature]] = [
    Empty,
    ExpiryTime,
    TimeToMaturity,
    LogMoneyness,
    MaxLogMoneyness,
    MaxMoneyness,
    Moneyness,
    PrevHedge,
    Variance,
    Volatility,
    Zeros,
    Spot,
    UnderlierSpot,
]

for cls in FEATURES:
    FeatureFactory().register_feature(str(cls()), cls)
