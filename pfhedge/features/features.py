from typing import List
from typing import Optional

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float
from pfhedge.instruments.derivative.base import BaseOption

from ._base import Feature
from ._base import StateIndependentFeature
from .functional import barrier


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

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        return self.derivative.moneyness(time_step, log=self.log).unsqueeze(-1)


class LogMoneyness(Moneyness):
    """Log moneyness of the underlying instrument of the derivative."""

    derivative: BaseOption

    def __init__(self) -> None:
        super().__init__(log=True)


class ExpiryTime(StateIndependentFeature):
    """Remaining time to the maturity of the derivative."""

    derivative: BaseOption

    def __str__(self) -> str:
        return "expiry_time"

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        return self.derivative.time_to_maturity(time_step).unsqueeze(-1)


class Volatility(StateIndependentFeature):
    """Volatility of the underlier of the derivative."""

    def __str__(self) -> str:
        return "volatility"

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        index = [time_step] if isinstance(time_step, int) else ...
        return self.derivative.ul().volatility[:, index].unsqueeze(-1)


class PrevHedge(Feature):
    """Previous holding of underlier."""

    hedger: Module

    def __str__(self) -> str:
        return "prev_hedge"

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
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

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        return barrier(
            time_step, derivative=self.derivative, threshold=self.threshold, up=self.up
        ).unsqueeze(-1)


class Zeros(StateIndependentFeature):
    """A feature of which value is always zero."""

    def __str__(self) -> str:
        return "zeros"

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        index = [time_step] if time_step is not None else ...
        return torch.zeros_like(self.derivative.ul().spot[..., index]).unsqueeze(-1)


class Empty(StateIndependentFeature):
    """A feature of which value is always empty."""

    def __str__(self) -> str:
        return "empty"

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
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

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        return self.derivative.max_moneyness(time_step, log=self.log).unsqueeze(-1)


class MaxLogMoneyness(MaxMoneyness):
    """Cumulative maximum of log Moneyness."""

    derivative: BaseOption

    def __init__(self) -> None:
        super().__init__(log=True)


class ModuleOutput(Feature, Module):
    """The feature computed as an output of a :class:`torch.nn.Module`.

    Args:
        module (torch.nn.Module): Module to compute the value of the feature.
            The input and output shapes should be :math:`(N, *, H_in) -> (N, *, 1)`,
            where :math:`N` stands for the number of Monte Carlo paths of
            the underlier of the derivative,
            :math:`H_in` stands for the number of input features
            (namely, ``len(inputs)``),
            and :math:`*` means any number of additional dimensions.
        inputs (list[Feature]): The input features to the module.

    Examples:

        >>> from torch.nn import Linear
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.simulate(n_paths=3)
        >>>
        >>> m = Linear(2, 1)
        >>> f = ModuleOutput(m, [Moneyness(), ExpiryTime()]).of(derivative)
        >>> f[0].size()
        torch.Size([3, 1, 1])
        >>> f
        ModuleOutput(
          inputs=['moneyness', 'expiry_time']
          (module): Linear(in_features=2, out_features=1, bias=True)
        )

        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.simulate(n_paths=3)
        >>> m = BlackScholes(derivative)
        >>> f = ModuleOutput(m, [LogMoneyness(), ExpiryTime(), Volatility()])
        >>> f = f.of(derivative)
        >>> f[0].size()
        torch.Size([3, 1, 1])
    """

    module: Module

    def __init__(self, module: Module, inputs: List[Feature]) -> None:
        super(Module, self).__init__()
        super(Feature, self).__init__()

        self.add_module("module", module)
        self.inputs = inputs

    def extra_repr(self) -> str:
        return "inputs=" + str(list(map(str, self.inputs)))

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input)

    def _get_input(self, time_step: Optional[int]) -> Tensor:
        return torch.cat([f[time_step] for f in self.inputs], -1)

    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        return self(self._get_input(time_step))

    def of(self, derivative=None, hedger=None):
        self.inputs = [feature.of(derivative, hedger) for feature in self.inputs]
        return self

    def is_state_dependent(self):
        return any(map(lambda f: f.is_state_dependent(), self.inputs))
