from typing import List

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float

from ._base import Feature
from .functional import barrier
from .functional import empty
from .functional import max_log_moneyness
from .functional import max_moneyness
from .functional import prev_hedge
from .functional import zeros


class Moneyness(Feature):
    """Moneyness of the underlying instrument of the derivative.

    Args:
        log (bool, default=False): If ``True``, represents log moneyness.
    """

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "log_moneyness" if self.log else "moneyness"

    def __getitem__(self, i: int) -> Tensor:
        if self.log:
            return self.derivative.log_moneyness(i)
        else:
            return self.derivative.moneyness(i)


class LogMoneyness(Moneyness):
    """Log moneyness of the underlying instrument of the derivative."""

    def __init__(self) -> None:
        super().__init__(log=True)


class ExpiryTime(Feature):
    """Remaining time to the maturity of the derivative."""

    def __str__(self) -> str:
        return "expiry_time"

    def __getitem__(self, i: int) -> Tensor:
        return self.derivative.time_to_maturity(i)


class Volatility(Feature):
    """Volatility of the underlier of the derivative."""

    def __str__(self) -> str:
        return "volatility"

    def __getitem__(self, i: int) -> Tensor:
        return self.derivative.ul().volatility[:, [i]]


class PrevHedge(Feature):
    """Previous holding of underlier."""

    def __str__(self) -> str:
        return "prev_hedge"

    def __getitem__(self, i: int) -> Tensor:
        return prev_hedge(i, derivative=self.derivative, hedger=self.hedger)


class Barrier(Feature):
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

    def __getitem__(self, i: int) -> Tensor:
        return barrier(
            i, derivative=self.derivative, threshold=self.threshold, up=self.up
        )


class Zeros(Feature):
    """A feature of which value is always zero."""

    def __str__(self) -> str:
        return "zeros"

    def __getitem__(self, i: int) -> Tensor:
        return zeros(i, derivative=self.derivative)


class Empty(Feature):
    """A feature of which value is always empty."""

    def __str__(self) -> str:
        return "empty"

    def __getitem__(self, i: int) -> Tensor:
        return empty(i, derivative=self.derivative)


class MaxMoneyness(Feature):
    """Cumulative maximum of moneyness.

    Args:
        log (bool, default=False): If ``True``, represents log moneyness.
    """

    def __init__(self, log: bool = False) -> None:
        super().__init__()
        self.log = log

    def __str__(self) -> str:
        return "max_log_moneyness" if self.log else "max_moneyness"

    def __getitem__(self, i: int) -> Tensor:
        if self.log:
            return max_log_moneyness(i, derivative=self.derivative)
        else:
            return max_moneyness(i, derivative=self.derivative)


class MaxLogMoneyness(MaxMoneyness):
    """Cumulative maximum of log Moneyness."""

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
        >>> m = Linear(2, 1)
        >>> f = ModuleOutput(m, [Moneyness(), ExpiryTime()]).of(derivative)
        >>> f[0]
        tensor([[...],
                [...],
                [...]], grad_fn=<AddmmBackward>)
        >>> f
        ModuleOutput(
          inputs=['moneyness', 'expiry_time']
          (module): Linear(in_features=2, out_features=1, bias=True)
        )

        >>> from pfhedge.nn import BlackScholes

        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.simulate(n_paths=3)
        >>> m = BlackScholes(derivative)
        >>> f = ModuleOutput(m, [LogMoneyness(), ExpiryTime(), Volatility()])
        >>> f = f.of(derivative)
        >>> f[0]
        tensor([[...],
                [...],
                [...]])
    """

    def __init__(self, module: Module, inputs: List[Feature]) -> None:
        super().__init__()

        self.module = module
        self.inputs = inputs

    def extra_repr(self) -> str:
        return "inputs=" + str([str(f) for f in self.inputs])

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input)

    def __getitem__(self, i: int) -> Tensor:
        return self(torch.cat([f[i] for f in self.inputs], 1))

    def of(self, derivative=None, hedger=None):
        super().of(derivative, hedger)
        self.inputs = [feature.of(derivative, hedger) for feature in self.inputs]
        return self
