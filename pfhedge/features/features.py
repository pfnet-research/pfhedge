import torch

from ._base import Feature


class Moneyness(Feature):
    """Moneyness of the underlying instrument of the derivative.

    Args:
        log (bool, default=False): If `True`, represents log moneyness.
    """

    def __init__(self, log=False):
        super().__init__()

        self.log = log

    def __str__(self):
        if self.log:
            return "log_moneyness"
        else:
            return "moneyness"

    def __getitem__(self, i):
        s = self.derivative.underlier.prices[i].reshape(-1, 1)
        output = s / self.derivative.strike
        if self.log:
            output = torch.log(output)
        return output


class LogMoneyness(Moneyness):
    """Log moneyness of the underlying instrument of the derivative."""

    def __init__(self):
        super().__init__(log=True)


class ExpiryTime(Feature):
    """Remaining time to the maturity of the derivative."""

    def __str__(self):
        return "expiry_time"

    def __getitem__(self, i):
        value = self.derivative.maturity - i * self.derivative.underlier.dt
        return torch.full_like(self.derivative.underlier.prices.T[:, :1], value)


class Volatility(Feature):
    """Volatility of the underlier of the derivative."""

    def __str__(self):
        return "volatility"

    def __getitem__(self, i):
        value = self.derivative.underlier.volatility
        return torch.full_like(self.derivative.underlier.prices.T[:, :1], value)


class PrevHedge(Feature):
    """Previous holding of underlier."""

    def __str__(self):
        return "prev_hedge"

    def __getitem__(self, i):
        if hasattr(self.hedger, "prev"):
            return self.hedger.prev.reshape(-1, 1)
        else:
            return torch.zeros_like(self.derivative.underlier.prices.T[:, :1])


class Barrier(Feature):
    """A feature which signifies whether the price of the underlier have reached
    the barrier. The output `1.0` means that the price have touched the barrier,
    and `0` otherwise.

    Args:
        barrier (float): The price level of the barrier.
        up (bool, default True): If `True`, signifies whether the price has exceeded
            the barrier upward.
            If `False`, signifies whether the price has exceeded the barrier downward.
    """

    def __init__(self, barrier, up=True):
        super().__init__()

        self.barrier = barrier
        self.up = up

    def __repr__(self):
        return self.__class__.__name__ + f"({self.barrier}, up={self.up})"

    def __getitem__(self, i):
        if self.up:
            touch_barrier = self.derivative.underlier.prices[: i + 1] >= self.barrier
        else:
            touch_barrier = self.derivative.underlier.prices[: i + 1] <= self.barrier
        return (
            touch_barrier.any(dim=0).reshape(-1, 1).to(self.derivative.underlier.prices)
        )


class Zero(Feature):
    """A feature of which value is always zero."""

    def __str__(self):
        return "zero"

    def __getitem__(self, i):
        return torch.zeros_like(self.derivative.underlier.prices.T[:, :1])


class MaxMoneyness(Feature):
    """Cumulative maximum of moneyness.

    Args:
        log (bool, default=False): If `True`, represents log moneyness.
    """

    def __init__(self, log=False):
        self.log = log
        self.moneyness = Moneyness(log=log)

    def __str__(self):
        if self.log:
            return "max_log_moneyness"
        else:
            return "max_moneyness"

    def __getitem__(self, i):
        s = self.derivative.underlier.prices[: i + 1].max(dim=0).values.reshape(-1, 1)
        output = s / self.derivative.strike
        if self.log:
            output = torch.log(output)
        return output

    def of(self, derivative=None, hedger=None):
        super().of(derivative)
        self.moneyness.of(derivative=derivative, hedger=hedger)
        return self


class MaxLogMoneyness(MaxMoneyness):
    """Cumulative maximum of log Moneyness."""

    def __init__(self):
        super().__init__(log=True)


class ModuleOutput(Feature, torch.nn.Module):
    """The feature computed as an output of a `torch.nn.Module`.

    Args:
        module (torch.nn.Module): Module to compute the value of the feature.
            The input and output shapes should be `(N, *, H_in) -> (N, *, 1)`,
            where `N` stands for the number of Monte Carlo paths of the underlier of
            the derivative, `H_in` stands for the number of input features
            (namely, `H_in = len(features)`),
            and `*` means any number of additional dimensions.
        features (list[Feature]): The input features to the module.

    Examples:

        >>> from torch.nn import Linear
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> deriv = EuropeanOption(BrownianStock())
        >>> deriv.simulate(n_paths=3)
        >>> m = Linear(2, 1)
        >>> f = ModuleOutput(m, [Moneyness(), ExpiryTime()]).of(deriv)
        >>> f[0]
        tensor([[...],
                [...],
                [...]], grad_fn=<AddmmBackward>)
        >>> f
        ModuleOutput(
          features=['moneyness', 'expiry_time'],
          (module): Linear(in_features=2, out_features=1, bias=True)
        )

        >>> from pfhedge.nn import BlackScholes

        >>> _ = torch.manual_seed(42)
        >>> deriv = EuropeanOption(BrownianStock())
        >>> deriv.simulate(n_paths=3)
        >>> m = BlackScholes(deriv)
        >>> f = ModuleOutput(m, [LogMoneyness(), ExpiryTime(), Volatility()]).of(deriv)
        >>> f[0]
        tensor([[...],
                [...],
                [...]])
    """

    def __init__(self, module, features):
        super().__init__()

        self.module = module
        self.features = features

    def extra_repr(self):
        return f"features={[str(f) for f in self.features]},"

    def forward(self, input):
        return self.module(input)

    def __getitem__(self, i):
        return self(torch.cat([f[i] for f in self.features], 1))

    def of(self, derivative=None, hedger=None):
        super().of(derivative, hedger)
        self.features = [feature.of(derivative, hedger) for feature in self.features]
        return self
