from abc import abstractmethod
from typing import Optional
from typing import TypeVar

import torch
from torch import Tensor

from ..base import Instrument

T = TypeVar("T", bound="Derivative")


class Derivative(Instrument):
    """Base class for all derivatives.

    A derivative is a financial instrument whose payoff is contingent on
    a primary instrument (or a set of primary instruments).
    A (over-the-counter) derivative is not traded on the market and therefore the price
    is not directly accessible.
    Examples include options and swaps.

    A derivative relies on primary assets (See :class:`Primary` for details), such as
    stocks, bonds, commodities, and currencies.

    Attributes:
        underlier (:class:`Primary`): The underlying asset on which the derivative's
            payoff relies.
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.
    """

    underlier: "Primary"
    maturity: float

    @property
    def dtype(self) -> torch.dtype:
        return self.underlier.dtype

    @property
    def device(self) -> torch.device:
        return self.underlier.device

    def simulate(
        self, n_paths: int = 1, init_state: Optional[tuple] = None, **kwargs
    ) -> None:
        """Simulate time series associated with the underlier.

        Args:
            n_paths (int): The number of paths to simulate.
            init_state (tuple, optional): The initial state of the underlying
                instrument. If `None` (default), sensible default values are used.
            **kwargs: Other parameters passed to `self.underlier.simulate()`.
        """
        self.underlier.simulate(
            n_paths=n_paths, time_horizon=self.maturity, init_state=init_state, **kwargs
        )

    def to(self: T, *args, **kwargs) -> T:
        self.underlier.to(*args, **kwargs)
        return self

    @abstractmethod
    def payoff(self) -> Tensor:
        """Returns the payoffs of the derivative.

        Shape:
            - Output: :math:`(N)` where :math:`N` stands for the number of simulated
              paths.

        Returns:
            torch.Tensor
        """


class OptionMixin:
    """Mixin class of options."""

    underlier: "Primary"
    strike: float
    maturity: float

    def moneyness(self, time_step: int = None) -> Tensor:
        """Returns the moneyness of self.

        Args:
            time_step (int, optional): The time step to calculate
                the moneyness. If `None` (default), the moneyness is calculated
                at all time steps.

        Shape:
            - Output: :math:`(N, T)` where :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If `time_step` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        spot = self.underlier.spot
        if time_step is not None:
            spot = spot[..., time_step]
        return spot / self.strike

    def log_moneyness(self, time_step: int = None) -> Tensor:
        """Returns the log moneyness of self.

        Args:
            time_step (int, optional): The time step to calculate the log
                moneyness. If `None` (default), the moneyness is calculated
                at all time steps.

        Shape:
            - Output: :math:`(N, T)` where :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If `time_step` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        return self.moneyness(time_step=time_step).log()

    def time_to_maturity(self, time_step: int = None) -> Tensor:
        """Returns the time to maturity of self.

        Args:
            time_step (int, optional): The time step to calculate
                the time to maturity. If `None` (default), the time to
                maturity is calculated at all time steps.

        Shape:
            - Output: :math:`(N, T)` where :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If `time_step` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        n_paths, n_steps = self.underlier.spot.size()
        if time_step is None:
            t = self.underlier.dt * torch.arange(n_steps).repeat(n_paths, 1)
            t.to(self.underlier.device)
            return self.maturity - t
        else:
            t = torch.full_like(
                self.underlier.spot[:, 0], time_step * self.underlier.dt
            )
            return self.maturity - t


# Assign docstrings so they appear in Sphinx documentation
Derivative.to.__doc__ = Instrument.to.__doc__
Derivative.cpu = Instrument.cpu
Derivative.cpu.__doc__ = Instrument.cpu.__doc__
Derivative.cuda = Instrument.cuda
Derivative.cuda.__doc__ = Instrument.cuda.__doc__
Derivative.double = Instrument.double
Derivative.double.__doc__ = Instrument.double.__doc__
Derivative.float = Instrument.float
Derivative.float.__doc__ = Instrument.float.__doc__
Derivative.half = Instrument.half
Derivative.half.__doc__ = Instrument.half.__doc__
