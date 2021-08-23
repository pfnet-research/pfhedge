from abc import abstractmethod
from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring

from ..base import Instrument
from ..primary.base import Primary

T = TypeVar("T", bound="Derivative")
TensorOrFloat = Union[Tensor, float]


def _addindent(string: str, n_spaces: int = 2) -> str:
    lines = []
    for line in string.split("\n"):
        lines.append(" " * n_spaces + line)
    return "\n".join(lines)


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

    underlier: Primary
    maturity: float
    pricer: Optional[Callable[[Any], Tensor]]
    cost: float

    def __init__(self):
        super().__init__()
        self.pricer = None
        self.cost = 0.0

    @property
    def dtype(self) -> torch.dtype:
        return self.underlier.dtype

    @property
    def device(self) -> torch.device:
        return self.underlier.device

    def simulate(
        self, n_paths: int = 1, init_state: Optional[Tuple[TensorOrFloat, ...]] = None
    ) -> None:
        """Simulate time series associated with the underlier.

        Args:
            n_paths (int): The number of paths to simulate.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the underlier.
            **kwargs: Other parameters passed to ``self.underlier.simulate()``.
        """
        self.underlier.simulate(
            n_paths=n_paths, time_horizon=self.maturity, init_state=init_state
        )

    def ul(self) -> Primary:
        """Alias for ``self.underlier``."""
        return self.underlier

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

    def list(self: T, pricer: Callable[[T], Tensor], cost: float = 0.0) -> None:
        """Make self a listed derivative.

        After this method self will be a exchange-traded derivative which can be transacted
        at any time with the spot price given by ``self.spot``.

        See an example in :class:`EuropeanOption` for a usage.

        Args:
            pricer (Callable[[Derivative], Tensor]]): A function that takes self
                and returns the spot price tensor of self.
            cost (float, optional): The transaction cost rate.
        """
        self.pricer = pricer
        self.cost = cost

    @property
    def spot(self) -> Tensor:
        """Returns ``self.pricer(self)`` if self is listed.

        See :func:`list()` for details.
        """
        if self.pricer is None:
            raise ValueError("self is not listed.")
        return self.pricer(self)

    def __repr__(self) -> str:
        main_str = self._get_name() + "(\n  "
        main_str += self.extra_repr() + "\n"
        main_str += _addindent("(underlier): " + repr(self.ul()))
        main_str += "\n)"
        return main_str


class BaseOption(Derivative):
    """Base class of options."""

    underlier: Primary
    strike: float
    maturity: float

    def moneyness(self, time_step: Optional[int] = None, log: bool = False) -> Tensor:
        """Returns the moneyness of self.

        Args:
            time_step (int, optional): The time step to calculate
                the moneyness. If ``None`` (default), the moneyness is calculated
                at all time steps.
            log (bool, default=False): If ``True``, returns log moneyness.

        Shape:
            - Output: :math:`(N, T)` where :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If ``time_step`` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        index = ... if time_step is None else [time_step]
        output = self.underlier.spot[..., index] / self.strike
        if log:
            output = output.log()
        return output

    def log_moneyness(self, time_step: Optional[int] = None) -> Tensor:
        """Returns ``self.moneyness(time_step).log()``."""
        return self.moneyness(time_step=time_step, log=True)

    def time_to_maturity(self, time_step: Optional[int] = None) -> Tensor:
        """Returns the time to maturity of self.

        Args:
            time_step (int, optional): The time step to calculate
                the time to maturity. If ``None`` (default), the time to
                maturity is calculated at all time steps.

        Shape:
            - Output: :math:`(N, T)` where :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If ``time_step`` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        n_paths, n_steps = self.underlier.spot.size()
        if time_step is None:
            # Time passed from the beginning
            t = torch.arange(n_steps).to(self.underlier.spot) * self.underlier.dt
            return (t[-1] - t).unsqueeze(0).expand(n_paths, -1)
        else:
            time = n_steps - (time_step % n_steps) - 1
            t = torch.tensor([[time]]).to(self.underlier.spot) * self.underlier.dt
            return t.expand(n_paths, -1)

    def max_moneyness(self, time_step: Optional[int] = None, log=False) -> Tensor:
        """Returns the cumulative maximum of the moneyness.

        Args:
            time_step (int, optional): The time step to calculate
                the time to maturity. If ``None`` (default), the time to
                maturity is calculated at all time steps.
            log (bool, default=False): If ``True``, returns the cumulative
                maximum of the log moneyness.

        Shape:
            - Output: :math:`(N, T)` where :math:`N` is the number of paths and
              :math:`T` is the number of time steps.
              If ``time_step`` is given, the shape is :math:`(N, 1)`.

        Returns:
            torch.Tensor
        """
        moneyness = self.moneyness(None, log=log)
        if time_step is None:
            return moneyness.cummax(dim=-1).values
        else:
            return moneyness[..., : time_step + 1].max(dim=-1, keepdim=True).values

    def max_log_moneyness(self, time_step: Optional[int] = None) -> Tensor:
        """Returns ``self.max_moneyness(time_step).log()``."""
        return self.max_moneyness(time_step, log=True)


# Assign docstrings so they appear in Sphinx documentation
set_docstring(Derivative, "to", Instrument.to)
set_attr_and_docstring(Derivative, "cpu", Instrument.cpu)
set_attr_and_docstring(Derivative, "cuda", Instrument.cuda)
set_attr_and_docstring(Derivative, "double", Instrument.double)
set_attr_and_docstring(Derivative, "float", Instrument.float)
set_attr_and_docstring(Derivative, "half", Instrument.half)
