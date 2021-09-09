from abc import abstractmethod
from collections import OrderedDict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TypeVar

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _addindent
from pfhedge._utils.typing import TensorOrScalar

from ..base import Instrument
from ..primary.base import Primary

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

    underlier: Primary
    cost: float
    maturity: float
    pricer: Optional[Callable[[Any], Tensor]]
    _clauses: Dict[str, Callable[["Derivative", Tensor], Tensor]]

    def __init__(self):
        super().__init__()
        self.pricer = None
        self.cost = 0.0
        self._clauses = OrderedDict()

    @property
    def dtype(self) -> Optional[torch.dtype]:
        return self.underlier.dtype

    @property
    def device(self) -> Optional[torch.device]:
        return self.underlier.device

    def simulate(
        self, n_paths: int = 1, init_state: Optional[Tuple[TensorOrScalar, ...]] = None
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
    def payoff_fn(self) -> Tensor:
        """Defines the payoff function of the derivative.

        This should be overridden by all subclasses.

        Note:
            Although the payoff function needs to be defined within this function,
            one should use the :meth:`payoff` method afterwards instead of this
            since the former takes care of applying the registered clauses
            (See :meth:`add_clause`)
            while the latter silently ignores them.

        Shape:
            - Output: :math:`(N)` where
              :math:`N` stands for the number of simulated paths.

        Returns:
            torch.Tensor
        """

    def payoff(self) -> Tensor:
        """Returns the payoff of the derivative.

        Shape:
            - Output: :math:`(N)` where
              :math:`N` stands for the number of simulated paths.

        Returns:
            torch.Tensor
        """
        payoff = self.payoff_fn()
        for clause in self._clauses.values():
            payoff = clause(self, payoff)
        return payoff

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

    def add_clause(
        self, name: str, clause: Callable[["Derivative", Tensor], Tensor]
    ) -> None:
        """Adds a clause to the derivative.

        The clause will be called after :meth:`payoff_fn` method
        has computed the payoff and modify the payoff tensor.
        It should have the following signature::

            clause(derivative, payoff) -> modified payoff

        Args:
            name (str): The name of the clause.
            clause (callable[[Derivative, torch.Tensor], torch.Tensor]):
                The clause to add.
        """
        if not isinstance(name, torch._six.string_classes):
            raise TypeError(
                "clause name should be a string. Got {}".format(torch.typename(name))
            )
        elif hasattr(self, name) and name not in self._clauses:
            raise KeyError("attribute '{}' already exists".format(name))
        elif "." in name:
            raise KeyError('clause name can\'t contain ".", got: {}'.format(name))
        elif name == "":
            raise KeyError('clause name can\'t be empty string ""')
        self._clauses[name] = clause

    @property
    def spot(self) -> Tensor:
        """Returns ``self.pricer(self)`` if self is listed.

        See :meth:`list()` for details.
        """
        if self.pricer is None:
            raise ValueError("self is not listed.")
        return self.pricer(self)

    def __repr__(self) -> str:
        params_str = ""
        if self.extra_repr() != "":
            params_str += self.extra_repr() + "\n"
        if self._clauses:
            params_str += "clauses=" + repr(list(self._clauses.keys())) + "\n"
        params_str += "(underlier): " + repr(self.ul())
        if params_str != "":
            params_str = "\n" + _addindent(params_str) + "\n"
        return self._get_name() + "(" + params_str + ")"


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
            - Output: :math:`(N, T)` where
              :math:`N` is the number of paths and
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
        """Returns ``self.moneyness(time_step).log()``.

        Returns:
            torch.Tensor
        """
        return self.moneyness(time_step=time_step, log=True)

    def time_to_maturity(self, time_step: Optional[int] = None) -> Tensor:
        """Returns the time to maturity of self.

        Args:
            time_step (int, optional): The time step to calculate
                the time to maturity. If ``None`` (default), the time to
                maturity is calculated at all time steps.

        Shape:
            - Output: :math:`(N, T)` where
              :math:`N` is the number of paths and
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
            - Output: :math:`(N, T)` where
              :math:`N` is the number of paths and
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
        """Returns ``self.max_moneyness(time_step).log()``.

        Returns:
            torch.Tensor
        """
        return self.max_moneyness(time_step, log=True)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(Derivative, "to", Instrument.to)
_set_attr_and_docstring(Derivative, "cpu", Instrument.cpu)
_set_attr_and_docstring(Derivative, "cuda", Instrument.cuda)
_set_attr_and_docstring(Derivative, "double", Instrument.double)
_set_attr_and_docstring(Derivative, "float", Instrument.float)
_set_attr_and_docstring(Derivative, "half", Instrument.half)
