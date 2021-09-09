from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_payoff

from ..primary.base import Primary
from .base import BaseOption
from .base import Derivative


class EuropeanOption(BaseOption):
    r"""A European option.

    A European option provides its holder the right to buy (for call option)
    or sell (for put option) an underlying asset with the strike price
    on the date of maturity.

    The payoff of a European call option is given by:

    .. math::

        \mathrm{payoff} = \max(S - K, 0)

    Here, :math:`S` is the underlying asset's price at maturity and
    :math:`K` is the strike price (`strike`) of the option.

    The payoff of a European put option is given by:

    .. math::

        \mathrm{payoff} = \max(K - S, 0)

    .. seealso::
        :func:`pfhedge.nn.functional.european_payoff`: Payoff function.

    Args:
        underlier (:class:`Primary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.
        maturity (float, default=20/250): The maturity of the option.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:

        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> derivative.payoff()
        tensor([0.0000, 0.0179])

        Using custom ``dtype`` and ``device``.

        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.to(dtype=torch.float64, device="cuda:0")
        EuropeanOption(
          ...
          (underlier): BrownianStock(..., dtype=torch.float64, device='cuda:0')
        )

        Make ``self`` a listed derivative.

        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> pricer = lambda derivative: BlackScholes(derivative).price(
        ...     log_moneyness=derivative.log_moneyness(),
        ...     time_to_maturity=derivative.time_to_maturity(),
        ...     volatility=derivative.ul().volatility)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.list(pricer, cost=1e-4)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.ul().spot
        tensor([[1.0000, 0.9788, 0.9665, 0.9782, 0.9947, 1.0049],
                [1.0000, 0.9905, 1.0075, 1.0162, 1.0119, 1.0220]])
        >>> derivative.spot
        tensor([[0.0113, 0.0028, 0.0006, 0.0009, 0.0028, 0.0049],
                [0.0113, 0.0060, 0.0130, 0.0180, 0.0131, 0.0220]])

        Add a knock-out clause with a barrier at 1.03:

        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=8)
        >>> derivative.payoff()
        tensor([0.0000, 0.0000, 0.0113, 0.0414, 0.0389, 0.0008, 0.0000, 0.0000])
        >>>
        >>> def knockout(derivative, payoff):
        ...     max = derivative.underlier.spot.max(-1).values
        ...     return payoff.where(max < 1.03, torch.zeros_like(max))
        >>>
        >>> derivative.add_clause("knockout", knockout)
        >>> derivative
        EuropeanOption(
          strike=1., maturity=0.0200
          clauses=['knockout']
          (underlier): BrownianStock(sigma=0.2000, dt=0.0040)
        )
        >>> derivative.payoff()
        tensor([0.0000, 0.0000, 0.0113, 0.0000, 0.0000, 0.0008, 0.0000, 0.0000])
    """

    def __init__(
        self,
        underlier: Primary,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 20 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.underlier = underlier
        self.call = call
        self.strike = strike
        self.maturity = maturity

        # TODO(simaki): Remove later. Deprecated for > v0.12.3
        if dtype is not None or device is not None:
            self.to(dtype=dtype, device=device)
            raise DeprecationWarning(
                "Specifying device and dtype when constructing a Derivative is deprecated."
                "Specify them in the constructor of the underlier instead."
            )

    def extra_repr(self):
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        params.append("maturity=" + _format_float(self.maturity))
        return ", ".join(params)

    def payoff_fn(self) -> Tensor:
        return european_payoff(self.ul().spot, call=self.call, strike=self.strike)


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(EuropeanOption, "simulate", Derivative.simulate)
_set_attr_and_docstring(EuropeanOption, "to", Derivative.to)
_set_attr_and_docstring(EuropeanOption, "ul", Derivative.ul)
_set_attr_and_docstring(EuropeanOption, "list", Derivative.list)
_set_docstring(EuropeanOption, "payoff", Derivative.payoff)
_set_attr_and_docstring(EuropeanOption, "moneyness", BaseOption.moneyness)
_set_attr_and_docstring(EuropeanOption, "log_moneyness", BaseOption.log_moneyness)
_set_attr_and_docstring(EuropeanOption, "time_to_maturity", BaseOption.time_to_maturity)
