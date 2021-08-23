from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_payoff

from ..primary.base import Primary
from .base import BaseOption
from .base import Derivative


class EuropeanOption(BaseOption):
    """A European option.

    A European option provides its holder the right to buy (for call option)
    or sell (for put option) an underlying asset with the strike price
    on the date of maturity.

    The payoff of a European call option is given by:

    .. math::

        \\mathrm{payoff} = \\max(S - K, 0)

    Here, :math:`S` is the underlying asset's price at maturity and
    :math:`K` is the strike price (`strike`) of the option.

    The payoff of a European put option is given by:

    .. math::

        \\mathrm{payoff} = \\max(K - S, 0)

    Args:
        underlier (:class:`Primary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.
        maturity (float, default=20/250): The maturity of the option.
        dtype (torch.dtype, optional): Desired device of returned tensor.
            Default: If None, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

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
          strike=..., maturity=...
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

        self.to(dtype=dtype, device=device)

    def extra_repr(self):
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        params.append("maturity=" + _format_float(self.maturity))
        return ", ".join(params)

    def payoff(self) -> Tensor:
        return european_payoff(self.ul().spot, call=self.call, strike=self.strike)


# Assign docstrings so they appear in Sphinx documentation
set_attr_and_docstring(EuropeanOption, "simulate", Derivative.simulate)
set_attr_and_docstring(EuropeanOption, "to", Derivative.to)
set_attr_and_docstring(EuropeanOption, "ul", Derivative.ul)
set_attr_and_docstring(EuropeanOption, "list", Derivative.list)
set_docstring(EuropeanOption, "payoff", Derivative.payoff)
set_attr_and_docstring(EuropeanOption, "moneyness", BaseOption.moneyness)
set_attr_and_docstring(EuropeanOption, "log_moneyness", BaseOption.log_moneyness)
set_attr_and_docstring(EuropeanOption, "time_to_maturity", BaseOption.time_to_maturity)
