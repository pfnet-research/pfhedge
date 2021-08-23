from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import american_binary_payoff

from ..primary.base import Primary
from .base import BaseOption
from .base import Derivative


class AmericanBinaryOption(BaseOption):
    """An American binary Option.

    An American binary call option pays an unit amount of cash if and only if
    the maximum of the underlying asset's price until maturity is equal or greater
    than the strike price.

    An American binary put option pays an unit amount of cash if and only if
    the minimum of the underlying asset's price until maturity is equal or smaller
    than the strike price.

    The payoff of an American binary call option is given by:

    .. math ::

        \\mathrm{payoff} =
        \\begin{cases}
            1 & (\\mathrm{Max} \\geq K) \\\\
            0 & (\\text{otherwise})
        \\end{cases}

    Here, :math:`\\mathrm{Max}` is the maximum of the underlying asset's price
    until maturity and :math:`K` is the strike price (`strike`) of the option.

    The payoff of an American binary put option is given by:

    .. math ::

        \\mathrm{payoff} =
        \\begin{cases}
            1 & (\\mathrm{Min} \\leq K) \\\\
            0 & (\\text{otherwise})
        \\end{cases}

    Here, :math:`\\mathrm{Min}` is the minimum of the underlying asset's price.

    Args:
        underlier (:class:`Primary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.
        maturity (float, default=20/250): The maturity of the option.
        dtype (torch.device, optional): Desired device of returned tensor.
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
        >>> from pfhedge.instruments import AmericanBinaryOption
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = AmericanBinaryOption(BrownianStock(), \
maturity=5/250, strike=1.01)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> derivative.payoff()
        tensor([0., 1.])
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
        return american_binary_payoff(
            self.ul().spot, call=self.call, strike=self.strike
        )


# Assign docstrings so they appear in Sphinx documentation
set_attr_and_docstring(AmericanBinaryOption, "simulate", Derivative.simulate)
set_attr_and_docstring(AmericanBinaryOption, "to", Derivative.to)
set_attr_and_docstring(AmericanBinaryOption, "ul", Derivative.ul)
set_attr_and_docstring(AmericanBinaryOption, "list", Derivative.list)
set_docstring(AmericanBinaryOption, "payoff", Derivative.payoff)
set_attr_and_docstring(AmericanBinaryOption, "moneyness", BaseOption.moneyness)
set_attr_and_docstring(AmericanBinaryOption, "log_moneyness", BaseOption.log_moneyness)
set_attr_and_docstring(
    AmericanBinaryOption, "time_to_maturity", BaseOption.time_to_maturity
)
