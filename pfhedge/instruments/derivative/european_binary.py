from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_binary_payoff

from ..primary.base import BasePrimary
from .base import BaseDerivative
from .base import OptionMixin


class EuropeanBinaryOption(BaseDerivative, OptionMixin):
    r"""European binary option.

    The payoff of a European binary call option is given by

    .. math::
        \mathrm{payoff} =
        \begin{cases}
            1 & (S \geq K) \\
            0 & (\text{otherwise})
        \end{cases} ,

    where
    :math:`S` is the underlier's spot price at maturity and
    :math:`K` is the strike.

    The payoff of a European binary put option is given by

    .. math::
        \mathrm{payoff} =
        \begin{cases}
            1 & (S \leq K) \\
            0 & (\text{otherwise})
        \end{cases} .

    .. seealso::
        - :func:`pfhedge.nn.functional.european_binary_payoff`

    Args:
        underlier (:class:`BasePrimary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1): The strike price of the option.
        maturity (float, default=20/250): The maturity of the option.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:
        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanBinaryOption
        ...
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanBinaryOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> derivative.payoff()
        tensor([0., 1.])
    """

    def __init__(
        self,
        underlier: BasePrimary,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 20 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
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

    def extra_repr(self) -> str:
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        params.append("maturity=" + _format_float(self.maturity))
        return ", ".join(params)

    def payoff_fn(self) -> Tensor:
        return european_binary_payoff(
            self.ul().spot, call=self.call, strike=self.strike
        )


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(EuropeanBinaryOption, "simulate", BaseDerivative.simulate)
_set_attr_and_docstring(EuropeanBinaryOption, "to", BaseDerivative.to)
_set_attr_and_docstring(EuropeanBinaryOption, "ul", BaseDerivative.ul)
_set_attr_and_docstring(EuropeanBinaryOption, "list", BaseDerivative.list)
_set_docstring(EuropeanBinaryOption, "payoff", BaseDerivative.payoff)
_set_attr_and_docstring(EuropeanBinaryOption, "moneyness", OptionMixin.moneyness)
_set_attr_and_docstring(
    EuropeanBinaryOption, "log_moneyness", OptionMixin.log_moneyness
)
_set_attr_and_docstring(
    EuropeanBinaryOption, "time_to_maturity", OptionMixin.time_to_maturity
)
