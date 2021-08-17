from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring

from ...nn.functional import european_payoff
from ..primary.base import Primary
from .base import Derivative
from .base import OptionMixin


class EuropeanOption(Derivative, OptionMixin):
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
            (see ``torch.set_default_tensor_type()``).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see ``torch.set_default_tensor_type()``).
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
        EuropeanOption(..., dtype=torch.float64, device='cuda:0')
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

    def __repr__(self):
        params = [f"{self.ul().__class__.__name__}(...)"]
        if not self.call:
            params.append(f"call={self.call}")
        params.append(f"strike={self.strike}")
        params.append(f"maturity={self.maturity:.2e}")
        params += self.dinfo
        return self.__class__.__name__ + "(" + ", ".join(params) + ")"

    def payoff(self) -> Tensor:
        return european_payoff(self.underlier.spot, call=self.call, strike=self.strike)


# Assign docstrings so they appear in Sphinx documentation
set_attr_and_docstring(EuropeanOption, "simulate", Derivative.simulate)
set_attr_and_docstring(EuropeanOption, "to", Derivative.to)
set_attr_and_docstring(EuropeanOption, "ul", Derivative.ul)
set_docstring(EuropeanOption, "payoff", Derivative.payoff)
set_attr_and_docstring(EuropeanOption, "moneyness", OptionMixin.moneyness)
set_attr_and_docstring(EuropeanOption, "log_moneyness", OptionMixin.log_moneyness)
set_attr_and_docstring(EuropeanOption, "time_to_maturity", OptionMixin.time_to_maturity)
