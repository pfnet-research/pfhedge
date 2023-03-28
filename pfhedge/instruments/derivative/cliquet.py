from math import floor

from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import european_forward_start_payoff

from ..primary.base import BasePrimary
from .base import BaseDerivative


class EuropeanForwardStartOption(BaseDerivative):
    r"""European forward start option.

    The payoff is given by

    .. math ::
        \mathrm{payoff} = \max(S_T / S_{T'} - K, 0) ,

    where
    :math:`S_T` is the underlier's spot price at maturity,
    :math:`S_{T'}` is the underlier's spot price at ``start``, and
    :math:`K` is the strike.

    Note:
        If ``start`` is not divisible by the interval of the time step,
        it is rounded off so that the start time is the largest value
        that is divisible by the interval and less than ``start``.

    .. seealso::
        - :func:`pfhedge.nn.functional.european_forward_start_payoff`

    Args:
        underlier (:class:`BasePrimary`): The underlying instrument of the option.
        strike (float, default=1.0): The strike value of the option.
        maturity (float, default=20/250): The maturity of the option.
        start (float, default=10/250): The start of the option.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:
        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanForwardStartOption
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanForwardStartOption(BrownianStock(), maturity=5/250, start=2/250)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> derivative.underlier.spot[:, -1] / derivative.underlier.spot[:, 2]
        tensor([0.9862, 1.0203])
        >>> derivative.payoff()
        tensor([0.0000, 0.0203])
    """

    def __init__(
        self,
        underlier: BasePrimary,
        strike: float = 1.0,
        maturity: float = 20 / 250,
        start: float = 10 / 250,
    ) -> None:
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.strike = strike
        self.maturity = maturity
        self.start = start

    def extra_repr(self) -> str:
        params = [
            "strike=" + _format_float(self.strike),
            "maturity=" + _format_float(self.maturity),
            "start=" + _format_float(self.start),
        ]
        return ", ".join(params)

    def _start_index(self) -> int:
        return floor(self.start / self.ul().dt)

    def payoff_fn(self) -> Tensor:
        return european_forward_start_payoff(
            self.ul().spot, strike=self.strike, start_index=self._start_index()
        )


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(EuropeanForwardStartOption, "simulate", BaseDerivative.simulate)
_set_attr_and_docstring(EuropeanForwardStartOption, "to", BaseDerivative.to)
_set_attr_and_docstring(EuropeanForwardStartOption, "ul", BaseDerivative.ul)
_set_attr_and_docstring(EuropeanForwardStartOption, "list", BaseDerivative.list)
_set_docstring(EuropeanForwardStartOption, "payoff", BaseDerivative.payoff)
