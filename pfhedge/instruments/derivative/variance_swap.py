from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import realized_variance

from ..primary.base import Primary
from .base import Derivative


class VarianceSwap(Derivative):
    """A variance swap.

    A variance swap pays cash in the amount of the realized variance
    until the maturity and levies the cash of the strike variance.

    The payoff of a variance swap is given by

    .. math::

        \\mathrm{payoff} = \\sigma^2 - K

    where :math:`\\sigma^2` is the realized variance of the underlying asset
    until maturity and :math:`K` is the strike variance (``strike``).
    See :func:`pfhedge.nn.functional.realized_variance` for the definition of
    the realized variance.

    Args:
        underlier (:class:`Primary`): The underlying instrument.
        strike (float, default=0.04): The strike variance of the swap.
        maturity (float, default=20/250): The maturity of the derivative.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:

        >>> import torch
        >>> from pfhedge.nn.functional import realized_variance
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import VarianceSwap
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = VarianceSwap(BrownianStock(), strike=0.04, maturity=5/250)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])
        >>> realized_variance(derivative.ul().spot, dt=derivative.ul().dt)
        tensor([0.0114, 0.0129])
        >>> derivative.payoff()
        tensor([-0.0286, -0.0271])
    """

    def __init__(
        self,
        underlier: Primary,
        strike: float = 0.04,
        maturity: float = 20 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.underlier = underlier
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
        return ", ".join(
            (
                "strike=" + _format_float(self.strike),
                "maturity=" + _format_float(self.maturity),
            )
        )

    def payoff_fn(self) -> Tensor:
        return realized_variance(self.ul().spot, dt=self.ul().dt) - self.strike


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(VarianceSwap, "simulate", Derivative.simulate)
_set_attr_and_docstring(VarianceSwap, "to", Derivative.to)
_set_attr_and_docstring(VarianceSwap, "ul", Derivative.ul)
_set_attr_and_docstring(VarianceSwap, "list", Derivative.list)
_set_docstring(VarianceSwap, "payoff", Derivative.payoff)
