import torch
from torch import Tensor

from ..nn.functional import european_binary_payoff
from .base import Derivative


class EuropeanBinaryOption(Derivative):
    """A European binary option.

    An American binary call option pays an unit amount of cash if and only if
    the underlying asset's price at maturity is equal or greater than the strike price.

    An American binary put option pays an unit amount of cash if and only if
    the underlying asset's price at maturity is equal or smaller than the strike price.

    The payoff of an American binary call option is given by:

    .. math::

        \\mathrm{payoff} =
        \\begin{cases}
            1 & (S \\geq K) \\\\
            0 & (\\text{otherwise})
        \\end{cases}

    with :math:`S` being the underlying asset's price at maturity and
    :math:`K` being the strike price (`strike`) of the option

    The payoff of an American binary put option is given by:

    .. math::

        \\mathrm{payoff} =
        \\begin{cases}
            1 & (S \\leq K) \\\\
            0 & (\\text{otherwise})
        \\end{cases}

    Args:
        underlier (:class:`Primary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1): The strike price of the option.
        maturity (float, default=20/250) The maturity of the option.
        dtype (torch.dtype, optional): Desired device of returned tensor.
            Default: If None, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.

    Examples:

        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanBinaryOption
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanBinaryOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=2)
        >>> derivative.underlier.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930],
                [1.0000, 1.0282, 1.0199, 1.0258, 1.0292]])
        >>> derivative.payoff()
        tensor([0., 1.])
    """

    def __init__(
        self,
        underlier,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 20 / 250,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()
        self.underlier = underlier
        self.call = call
        self.strike = strike
        self.maturity = maturity

        self.to(dtype=dtype, device=device)

    def __repr__(self):
        params = [f"{self.underlier.__class__.__name__}(...)"]
        if not self.call:
            params.append(f"call={self.call}")
        params.append(f"strike={self.strike}")
        params.append(f"maturity={self.maturity:.2e}")
        params += self.dinfo
        return self.__class__.__name__ + "(" + ", ".join(params) + ")"

    def payoff(self) -> Tensor:
        return european_binary_payoff(
            self.underlier.spot, call=self.call, strike=self.strike
        )


# Assign docstrings so they appear in Sphinx documentation
EuropeanBinaryOption.simulate = Derivative.simulate
EuropeanBinaryOption.simulate.__doc__ = Derivative.simulate.__doc__
EuropeanBinaryOption.to = Derivative.to
EuropeanBinaryOption.to.__doc__ = Derivative.to.__doc__
EuropeanBinaryOption.payoff.__doc__ = Derivative.payoff.__doc__
