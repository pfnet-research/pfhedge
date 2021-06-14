from torch import Tensor

from ..nn.functional import american_binary_payoff
from ._base import Derivative


class AmericanBinaryOption(Derivative):
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
        maturity (float, default=30 / 365): The maturity of the option.
        dtype (torch.device, optional): Desired device of returned tensor.
            Default: If None, uses a global default (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

    Examples:

        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import AmericanBinaryOption
        >>> _ = torch.manual_seed(42)
        >>> deriv = AmericanBinaryOption(BrownianStock(), maturity=5 / 365, strike=1.01)
        >>> deriv.simulate(n_paths=2)
        >>> deriv.underlier.prices
        tensor([[1.0000, 1.0000],
                [1.0024, 1.0024],
                [0.9906, 1.0004],
                [1.0137, 0.9936],
                [1.0186, 0.9964]])
        >>> deriv.payoff()
        tensor([1., 0.])
    """

    def __init__(
        self,
        underlier,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 30 / 365,
        dtype=None,
        device=None,
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
        return american_binary_payoff(
            self.underlier.prices, call=self.call, strike=self.strike
        )
