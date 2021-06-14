from torch import Tensor

from ..nn.functional import lookback_payoff
from ._base import Derivative


class LookbackOption(Derivative):
    """A lookback option with fixed strike.

    A lookback call option provides its holder the right to buy an underlying with
    the strike price and to sell with the highest price until the date of maturity.

    A lookback put option provides its holder the right to sell an underlying with
    the strike price and to buy with the lowest price until the date of maturity.

    The payoff of a lookback call option is given by:

    .. math::

        \\mathrm{payoff} = \\max(\\mathrm{Max} - K, 0)

    Here, :math:`\\mathrm{Max}` is the maximum of the underlying asset's price
    until maturity and :math:`K` is the strike price (`strike`) of the option.

    The payoff of a lookback put option is given by:

    .. math::

        \\mathrm{payoff} = \\max(K - \\mathrm{Min}, 0)

    Here, :math:`\\mathrm{Min}` is the minimum of the underlying asset's price.

    Args:
        underlier (:class:`Primary`): The underlying instrument of the option.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.
        maturity (float, default=30/365): The maturity of the option.
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
        >>> from pfhedge.instruments import LookbackOption
        >>> _ = torch.manual_seed(42)
        >>> deriv = LookbackOption(BrownianStock(), maturity=5 / 365)
        >>> deriv.simulate(n_paths=2)
        >>> deriv.underlier.prices
        tensor([[1.0000, 1.0000],
                [1.0024, 1.0024],
                [0.9906, 1.0004],
                [1.0137, 0.9936],
                [1.0186, 0.9964]])
        >>> deriv.payoff()
        tensor([0.0186, 0.0024])
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
        return lookback_payoff(
            self.underlier.prices, call=self.call, strike=self.strike
        )
