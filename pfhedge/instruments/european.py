from torch import Tensor

from ..nn.functional import european_payoff
from ._base import Derivative


class EuropeanOption(Derivative):
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
        >>> from pfhedge.instruments import EuropeanOption
        >>> _ = torch.manual_seed(42)
        >>> deriv = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> deriv.simulate(n_paths=2)
        >>> deriv.underlier.prices
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930],
                [1.0000, 1.0282, 1.0199, 1.0258, 1.0292]])
        >>> deriv.payoff()
        tensor([0.0000, 0.0292])

        Using custom `dtype` and `device`.

        >>> deriv = EuropeanOption(BrownianStock())
        >>> deriv.to(dtype=torch.float64, device="cuda:0")
        EuropeanOption(BrownianStock(...), ..., dtype=torch.float64, device='cuda:0')
    """

    def __init__(
        self,
        underlier,
        call: bool = True,
        strike: float = 1.0,
        maturity: float = 20 / 250,
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
        return european_payoff(
            self.underlier.prices, call=self.call, strike=self.strike
        )
