import torch
from torch import Tensor

from ..._utils.bisect import bisect
from ._base import BSModuleMixin


class BSEuropeanBinaryOption(BSModuleMixin):
    """Black-Scholes formula for a European binary option.

    Args:
        derivative (:class:`pfhedge.instruments.EuropeanBinaryOption`. optional):
            The derivative to get the Black-Scholes formula.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 3)`, where :math:`*` means any number of additional dimensions.
          See `features()` for input features.
        - Output: :math:`(N, *, 1)` Delta of the derivative.
          All but the last dimension are the same shape as the input.

    .. seealso ::

        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.

    Examples:

        The `forward` method returns delta of the derivative.

        >>> from pfhedge.nn import BSEuropeanBinaryOption
        >>> m = BSEuropeanBinaryOption(strike=1.0)
        >>> m.features()
        ['log_moneyness', 'expiry_time', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, 0.1, 0.2],
        ...     [ 0.00, 0.1, 0.2],
        ...     [ 0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[6.2576],
                [6.3047],
                [6.1953]])

        One can instantiate it using an
        :class:`pfhedge.instruments.EuropeanBinaryOption`.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanBinaryOption
        >>> deriv = EuropeanBinaryOption(BrownianStock(), strike=1.1)
        >>> m = BSEuropeanBinaryOption(deriv)
        >>> m
        BSEuropeanBinaryOption(strike=1.1)
    """

    def __init__(self, derivative=None, call: bool = True, strike: float = 1.0):
        super().__init__()

        if derivative is not None:
            self.call = derivative.call
            self.strike = derivative.strike
        else:
            self.call = call
            self.strike = strike

        if not self.call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

    def extra_repr(self):
        params = []
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        return ", ".join(params)

    def features(self) -> list:
        return ["log_moneyness", "expiry_time", "volatility"]

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Tensor,
        expiry_time: Tensor,
        volatility: Tensor,
        create_graph: bool = False,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness: (torch.Tensor): Log moneyness of the underlying asset.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.
            create_graph (bool, default=False): If True, graph of the derivative will
                be constructed. This option is used to compute gamma.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        delta = self.N.pdf(self.d2(s, t, v)) / (
            self.strike * torch.exp(s) * v * torch.sqrt(t)
        )
        return delta

    def gamma(
        self, log_moneyness: Tensor, expiry_time: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        prices = self.strike * torch.exp(torch.as_tensor(log_moneyness))
        prices = torch.Tensor.requires_grad_(prices)

        s = torch.log(prices / self.strike)
        t, v = map(torch.as_tensor, (expiry_time, volatility))

        delta = self.delta(s, t, v, create_graph=True)
        gamma = torch.autograd.grad(
            delta, prices, grad_outputs=torch.ones_like(prices)
        )[0]

        return gamma

    def price(
        self, log_moneyness: Tensor, expiry_time: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        price = self.N.cdf(self.d2(s, t, v))
        price = 1.0 - price if not self.call else price  # put-call parity

        return price

    def implied_volatility(
        self,
        log_moneyness: Tensor,
        expiry_time: Tensor,
        price: Tensor,
        precision: float = 1e-6,
    ) -> Tensor:
        """
        Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            expiry_time (torch.Tensor): Time to expiry of the option.
            price (torch.Tensor): Price of the derivative.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, t, p = map(torch.as_tensor, (log_moneyness, expiry_time, price))
        get_price = lambda v: self.price(s, t, v)
        return bisect(get_price, p, lower=0.001, upper=1.000, precision=precision)
