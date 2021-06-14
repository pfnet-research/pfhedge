import torch
from torch import Tensor

from ..._utils.bisect import bisect
from ._base import BSModuleMixin


class BSLookbackOption(BSModuleMixin):
    """Black-Scholes formula for a lookback option with a fixed strike.

    Args:
        derivative (:class:`pfhedge.instruments.LookbackOption`, optional):
            The derivative to get the Black-Scholes formula.
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 4)`, where :math:`*` means any number of additional
          dimensions.  See `features()` for input features.
        - Output: :math:`(N, *, 1)`. Delta of the derivative.
          All but the last dimension are the same shape as the input.

    .. seealso ::

        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.

    Examples:

        The `forward` method returns delta of the derivative.

        >>> from pfhedge.nn import BSLookbackOption
        >>> m = BSLookbackOption()
        >>> m.features()
        ['log_moneyness', 'max_log_moneyness', 'expiry_time', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, -0.01, 0.1, 0.2],
        ...     [ 0.00,  0.00, 0.1, 0.2],
        ...     [ 0.01,  0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[0.9208],
                [1.0515],
                [1.0515]])

        One can instantiate it using an
        :class:`pfhedge.instruments.LookbackOption`.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import LookbackOption
        >>> deriv = LookbackOption(BrownianStock(), strike=1.1)
        >>> m = BSLookbackOption(deriv)
        >>> m
        BSLookbackOption(strike=1.1)
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
        return ["log_moneyness", "max_log_moneyness", "expiry_time", "volatility"]

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        expiry_time: Tensor,
        volatility: Tensor,
        create_graph: bool = False,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the prices of the underlying
                asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.
            create_graph (bool, default=False): If True, graph of the derivative
                will be constructed. This option is used to compute gamma.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - max_log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`


        Returns:
            Tensor
        """
        prices = self.strike * torch.exp(torch.as_tensor(log_moneyness))
        prices = Tensor.requires_grad_(prices)

        s = torch.log(prices / self.strike)
        m, t, v = map(torch.as_tensor, (max_log_moneyness, expiry_time, volatility))
        m = torch.max(s, m)  # `s + epsilon` may be greater than `m`

        price = self.price(s, m, t, v)
        delta = torch.autograd.grad(
            price,
            inputs=prices,
            grad_outputs=torch.ones_like(prices),
            create_graph=create_graph,
        )[0]

        return delta

    @torch.enable_grad()
    def gamma(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        expiry_time: Tensor,
        volatility: Tensor,
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the prices of the
                underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor):
                Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - max_log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            Tensor
        """
        prices = self.strike * torch.exp(torch.as_tensor(log_moneyness))
        prices = Tensor.requires_grad_(prices)

        s = torch.log(prices / self.strike)
        m, t, v = map(torch.as_tensor, (max_log_moneyness, expiry_time, volatility))
        m = torch.max(s, m)  # `s + epsilon` may be greater than `m`

        delta = self.delta(s, m, t, v, create_graph=True)
        gamma = torch.autograd.grad(
            delta, prices, grad_outputs=torch.ones_like(prices)
        )[0]

        return gamma

    def price(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        expiry_time: Tensor,
        volatility: Tensor,
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the prices of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - max_log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            Tensor
        """
        s, m, t, v = map(
            torch.as_tensor, (log_moneyness, max_log_moneyness, expiry_time, volatility)
        )

        d1 = self.d1(s, t, v)
        d2 = self.d2(s, t, v)
        e1 = (s - m + (v ** 2 / 2) * t) / (v * torch.sqrt(t))  # d' in paper
        e2 = (s - m - (v ** 2 / 2) * t) / (v * torch.sqrt(t))

        # when max moneyness < strike
        price_0 = self.strike * (
            torch.exp(s) * self.N.cdf(d1)
            - self.N.cdf(d2)
            + torch.exp(s) * v * torch.sqrt(t) * (d1 * self.N.cdf(d1) + self.N.pdf(d1))
        )
        # when max moneyness >= strike
        price_1 = self.strike * (
            torch.exp(s) * self.N.cdf(e1)
            - torch.exp(m) * self.N.cdf(e2)
            + torch.exp(m)
            - 1
            + torch.exp(s) * v * torch.sqrt(t) * (e1 * self.N.cdf(e1) + self.N.pdf(e1))
        )

        return torch.where(m < 0, price_0, price_1)

    def implied_volatility(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        expiry_time: Tensor,
        price: Tensor,
        precision: float = 1e-6,
    ) -> Tensor:
        """Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the prices of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            expiry_time (torch.Tensor): Time to expiry of the option.
            price (torch.Tensor): Price of the derivative.
            precision (float, default=1e-6): Computational precision of the implied volatility.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - max_log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - price: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            Tensor
        """
        s, m, t, p = map(
            torch.as_tensor, (log_moneyness, max_log_moneyness, expiry_time, price)
        )
        get_price = lambda volatility: self.price(s, m, t, volatility)
        return bisect(get_price, p, lower=0.001, upper=1.000, precision=precision)
