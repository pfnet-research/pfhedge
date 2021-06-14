import torch
from torch import Tensor

from ..._utils.bisect import bisect
from ._base import BSModuleMixin


class BSEuropeanOption(BSModuleMixin):
    """Black-Scholes formula for a European option.

    Args:
        derivative (:class:`pfhedge.instruments.EuropeanOption`, optional):
            The derivative to get the Black-Scholes formula.
        call (bool, default=True): Specifies whether the option is
            call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input : :math:`(N, *, 3)`, where :math:`*` means any number
          of additional dimensions. See `features()` for input features.
        - Output: :math:`(N, *, 1)`. Delta of the derivative.
          All but the last dimension are the same shape as the input.

    .. seealso ::

        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.

    Examples:

        The `forward` method returns delta of the derivative.

        >>> from pfhedge.nn import BSEuropeanOption
        >>> m = BSEuropeanOption()
        >>> m.features()
        ['log_moneyness', 'expiry_time', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, 0.1, 0.2],
        ...     [ 0.00, 0.1, 0.2],
        ...     [ 0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[0.4497],
                [0.5126],
                [0.5752]])

        One can instantiate it using a
        :class:`pfhedge.instruments.EuropeanOption`.

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> deriv = EuropeanOption(BrownianStock(), call=False)
        >>> m = BSEuropeanOption(deriv)
        >>> m
        BSEuropeanOption(call=False)
    """

    def __init__(self, derivative=None, call=True, strike=1.0):
        super().__init__()

        if derivative is not None:
            self.call = derivative.call
            self.strike = derivative.strike
        else:
            self.call = call
            self.strike = strike

    def extra_repr(self):
        params = []
        if not self.call:
            params.append(f"call={self.call}")
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        return ", ".join(params)

    def delta(
        self, log_moneyness: Tensor, expiry_time: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns delta of the derivative.

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
            Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        delta = self.N.cdf(self.d1(s, t, v))
        delta = delta - 1 if not self.call else delta  # put-call parity

        return delta

    def gamma(
        self, log_moneyness: Tensor, expiry_time: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness: (torch.Tensor): Log moneyness of the underlying asset.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            Tensor
        """
        if not self.call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))
        price = self.strike * torch.exp(s)
        gamma = self.N.pdf(self.d1(s, t, v)) / (price * v * torch.sqrt(t))

        return gamma

    def price(
        self, log_moneyness: Tensor, expiry_time: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            expiry_time (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        n1 = self.N.cdf(self.d1(s, t, v))
        n2 = self.N.cdf(self.d2(s, t, v))

        price = self.strike * (torch.exp(s) * n1 - n2)

        if not self.call:
            price += self.strike * (torch.exp(s) - 1)  # put-call parity

        return price

    def implied_volatility(
        self, log_moneyness: Tensor, expiry_time: Tensor, price: Tensor, precision=1e-6
    ) -> Tensor:
        """
        Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            expiry_time (torch.Tensor): Time to expiry of the option.
            price (torch.Tensor): Price of the derivative.
            precision (float): Computational precision of the
                implied volatility.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - expiry_time: :math:`(N, *)`
            - price: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns
            Tensor
        """
        s, t, p = map(torch.as_tensor, (log_moneyness, expiry_time, price))
        get_price = lambda v: self.price(s, t, v)
        return bisect(get_price, p, lower=0.001, upper=1.000, precision=precision)
