import torch
from torch import Tensor

import pfhedge.autogreek as autogreek
from pfhedge._utils.bisect import bisect

from ._base import BSModuleMixin


class BSEuropeanBinaryOption(BSModuleMixin):
    """Black-Scholes formula for a European binary option.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 3)`, where :math:`*` means any number of additional
          dimensions. See `inputs()` for the names of input features.
        - Output: :math:`(N, *, 1)` Delta of the derivative.
          All but the last dimension are the same shape as the input.

    .. seealso ::

        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.

    Examples:

        The `forward` method returns delta of the derivative.

        >>> from pfhedge.nn import BSEuropeanBinaryOption
        >>>
        >>> m = BSEuropeanBinaryOption(strike=1.0)
        >>> m.inputs()
        ['log_moneyness', 'expiry_time', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, 0.1, 0.2],
        ...     [ 0.00, 0.1, 0.2],
        ...     [ 0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[6.2576],
                [6.3047],
                [6.1953]])

    References:
        John C. Hull, 2003. Options futures and other derivatives. Pearson.
    """

    def __init__(self, call: bool = True, strike: float = 1.0):
        if not call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

        super().__init__()
        self.call = call
        self.strike = strike

    @classmethod
    def from_derivative(cls, derivative):
        """Initialize a module from a derivative.

        Args:
            derivative (:class:`pfhedge.instruments.EuropeanBinaryOption`):
                The derivative to get the Black-Scholes formula.

        Returns:
            BSEuropeanBinaryOption

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanBinaryOption
            >>>
            >>> derivative = EuropeanBinaryOption(BrownianStock(), strike=1.1)
            >>> m = BSEuropeanBinaryOption.from_derivative(derivative)
            >>> m
            BSEuropeanBinaryOption(strike=1.1)
        """
        return cls(call=derivative.call, strike=derivative.strike)

    def extra_repr(self):
        params = []
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        return ", ".join(params)

    def inputs(self) -> list:
        return ["log_moneyness", "expiry_time", "volatility"]

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

    @torch.enable_grad()
    def delta(
        self, log_moneyness: Tensor, expiry_time: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns delta of the derivative.

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
            torch.Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        delta = self.N.pdf(self.d2(s, t, v)) / (self.strike * s.exp() * v * t.sqrt())
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
        return autogreek.gamma(
            self.price,
            strike=self.strike,
            log_moneyness=log_moneyness,
            expiry_time=expiry_time,
            volatility=volatility,
        )

    def implied_volatility(
        self,
        log_moneyness: Tensor,
        expiry_time: Tensor,
        price: Tensor,
        precision: float = 1e-6,
    ) -> Tensor:
        """Returns implied volatility of the derivative.

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
        pricer = lambda v: self.price(s, t, v)
        return bisect(pricer, p, lower=0.001, upper=1.000, precision=precision)


# Assign docstrings so they appear in Sphinx documentation
BSEuropeanBinaryOption.inputs.__doc__ = BSModuleMixin.inputs.__doc__
BSEuropeanBinaryOption.forward = BSModuleMixin.forward
BSEuropeanBinaryOption.forward.__doc__ = BSModuleMixin.forward.__doc__
