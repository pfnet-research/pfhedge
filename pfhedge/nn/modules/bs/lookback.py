from typing import Optional

import torch
from torch import Tensor

import pfhedge.autogreek as autogreek
from pfhedge._utils.bisect import bisect

from ._base import BSModuleMixin


class BSLookbackOption(BSModuleMixin):
    """Black-Scholes formula for a lookback option with a fixed strike.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 4)`, where :math:`*` means any number of additional
          dimensions.  See `inputs()` for the names of input features.
        - Output: :math:`(N, *, 1)`. Delta of the derivative.
          All but the last dimension are the same shape as the input.

    .. seealso ::

        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.

    Examples:

        The `forward` method returns delta of the derivative.

        >>> from pfhedge.nn import BSLookbackOption
        >>>
        >>> m = BSLookbackOption()
        >>> m.inputs()
        ['log_moneyness', 'max_log_moneyness', 'expiry_time', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, -0.01, 0.1, 0.2],
        ...     [ 0.00,  0.00, 0.1, 0.2],
        ...     [ 0.01,  0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[0.9208],
                [1.0515],
                [1.0515]])

    References:
        Conze, A., 1991. Path dependent options: The case of lookback options.
        The Journal of Finance, 46(5), pp.1893-1907.
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
            derivative (:class:`pfhedge.instruments.LookbackOption`):
                The derivative to get the Black-Scholes formula.

        Returns:
            BSLookbackOption

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import LookbackOption
            >>>
            >>> derivative = LookbackOption(BrownianStock(), strike=1.1)
            >>> m = BSLookbackOption.from_derivative(derivative)
            >>> m
            BSLookbackOption(strike=1.1)
        """
        return cls(call=derivative.call, strike=derivative.strike)

    def extra_repr(self):
        params = []
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        return ", ".join(params)

    def inputs(self) -> list:
        return ["log_moneyness", "max_log_moneyness", "expiry_time", "volatility"]

    def price(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        expiry_time: Tensor,
        volatility: Tensor,
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
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
        e1 = (s - m + (v ** 2 / 2) * t) / (v * t.sqrt())  # d' in paper
        e2 = (s - m - (v ** 2 / 2) * t) / (v * t.sqrt())

        # when max moneyness < strike
        price_0 = self.strike * (
            s.exp() * self.N.cdf(d1)
            - self.N.cdf(d2)
            + s.exp() * v * t.sqrt() * (d1 * self.N.cdf(d1) + self.N.pdf(d1))
        )
        # when max moneyness >= strike
        price_1 = self.strike * (
            s.exp() * self.N.cdf(e1)
            - m.exp() * self.N.cdf(e2)
            + m.exp()
            - 1
            + s.exp() * v * t.sqrt() * (e1 * self.N.cdf(e1) + self.N.pdf(e1))
        )

        return torch.where(m < 0, price_0, price_1)

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        expiry_time: Tensor,
        volatility: Tensor,
        create_graph: bool = False,
        strike: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
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
        return autogreek.delta(
            self.price,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            expiry_time=expiry_time,
            volatility=volatility,
            create_graph=create_graph,
        )

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
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
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
        return autogreek.gamma(
            self.price,
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            expiry_time=expiry_time,
            volatility=volatility,
        )

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
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            expiry_time (torch.Tensor): Time to expiry of the option.
            price (torch.Tensor): Price of the derivative.
            precision (float, default=1e-6): Precision of the implied volatility.

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
        pricer = lambda volatility: self.price(s, m, t, volatility)
        return bisect(pricer, p, lower=0.001, upper=1.000, precision=precision)


# Assign docstrings so they appear in Sphinx documentation
BSLookbackOption.inputs.__doc__ = BSModuleMixin.inputs.__doc__
BSLookbackOption.forward = BSModuleMixin.forward
BSLookbackOption.forward.__doc__ = BSModuleMixin.forward.__doc__
