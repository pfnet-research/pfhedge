from math import sqrt

import torch
from torch import Tensor
from torch.distributions.utils import broadcast_all

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import d1
from pfhedge.nn.functional import d2
from pfhedge.nn.functional import ncdf
from pfhedge.nn.functional import npdf

from ._base import BSModuleMixin


class BSAmericanBinaryOption(BSModuleMixin):
    """Black-Scholes formula for an American Binary Option.

    Note:
        Risk-free rate is set to zero.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 4)` where
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.

    .. seealso::
        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.
        - :class:`pfhedge.instruments.AmericanBinaryOption`:
          Corresponding derivative.

    References:
        - Dai, M., 2000. A closed-form solution for perpetual American floating strike
          lookback options. Journal of Computational Finance, 4(2), pp.63-68.

    Examples:

        >>> from pfhedge.nn import BSAmericanBinaryOption
        >>>
        >>> m = BSAmericanBinaryOption(strike=1.0)
        >>> m.inputs()
        ['log_moneyness', 'max_log_moneyness', 'time_to_maturity', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, -0.01, 0.1, 0.2],
        ...     [ 0.00,  0.00, 0.1, 0.2],
        ...     [ 0.01,  0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[1.1285],
                [0.0000],
                [0.0000]])
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
            derivative (:class:`pfhedge.instruments.AmericanBinaryOption`):
                The derivative to get the Black-Scholes formula.

        Returns:
            BSAmericanBinaryOption

        Examples:

            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import AmericanBinaryOption
            >>>
            >>> derivative = AmericanBinaryOption(BrownianStock(), strike=1.1)
            >>> m = BSAmericanBinaryOption.from_derivative(derivative)
            >>> m
            BSAmericanBinaryOption(strike=1.1000)
        """
        return cls(call=derivative.call, strike=derivative.strike)

    def extra_repr(self) -> str:
        params = []
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def price(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        time_to_maturity: Tensor,
        volatility: Tensor,
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, m, t, v = broadcast_all(
            log_moneyness, max_log_moneyness, time_to_maturity, volatility
        )

        n1 = ncdf(d1(s, t, v) / sqrt(2))
        n2 = ncdf(d2(s, t, v) / sqrt(2))
        p = (1 / 2) * (s.exp() * (1 + n1) + n2)

        return p.where(m < 0, torch.ones_like(p))

    def delta(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        time_to_maturity: Tensor,
        volatility: Tensor,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, m, t, v = broadcast_all(
            log_moneyness, max_log_moneyness, time_to_maturity, volatility
        )

        c1 = ncdf(d1(s, t, v) / sqrt(2))
        p1 = npdf(d1(s, t, v) / sqrt(2))
        p2 = npdf(d2(s, t, v) / sqrt(2))
        d = (1 + c1 + (p1 + p2)) / (2 * self.strike)

        return d.where(m < 0, torch.zeros_like(d))

    @torch.enable_grad()
    def gamma(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        time_to_maturity: Tensor,
        volatility: Tensor,
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        return super().gamma(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    def implied_volatility(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        time_to_maturity: Tensor,
        price: Tensor,
        precision: float = 1e-6,
    ) -> Tensor:
        """Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.
            precision (float, default=1e-6): Computational precision of the implied
                volatility.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        return find_implied_volatility(
            self.price,
            price=price,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(BSAmericanBinaryOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSAmericanBinaryOption, "forward", BSModuleMixin.forward)
