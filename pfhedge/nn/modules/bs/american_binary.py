from math import sqrt
from typing import Optional

import torch
from torch import Tensor
from torch.distributions.utils import broadcast_all

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import AmericanBinaryOption
from pfhedge.nn.functional import d1
from pfhedge.nn.functional import d2
from pfhedge.nn.functional import ncdf
from pfhedge.nn.functional import npdf

from ._base import BSModuleMixin
from ._base import acquire_params_from_derivative_0
from ._base import acquire_params_from_derivative_2


class BSAmericanBinaryOption(BSModuleMixin):
    """Black-Scholes formula for an american binary option.

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
        - Shreve, S.E., 2004. Stochastic calculus for finance II:
          Continuous-time models (Vol. 11). Springer Science & Business Media.

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
        tensor([[...],
                [...],
                [...]])
    """

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[AmericanBinaryOption] = None,
    ):
        if not call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

        super().__init__()
        self.call = call
        self.strike = strike
        self.derivative = derivative

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
        return cls(
            call=derivative.call, strike=derivative.strike, derivative=derivative
        )

    def extra_repr(self) -> str:
        params = []
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def price(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor

        Note:
            Parameters are not optional if the module has not accepted a derivative in its initialization.
        """
        # This formula is derived using the results in Section 7.3.3 of Shreve's book.
        # Price is I_2 - I_4 where the interval of integration is [k --> -inf, b].
        # By this substitution we get N([log(S(0) / K) + ...] / sigma T) --> 1.
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )

        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
        p = ncdf(d2(s, t, v)) + s.exp() * (1 - ncdf(d2(-s, t, v)))

        return p.where(max_log_moneyness < 0, torch.ones_like(p))

    def delta(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor

        Note:
            Parameters are not optional if the module has not accepted a derivative in its initialization.
        """
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
        spor = s.exp() * self.strike
        # ToDo: fix 0/0 issue
        p = (
            npdf(d2(s, t, v)) / (spor * v * t.sqrt())
            - (1 - ncdf(d2(-s, t, v))) / self.strike
            + npdf(d2(-s, t, v)) / (self.strike * v * t.sqrt())
        )

        return p.where(max_log_moneyness < 0, torch.zeros_like(p))

    @torch.enable_grad()
    def gamma(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor

        Note:
            args are not optional if it doesn't accept derivative in this initialization.
        """
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().gamma(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    @torch.enable_grad()
    def vega(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns vega of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor

        Note:
            args are not optional if it doesn't accept derivative in this initialization.
        """
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().vega(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    @torch.enable_grad()
    def theta(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns theta of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Note:
            Risk-free rate is set to zero.

        Returns:
            torch.Tensor

        Note:
            args are not optional if it doesn't accept derivative in this initialization.
        """
        (
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        ) = acquire_params_from_derivative_2(
            self.derivative,
            log_moneyness,
            max_log_moneyness,
            time_to_maturity,
            volatility,
        )
        return super().theta(
            strike=self.strike,
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    def implied_volatility(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        price: Optional[Tensor] = None,
        precision: float = 1e-6,
    ) -> Tensor:
        """Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.
            precision (float): Computational precision of the implied volatility.
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

        Note:
            args are not optional if it doesn't accept derivative in this initialization.
            price seems optional in typing, but it isn't. It is set for the compatibility to the previous versions.
        """
        (log_moneyness, time_to_maturity) = acquire_params_from_derivative_0(
            self.derivative, log_moneyness, time_to_maturity
        )
        if price is None:
            raise ValueError(
                "price is required in this method. None is set only for compatibility to the previous versions."
            )
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
