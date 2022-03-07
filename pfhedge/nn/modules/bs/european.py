from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.distributions.utils import broadcast_all

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import EuropeanOption
from pfhedge.nn.functional import d1
from pfhedge.nn.functional import d2
from pfhedge.nn.functional import ncdf
from pfhedge.nn.functional import npdf

from ._base import BSModuleMixin


class BSEuropeanOption(BSModuleMixin):
    """Black-Scholes formula for a European option.

    Note:
        Risk-free rate is set to zero.

    .. seealso::
        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.
        - :class:`pfhedge.instruments.EuropeanOption`:
          Corresponding derivative.

    References:
        - John C. Hull, 2003. Options futures and other derivatives. Pearson.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 3)` where
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.

    Examples:
        >>> from pfhedge.nn import BSEuropeanOption
        >>>
        >>> m = BSEuropeanOption()
        >>> m.inputs()
        ['log_moneyness', 'time_to_maturity', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, 0.1, 0.2],
        ...     [ 0.00, 0.1, 0.2],
        ...     [ 0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[0.4497],
                [0.5126],
                [0.5752]])
    """

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[EuropeanOption] = None,
    ):
        super().__init__()
        self.call = call
        self.strike = strike
        self.derivative = derivative

    @classmethod
    def from_derivative(cls, derivative):
        """Initialize a module from a derivative.

        Args:
            derivative (:class:`pfhedge.instruments.EuropeanOption`):
                The derivative to get the Black-Scholes formula.

        Returns:
            BSEuropeanOption

        Examples:
            >>> from pfhedge.instruments import BrownianStock
            >>> from pfhedge.instruments import EuropeanOption
            >>>
            >>> derivative = EuropeanOption(BrownianStock(), call=False)
            >>> m = BSEuropeanOption.from_derivative(derivative)
            >>> m
            BSEuropeanOption(call=False, strike=1.)
        """
        return cls(
            call=derivative.call, strike=derivative.strike, derivative=derivative
        )

    def extra_repr(self) -> str:
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def _acquire_params_from_derivative(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if log_moneyness is None:
            if self.derivative is None:
                raise ValueError(
                    "log_moneyness is required if derivative is not set at this initialization."
                )
            if self.derivative.ul().spot is None:
                raise AttributeError("please simulate first")
            log_moneyness = self.derivative.log_moneyness()
        if time_to_maturity is None:
            if self.derivative is None:
                raise ValueError(
                    "time_to_maturity is required if derivative is not set at this initialization."
                )
            time_to_maturity = self.derivative.time_to_maturity()
        if volatility is None:
            if self.derivative is None:
                raise ValueError(
                    "time_to_maturity is required if derivative is not set at this initialization."
                )
            if self.derivative.ul().volatility is None:
                raise AttributeError(
                    "please simulate first and check if volatility exists in the derivative's underlier."
                )
            volatility = self.derivative.ul().volatility
        return log_moneyness, time_to_maturity, volatility

    def delta(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
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
            time_to_maturity,
            volatility,
        ) = self._acquire_params_from_derivative(
            log_moneyness, time_to_maturity, volatility
        )
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

        delta = ncdf(d1(s, t, v))
        delta = delta - 1 if not self.call else delta  # put-call parity

        return delta

    def gamma(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
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
            time_to_maturity,
            volatility,
        ) = self._acquire_params_from_derivative(
            log_moneyness, time_to_maturity, volatility
        )
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
        price = self.strike * s.exp()
        numerator = npdf(d1(s, t, v))
        denominator = price * v * t.sqrt()
        output = numerator / denominator
        return torch.where(
            (numerator == 0).logical_and(denominator == 0),
            torch.zeros_like(output),
            output,
        )

    def vega(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns vega of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
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
            time_to_maturity,
            volatility,
        ) = self._acquire_params_from_derivative(
            log_moneyness, time_to_maturity, volatility
        )
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
        price = self.strike * s.exp()
        vega = npdf(d1(s, t, v)) * price * t.sqrt()

        return vega

    def theta(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns theta of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
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
            time_to_maturity,
            volatility,
        ) = self._acquire_params_from_derivative(
            log_moneyness, time_to_maturity, volatility
        )
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)
        price = self.strike * s.exp()
        numerator = -npdf(d1(s, t, v)) * price * v
        denominator = 2 * t.sqrt()
        output = numerator / denominator
        return torch.where(
            (numerator == 0).logical_and(denominator == 0),
            torch.zeros_like(output),
            output,
        )

    def price(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
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
            time_to_maturity,
            volatility,
        ) = self._acquire_params_from_derivative(
            log_moneyness, time_to_maturity, volatility
        )
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

        n1 = ncdf(d1(s, t, v))
        n2 = ncdf(d2(s, t, v))

        price = self.strike * (s.exp() * n1 - n2)

        if not self.call:
            price += self.strike * (1 - s.exp())  # put-call parity

        return price

    def implied_volatility(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        price: Optional[Tensor] = None,
        precision: float = 1e-6,
    ) -> Tensor:
        """Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            price (torch.Tensor): Price of the derivative.
            precision (float): Computational precision of the
                implied volatility.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - time_to_maturity: :math:`(N, *)`
            - price: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns
            torch.Tensor

        Note:
            args are not optional if it doesn't accept derivative in this initialization.
            price seems optional in typing, but it isn't. It is set for the compatibility to the previous versions.
        """
        if log_moneyness is None:
            if self.derivative is None:
                raise ValueError(
                    "log_moneyness is required if derivative is not set at this initialization."
                )
            if self.derivative.ul().spot is None:
                raise AttributeError("please simulate first")
            log_moneyness = self.derivative.log_moneyness()
        if time_to_maturity is None:
            if self.derivative is None:
                raise ValueError(
                    "time_to_maturity is required if derivative is not set at this initialization."
                )
            time_to_maturity = self.derivative.time_to_maturity()
        if price is None:
            raise ValueError(
                "price is required in this method. None is set only for compatibility to the previous versions."
            )

        return find_implied_volatility(
            self.price,
            price=price,
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


# Assign docstrings so they appear in Sphinx documentation
_set_attr_and_docstring(BSEuropeanOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSEuropeanOption, "forward", BSModuleMixin.forward)
