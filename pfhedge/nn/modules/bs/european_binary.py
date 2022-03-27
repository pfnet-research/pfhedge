from typing import List
from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import EuropeanBinaryOption
from pfhedge.nn.functional import bs_european_binary_delta
from pfhedge.nn.functional import bs_european_binary_gamma
from pfhedge.nn.functional import bs_european_binary_price
from pfhedge.nn.functional import bs_european_binary_theta
from pfhedge.nn.functional import bs_european_binary_vega

from ._base import BSModuleMixin
from ._base import acquire_params_from_derivative_0
from ._base import acquire_params_from_derivative_1
from .black_scholes import BlackScholesModuleFactory


class BSEuropeanBinaryOption(BSModuleMixin):
    """Black-Scholes formula for a European binary option.

    Note:
        Risk-free rate is set to zero.

    .. seealso::
        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.
        - :class:`pfhedge.instruments.EuropeanBinaryOption`:
          Corresponding derivative.

    References:
        - John C. Hull, 2003. Options futures and other derivatives. Pearson.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 3)`, where
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.

    Examples:
        >>> from pfhedge.nn import BSEuropeanBinaryOption
        ...
        >>> m = BSEuropeanBinaryOption(strike=1.0)
        >>> m.inputs()
        ['log_moneyness', 'time_to_maturity', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, 0.1, 0.2],
        ...     [ 0.00, 0.1, 0.2],
        ...     [ 0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[6.2576],
                [6.3047],
                [6.1953]])
    """

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[EuropeanBinaryOption] = None,
    ) -> None:
        super().__init__()
        self.call = call
        self.strike = strike
        self.derivative = derivative

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
            BSEuropeanBinaryOption(strike=1.1000)
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

    def inputs(self) -> List[str]:
        return ["log_moneyness", "time_to_maturity", "volatility"]

    def price(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional)): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional)): Time to expiry of the option.
            volatility (torch.Tensor, optional)): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
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
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_price(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
        )

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness: (torch.Tensor, optional): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
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
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_delta(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

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
            - log_moneyness: :math:`(N, *)`
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
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_gamma(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
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
            - log_moneyness: :math:`(N, *)`
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
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_vega(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

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
            - log_moneyness: :math:`(N, *)`
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
        ) = acquire_params_from_derivative_1(
            self.derivative, log_moneyness, time_to_maturity, volatility
        )
        return bs_european_binary_theta(
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            call=self.call,
            strike=self.strike,
        )

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
            - log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - price: :math:`(N, *)`
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
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


factory = BlackScholesModuleFactory()
factory.register_module("EuropeanBinaryOption", BSEuropeanBinaryOption)

# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BSEuropeanBinaryOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSEuropeanBinaryOption, "forward", BSModuleMixin.forward)
