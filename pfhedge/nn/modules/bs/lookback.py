from typing import List
from typing import Optional

import torch
from torch import Tensor

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.instruments import LookbackOption
from pfhedge.nn.functional import bs_lookback_delta
from pfhedge.nn.functional import bs_lookback_gamma
from pfhedge.nn.functional import bs_lookback_price
from pfhedge.nn.functional import bs_lookback_theta
from pfhedge.nn.functional import bs_lookback_vega

from ._base import BSModuleMixin
from ._base import acquire_params_from_derivative_0
from ._base import acquire_params_from_derivative_2
from .black_scholes import BlackScholesModuleFactory


class BSLookbackOption(BSModuleMixin):
    """Black-Scholes formula for a lookback option with a fixed strike.

    Note:
        - The formulas are for continuous monitoring while
          :class:`pfhedge.instruments.LookbackOption` monitors spot prices discretely.
          To get adjustment for discrete monitoring, see, for instance,
          Broadie, Glasserman, and Kou (1999).

    .. seealso::
        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.
        - :class:`pfhedge.instruments.LookbackOption`:
          Corresponding derivative.

    References:
        - Conze, A., 1991. Path dependent options: The case of lookback options.
          The Journal of Finance, 46(5), pp.1893-1907.
        - Broadie, M., Glasserman, P. and Kou, S.G., 1999.
          Connecting discrete and continuous path-dependent options.
          Finance and Stochastics, 3(1), pp.55-82.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 4)` where
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.

    Examples:
        >>> from pfhedge.nn import BSLookbackOption
        >>>
        >>> m = BSLookbackOption()
        >>> m.inputs()
        ['log_moneyness', 'max_log_moneyness', 'time_to_maturity', 'volatility']
        >>> input = torch.tensor([
        ...     [-0.01, -0.01, 0.1, 0.2],
        ...     [ 0.00,  0.00, 0.1, 0.2],
        ...     [ 0.01,  0.01, 0.1, 0.2]])
        >>> m(input)
        tensor([[0.9208],
                [1.0515],
                [1.0515]])
    """

    def __init__(
        self,
        call: bool = True,
        strike: float = 1.0,
        derivative: Optional[LookbackOption] = None,
    ) -> None:
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
            BSLookbackOption(strike=1.1000)
        """
        return cls(
            call=derivative.call, strike=derivative.strike, derivative=derivative
        )

    def extra_repr(self) -> str:
        params = []
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def inputs(self) -> List[str]:
        return ["log_moneyness", "max_log_moneyness", "time_to_maturity", "volatility"]

    def price(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Returns price of the derivative.

        The price is given by:

        .. math::
            \begin{cases}
                S(0) \{
                    N(d_1) + \sigma \sqrt{T} [N'(d_1) + d_1 N(d_1)]
                \} - K N(d_2)
                & (M \leq K) \\
                S(0) \{
                    N(d_1') + \sigma \sqrt{T} [N'(d_1') + d_1' N(d_1')]
                \} - K + M [1 - N(d_2')]
                & (M > K) \\
            \end{cases}
        where
        :math:`M = \max_{t < 0} S(t)`,
        :math:`d_1' = [\log(S(0) / M) + \frac12 \sigma^2 T] / \sigma \sqrt{T}`, and
        :math:`d_2' = [\log(S(0) / M) - \frac12 \sigma^2 T] / \sigma \sqrt{T}`.

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
        return bs_lookback_price(
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            strike=self.strike,
        )

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Optional[Tensor] = None,
        max_log_moneyness: Optional[Tensor] = None,
        time_to_maturity: Optional[Tensor] = None,
        volatility: Optional[Tensor] = None,
        create_graph: bool = False,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor, optional): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor, optional): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor, optional): Time to expiry of the option.
            volatility (torch.Tensor, optional): Volatility of the underlying asset.
            create_graph (bool, default=False): If True, graph of the derivative
                will be constructed. This option is used to compute gamma.

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
        return super().delta(
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            create_graph=create_graph,
            strike=self.strike,
        )

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
            volatility (torch.Tensor, optional):
                Volatility of the underlying asset.

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
            volatility (torch.Tensor, optional):
                Volatility of the underlying asset.

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
            volatility (torch.Tensor, optional):
                Volatility of the underlying asset.

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
            price (torch.Tensor): Price of the derivative.
            precision (float, default=1e-6): Precision of the implied volatility.

        Shape:
            - log_moneyness: :math:`(N, *)` where
              :math:`*` means any number of additional dimensions.
            - max_log_moneyness: :math:`(N, *)`
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
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


factory = BlackScholesModuleFactory()
factory.register_module("LookbackOption", BSLookbackOption)

# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BSLookbackOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSLookbackOption, "forward", BSModuleMixin.forward)
