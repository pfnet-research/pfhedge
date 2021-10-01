from typing import List

import torch
from torch import Tensor

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import d1
from pfhedge.nn.functional import d2
from pfhedge.nn.functional import ncdf
from pfhedge.nn.functional import npdf

from ._base import BSModuleMixin


class BSLookbackOption(BSModuleMixin):
    """Black-Scholes formula for a lookback option with a fixed strike.

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
        - :class:`pfhedge.instruments.LookBackOption`:
          Corresponding derivative.

    References:
        - Conze, A., 1991. Path dependent options: The case of lookback options.
          The Journal of Finance, 46(5), pp.1893-1907.

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
            BSLookbackOption(strike=1.1000)
        """
        return cls(call=derivative.call, strike=derivative.strike)

    def extra_repr(self) -> str:
        params = []
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def inputs(self) -> List[str]:
        return ["log_moneyness", "max_log_moneyness", "time_to_maturity", "volatility"]

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
        s, m, t, v = map(
            torch.as_tensor,
            (log_moneyness, max_log_moneyness, time_to_maturity, volatility),
        )

        d1_ = d1(s, t, v)
        d2_ = d2(s, t, v)
        e1 = (s - m + (v.square() / 2) * t) / (v * t.sqrt())  # d' in paper
        e2 = (s - m - (v.square() / 2) * t) / (v * t.sqrt())

        # when max moneyness < strike
        price_0 = self.strike * (
            s.exp() * ncdf(d1_)
            - ncdf(d2_)
            + s.exp() * v * t.sqrt() * (d1_ * ncdf(d1_) + npdf(d1_))
        )
        # when max moneyness >= strike
        price_1 = self.strike * (
            s.exp() * ncdf(e1)
            - m.exp() * ncdf(e2)
            + m.exp()
            - 1
            + s.exp() * v * t.sqrt() * (e1 * ncdf(e1) + npdf(e1))
        )

        return torch.where(m < 0, price_0, price_1)

    @torch.enable_grad()
    def delta(
        self,
        log_moneyness: Tensor,
        max_log_moneyness: Tensor,
        time_to_maturity: Tensor,
        volatility: Tensor,
        create_graph: bool = False,
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            max_log_moneyness (torch.Tensor): Cumulative maximum of the log moneyness.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.
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
        """
        return super().delta(
            log_moneyness=log_moneyness,
            max_log_moneyness=max_log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
            create_graph=create_graph,
        )

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
            volatility (torch.Tensor):
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
_set_docstring(BSLookbackOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSLookbackOption, "forward", BSModuleMixin.forward)
