from typing import List

import torch
from torch import Tensor
from torch.distributions.utils import broadcast_all

from pfhedge._utils.bisect import find_implied_volatility
from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge.nn.functional import d1
from pfhedge.nn.functional import d2
from pfhedge.nn.functional import ncdf
from pfhedge.nn.functional import npdf

from ._base import BSModuleMixin


class BSEuropeanBinaryOption(BSModuleMixin):
    """Black-Scholes formula for a European binary option.

    Note:
        Risk-free rate is set to zero.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input: :math:`(N, *, 3)`, where
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.

    .. seealso::
        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.
        - :class:`pfhedge.instruments.EuropeanBinaryOption`:
          Corresponding derivative.

    References:
        - John C. Hull, 2003. Options futures and other derivatives. Pearson.

    Examples:

        >>> from pfhedge.nn import BSEuropeanBinaryOption
        >>>
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
            BSEuropeanBinaryOption(strike=1.1000)
        """
        return cls(call=derivative.call, strike=derivative.strike)

    def extra_repr(self) -> str:
        params = []
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def inputs(self) -> List[str]:
        return ["log_moneyness", "time_to_maturity", "volatility"]

    def price(
        self, log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns price of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

        price = ncdf(d2(s, t, v))
        price = 1.0 - price if not self.call else price  # put-call parity

        return price

    @torch.enable_grad()
    def delta(
        self, log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns delta of the derivative.

        Args:
            log_moneyness: (torch.Tensor): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        s, t, v = broadcast_all(log_moneyness, time_to_maturity, volatility)

        delta = npdf(d2(s, t, v)) / (self.strike * s.exp() * v * t.sqrt())
        return delta

    def gamma(
        self, log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns gamma of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - volatility: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        return super().gamma(
            strike=self.strike,
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            volatility=volatility,
        )

    def implied_volatility(
        self,
        log_moneyness: Tensor,
        time_to_maturity: Tensor,
        price: Tensor,
        precision: float = 1e-6,
    ) -> Tensor:
        """Returns implied volatility of the derivative.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            price (torch.Tensor): Price of the derivative.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns:
            torch.Tensor
        """
        return find_implied_volatility(
            self.price,
            price=price,
            log_moneyness=log_moneyness,
            time_to_maturity=time_to_maturity,
            precision=precision,
        )


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BSEuropeanBinaryOption, "inputs", BSModuleMixin.inputs)
_set_attr_and_docstring(BSEuropeanBinaryOption, "forward", BSModuleMixin.forward)
