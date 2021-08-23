import torch
from torch import Tensor

from pfhedge._utils.bisect import bisect
from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.str import _format_float

from ._base import BSModuleMixin


class BSEuropeanOption(BSModuleMixin):
    """Black-Scholes formula for a European option.

    Args:
        call (bool, default=True): Specifies whether the option is call or put.
        strike (float, default=1.0): The strike price of the option.

    Shape:
        - Input : :math:`(N, *, 3)`, where :math:`*` means any number
          of additional dimensions. See ``inputs`` for the names of input features.
        - Output: :math:`(N, *, 1)`. Delta of the derivative.
          All but the last dimension are the same shape as the input.

    .. seealso ::

        - :class:`pfhedge.nn.BlackScholes`:
          Initialize Black-Scholes formula module from a derivative.

    Examples:

        The ``forward`` method returns delta of the derivative.

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

    References:
        John C. Hull, 2003. Options futures and other derivatives. Pearson.
    """

    def __init__(self, call: bool = True, strike: float = 1.0):
        super().__init__()
        self.call = call
        self.strike = strike

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
        return cls(call=derivative.call, strike=derivative.strike)

    def extra_repr(self) -> str:
        params = []
        if not self.call:
            params.append("call=" + str(self.call))
        params.append("strike=" + _format_float(self.strike))
        return ", ".join(params)

    def delta(
        self, log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns delta of the derivative.

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
            Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, time_to_maturity, volatility))

        delta = self.N.cdf(self.d1(s, t, v))
        delta = delta - 1 if not self.call else delta  # put-call parity

        return delta

    def gamma(
        self, log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns gamma of the derivative.

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
            Tensor
        """
        if not self.call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

        s, t, v = map(torch.as_tensor, (log_moneyness, time_to_maturity, volatility))
        price = self.strike * s.exp()
        gamma = self.N.log_prob(self.d1(s, t, v)).exp() / (price * v * t.sqrt())

        return gamma

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
            Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, time_to_maturity, volatility))

        n1 = self.N.cdf(self.d1(s, t, v))
        n2 = self.N.cdf(self.d2(s, t, v))

        price = self.strike * (s.exp() * n1 - n2)

        if not self.call:
            price += self.strike * (1 - s.exp())  # put-call parity

        return price

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
            precision (float): Computational precision of the
                implied volatility.

        Shape:
            - log_moneyness: :math:`(N, *)`
            - time_to_maturity: :math:`(N, *)`
            - price: :math:`(N, *)`
            - output: :math:`(N, *)`

        Returns
            Tensor
        """
        s, t, p = map(torch.as_tensor, (log_moneyness, time_to_maturity, price))
        pricer = lambda v: self.price(s, t, v)
        return bisect(pricer, p, lower=0.001, upper=1.000, precision=precision)


# Assign docstrings so they appear in Sphinx documentation
set_attr_and_docstring(BSEuropeanOption, "inputs", BSModuleMixin.inputs)
set_attr_and_docstring(BSEuropeanOption, "forward", BSModuleMixin.forward)
