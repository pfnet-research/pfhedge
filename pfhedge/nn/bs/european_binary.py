import torch

from ..._utils.bisect import bisect
from ._base import BSModuleMixin


class BSEuropeanBinaryOption(BSModuleMixin):
    """
    Black-Scholes formula for a European binary option.

    Parameters
    ----------
    - derivative : EuropeanBinaryOption
        The derivative to get the Black-Scholes formula.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.

    Shape
    -----
    - Input : (N, *, 3)
        See `features()` for input features.
        Here, `*` means any number of additional dimensions.
    - Output : (N, *, 1)
        Delta of the derivative.
        All but the last dimension are the same shape as the input.

    Examples
    --------
    >>> m = BSEuropeanBinaryOption(strike=1.0)
    >>> m.features()
    ['log_moneyness', 'expiry_time', 'volatility']
    >>> x = torch.tensor([
    ...     [-0.01, 0.1, 0.2],
    ...     [ 0.00, 0.1, 0.2],
    ...     [ 0.01, 0.1, 0.2]])
    >>> m(x)
    tensor([[6.2576],
            [6.3047],
            [6.1953]])
    """

    def __init__(self, derivative=None, call=True, strike=1.0):
        super().__init__()

        if derivative is not None:
            self.call = derivative.call
            self.strike = derivative.strike
        else:
            self.call = call
            self.strike = strike

        if not self.call:
            raise ValueError(
                f"{self.__class__.__name__} for a put option is not yet supported."
            )

    def extra_repr(self):
        params = []
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        return ", ".join(params)

    def features(self):
        return ["log_moneyness", "expiry_time", "volatility"]

    @torch.enable_grad()
    def delta(
        self, log_moneyness, expiry_time, volatility, create_graph=False
    ) -> torch.Tensor:
        """
        Returns delta of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - volatility : Tensor, shape (N, *)
            Volatility of the underlying asset.
        - create_graph : bool, default False
            If True, graph of the derivative will be constructed.
            This option is used to compute gamma.

        Returns
        -------
        delta : Tensor, shape (N, *)
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        delta = self.N.pdf(self.d2(s, t, v)) / (
            self.strike * torch.exp(s) * v * torch.sqrt(t)
        )
        return delta

    def gamma(self, log_moneyness, expiry_time, volatility) -> torch.Tensor:
        """
        Returns gamma of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - volatility : Tensor, shape (N, *)
            Volatility of the underlying asset.

        Returns
        -------
        gamma : Tensor, shape (N, *)
        """
        prices = self.strike * torch.exp(torch.as_tensor(log_moneyness))
        prices = torch.Tensor.requires_grad_(prices)

        s = torch.log(prices / self.strike)
        t, v = map(torch.as_tensor, (expiry_time, volatility))

        delta = self.delta(s, t, v, create_graph=True)
        gamma = torch.autograd.grad(
            delta, prices, grad_outputs=torch.ones_like(prices)
        )[0]

        return gamma

    def price(self, log_moneyness, expiry_time, volatility) -> torch.Tensor:
        """
        Returns price of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - volatility : Tensor, shape (N, *)
            Volatility of the underlying asset.

        Returns
        -------
        price : Tensor, shape (N, *)
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))

        price = self.N.cdf(self.d2(s, t, v))
        price = 1.0 - price if not self.call else price  # put-call parity

        return price

    def implied_volatility(
        self, log_moneyness, expiry_time, price, precision=1e-6
    ) -> torch.Tensor:
        """
        Returns implied volatility of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - price : Tensor, shape (N, *)
            Price of the derivative.
        - precision : float, default 1e-6
            Computational precision of the implied volatility.

        Returns
        -------
        implied_volatility : Tensor, shape (N, *)
        """
        s, t, p = map(torch.as_tensor, (log_moneyness, expiry_time, price))
        get_price = lambda v: self.price(s, t, v)
        return bisect(get_price, p, lower=0.001, upper=1.000, precision=precision)
