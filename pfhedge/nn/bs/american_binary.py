import torch

# from ..._utils.bisect import bisect
from ._base import BSModuleMixin


class BSAmericanBinaryOption(BSModuleMixin):
    """
    Black-Scholes formulas for American Binary Option.

    Parameters
    ----------
    - derivative : AmericanBinaryOption
        The derivative to get the Black-Scholes formula.
    - call : bool, default True
        Specify whether the option is call or put.
    - strike : float, default 1.0
        The strike price of the option.

    Shape
    -----
    - Input : (N, *, 4)
        See `features()` for input features.
        Here, `*` means any number of additional dimensions.
    - Output : (N, *, 1)
        Delta of the derivative.
        All but the last dimension are the same shape as the input.

    Examples
    --------
    >>> m = BSAmericanBinaryOption(strike=1.0)
    >>> m.features()
    ['log_moneyness', 'max_log_moneyness', 'expiry_time', 'volatility']
    >>> x = torch.tensor([
    ...     [-0.01, -0.01, 0.1, 0.2],
    ...     [ 0.00,  0.00, 0.1, 0.2],
    ...     [ 0.01,  0.01, 0.1, 0.2]])
    >>> m(x)
    tensor([[1.1285],
            [0.0000],
            [0.0000]])
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
            raise NotImplementedError(
                f"{self.__class__.__name__} for a put option is not yet implemented."
            )

    def extra_repr(self):
        params = []
        # if not self.call:
        #     params.append(f"call={self.call}")
        if self.strike != 1.0:
            params.append(f"strike={self.strike}")
        return ", ".join(params)

    def delta(
        self, log_moneyness, max_log_moneyness, expiry_time, volatility
    ) -> torch.Tensor:
        """
        Returns delta of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - max_log_moneyness : Tensor, shape (N, *)
            Cumulative maximum of the log moneyness.
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
        s, m, t, v = map(
            torch.as_tensor, (log_moneyness, max_log_moneyness, expiry_time, volatility)
        )

        sqrt2 = torch.sqrt(torch.as_tensor(2.0)).item()
        d1 = self.d1(s, t, v)
        d2 = self.d2(s, t, v)
        c1 = self.N.cdf(d1 / sqrt2)
        p1 = self.N.pdf(d1 / sqrt2)
        p2 = self.N.pdf(d2 / sqrt2)

        d = (1 + c1 + (p1 + p2)) / (2 * self.strike)

        return torch.where(m < 0, d, torch.zeros_like(d))

    def gamma(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns gamma of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - max_log_moneyness : Tensor, shape (N, *)
            Cumulative maximum of the log moneyness.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - volatility : Tensor, shape (N, *)
            Volatility of the underlying asset.

        Returns
        -------
        gamma : Tensor, shape (N, *)
        """
        raise NotImplementedError(
            f"gamma of {self.__class__.__name__} is not yet implemented."
        )

    def price(
        self, log_moneyness, max_log_moneyness, expiry_time, volatility
    ) -> torch.Tensor:
        """
        Returns price of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - max_log_moneyness : Tensor, shape (N, *)
            Cumulative maximum of the log moneyness.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - volatility : Tensor, shape (N, *)
            Volatility of the underlying asset.

        Returns
        -------
        price : Tensor, shape (N, *)
        """
        s, m, t, v = map(
            torch.as_tensor, (log_moneyness, max_log_moneyness, expiry_time, volatility)
        )

        sqrt2 = torch.sqrt(torch.as_tensor(2.0)).item()
        n1 = self.N.cdf(self.d1(s, t, v) / sqrt2)
        n2 = self.N.cdf(self.d2(s, t, v) / sqrt2)

        p = (1 / 2) * (torch.exp(s) * (1 + n1) + n2)

        return torch.where(m < 0, p, torch.ones_like(p))

    def implied_volatility(self, *args, **kwargs) -> torch.Tensor:
        """
        Returns implied volatility of the derivative.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (N, *)
            Log moneyness of the prices of the underlying asset.
        - max_log_moneyness : Tensor, shape (N, *)
            Cumulative maximum of the log moneyness.
        - expiry_time : Tensor, shape (N, *)
            Time to expiry of the option.
        - volatility : Tensor, shape (N, *)
            Volatility of the underlying asset.
        - precision : float, default 1e-6
            Computational precision of the implied volatility.

        Returns
        -------
        implied_volatility : Tensor, shape (N, *)
        """
        raise NotImplementedError(
            f"implied volatility of {self.__class__.__name__} is not yet implemented."
        )
