import abc
from inspect import signature

import torch


class _Normal(torch.distributions.normal.Normal):
    """
    Creates a normal (also called Gaussian) distribution parameterized
    by `loc` and `scale`.
    It has `pdf` method to compute the probability density function.

    See `torch.distributions.normal.Normal` for details and other methods.
    """

    def pdf(self, value: torch.Tensor) -> torch.Tensor:
        """
        Returns the probability density function evaluated at value.

        Parameters
        ----------
        - value : Tensor, shape (*)

        Returns
        -------
        pdf : Tensor, shape (*)
        """
        return self.log_prob(value).exp()


class BSModuleMixin(torch.nn.Module):
    """
    A mixin class for Black-Scholes formula modules.

    Shape
    -----
    - Input : (N, *, H_in)
        See `features()` for input features.
        Here, `*` means any number of additional dimensions.
    - Output : (N, *, 1)
        Delta of the derivative.
        All but the last dimension are the same shape as the input.
    """

    def forward(self, input) -> torch.Tensor:
        """
        Returns delta of the derivative.

        Parameters
        ----------
        - input : torch.Tensor, shape (N, *, H_in)
            The input tensor.
            Features are concatenated along the last dimension.

        Returns
        -------
        output : torch.Tensor, shape (N, *, 1)
        """
        return self.delta(*(input[..., [i]] for i in range(input.size()[-1])))

    @abc.abstractmethod
    def delta(*args, **kwargs) -> torch.Tensor:
        """
        Returns delta of the derivative.

        Returns
        -------
        delta : Tensor, shape (N, *)
        """

    def features(self) -> list:
        """
        Returns a list of names of input features.

        By default, this method infers the names of features from the signature of
        the `delta` method.
        Please override this method if the signature is different from
        the feature names.
        """
        return list(signature(self.delta).parameters.keys())

    @property
    def N(self):
        """
        Returns normal distribution with zero mean and unit standard deviation.
        """
        return _Normal(torch.tensor(0.0), torch.tensor(1.0))

    @staticmethod
    def d1(log_moneyness, expiry_time, volatility) -> torch.Tensor:
        """
        Returns `d1` in the Black-Scholes formula.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (*)
            Log moneyness of the underlying asset.
        - expiry_time : Tensor, shape (*)
            Time to expiry of the option.
        - volatility : Tensor, shape (*)
            Volatility of the underlying asset.

        Returns
        -------
        d1 : Tensor, shape (*)
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))
        return (s + (v ** 2 / 2) * t) / (v * torch.sqrt(t))

    @staticmethod
    def d2(log_moneyness, expiry_time, volatility) -> torch.Tensor:
        """
        Returns `d2` in the Black-Scholes formula.

        Parameters
        ----------
        - log_moneyness : Tensor, shape (*)
            Log moneyness of the underlying asset.
        - expiry_time : Tensor, shape (*)
            Time to expiry of the option.
        - volatility : Tensor, shape (*)
            Volatility of the underlying asset.

        Returns
        -------
        d2 : Tensor, shape (*)
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, expiry_time, volatility))
        return (s - (v ** 2 / 2) * t) / (v * torch.sqrt(t))
