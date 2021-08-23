import abc
from inspect import signature
from typing import List
from typing import no_type_check

import torch
from torch import Tensor
from torch.distributions.normal import Normal
from torch.nn import Module


class BSModuleMixin(Module):
    """A mixin class for Black-Scholes formula modules.

    Shape:
        - Input: :math:`(N, *, H_\\text{in})`, where :math:`*` means any number of
          additional dimensions. See :func:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`: All but the last dimension are the same shape
          as the input.
    """

    def forward(self, input: Tensor) -> Tensor:
        """Returns delta of the derivative.

        Args:
            input (torch.Tensor): The input tensor. Features are concatenated along
                the last dimension.
                See :func:`inputs()` for the names of the input features.

        Returns:
            torch.Tensor
        """
        return self.delta(*(input[..., [i]] for i in range(input.size(-1))))

    @abc.abstractmethod
    @no_type_check
    def delta(self, *args, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """

    def inputs(self) -> List[str]:
        """Returns the names of input features.

        Returns:
            list
        """
        return list(signature(self.delta).parameters.keys())

    @property
    def N(self) -> Normal:
        """Returns normal distribution with zero mean and unit standard deviation."""
        return Normal(torch.tensor(0.0), torch.tensor(1.0))

    @staticmethod
    def d1(
        log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns :math:`d_1` in the Black-Scholes formula.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Returns:
            torch.Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, time_to_maturity, volatility))
        return (s + (v ** 2 / 2) * t) / (v * t.sqrt())

    @staticmethod
    def d2(
        log_moneyness: Tensor, time_to_maturity: Tensor, volatility: Tensor
    ) -> Tensor:
        """Returns :math:`d_2` in the Black-Scholes formula.

        Args:
            log_moneyness (torch.Tensor): Log moneyness of the underlying asset.
            time_to_maturity (torch.Tensor): Time to expiry of the option.
            volatility (torch.Tensor): Volatility of the underlying asset.

        Returns:
            torch.Tensor
        """
        s, t, v = map(torch.as_tensor, (log_moneyness, time_to_maturity, volatility))
        return (s - (v ** 2 / 2) * t) / (v * t.sqrt())
