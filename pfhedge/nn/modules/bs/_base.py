from inspect import signature
from typing import List
from typing import no_type_check

import torch
from torch import Tensor
from torch.nn import Module

import pfhedge.autogreek as autogreek


class BSModuleMixin(Module):
    """A mixin class for Black-Scholes formula modules.

    Shape:
        - Input: :math:`(N, *, H_\\text{in})` where
          :math:`H_\\text{in}` is the number of input features and
          :math:`*` means any number of additional dimensions.
          See :meth:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.
    """

    def forward(self, input: Tensor) -> Tensor:
        """Returns delta of the derivative.

        Args:
            input (torch.Tensor): The input tensor. Features are concatenated along
                the last dimension.
                See :meth:`inputs()` for the names of the input features.

        Returns:
            torch.Tensor
        """
        return self.delta(*(input[..., [i]] for i in range(input.size(-1))))

    @no_type_check
    def price(self, *args, **kwargs) -> Tensor:
        """Returns price of the derivative.

        Returns:
            torch.Tensor
        """

    @no_type_check
    def delta(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.delta(self.price, **kwargs)

    @no_type_check
    def gamma(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.gamma(self.price, **kwargs)

    @no_type_check
    def vega(self, **kwargs) -> Tensor:
        """Returns delta of the derivative.

        Returns:
            torch.Tensor
        """
        return autogreek.vega(self.price, **kwargs)

    def inputs(self) -> List[str]:
        """Returns the names of input features.

        Returns:
            list
        """
        return list(signature(self.delta).parameters.keys())
