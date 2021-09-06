import abc
from inspect import signature
from typing import List
from typing import no_type_check

import torch
from torch import Tensor
from torch.nn import Module


class BSModuleMixin(Module):
    """A mixin class for Black-Scholes formula modules.

    Shape:
        - Input: :math:`(N, *, H_\\text{in})` where
          :math:`H_\\text{in}` is the number of input features and
          :math:`*` means any number of additional dimensions.
          See :func:`inputs` for the names of input features.
        - Output: :math:`(N, *, 1)`.
          All but the last dimension are the same shape as the input.
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
