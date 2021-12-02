from typing import Optional

from torch import Tensor
from torch.nn import Module

from pfhedge._utils.str import _format_float

from ..functional import clamp
from ..functional import leaky_clamp


class LeakyClamp(Module):
    r"""Leakily clamp all elements in ``input`` into the range :math:`[\min, \max]`.

    The bounds :math:`\min` and :math:`\max` can be tensors.

    If :math:`\min \leq \max`:

    .. math::
        \text{output} = \begin{cases}
            \min + \text{clampled_slope} * (\text{input} - \min) &
            \text{input} < \min \\
            \text{input} & \min \leq \text{input} \leq \max \\
            \max + \text{clampled_slope} * (\text{input} - \max) &
            \max < \text{input}
        \end{cases}

    If :math:`\min > \max`:

    .. math::

        \text{output} = \begin{cases}
            \frac12 (\min + \max)
            & \text{inverted_output} = \text{'mean'} \\
            \max
            & \text{inverted_output} = \text{'max'} \\
        \end{cases}

    .. seealso::
        - :func:`pfhedge.nn.functional.leaky_clamp`

    Args:
        clamped_slope (float, default=0.01):
            Controls the slope in the clampled regions.
        inverted_output ({'mean', ''max'}, default='mean'):
            Controls the output when :math:`\min > \max`.
            'max' is consistent with :func:`torch.clamp`.

    Shape:
        - input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - min: :math:`(N, *)` or any size broadcastable to ``input``.
        - max: :math:`(N, *)` or any size broadcastable to ``input``.
        - output: :math:`(N, *)`, same shape as the input.

    Examples:
        >>> import torch
        >>> from pfhedge.nn import LeakyClamp
        >>> m = LeakyClamp()
        >>> input = torch.linspace(-2, 12, 15) * 0.1
        >>> input
        tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
                 0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
        >>> m(input, 0.0, 1.0)
        tensor([-2.0000e-03, -1.0000e-03,  0.0000e+00,  1.0000e-01,  2.0000e-01,
                 3.0000e-01,  4.0000e-01,  5.0000e-01,  6.0000e-01,  7.0000e-01,
                 8.0000e-01,  9.0000e-01,  1.0000e+00,  1.0010e+00,  1.0020e+00])
    """

    def __init__(self, clamped_slope: float = 0.01, inverted_output: str = "mean"):
        super().__init__()
        self.clamped_slope = clamped_slope
        self.inverted_output = inverted_output

    def extra_repr(self) -> str:
        return "clamped_slope=" + _format_float(self.clamped_slope)

    def forward(
        self, input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        r"""Clamp all elements in ``input`` into the range :math:`[\min, \max]`.

        Args:
            input (torch.Tensor): The input tensor.
            min (torch.Tensor, optional): Lower-bound of the range to be clamped to.
            max (torch.Tensor, optional): Upper-bound of the range to be clamped to.

        Returns:
            torch.Tensor
        """
        return leaky_clamp(input, min=min, max=max, clamped_slope=self.clamped_slope)


class Clamp(Module):
    r"""Clamp all elements in ``input`` into the range :math:`[\min, \max]`.

    The bounds :math:`\min` and :math:`\max` can be tensors.

    If :math:`\min \leq \max`:

    .. math::

        \text{output} = \begin{cases}
        \min & \text{input} < \min \\
        \text{input} & \min \leq \text{input} \leq \max \\
        \max & \max < \text{input}
        \end{cases}

    If :math:`\min > \max`:

    .. math::

        \text{output} = \begin{cases}
            \frac12 (\min + \max)
            & \text{inverted_output} = \text{'mean'} \\
            \max
            & \text{inverted_output} = \text{'max'} \\
        \end{cases}

    .. seealso::
        - :func:`torch.clamp`
        - :func:`pfhedge.nn.functional.clamp`

    Args:
        inverted_output ({'mean', ''max'}, default='mean'):
            Controls the output when :math:`\min > \max`.
            'max' is consistent with :func:`torch.clamp`.

    Shape:
        - input: :math:`(N, *)` where
          :math:`*` means any number of additional dimensions.
        - min: :math:`(N, *)` or any size broadcastable to ``input``.
        - max: :math:`(N, *)` or any size broadcastable to ``input``.
        - output: :math:`(N, *)`, same shape as the input.

    Examples:

        >>> import torch
        >>> from pfhedge.nn import Clamp
        >>>
        >>> m = Clamp()
        >>> input = torch.linspace(-2, 12, 15) * 0.1
        >>> input
        tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
                 0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
        >>> m(input, 0.0, 1.0)
        tensor([0.0000, 0.0000, 0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000,
                0.7000, 0.8000, 0.9000, 1.0000, 1.0000, 1.0000])

        When :math:`\min > \max`, returns the mean of :math:`\min` and :math:`\max`.

        >>> input = torch.tensor([1.0, 0.0])
        >>> min = torch.tensor([0.0, 1.0])
        >>> max = torch.tensor([0.0, 0.0])
        >>> m(input, min, max)
        tensor([0.0000, 0.5000])
    """

    def forward(
        self, input: Tensor, min: Optional[Tensor] = None, max: Optional[Tensor] = None
    ) -> Tensor:
        r"""Clamp all elements in ``input`` into the range :math:`[\min, \max]`.

        Args:
            input (torch.Tensor): The input tensor.
            min (torch.Tensor, optional): Lower-bound of the range to be clamped to.
            max (torch.Tensor, optional): Upper-bound of the range to be clamped to.

        Returns:
            torch.Tensor
        """
        return clamp(input, min=min, max=max)
