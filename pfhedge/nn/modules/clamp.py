import torch

from ..functional import clamp
from ..functional import leaky_clamp


class LeakyClamp(torch.nn.Module):
    """
    Clamp all elements in the input tensor into the range [`min_value`, `max_value`]
    and return a resulting tensor.
    The bounds `min_value` and `max_value` can be tensors.

    If min_value <= max_value:

        output = min_value + clamped_slope * (input - min_value) if input < min_value
                 input if min_value <= input <= max_value
                 max_value + clamped_slope * (input - max_value) if input > max_value

    If min_value > max_value:

        output = (min_value + max_value) / 2

    Parameters
    ----------
    - clamped_slope : float, default 0.01
        Controls the slope in the clampled regions.

    Shape
    -----
    Inputs: input, min_value, max_value
    Output: output
    - input : Tensor, shape (*)
        The input tensor.
    - min_value : float or Tensor, shape (*), optional
        Lower-bound of the range to be clamped to.
    - max_value : float or Tensor, shape (*), optional
        Upper-bound of the range to be clamped to.
    - output : Tensor, shape (*)
        The output tensor.

    Examples
    --------
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

    def __init__(self, clamped_slope=0.01):
        super().__init__()
        self.clamped_slope = clamped_slope

    def extra_repr(self):
        return f"clamped_slope={self.clamped_slope}" if self.clamped_slope != 0 else ""

    def forward(self, input, min_value=None, max_value=None) -> torch.Tensor:
        return leaky_clamp(
            input, min_value, max_value, clamped_slope=self.clamped_slope
        )


class Clamp(torch.nn.Module):
    """
    Clamp all elements in the input tensor into the range [`min_value`, `max_value`]
    and return a resulting tensor.
    The bounds `min_value` and `max_value` can be tensors.

    If min_value <= max_value:

        output = min_value if input < min_value
                 input if min_value <= input <= max_value
                 max_value if input > max_value

    If min_value > max_value:

        output = (min_value + max_value) / 2

    Shape
    -----
    Inputs: input, min_value, max_value
    Output: output
    - input : Tensor, shape (*)
        The input tensor.
    - min_value : float or Tensor, shape (*), optional
        Lower-bound of the range to be clamped to.
    - max_value : float or Tensor, shape (*), optional
        Upper-bound of the range to be clamped to.
    - output : Tensor, shape (*)
        The output tensor.

    Returns
    -------
    output : Tensor, shape (*)
        The output tensor.

    Examples
    --------
    >>> m = Clamp()
    >>> input = torch.linspace(-2, 12, 15) * 0.1
    >>> input
    tensor([-0.2000, -0.1000,  0.0000,  0.1000,  0.2000,  0.3000,  0.4000,  0.5000,
             0.6000,  0.7000,  0.8000,  0.9000,  1.0000,  1.1000,  1.2000])
    >>> m(input, 0.0, 1.0)
    tensor([0.0000, 0.0000, 0.0000, 0.1000, 0.2000, 0.3000, 0.4000, 0.5000, 0.6000,
            0.7000, 0.8000, 0.9000, 1.0000, 1.0000, 1.0000])

    >>> input = torch.tensor([1.0, 0.0])
    >>> m(input, [0.0, 1.0], 0.0)
    tensor([0.0000, 0.5000])
    """

    def forward(self, input, min_value=None, max_value=None):
        return clamp(input, min_value, max_value)
