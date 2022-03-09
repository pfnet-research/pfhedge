from typing import Optional

import torch
from torch import Tensor

from .engine import RandnSobolBoxMuller


def randn_antithetic(
    *size: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    dim: int = 0,
    shuffle: bool = True
) -> Tensor:
    """Returns a tensor filled with random numbers obtained by an antithetic sampling.

    The output should be a normal distribution with mean 0 and variance 1
    (also called the standard normal distribution).

    Parameters:
        size (``int``...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.stochastic import randn_antithetic
        >>>
        >>> _ = torch.manual_seed(42)
        >>> output = randn_antithetic(4, 3)
        >>> output
        tensor([[-0.3367, -0.1288, -0.2345],
                [ 0.2303, -1.1229, -0.1863],
                [-0.2303,  1.1229,  0.1863],
                [ 0.3367,  0.1288,  0.2345]])
        >>> output.mean(dim=0).allclose(torch.zeros(3), atol=1e-07, rtol=0.0)
        True
    """
    if dim != 0:
        raise ValueError("dim != 0 is not supported.")

    size_list = list(size)
    size_half = [-(-size_list[0] // 2)] + size_list[1:]
    randn = torch.randn(*size_half, dtype=dtype, device=device)

    output = torch.cat((randn, -randn), dim=0)

    if shuffle:
        output = output[torch.randperm(output.size(dim))]

    output = output[: size_list[0]]

    return output


def randn_sobol_boxmuller(
    *size: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    scramble: bool = True,
    seed: Optional[int] = None
) -> Tensor:
    """Returns a tensor filled with random numbers obtained by a Sobol sequence
    applied with the Box-Muller transformation.

    The outputs should be normal distribution with mean 0 and variance 1
    (also called the standard normal distribution).

    Parameters:
        size (``int``...): a sequence of integers defining the shape of the output tensor.
            Can be a variable number of arguments or a collection like a list or tuple.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.stochastic import randn_sobol_boxmuller
        >>>
        >>> _ = torch.manual_seed(42)
        >>> output = randn_sobol_boxmuller(4, 3)
        >>> output
        tensor([[ 0.0559,  0.4954, -0.8578],
                [-0.7492, -1.0370, -0.4778],
                [ 0.1651,  0.0430, -2.0368],
                [ 1.1309, -0.1779,  0.0796]])
    """
    engine = RandnSobolBoxMuller(scramble=scramble, seed=seed)
    return engine(*size, dtype=dtype, device=device)
