from copy import deepcopy

import torch


def randn_antithetic(*size, dtype=None, device=None, dim=0, shuffle=True):
    """Returns a tensor filled with random numbers obtained by an antithetic
    sampling of a normal distribution with mean 0 and variance 1
    (also called the standard normal distribution).

    Parameters:
        size (``int`` ...): a sequence of integers defining the shape of the output tensor.
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
        >>> output = randn_antithetic((4, 3))
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

    size = list(*size)
    size_half = [-(-size[0] // 2)] + size[1:]
    randn = torch.randn(*size_half, dtype=dtype, device=device)

    output = torch.cat((randn, -randn), dim=0)

    if shuffle:
        output = output[torch.randperm(output.size(dim))]

    output = output[: size[0]]

    return output
