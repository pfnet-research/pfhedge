from typing import Optional

import torch
from torch import Tensor


def generate_brownian(
    n_paths: int,
    n_steps: int,
    init_value: float = 0.0,
    volatility: float = 0.2,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Returns time series following the Brownian motion.

    The drift of the time series is assumed to be vanishing.

    The time evolution of the process is given by:

    .. math ::

        dS(t) = \\sigma dW(t) \\,.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_value (float, default=0.0): The initial value of the time series.
        volatility (float, default=0.2): The volatility of the Brownian motion.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If `None`, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Shape:
        - Output: :math:`(N, T)`, where :math:`N` is the number of paths and
          :math:`T` is the number of time steps.

    Returns:
        torch.Tensor

    Examples:

        >>> from pfhedge.stochastic import generate_brownian
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_brownian(2, 5)
        tensor([[ 0.0000,  0.0016,  0.0046,  0.0075, -0.0067],
                [ 0.0000,  0.0279,  0.0199,  0.0257,  0.0291]])
    """
    randn = torch.randn((n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    return init_value + volatility * torch.tensor(dt).to(randn).sqrt() * randn.cumsum(1)


def generate_geometric_brownian(
    n_paths: int,
    n_steps: int,
    init_value: float = 1.0,
    volatility: float = 0.2,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    """Returns time series following the geometric Brownian motion.

    The drift of the time series is assumed to be vanishing.

    The time evolution of the process is given by:

    .. math ::

        dS(t) = \\sigma S(t) dW(t) \\,.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_value (float, default=0.0): The initial value of the time series.
        volatility (float, default=0.2): The volatility of the Brownian motion.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If `None`, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Shape:
        - Output: :math:`(N, T)`, where :math:`T` is the number of time steps and
          :math:`N` is the number of paths.

    Returns:
        torch.Tensor

    Examples:

        >>> from pfhedge.stochastic import generate_brownian
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_geometric_brownian(2, 5)
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930],
                [1.0000, 1.0282, 1.0199, 1.0258, 1.0292]])
    """
    brownian = generate_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        init_value=0.0,
        volatility=volatility,
        dt=dt,
        dtype=dtype,
        device=device,
    )
    t = dt * torch.arange(n_steps).to(brownian).reshape(1, -1)
    return init_value * (brownian - (volatility ** 2) * t / 2).exp()
