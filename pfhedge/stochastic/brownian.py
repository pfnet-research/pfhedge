import torch
from torch import Tensor


def generate_brownian(
    n_steps: int,
    n_paths: int,
    init_value: float = 0.0,
    volatility: float = 0.2,
    dt: float = 1.0 / 365,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tensor:
    """Returns time series following the Brownian motion.

    The drift of the time series is assumed to be vanishing.

    Args:
        n_steps (int): The number of time steps.
        n_paths (int): The number of simulated paths.
        init_value (float, default=0.0): The initial value of the time series.
        volatility (float, default=0.2): The volatility of the Brownian motion.
        dt (float, default=1/365): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If `None`, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Shape:
        - Output: :math:`(T, N)`, where :math:`T` is the number of time steps and
          :math:`N` is the number of paths.

    Returns:
        torch.Tensor

    Examples:

        >>> _ = torch.manual_seed(42)
        >>> generate_brownian(5, 2)
        tensor([[ 0.0000,  0.0000],
                [ 0.0025,  0.0024],
                [-0.0093,  0.0005],
                [ 0.0138, -0.0062],
                [ 0.0186, -0.0034]])
    """
    randn = torch.randn((n_steps, n_paths), dtype=dtype, device=device)
    randn[0] = 0.0
    return init_value + volatility * torch.tensor(dt).to(randn).sqrt() * randn.cumsum(0)


def generate_geometric_brownian(
    n_steps: int,
    n_paths: int,
    init_value: float = 1.0,
    volatility: float = 0.2,
    dt: float = 1.0 / 365,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tensor:
    """Returns time series following the geometric Brownian motion.

    The drift of the time series is assumed to be vanishing.

    Args:
        n_steps (int): The number of time steps.
        n_paths (int): The number of simulated paths.
        init_value (float, default=0.0): The initial value of the time series.
        volatility (float, default=0.2): The volatility of the Brownian motion.
        dt (float, default=1/365): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If `None`, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Shape:
        - Output: :math:`(T, N)`, where :math:`T` is the number of time steps and
          :math:`N` is the number of paths.

    Returns:
        torch.Tensor

    Examples:

        >>> _ = torch.manual_seed(42)
        >>> generate_geometric_brownian(5, 2)
        tensor([[1.0000, 1.0000],
                [1.0024, 1.0024],
                [0.9906, 1.0004],
                [1.0137, 0.9936],
                [1.0186, 0.9964]])
    """
    brownian = generate_brownian(
        n_steps=n_steps,
        n_paths=n_paths,
        init_value=0.0,
        volatility=volatility,
        dt=dt,
        dtype=dtype,
        device=device,
    )
    t = dt * torch.arange(n_steps).to(brownian).reshape(-1, 1)
    return init_value * torch.exp(brownian - (volatility ** 2) * t / 2)
