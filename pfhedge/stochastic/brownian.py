from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor

TensorOrFloat = Union[Tensor, float]


def generate_brownian(
    n_paths: int,
    n_steps: int,
    init_state: Tuple[TensorOrFloat, ...] = (0.0,),
    sigma: float = 0.2,
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
        init_state (tuple[torch.Tensor | float], default=(0.0,)): The initial state of
            the time series.
            This is specified by ``(S0,)``, where ``S0`` is
            the initial value of :math:`S`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        sigma (float, default=0.2): The parameter :math:`sigma`, which stands for
            the volatility of the time series.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
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
    # Accept Union[float, Tensor] as well because making a tuple with a single element
    # is troublesome
    if isinstance(init_state, (float, Tensor)):
        init_state = (init_state,)

    # Cast to init_state: Tuple[Tensor, ...] with desired dtype and device
    init_state = cast(Tuple[Tensor, ...], tuple(map(torch.as_tensor, init_state)))
    init_state = tuple(map(lambda t: t.to(dtype=dtype, device=device), init_state))

    init_value = init_state[0]
    randn = torch.randn((n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    return init_value + sigma * torch.tensor(dt).to(randn).sqrt() * randn.cumsum(1)


def generate_geometric_brownian(
    n_paths: int,
    n_steps: int,
    init_state: Tuple[TensorOrFloat, ...] = (1.0,),
    sigma: float = 0.2,
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
        init_state (tuple[torch.Tensor | float], default=(1.0,)): The initial state of
            the time series.
            This is specified by ``(S0,)``, where ``S0`` is the initial value of :math:`S`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        sigma (float, default=0.2): The parameter :math:`sigma`, which stands for
            the volatility of the time series.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
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
    # Accept Union[float, Tensor] as well because making a tuple with a single element
    # is troublesome
    if isinstance(init_state, (float, Tensor)):
        init_state = (init_state,)

    # Cast to init_state: Tuple[Tensor, ...] with desired dtype and device
    init_state = cast(Tuple[Tensor, ...], tuple(map(torch.as_tensor, init_state)))
    init_state = tuple(map(lambda t: t.to(dtype=dtype, device=device), init_state))

    brownian = generate_brownian(
        n_paths=n_paths,
        n_steps=n_steps,
        init_state=(0.0,),
        sigma=sigma,
        dt=dt,
        dtype=dtype,
        device=device,
    )
    t = dt * torch.arange(n_steps).to(brownian).reshape(1, -1)
    return init_state[0] * (brownian - (sigma ** 2) * t / 2).exp()
