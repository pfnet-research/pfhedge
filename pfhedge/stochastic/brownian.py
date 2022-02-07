from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar


def generate_brownian(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (0.0,),
    sigma: float = 0.2,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Returns time series following the Brownian motion.

    The time evolution of the process is given by:

    .. math::
        dS(t) = \sigma dW(t) \,.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], default=(0.0,)): The initial state of
            the time series.
            This is specified by a tuple :math:`(S(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        sigma (float, default=0.2): The parameter :math:`\sigma`,
            which stands for the volatility of the time series.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.
        engine (callable, default=torch.randn): The desired generator of random numbers
            from a standard normal distribution.
            A function call ``engine(size, dtype=None, device=None)``
            should return a tensor filled with random numbers
            from a standard normal distribution.

    Shape:
        - Output: :math:`(N, T)` where
          :math:`N` is the number of paths and
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
    # randn = torch.randn((n_paths, n_steps), dtype=dtype, device=device)
    randn = engine((n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    return sigma * randn.new_tensor(dt).sqrt() * randn.cumsum(1) + init_value


def generate_geometric_brownian(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    sigma: float = 0.2,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Returns time series following the geometric Brownian motion.

    The time evolution of the process is given by:

    .. math::

        dS(t) = \sigma S(t) dW(t) \,.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], default=(0.0,)): The initial state of
            the time series.
            This is specified by a tuple :math:`(S(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        sigma (float, default=0.2): The parameter :math:`sigma`,
            which stands for the volatility of the time series.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.
        engine (callable, default=torch.randn): The desired generator of random numbers
            from a standard normal distribution.
            A function call ``engine(size, dtype=None, device=None)``
            should return a tensor filled with random numbers
            from a standard normal distribution.

    Shape:
        - Output: :math:`(N, T)` where
          :math:`N` is the number of paths and
          :math:`T` is the number of time steps.

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
        engine=engine,
    )
    t = dt * torch.arange(n_steps).to(brownian).unsqueeze(0)
    return init_state[0] * (brownian - (sigma ** 2) * t / 2).exp()
