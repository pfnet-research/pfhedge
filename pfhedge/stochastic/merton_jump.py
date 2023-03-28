from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic._utils import cast_state


def generate_merton_jump(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    mu: float = 0.0,
    sigma: float = 0.2,
    jump_per_year: float = 68.2,
    jump_mean: float = 0.0,
    jump_std: float = 0.02,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Returns time series following the Merton's Jump Diffusion Model.

    The time evolution of the process is given by:

    .. math::

        \frac{dS(t)}{S(t)} = (\mu - \lambda k) dt + \sigma dW(t) + dJ(t) .

    Reference:
     - Merton, R.C., Option pricing when underlying stock returns are discontinuous,
       Journal of Financial Economics, 3 (1976), 125-144.
     - Gugole, N. (2016). Merton jump-diffusion model versus the black and scholes approach for the log-returns and volatility
       smile fitting. International Journal of Pure and Applied Mathematics, 109(3), 719-736.
       (Especially p.730 for parameter calibulation.)

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], default=(1.0,)): The initial state of
            the time series.
            This is specified by a tuple :math:`(S(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        mu (float, default=0.0): The parameter :math:`\mu`,
            which stands for the dirft coefficient of the time series.
        sigma (float, default=0.2): The parameter :math:`\sigma`,
            which stands for the volatility of the time series.
        jump_per_year (float, default=68.2): The frequency of jumps in one year.
        jump_mean (float, default=0.0): The mean of jumnp sizes.
        jump_std (float, default=0.02): The deviation of jump sizes.
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
        >>> from pfhedge.stochastic import generate_merton_jump
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_merton_jump(2, 5)
        tensor([[1.0000, 0.9904, 1.0074, 1.0160, 1.0117],
                [1.0000, 1.0034, 1.0040, 1.0654, 1.0621]])
    """
    # https://www.codearmo.com/python-tutorial/merton-jump-diffusion-model-python
    init_state = cast_state(init_state, dtype=dtype, device=device)

    poisson = torch.distributions.poisson.Poisson(rate=jump_per_year * dt)
    n_jumps = poisson.sample((n_paths, n_steps - 1)).to(dtype=dtype, device=device)

    jump_size = (
        jump_mean
        + engine(*(n_paths, n_steps - 1), dtype=dtype, device=device) * jump_std
    )
    jump = n_jumps * jump_size
    jump = torch.cat(
        [torch.zeros((n_paths, 1), dtype=dtype, device=device), jump], dim=1
    )

    randn = engine(*(n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    drift = (
        (mu - (sigma ** 2) / 2 - jump_per_year * (jump_mean + jump_std ** 2 / 2))
        * dt
        * torch.arange(n_steps).to(randn)
    )
    brown = randn.new_tensor(dt).sqrt() * randn.cumsum(1)
    return init_state[0] * (drift + sigma * brown + jump.cumsum(1)).exp()
