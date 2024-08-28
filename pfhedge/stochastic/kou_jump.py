import math
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


def generate_kou_jump(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    sigma: float = 0.2,
    mu: float = 0.0,
    jump_per_year: float = 68.0,
    jump_mean_up: float = 0.02,
    jump_mean_down: float = 0.05,
    jump_up_prob: float = 0.5,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    engine: Callable[..., Tensor] = torch.randn,
) -> Tensor:
    r"""Kou's Jump Diffusion Model for stock prices.
       Assumes number of jumps to be poisson distribution
       with ASYMMETRIC jump for up and down movement; the
       lof of these jump follows exponential distribution
       with mean jump_mean_up and jump_mean_down resp.

       See Glasserman, Paul. Monte Carlo Methods in Financial
       Engineering. New York: Springer-Verlag, 2004.for details.
       Copy available at "https://www.bauer.uh.edu/spirrong/
       Monte_Carlo_Methods_In_Financial_Enginee.pdf"

       Combined with the original paper by Kou:
       A Jump-Diffusion Model for Option Pricing
       https://www.columbia.edu/~sk75/MagSci02.pdf

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], default=(0.0,)): The initial state of
            the time series.
            This is specified by a tuple :math:`(S(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
            The shape of torch.Tensor must be (1,) or (n_paths,).
        sigma (float, default=0.2): The parameter :math:`\sigma`,
            which stands for the volatility of the time series.
        mu (float, default=0.0): The parameter :math:`\mu`,
            which stands for the drift of the time series.
        jump_per_year (float, optional): Jump poisson process annual
            lambda: Average number of annual jumps. Defaults to 1.0.
        jump_mean_up (float, optional): Mu for the up jumps:
            Instaneous value. Defaults to 0.02.
            This has to be postive and smaller than 1.
        jump_mean_down (float, optional): Mu for the down jumps:
            Instaneous value. Defaults to 0.05.
            This has to be larger than 0.
        jump_up_prob (float, optional): Given a jump occurs,
            this is conditional prob for up jump.
            Down jump occurs with prob 1-jump_up_prob.
            Has to be in [0,1].
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
            Only to be used for the normal component,
            jupms uses poisson distribution.

    Shape:
        - Output: :math:`(N, T)` where
          :math:`N` is the number of paths and
          :math:`T` is the number of time steps.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.stochastic import generate_kou_jump
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_kou_jump(2, 5, jump_per_year = 10.0)
        tensor([[1.0000, 1.0021, 1.0055, 1.0089, 0.9952],
                [1.0000, 1.0288, 1.0210, 1.0275, 1.0314]])
    """
    if not (0 < jump_mean_up < 1.0):
        raise ValueError("jump_mean_up must be postive and smaller than 1")

    if not jump_mean_down > 0:
        raise ValueError("jump_mean_down must be postive")

    if not (0 <= jump_up_prob <= 1.0):
        raise ValueError("jump prob must be in 0 and 1 incl")

    # change means to rate of exponential distributions
    jump_eta_up = 1 / jump_mean_up
    jump_eta_down = 1 / jump_mean_down

    init_state = cast_state(init_state, dtype=dtype, device=device)

    init_value = init_state[0]

    t = dt * torch.arange(n_steps, device=device, dtype=dtype)[None, :]
    returns = (
        engine(*(n_paths, n_steps), dtype=dtype, device=device) * math.sqrt(dt) * sigma
    )

    returns[:, 0] = 0.0

    # Generate jump components
    poisson = torch.distributions.poisson.Poisson(rate=jump_per_year * dt)
    n_jumps = poisson.sample((n_paths, n_steps - 1)).to(dtype=dtype, device=device)

    # if n_steps is greater than 1
    if (n_steps - 1) > 0:
        # max jumps used to aggregte jump in between dt time
        max_jumps = int(n_jumps.max())
        size_paths = torch.Size([n_paths, n_steps - 1, max_jumps])

        # up exp generator
        up_exp_dist = torch.distributions.exponential.Exponential(rate=jump_eta_up)

        # down exp generator
        down_exp_dist = torch.distributions.exponential.Exponential(rate=jump_eta_down)

        # up or down generator
        direction_uni_dist = torch.distributions.uniform.Uniform(0.0, 1.0)

        log_jump = torch.where(
            direction_uni_dist.sample(size_paths) < jump_up_prob,
            up_exp_dist.sample(size_paths),
            -down_exp_dist.sample(size_paths),
        ).to(returns)

        # for no jump condition
        log_jump = torch.cat(
            (torch.zeros(n_paths, n_steps - 1, 1).to(log_jump), log_jump), dim=-1
        )

        exp_jump_ind = torch.exp(log_jump)

        # filter out jump movements that did not occur in dt time
        indices_expanded = n_jumps[..., None]
        k_range = torch.arange(max_jumps + 1).to(returns)
        mask = k_range > indices_expanded
        # exp(0) as to no jump after n_jump
        exp_jump_ind[mask] = 1.0

        # aggregate jumps in time dt--> multiplication of exponent
        exp_jump = torch.prod(exp_jump_ind, dim=-1)

        # no jump at time 0--> exp(0.0)=1.0
        exp_jump = torch.cat((torch.ones(n_paths, 1).to(exp_jump), exp_jump), dim=1)

    else:
        exp_jump = torch.ones(n_paths, 1).to(returns)

    # aggregate jumps upto time t
    exp_jump_agg = torch.cumprod(exp_jump, dim=-1)

    # jump correction for drift: see the paper
    m = (
        (1 - jump_up_prob) * (jump_eta_down / (jump_eta_down + 1))
        + (jump_up_prob) * (jump_eta_up / (jump_eta_up - 1))
        - 1
    )

    prices = (
        torch.exp(
            (mu - jump_per_year * m) * t + returns.cumsum(1) - (sigma ** 2) * t / 2
        )
        * init_value.view(-1, 1)
        * exp_jump_agg
    )

    return prices
