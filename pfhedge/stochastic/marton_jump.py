from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic._utils import cast_state


def generate_marton_jump(
    n_paths: int,
    n_steps: int,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    sigma: float = 0.2,
    mu: float = 0.0,
    jump_per_year=1.0,
    jump_mean=0.0,
    jump_std=0.3,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    # https://github.com/federicomariamassari/financial-engineering/blob/master/python-modules/jump_diffusion.py
    init_state = cast_state(init_state, dtype=dtype, device=device)

    poisson = torch.distributions.poisson.Poisson(rate=jump_per_year * dt)
    n_jumps = poisson.sample((n_paths, n_steps - 1)).to(dtype=dtype, device=device)

    jump_size = (
        jump_mean
        + torch.randn((n_paths, n_steps - 1), dtype=dtype, device=device) * jump_std
    )
    jump = n_jumps * jump_size
    jump = torch.cat(
        [torch.zeros((n_paths, 1), dtype=dtype, device=device), jump], dim=1
    )

    randn = torch.randn((n_paths, n_steps), dtype=dtype, device=device)
    randn[:, 0] = 0.0
    drift = (mu - (sigma**2) / 2) * dt * torch.arange(n_steps).to(randn)
    brown = randn.new_tensor(dt).sqrt() * randn.cumsum(1)
    return init_state[0] * (drift + sigma * brown + jump.cumsum(1)).exp()
