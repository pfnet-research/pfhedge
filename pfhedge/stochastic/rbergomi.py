from typing import Optional
from typing import Tuple

import torch

from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic._utils import cast_state
from pfhedge.stochastic.heston import HestonTuple


class RBergomiTuple(HestonTuple):
    pass


def generate_rbergomi(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    alpha: float = -0.4,
    rho: float = -0.9,
    eta: float = 1.9,
    xi: float = 0.2,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> RBergomiTuple:
    # https://github.com/ryanmccrickerd/rough_bergomi/

    if init_state is None:
        init_state = (1.0, xi)

    init_state = cast_state(init_state, dtype=dtype, device=device)
    alpha_tensor, rho_tensor, eta_tensor = cast_state(
        (alpha, rho, eta), dtype=dtype, device=device
    )

    _dW1_cov1 = dt ** (alpha + 1) / (alpha + 1)
    _dW1_cov2 = dt ** (2 * alpha + 1) / (2 * alpha + 1)
    _dW1_covariance_matrix = torch.as_tensor(
        [[dt, _dW1_cov1], [_dW1_cov1, _dW1_cov2]], dtype=dtype, device=device
    )
    _dW1_generator = torch.distributions.multivariate_normal.MultivariateNormal(
        loc=torch.as_tensor([0.0, 0.0], dtype=dtype, device=device),
        covariance_matrix=_dW1_covariance_matrix,
    )

    dW1 = _dW1_generator.sample([n_paths, n_steps - 1])
    dW2 = torch.randn([n_paths, n_steps - 1], dtype=dtype, device=device)
    dW2 *= dW2.new_tensor(dt).sqrt()

    _Y1 = torch.cat(
        [torch.zeros([n_paths, 1], dtype=dtype, device=device), dW1[:, :, 1]], dim=-1
    )
    discrete_TBSS_fn = lambda k, a: ((k ** (a + 1) - (k - 1) ** (a + 1)) / (a + 1)) ** (
        1 / a
    )
    _gamma = (
        discrete_TBSS_fn(torch.arange(2, n_steps, dtype=dtype, device=device), alpha)
        / (n_steps - 1)
    ) ** alpha
    _gamma = torch.cat([torch.zeros([2], dtype=dtype, device=device), _gamma], dim=0)
    _Xi = dW1[:, :, 0]
    _GXi_convolve = torch.nn.functional.conv1d(
        _gamma.__reversed__().repeat(1, 1, 1),
        _Xi.unsqueeze(dim=1),
        padding=_Xi.size(1) - 1,
    ).squeeze(dim=0)
    _Y2 = _GXi_convolve[:, torch.arange(-1, -1 - n_steps, -1)]
    Y = torch.sqrt(2 * alpha_tensor + 1) * (_Y1 + _Y2)
    dB = rho_tensor * dW1[:, :, 0] + torch.sqrt(1 - rho_tensor.square()) * dW2
    variance = init_state[1] * torch.exp(
        eta_tensor * Y
        - 0.5
        * eta_tensor.square()
        * (torch.arange(0, n_steps, dtype=dtype, device=device) * dt)
        ** (2 * alpha_tensor + 1)
    )

    _increments = variance[:, :-1].sqrt() * dB - 0.5 * variance[:, :-1] * dt
    _integral = torch.cumsum(_increments, dim=1)
    log_return = torch.cat(
        [torch.zeros((n_paths, 1), dtype=dtype, device=device), _integral], dim=-1
    )
    prices = init_state[0] * log_return.exp()

    return RBergomiTuple(prices, variance)
