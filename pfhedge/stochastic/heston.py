from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from .cir import generate_cir


def generate_heston(
    n_paths: int,
    n_steps: int,
    init_state: Tuple[float, float] = (1.0, 0.04),
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 2.0,
    rho: float = -0.7,
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, Tensor]:
    """Returns time series following Heston model.

    The time evolution of Heston process is given by:

    .. math ::

        dS(t) = S(t) \\sqrt{V(t)} dW_1(t) \\,, \\\\
        dV(t) = \\kappa (\\theta - V(t)) + \\sigma \\sqrt{V(t)} dW_2(t) \\,.

    The correlation between :math:`dW_1` and :math:`dW_2` is :math:`\\rho`.
    The correlation between :math:`dW_1` and :math:`dW_2` is `\\rho`.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple, default=(1.0, 0.04)): Initial state of a simulation.
            `init_state` should be a 2-tuple `(spot, variance)` where `spot` is the
            initial spot price and `variance` of the initial variance.
        kappa (float, default=1.0): The parameter :math:`\\kappa`.
        theta (float, default=0.04): The parameter :math:`\\theta`.
        sigma (float, default=2.0): The parameter :math:`\\sigma`.
        rho (float, default=-0.7): The parameter :math:`\\rho`.
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
        (torch.Tensor, torch.Tensor): A tuple of spot and variance.

    Examples:

        >>> from pfhedge.stochastic import generate_heston
        >>>
        >>> _ = torch.manual_seed(42)
        >>> spot, variance = generate_heston(2, 5)
        >>> spot
        tensor([[1.0000, 0.9958, 0.9940, 0.9895, 0.9765],
                [1.0000, 1.0064, 1.0117, 1.0116, 1.0117]])
        >>> variance
        tensor([[0.0400, 0.0433, 0.0406, 0.0423, 0.0441],
                [0.0400, 0.0251, 0.0047, 0.0000, 0.0000]])

    References:
        - Andersen, Leif B.G., Efficient Simulation of the Heston Stochastic
          Volatility Model (January 23, 2007). Available at SSRN:
          https://ssrn.com/abstract=946405 or http://dx.doi.org/10.2139/ssrn.946404
    """
    GAMMA1 = 0.5
    GAMMA2 = 0.5

    init_spot, init_var = init_state

    variance = generate_cir(
        n_paths=n_paths,
        n_steps=n_steps,
        init_value=init_var,
        kappa=kappa,
        theta=theta,
        sigma=sigma,
        dt=dt,
        dtype=dtype,
        device=device,
    )

    log_spot = torch.empty_like(variance)
    log_spot[:, 0] = torch.tensor(init_spot).log()
    randn = torch.randn_like(variance)

    for i_step in range(n_steps - 1):
        # Compute log S(t + 1): Eq(33)
        k0 = -rho * kappa * theta * dt / sigma
        k1 = GAMMA1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        k2 = GAMMA2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        k3 = GAMMA1 * dt * (1 - rho ** 2)
        k4 = GAMMA2 * dt * (1 - rho ** 2)
        v0 = variance[:, i_step]
        v1 = variance[:, i_step + 1]
        log_spot[:, i_step + 1] = sum(
            (
                log_spot[:, i_step],
                k0 + k1 * v0 + k2 * v1,
                (k3 * v0 + k4 * v1).sqrt() * randn[:, i_step],
            )
        )

    return (log_spot.exp(), variance)
