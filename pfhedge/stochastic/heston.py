import torch
from torch import Tensor

from .cir import generate_cir


def generate_heston(
    n_paths: int,
    n_steps: int,
    init_state: tuple = (1.0, 0.04),
    kappa: float = 0.5,
    theta: float = 0.04,
    sigma: float = 1.0,
    rho: float = -0.9,
    dt: float = 1 / 250,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> tuple:
    """Returns time series following Heston model.

    The time evolution of Heston process is given by:

    .. math ::

        dS(t) = S(t) \\sqrt{V(t)} dW_1(t) \\,, \\\\
        dV(t) = \\kappa (\\theta - V(t)) + \\sigma \\sqrt{V(t)} dW_2(t) \\,.

    The correlation between :math:`dW_1` and :math:`dW_2` is `\\rho`.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple, default=(1.0, 0.04)):
        kappa (float): The parameter :math:`\\kappa`.
        theta (float): The parameter :math:`\\theta`.
        sigma (float): The parameter :math:`\\sigma`.
        rho (float): The parameter :math:`\\rho`.
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

        >>> from pfhedge.stochastic import generate_heston
        >>>
        >>> _ = torch.manual_seed(42)
        >>> spot, variance = generate_heston(2, 5)
        >>> spot
        tensor([[1.0000, 0.9944, 0.9923, 0.9876, 0.9773],
                [1.0000, 1.0055, 0.9609, 0.9683, 0.9602]])
        >>> variance
        tensor([[0.0400, 0.0441, 0.0446, 0.0470, 0.0495],
                [0.0400, 0.0347, 0.0823, 0.0670, 0.0754]])
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
    log_spot = torch.empty(n_paths, n_steps)
    log_spot[:, 0] = torch.tensor(init_spot).log()

    randn = torch.randn(n_paths, n_steps)

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
