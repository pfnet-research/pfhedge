import torch
from torch import Tensor


def generate_cir(
    n_paths:int,
    n_steps:int,
    init_value:float = 0.04,
    kappa: float=0.0,
    theta: float=0.04,
    sigma: float=0.04,
    dt:float=1/250,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tensor:
    """Returns CIR process

    Examples:

        >>> from pfhedge.stochastic import generate_cir
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_cir(2, 5)

    .. cite:
       ...
    """
    assert dtype is None
    assert device is None

    # PSI_CRIT in [1.0, 2.0]. See section 3.2.3
    PSI_CRIT = 1.5

    var = torch.empty((n_paths, n_steps), dtype=dtype, device=device)
    var[:, 0] = init_value

    randn = torch.randn(n_paths, n_steps, dtype=dtype, device=device)
    rand = torch.rand(n_paths, n_steps, dtype=dtype, device=device)

    for i_step in range(n_steps - 1):
        # Compute m, s, psi; Equation (17), (18)
        _expkdt = (-torch.tensor(kappa) * dt).exp()
        m = theta + (v[:, i_step] - theta) * _expkdt
        s2 = (
            (v[:, i_step] * (sigma ** 2) * _expkdt / kappa)
            * (1 - _expkdt)
            + (theta * (sigma ** 2) / (2 * kappa))
            * (1 - _expkdt) ** 2
        )
        psi = s2 / (m ** 2)

        # Compute a and b; Equation (27) (28)
        b = ((2 / psi) - 1 + (2 / psi).sqrt() * (2 / psi - 1).sqrt()).sqrt()
        a = m / (1 + b ** 2)

        if psi < PSI_CRIT:
            z = randn[:, i_step]
            v_next = a * (b + z) ** 2
        else:
            u = rand[:, i_step]
            p = (psi - 1) / (psi + 1)
            beta = (1 - p) / m
            v_next = torch.where(u > p, ((1 - p) / (1 - u)).log() / beta, 0)

        var[:, i_step + 1] = v_next

    return var


def generate_heston(
    n_paths:int,
    n_steps:int,
    init_state:tuple = (1.0, 0.04),
    kappa: float=0.0,
    theta: float=0.04,
    sigma: float=0.04,
    rho: float=0.0,
    dt:float=1/250,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> tuple:
    """Returns time series following Heston model.

    The drift of the time series is assumed to be vanishing.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple, default=(1.0, 0.04)):
        kappa (float, default=):
        theta (float, default=):
        sigma (float, default=):
        rho (float, default=):
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
        >>> generate_heston(2, 5)
    """
    assert dtype is None
    assert device is None

    GAMMA1 = 0.5
    GAMMA2 = 0.5

    init_spot, init_var = init_state

    # Variance: size (N, T)
    var = generate_cir(
        n_paths= n_paths,
        n_steps= n_steps,
        init_value= init_var,
        kappa= kappa,
        theta= theta,
        sigma= sigma,
        dt= dt,
        dtype= dtype,
        device= device,
    )

    log_spot = torch.empty(n_paths, n_steps)
    log_spot[:, 0] = torch.tensor(init_spot).log()

    for i_step in range(n_steps):
        k0 = - rho * kappa * theta * dt / epsilon
        k1 = GAMMA1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
        k2 = GAMMA2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
        k3 = GAMMA1 * dt * (1 - rho ** 2)
        k4 = GAMMA2 * dt * (1 - rho ** 2)
        v0 = var[:, i_step]
        v1 = var[:, i_step]
        log_spot[:, i_step + 1] = (
            log_spot[:, i_step] + k0 + k1 * v0 + k2 * v1
            + (k3 * v0 + k4 * v1).sqrt() * torch.randn(n_paths)
        )

    spot = log_spot.exp()

    return (spot, var)
