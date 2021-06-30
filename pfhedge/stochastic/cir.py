import torch
from torch import Tensor



def generate_cir(
    n_paths: int,
    n_steps: int,
    init_value: float = 0.04,
    kappa: float = 0.5,
    theta: float = 0.04,
    sigma: float = 1.0,
    dt: float = 1 / 250,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tensor:
    """Cox-Ingersoll-Ross process

    Examples:

        >>> from pfhedge.stochastic import generate_cir
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_cir(2, 5)
        tensor([[0.0400, 0.0201, 0.0132, 0.0067, 0.0109],
                [0.0400, 0.0201, 0.0115, 0.0048, 0.0040]])

    References:
        - Leif Andersen, Efficient Simulation of the Heston Stochastic Volatility
          Model.
    """
    assert dtype is None
    assert device is None

    # PSI_CRIT in [1.0, 2.0]. See section 3.2.3
    PSI_CRIT = 1.5

    var = torch.empty((n_paths, n_steps), dtype=dtype, device=device)
    var[:, 0] = init_value

    randn = torch.randn_like(var)
    rand = torch.rand_like(var)

    kappa = torch.tensor(kappa, dtype=dtype, device=device)
    theta = torch.tensor(theta, dtype=dtype, device=device)
    sigma = torch.tensor(sigma, dtype=dtype, device=device)

    for i_step in range(n_steps - 1):
        # Compute m, s, psi; Equation (17), (18)
        m = theta + (var[:, i_step] - theta) * (-kappa * dt).exp()
        s2 = (var[:, i_step] * (sigma ** 2) * (-kappa * dt).exp() / kappa) * (
            1 - (-kappa * dt).exp()
        ) + (theta * (sigma ** 2) / (2 * kappa)) * (1 - (-kappa * dt).exp()) ** 2
        psi = s2 / (m ** 2)

        # Compute a and b; Equation (27) (28)
        b = ((2 / psi) - 1 + (2 / psi).sqrt() * (2 / psi - 1).sqrt()).sqrt()
        a = m / (1 + b ** 2)

        # where psi < PSI_CRIT
        z = randn[:, i_step]
        v_next_0 = a * (b + z) ** 2
        # where psi >= PSI_CRIT
        u = rand[:, i_step]
        p = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p) / m
        v_next_1 = torch.where(
            u > p, ((1.0 - p) / (1.0 - u)).log() / beta, torch.zeros_like(u)
        )

        v_next = torch.where(psi > PSI_CRIT, v_next_0, v_next_1)

        var[:, i_step + 1] = v_next

    return var
