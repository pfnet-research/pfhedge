import torch
from torch import Tensor


def generate_cir(
    n_paths: int,
    n_steps: int,
    init_value: float = 0.04,
    kappa: float = 1.0,
    theta: float = 0.04,
    sigma: float = 2.0,
    dt: float = 1 / 250,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> Tensor:
    """Returns time series following Cox-Ingersoll-Ross process.

    The time evolution of CIR process is given by:

    .. math ::

        dX(t) = \\kappa (\\theta - X(t)) + \\sigma \\sqrt{X(t)} dW(t) \\,.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_value (float, default=0.04): The initial value of the time series.
        kappa (float, default=1.0): The parameter :math:`\\kappa`.
        theta (float, default=0.04): The parameter :math:`\\theta`.
        sigma (float, default=2.0): The parameter :math:`\\sigma`.
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

        >>> from pfhedge.stochastic import generate_cir
        >>>
        >>> _ = torch.manual_seed(42)
        >>> generate_cir(2, 5)
        tensor([[0.0400, 0.0433, 0.0406, 0.0423, 0.0441],
                [0.0400, 0.0251, 0.0047, 0.0000, 0.0000]])

    References:
        - Andersen, Leif B.G., Efficient Simulation of the Heston Stochastic
          Volatility Model (January 23, 2007). Available at SSRN:
          https://ssrn.com/abstract=946405 or http://dx.doi.org/10.2139/ssrn.946404
    """
    # PSI_CRIT in [1.0, 2.0]. See section 3.2.3
    PSI_CRIT = 1.5
    # Prevent zero division
    EPSILON = 1e-8

    output = torch.empty((n_paths, n_steps), dtype=dtype, device=device)
    output[:, 0] = init_value

    randn = torch.randn_like(output)
    rand = torch.rand_like(output)

    kappa = torch.tensor(kappa, dtype=dtype, device=device)
    theta = torch.tensor(theta, dtype=dtype, device=device)
    sigma = torch.tensor(sigma, dtype=dtype, device=device)
    dt = torch.tensor(dt, dtype=dtype, device=device)

    for i_step in range(n_steps - 1):
        v = output[:, i_step]

        # Compute m, s, psi: Eq(17,18)
        exp = (-kappa * dt).exp()
        m = theta + (v - theta) * exp
        s2 = sum(
            v * (sigma ** 2) * exp * (1 - exp) / kappa,
            theta * (sigma ** 2) * ((1 - exp) ** 2) / (2 * kappa),
        )
        psi = s2 / (m ** 2 + EPSILON)

        # Compute V(t + dt) where psi <= PSI_CRIT: Eq(23, 27, 28)
        b = ((2 / psi) - 1 + (2 / psi).sqrt() * (2 / psi - 1).sqrt()).sqrt()
        a = m / (1 + b ** 2)
        next_0 = a * (b + randn[:, i_step]) ** 2

        # Compute V(t + dt) where psi > PSI_CRIT: Eq(25)
        u = rand[:, i_step]
        p = (psi - 1) / (psi + 1)
        beta = (1 - p) / (m + EPSILON)
        pinv = ((1 - p) / (1 - u + EPSILON)).log() / beta
        next_1 = torch.where(u > p, pinv, torch.zeros_like(u))

        output[:, i_step + 1] = torch.where(psi <= PSI_CRIT, next_0, next_1)

    return output
