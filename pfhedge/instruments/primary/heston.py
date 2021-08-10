from typing import Optional
from typing import Tuple

import torch

from ...stochastic import generate_heston
from .base import Primary


class HestonStock(Primary):
    """A stock of which spot price and variance follow Heston process.

    See :func:`pfhedge.stochastic.generate_heston` for details of the process.

    Args:
        kappa (float, default=1.0): The parameter :math:`\\kappa`.
        theta (float, default=0.04): The parameter :math:`\\theta`.
        sigma (float, default=2.0): The parameter :math:`\\sigma`.
        rho (float, default=-0.7): The parameter :math:`\\rho`.
        cost (float, default=0.0): The transaction cost rate.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.device, optional): Desired device of returned tensor.
            Default: If None, uses a global default
            (see `torch.set_default_tensor_type()`).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see `torch.set_default_tensor_type()`).
            `device` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

    Attributes:
        spot (torch.Tensor): The spot price of the instrument.
            This attribute is set by a method :func:`simulate()`.
            The shape is :math:`(N, T)`, where :math:`N` is the number of simulated
            paths and :math:`T` is the number of time steps.
        variance (torch.Tensor): The variance of the spot of the instrument.
            This attribute is set by a method :func:`simulate()`.
            The shape is :math:`(N, T)`.

    Examples:

        >>> from pfhedge.instruments import HestonStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = HestonStock()
        >>> stock.simulate(n_paths=2, time_horizon=5/250)
        >>> stock.spot
        tensor([[1.0000, 0.9958, 0.9940, 0.9895, 0.9765],
                [1.0000, 1.0064, 1.0117, 1.0116, 1.0117]])
        >>> stock.variance
        tensor([[0.0400, 0.0433, 0.0406, 0.0423, 0.0441],
                [0.0400, 0.0251, 0.0047, 0.0000, 0.0000]])
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 2.0,
        rho: float = -0.7,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, float]:
        return (1.0, self.theta)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[tuple] = None,
    ) -> None:
        """Simulate the spot price and add it as a buffer named `spot`.

        The shape of the spot is :math:`(N, T)`, where :math:`N` is the number of
        simulated paths and :math:`T` is the number of time steps.
        The number of time steps is determinded from `dt` and `time_horizon`.

        Args:
            n_paths (int, default=1): The number of paths to simulate.
            time_horizon (float, default=20/250): The period of time to simulate
                the price.
            init_state (tuple, optional): The initial state of the instrument.
                `init_state` should be a 2-tuple `(spot, variance)`
                where `spot` is the initial spot price and `variance` is the initial
                variance.
                If `None` (default), the default value is chosen
                (See :func:`default_init_state`).
        """
        if init_state is None:
            init_state = self.default_init_state

        spot, variance = generate_heston(
            n_paths=n_paths,
            n_steps=int(time_horizon / self.dt),
            init_state=init_state,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            rho=self.rho,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)
        self.register_buffer("variance", variance)

    def __repr__(self):
        params = [
            f"kappa={self.kappa:.2e}",
            f"theta={self.theta:.2e}",
            f"sigma={self.sigma:.2e}",
            f"rho={self.rho:.2e}",
        ]
        if self.cost != 0.0:
            params.append(f"cost={self.cost:.2e}")
        params.append(f"dt={self.dt:.2e}")
        params += self.dinfo
        return self.__class__.__name__ + "(" + ", ".join(params) + ")"


# Assign docstrings so they appear in Sphinx documentation
HestonStock.default_init_state.__doc__ = Primary.default_init_state.__doc__
HestonStock.to = Primary.to
HestonStock.to.__doc__ = Primary.to.__doc__
