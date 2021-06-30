from typing import Optional

import torch

from ...stochastic import generate_heston
from .base import Primary


class HestonStock(Primary):
    """A stock of which spot prices follow the geometric Brownian motion.

    The drift of the spot prices is assumed to be vanishing.

    Args:
        volatility (float, default=0.2): The volatility of the price.
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
        spot (torch.Tensor): The spot prices of the instrument.
            This attribute is set by a method `simulate()`.
            The shape is :math:`(N, T)`, where :math:`T` is the number of time steps and
            :math:`N` is the number of simulated paths.
        variance (torch.Tensor): The variance of the spot of the instrument.
            This attribute is set by a method `simulate()`.
            The shape is :math:`(N, T)`.

    Examples:

        >>> from pfhedge.instruments import HestonStock
        >>>
        >>> _ = torch.manual_seed(1)
        >>> stock = HestonStock()
        >>> stock.simulate(n_paths=2, time_horizon=5/250)
        >>> stock.spot
        tensor([[1.0000, 1.0045, 1.0137, 1.0197, 1.0259],
                [1.0000, 0.9869, 0.9977, 0.9955, 1.0208]])
        >>> stock.variance
        tensor([[0.0400, 0.0570, 0.0607, 0.0556, 0.0751],
                [0.0400, 0.0257, 0.0462, 0.0520, 0.0120]])
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
    def default_init_state(self) -> tuple:
        """Returns the default initial state of simulation."""
        return (1.0, 0.04)

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
                where `spot` is the initial spot price and `variance` of the initial
                variance.
                If `None` (default), the default value is chosen
                (See :member:`default_init_state`).
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
HestonStock.to = Primary.to
HestonStock.to.__doc__ = Primary.to.__doc__
