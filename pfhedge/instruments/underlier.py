from typing import Optional

import torch

from ..stochastic import generate_geometric_brownian
from .base import Primary


class BrownianStock(Primary):
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

    Examples:

        >>> from pfhedge.instruments import BrownianStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = BrownianStock()
        >>> stock.simulate(n_paths=2, time_horizon=5 / 250)
        >>> stock.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930],
                [1.0000, 1.0282, 1.0199, 1.0258, 1.0292]])

        Using custom `dtype` and `device`.

        >>> stock = BrownianStock()
        >>> stock.to(dtype=torch.float64, device="cuda:0")
        BrownianStock(..., dtype=torch.float64, device='cuda:0')
    """

    def __init__(
        self,
        volatility: float = 0.2,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: torch.dtype = None,
        device: torch.device = None,
    ):
        super().__init__()

        self.volatility = volatility
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    def __repr__(self):
        params = [f"volatility={self.volatility:.2e}"]
        if self.cost != 0.0:
            params.append(f"cost={self.cost:.2e}")
        params.append(f"dt={self.dt:.2e}")
        params += self.dinfo
        return self.__class__.__name__ + "(" + ", ".join(params) + ")"

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 350,
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
                `init_state` should be a 1-tuple `(spot,)`
                where spot is the initial spot price.
                If `None` (default), the default value `(1.0,)` is chosen.

        Examples:

            >>> _ = torch.manual_seed(42)
            >>> stock = BrownianStock()
            >>> stock.simulate(n_paths=2, time_horizon=5 / 250, init_state=(2.0,))
            >>> stock.spot
            tensor([[2.0000, 2.0031, 2.0089, 2.0146, 1.9860],
                    [2.0000, 2.0565, 2.0398, 2.0516, 2.0584]])
        """
        if init_state is None:
            # Default value
            init_state = (1.0,)

        self.spot = generate_geometric_brownian(
            n_paths=n_paths,
            n_steps=int(time_horizon / self.dt),
            init_value=init_state[0],
            volatility=self.volatility,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )


# Assign docstrings so they appear in Sphinx documentation
BrownianStock.to = Primary.to
BrownianStock.to.__doc__ = Primary.to.__doc__
