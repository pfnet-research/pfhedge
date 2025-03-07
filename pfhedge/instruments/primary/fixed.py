from typing import List, Optional, Tuple
import math

import torch
from torch import Tensor

from pfhedge.instruments.primary.base import BasePrimary


class FixedStock(BasePrimary):
    """A primary instrument that returns fixed spot prices.

    Args:
        spots (List[float]): List of spot prices to return during simulation.
        dt (float, default=1/250): The intervals of the time steps.
        cost (float, default=0.0): The transaction cost rate.
        dtype (torch.dtype, optional): The desired dtype of returned tensor.
        device (torch.device, optional): The desired device of returned tensor.
    """

    def __init__(
        self,
        spots: List[float],
        dt: float = 1 / 250,
        cost: float = 0.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.spots = torch.tensor(spots, dtype=dtype, device=device)
        self.dt = dt
        self.cost = cost
        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (self.spots[0],)

    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        init_state: Optional[Tuple[Tensor, ...]] = None,
    ) -> None:
        """Simulate fixed spot prices.

        Args:
            n_paths (int): The number of paths to simulate.
            time_horizon (float): The period of time to simulate the price.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the instrument.

        Raises:
            ValueError: If the requested number of time steps exceeds the length of stored spots.
        """
        if init_state is None:
            init_state = self.default_init_state
        init_value = init_state[0]

        n_steps = math.ceil(time_horizon / self.dt + 1)
        if n_steps > len(self.spots):
            raise ValueError(
                f"Requested {n_steps} time steps but only {len(self.spots)} spots available"
            )

        scale = init_value / self.spots[0]
        spots = (self.spots[:n_steps] * scale).repeat(n_paths, 1)
        self.register_buffer('spot', spots)

    @property
    def volatility(self) -> Tensor:
        """Returns the realized volatility of the stored spots.

        The volatility is calculated as the standard deviation of log returns,
        annualized using the stored dt value.

        Returns:
            torch.Tensor: The annualized volatility of the stored spots.
        """
        # First compute the realized volatility value
        returns = torch.log(self.spots[1:] / self.spots[:-1])
        std_dev = returns.std()
        vol_value = std_dev / math.sqrt(self.dt)
        # If spot buffer exists, return with same shape as spot
        if hasattr(self, "_buffers") and "spot" in self._buffers:
            return torch.full_like(self.spot, vol_value)
        # Otherwise return the scalar value
        return vol_value

    @property
    def drift(self) -> Tensor:
        """Returns the drift of the stored spots."""
        returns = torch.log(self.spots[1:] / self.spots[:-1])
        return returns.mean() / self.dt
