import math
from itertools import islice
from typing import List
from typing import Optional
from typing import Tuple

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
            ValueError: If the requested number of paths exceeds the maximum number of paths.
        """
        if n_paths > self.max_n_paths(time_horizon):
            raise ValueError(
                f"Requested {n_paths} paths but only {self.max_n_paths(time_horizon)} available"
            )
        n_steps = self._n_steps(time_horizon)

        if init_state is None:
            init_state = self.default_init_state
        init_value = init_state[0]

        def spot_generator():
            yield self.spots[-n_steps:]
            start, end = -n_steps - 1, -1
            while start >= -len(self.spots):
                yield self.spots[start:end]
                start, end = start - 1, end - 1

        spots = [s * init_value / s[0] for s in islice(spot_generator(), n_paths)]
        self.register_buffer("spot", torch.stack(spots))

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

    def _n_steps(self, time_horizon: float) -> int:
        return math.ceil(time_horizon / self.dt + 1)

    def max_n_paths(self, time_horizon: float) -> int:
        """Returns the maximum number of paths that can be simulated."""
        return max(len(self.spots) - self._n_steps(time_horizon) + 1, 0)
