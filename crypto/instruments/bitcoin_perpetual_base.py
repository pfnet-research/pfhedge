"""
Base Bitcoin perpetual futures instrument for PFHedge.
"""

from abc import abstractmethod
from typing import Optional, Tuple
import torch
from torch import Tensor

from pfhedge.instruments import BasePrimary
from pfhedge._utils.typing import TensorOrScalar


class BitcoinPerpetualBase(BasePrimary):
    """Abstract base class for Bitcoin perpetual futures.

    This class defines the interface for Bitcoin perpetual instruments
    but does not implement simulation. Subclasses must implement
    the simulate() method according to their model (Brownian, Jump-Diffusion, Historical, etc.)

    Args:
        cost (float, default=0.0006): Transaction cost rate (taker fee).
        dt (float, default=1/24/12): Time step interval (default 5 minutes).
        leverage (float, default=20.0): Maximum leverage available.
        dtype (torch.dtype, optional): Desired dtype of tensors.
        device (torch.device, optional): Desired device of tensors.

    Attributes:
        cost (float): Transaction cost rate
        dt (float): Time step size
        leverage (float): Maximum leverage
        funding_interval (float): Funding payment interval (8 hours)
    """

    def __init__(
        self,
        cost: float = 0.0006,  # Lower cost for perpetual (taker fee)
        dt: float = 1 / 24 / 12,  # 5-minute bars
        leverage: float = 20.0,  # Maximum leverage
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Bitcoin perpetual base."""
        super().__init__()

        self.cost = cost
        self.dt = dt
        self.leverage = leverage
        # Funding every 8 hours (expressed in years to match dt units)
        self.funding_interval = (8 / 24) / 365

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        """Default initial state for simulation.

        Returns (1.0,) to match normalized pricing convention used in training.
        When training with normalized strikes (K/S0), we start at S0=1.0.
        """
        return (1.0,)

    @property
    def is_listed(self) -> bool:
        """Bitcoin perpetuals are always listed."""
        return True

    @property
    def has_funding(self) -> bool:
        """Perpetual instruments have funding rates."""
        return True

    @property
    def max_leverage(self) -> float:
        """Maximum leverage available for perpetual trading."""
        return self.leverage

    @abstractmethod
    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Simulate price paths for the perpetual.

        This method must be implemented by subclasses according to their
        specific model (Brownian motion, jump diffusion, historical replay, etc.)

        Args:
            n_paths: Number of paths to simulate
            time_horizon: Time period to simulate
            init_state: Initial state (price)
        """
        pass

    @property
    def funding_rate(self) -> Tensor:
        """Get the 8-hour funding rate.

        Returns:
            Tensor of funding rates, shape (n_paths, n_steps)
        """
        if not hasattr(self, "_buffers") or "_funding_rate" not in self._buffers:
            raise ValueError("No funding rate data. Call simulate() first.")
        return self.get_buffer("_funding_rate")

    def cumulative_funding_cost(self, position_size: Optional[Tensor] = None) -> Tensor:
        """Calculate cumulative funding cost/revenue.

        For long positions:
        - Pay funding when rate > 0
        - Receive funding when rate < 0

        For short positions (opposite):
        - Receive funding when rate > 0
        - Pay funding when rate < 0

        Args:
            position_size: Position size in BTC (positive=long, negative=short)
                         If None, assumes position size of 1.0

        Returns:
            Cumulative funding cost (positive = cost, negative = revenue)
        """
        if "_funding_rate" not in self._buffers:
            raise ValueError("No funding rate data loaded")

        funding = self.get_buffer("_funding_rate")
        spot = self.get_buffer("spot")

        if position_size is None:
            position_size = torch.ones(
                funding.shape[0], 1, dtype=self.dtype, device=self.device
            )
        elif not isinstance(position_size, Tensor):
            position_size = torch.tensor(
                position_size, dtype=self.dtype, device=self.device
            )

        # Ensure position_size has correct shape
        if position_size.dim() == 0:
            position_size = position_size.unsqueeze(0).unsqueeze(0)
        elif position_size.dim() == 1:
            position_size = position_size.unsqueeze(1)

        # Funding cost = position_size * funding_rate * notional_value
        funding_cost = position_size * funding * spot

        # Cumulative sum over time
        return funding_cost.cumsum(dim=1)

    def margin_requirement(self, position_size: float) -> float:
        """Calculate margin requirement for a position.

        For perpetual, this depends on leverage.

        Args:
            position_size: Size of the position in BTC

        Returns:
            Margin requirement in USD
        """
        current_price = self.spot[:, -1].mean().item() if self.buffers() else 50000.0
        notional_value = abs(position_size * current_price)
        return notional_value / self.leverage

    def funding_payment_times(self) -> Tensor:
        """Get indices where funding payments occur.

        Funding is paid every 8 hours in crypto markets.

        Returns:
            Boolean tensor indicating funding payment times
        """
        if not self.buffers():
            raise ValueError("No data loaded. Call simulate() first.")

        n_steps = self.get_buffer("spot").shape[1]
        steps_per_funding = int(self.funding_interval / self.dt)

        # Create boolean mask for funding times
        funding_times = torch.zeros(n_steps, dtype=torch.bool, device=self.device)
        funding_times[::steps_per_funding] = True

        return funding_times

    def __repr__(self) -> str:
        """String representation."""
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"leverage={self.leverage}",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"{self.__class__.__name__}({', '.join(params)})"
