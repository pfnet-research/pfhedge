"""
Abstract base class for Bitcoin instruments.
"""
from abc import abstractmethod
from math import ceil
from typing import Optional, Tuple
import torch
from torch import Tensor
import pandas as pd
import numpy as np

from pfhedge.instruments import BasePrimary
from pfhedge._utils.typing import TensorOrScalar


class BitcoinBase(BasePrimary):
    """Abstract base class for Bitcoin instruments.

    This class provides common functionality for Bitcoin spot and perpetual instruments,
    including historical data loading and buffer management.

    Args:
        cost (float, default=0.0): Transaction cost rate.
        dt (float, default=1/24/12): Time step interval (default 5 minutes).
        data_loader (Optional[CryptoDataLoader]): Data loader for historical data.
        dtype (torch.dtype, optional): Desired dtype of tensors.
        device (torch.device, optional): Desired device of tensors.
    """

    def __init__(
        self,
        cost: float = 0.0,
        dt: float = 1 / 24 / 12,  # 5-minute bars
        data_loader: Optional[object] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.cost = cost
        self.dt = dt
        self.data_loader = data_loader

        # Store data for reuse
        self._cached_data = None
        self._time_horizon = None

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        """Default initial state for simulation."""
        return (1.0,)  # Default initial price normalized to 1

    @property
    def volatility(self) -> Tensor:
        """Returns the volatility of the instrument.

        Calculated from historical data if available.
        """
        if not self.buffers():
            raise ValueError("No data loaded. Call simulate() first.")

        # Calculate rolling volatility from returns
        spot = self.get_buffer("spot")
        if spot.shape[1] > 1:
            log_returns = torch.log(spot[:, 1:] / spot[:, :-1])
            # Annualized volatility assuming 5-minute bars
            annualization_factor = np.sqrt(365 * 24 * 12)  # 5-min bars per year
            vol = log_returns.std(dim=1, keepdim=True) * annualization_factor
            # Expand to match spot shape
            return vol.expand_as(spot)
        else:
            return torch.zeros_like(spot)

    def load_historical_data(self, data: pd.DataFrame, n_paths: int = 1) -> None:
        """Load historical data into buffers.

        Args:
            data: DataFrame with price data (must have 'last_price' column)
            n_paths: Number of paths (for compatibility, usually 1 for historical)
        """
        if data is None or data.empty:
            raise ValueError("No data provided")

        # Ensure we have required columns
        required_cols = ["last_price"]
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")

        # Convert to tensors
        spot_prices = torch.tensor(
            data["last_price"].values, dtype=self.dtype, device=self.device
        ).unsqueeze(
            0
        )  # Shape: (1, n_steps)

        # Replicate for n_paths if needed
        if n_paths > 1:
            spot_prices = spot_prices.repeat(n_paths, 1)

        # Register core buffers
        self.register_buffer("spot", spot_prices)

        # Add bid/ask if available
        if "bid_price" in data.columns:
            bid_prices = torch.tensor(
                data["bid_price"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                bid_prices = bid_prices.repeat(n_paths, 1)
            self.register_buffer("bid", bid_prices)

        if "ask_price" in data.columns:
            ask_prices = torch.tensor(
                data["ask_price"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                ask_prices = ask_prices.repeat(n_paths, 1)
            self.register_buffer("ask", ask_prices)

        # Calculate and store mid price
        if "bid_price" in data.columns and "ask_price" in data.columns:
            mid_prices = (self.get_buffer("bid") + self.get_buffer("ask")) / 2
            self.register_buffer("mid", mid_prices)

    @abstractmethod
    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Simulate price paths for the instrument.

        This method must be implemented by subclasses according to their
        specific simulation model (Brownian motion, historical replay, etc.).

        Args:
            n_paths: Number of paths to simulate
            time_horizon: Time period to simulate
            init_state: Initial state
        """
        pass

    @abstractmethod
    def _load_data_for_simulation(self, time_horizon: float) -> pd.DataFrame:
        """Load appropriate data for simulation.

        This method should be implemented by subclasses to load
        either spot or perpetual data.

        Args:
            time_horizon: Time period to load data for

        Returns:
            DataFrame with price data
        """
        pass

    @property
    def is_listed(self) -> bool:
        """Bitcoin instruments are always listed (tradeable on exchanges)."""
        return True

    def to(self, *args, **kwargs):
        """Move buffers to device/dtype."""
        # Call parent implementation
        super().to(*args, **kwargs)

        # Update dtype and device attributes
        if "dtype" in kwargs:
            self.dtype = kwargs["dtype"]
        if "device" in kwargs:
            self.device = kwargs["device"]

        # Move all buffers
        for name, buffer in list(self._buffers.items()):
            if buffer is not None:
                self._buffers[name] = buffer.to(*args, **kwargs)

        return self

    def __repr__(self) -> str:
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"{self.__class__.__name__}({', '.join(params)})"
