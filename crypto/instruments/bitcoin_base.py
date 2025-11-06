from abc import abstractmethod
from math import ceil
from typing import Optional, Tuple, TYPE_CHECKING
import torch
from torch import Tensor
import pandas as pd
import numpy as np

from pfhedge.instruments import BasePrimary
from pfhedge._utils.typing import TensorOrScalar

if TYPE_CHECKING:
    from crypto.data.loader import CryptoDataLoader


class BitcoinBase(BasePrimary):

    def __init__(
        self,
        cost: float = 0.0,
        dt: float = 1 / 24 / 12,  # 5-minute bars
        data_loader: Optional["CryptoDataLoader"] = None,
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
        return (1.0,)  # Default initial price normalized to 1

    @property
    def volatility(self) -> Tensor:
        if not self.buffers():
            raise ValueError("No data loaded. Call simulate() first.")

        # Calculate rolling volatility from returns
        spot = self.get_buffer("spot")
        if spot.shape[1] > 1:
            log_returns = torch.log(spot[:, 1:] / spot[:, :-1])
            # Annualized volatility using actual dt (not hardcoded 5-min)
            annualization_factor = (
                np.sqrt(1.0 / self.dt) if self.dt > 0 else np.sqrt(365 * 24 * 12)
            )
            vol = log_returns.std(dim=1, keepdim=True) * annualization_factor
            # Expand to match spot shape
            return vol.expand_as(spot)
        else:
            return torch.zeros_like(spot)

    def load_historical_data(self, data: pd.DataFrame, n_paths: int = 1) -> None:
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
        pass

    @abstractmethod
    def _load_data_for_simulation(self, time_horizon: float) -> pd.DataFrame:
        pass

    @property
    def is_listed(self) -> bool:
        return True

    def to(self, *args, **kwargs):
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
