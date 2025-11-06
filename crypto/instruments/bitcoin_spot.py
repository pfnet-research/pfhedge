from typing import Optional, Tuple, TYPE_CHECKING
import pandas as pd
import torch

from pfhedge._utils.typing import TensorOrScalar
from .bitcoin_base import BitcoinBase

if TYPE_CHECKING:
    from crypto.data.loader import CryptoDataLoader


class BitcoinSpot(BitcoinBase):

    def __init__(
        self,
        cost: float = 0.001,  # Higher cost for spot (wider spreads)
        dt: float = 1 / 24 / 12,  # 5-minute bars
        data_loader: Optional["CryptoDataLoader"] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__(
            cost=cost, dt=dt, data_loader=data_loader, dtype=dtype, device=device
        )

        # Spot specific attributes
        self.instrument_type = "spot"
        self.leverage = 1.0  # No leverage for spot

    def _load_data_for_simulation(self, time_horizon: float) -> pd.DataFrame:
        if self.data_loader is None:
            raise ValueError("No data_loader provided")

        # Try to load actual spot data first
        try:
            spot_df = self.data_loader.load_spot_data()
            print("Using actual Bitcoin spot data")
        except FileNotFoundError:
            # Fallback to perpetual data as proxy for spot
            print("No spot data found, using perpetual as proxy")
            perpetual_df = self.data_loader.load_perpetual_data()

            if perpetual_df is None or perpetual_df.empty:
                # Try to load from cached data
                perpetual_df = self.data_loader.perpetual_data

            if perpetual_df is None or perpetual_df.empty:
                raise ValueError("No spot or perpetual data available in data_loader")

            # For spot, we use the perpetual prices but without funding
            # In reality, spot prices might differ slightly from perpetual
            spot_df = perpetual_df[["timestamp", "last_price"]].copy()

            # Add bid/ask if available from perpetual
            if "bid_price" in perpetual_df.columns:
                spot_df["bid_price"] = perpetual_df["bid_price"]
            if "ask_price" in perpetual_df.columns:
                spot_df["ask_price"] = perpetual_df["ask_price"]

        # Add spread if not present
        if "bid_price" in spot_df.columns and "ask_price" in spot_df.columns:
            spot_df["spread"] = spot_df["ask_price"] - spot_df["bid_price"]
            spot_df["spread_pct"] = spot_df["spread"] / spot_df["last_price"]

        return spot_df

    @property
    def has_funding(self) -> bool:
        return False

    @property
    def max_leverage(self) -> float:
        return 1.0

    def margin_requirement(self, position_size: float) -> float:
        current_price = self.spot[:, -1].mean().item() if self.buffers() else 50000.0
        return abs(position_size * current_price)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        from math import ceil

        if self.data_loader is None:
            raise ValueError("No data_loader provided. Cannot load historical data.")

        # Calculate number of steps needed
        n_steps = ceil(time_horizon / self.dt + 1)

        # Load spot data
        spot_data = self._load_data_for_simulation(time_horizon)

        # Ensure we have enough data
        if len(spot_data) < n_steps:
            import warnings

            warnings.warn(
                f"Requested {n_steps} steps but only {len(spot_data)} available. "
                f"Using all available data."
            )
            n_steps = len(spot_data)

        # Take only needed data
        data = spot_data.iloc[:n_steps].copy()

        # Load into buffers
        self.load_historical_data(data, n_paths)

    def __repr__(self) -> str:
        params = [f"cost={self.cost}", f"dt={self.dt}", "leverage=1.0", "type='spot'"]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinSpot({', '.join(params)})"
