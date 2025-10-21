"""
Bitcoin spot instrument for PFHedge.
"""
from typing import Optional, Tuple, TYPE_CHECKING
import pandas as pd
import torch

from pfhedge._utils.typing import TensorOrScalar
from .bitcoin_base import BitcoinBase

if TYPE_CHECKING:
    from crypto.data.loader import CryptoDataLoader


class BitcoinSpot(BitcoinBase):
    """Bitcoin spot instrument - direct exposure with full capital.

    Represents actual Bitcoin that would be bought/sold on spot exchanges.
    No leverage, no funding rates, simple direct exposure.

    Args:
        cost (float, default=0.001): Transaction cost rate (typically higher for spot).
        dt (float, default=1/24/12): Time step interval (default 5 minutes).
        data_loader (Optional[CryptoDataLoader]): Data loader for historical data.
        dtype (torch.dtype, optional): Desired dtype of tensors.
        device (torch.device, optional): Desired device of tensors.

    Examples:
        >>> from crypto.data.loader import CryptoDataLoader
        >>> from crypto.instruments import BitcoinSpot
        >>>
        >>> # Create data loader
        >>> loader = CryptoDataLoader("sample_data")
        >>> btc_spot = BitcoinSpot(cost=0.001, data_loader=loader)
        >>>
        >>> # Simulate (load historical data)
        >>> btc_spot.simulate(n_paths=1, time_horizon=5/250)
        >>> print(btc_spot.spot.shape)
        torch.Size([1, 25])  # 1 path, 25 time steps (5 days of 5-min bars)

    Attributes:
        spot (Tensor): Spot prices, shape (n_paths, n_steps)
        bid (Tensor): Bid prices, shape (n_paths, n_steps)
        ask (Tensor): Ask prices, shape (n_paths, n_steps)
        mid (Tensor): Mid prices, shape (n_paths, n_steps)
    """

    def __init__(
        self,
        cost: float = 0.001,  # Higher cost for spot (wider spreads)
        dt: float = 1 / 24 / 12,  # 5-minute bars
        data_loader: Optional["CryptoDataLoader"] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Bitcoin spot instrument."""
        super().__init__(
            cost=cost, dt=dt, data_loader=data_loader, dtype=dtype, device=device
        )

        # Spot specific attributes
        self.instrument_type = "spot"
        self.leverage = 1.0  # No leverage for spot

    def _load_data_for_simulation(self, time_horizon: float) -> pd.DataFrame:
        """Load spot price data for simulation.

        Args:
            time_horizon: Time period to load data for

        Returns:
            DataFrame with spot price data
        """
        if self.data_loader is None:
            raise ValueError("No data_loader provided")

        # Load perpetual data (we'll use it as proxy for spot)
        # In production, you might have separate spot data
        perpetual_df = self.data_loader.load_perpetual_data()

        if perpetual_df is None or perpetual_df.empty:
            # Try to load from sample data
            perpetual_df = self.data_loader.perpetual_data

        if perpetual_df is None or perpetual_df.empty:
            raise ValueError("No perpetual data available in data_loader")

        # For spot, we use the perpetual prices but without funding
        # In reality, spot prices might differ slightly from perpetual
        spot_df = perpetual_df[["timestamp", "last_price"]].copy()

        # Add bid/ask if available
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
        """Spot instruments do not have funding rates."""
        return False

    @property
    def max_leverage(self) -> float:
        """Maximum leverage for spot trading (none)."""
        return 1.0

    def margin_requirement(self, position_size: float) -> float:
        """Calculate margin requirement for a position.

        For spot, this is the full notional value.

        Args:
            position_size: Size of the position in BTC

        Returns:
            Margin requirement in USD
        """
        current_price = self.spot[:, -1].mean().item() if self.buffers() else 50000.0
        return abs(position_size * current_price)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Load historical spot price data.

        For spot, we typically use historical data rather than
        synthetic generation. If n_paths > 1, the same historical
        path is replicated.

        Args:
            n_paths: Number of paths (typically 1 for spot)
            time_horizon: Time period to load
            init_state: Not used for historical data
        """
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
        """String representation of BitcoinSpot."""
        params = [f"cost={self.cost}", f"dt={self.dt}", "leverage=1.0", "type='spot'"]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinSpot({', '.join(params)})"
