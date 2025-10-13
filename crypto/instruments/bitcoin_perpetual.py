"""
Bitcoin perpetual futures instrument for PFHedge.
"""
from typing import Optional
import pandas as pd
import torch
from torch import Tensor

from .bitcoin_base import BitcoinBase


class BitcoinPerpetual(BitcoinBase):
    """Bitcoin perpetual futures - leveraged instrument with funding rates.

    Represents perpetual futures contracts that track Bitcoin price through
    a funding rate mechanism. Most liquid instrument for hedging.

    Args:
        cost (float, default=0.0006): Transaction cost rate (taker fee).
        dt (float, default=1/24/12): Time step interval (default 5 minutes).
        leverage (float, default=20.0): Maximum leverage available.
        data_loader (Optional[CryptoDataLoader]): Data loader for historical data.
        dtype (torch.dtype, optional): Desired dtype of tensors.
        device (torch.device, optional): Desired device of tensors.

    Examples:
        >>> from crypto.data.loader import CryptoDataLoader
        >>> from crypto.instruments import BitcoinPerpetual
        >>>
        >>> # Create data loader
        >>> loader = CryptoDataLoader("sample_data")
        >>> btc_perp = BitcoinPerpetual(cost=0.0006, leverage=20, data_loader=loader)
        >>>
        >>> # Simulate (load historical data)
        >>> btc_perp.simulate(n_paths=1, time_horizon=5/250)
        >>> print(btc_perp.spot.shape)
        torch.Size([1, 25])  # 1 path, 25 time steps
        >>>
        >>> # Access funding rate
        >>> funding = btc_perp.funding_rate
        >>> print(f"Current funding rate: {funding[0, -1].item():.4%}")

    Attributes:
        spot (Tensor): Perpetual prices, shape (n_paths, n_steps)
        bid (Tensor): Bid prices, shape (n_paths, n_steps)
        ask (Tensor): Ask prices, shape (n_paths, n_steps)
        mid (Tensor): Mid prices, shape (n_paths, n_steps)
        funding_rate (Tensor): 8-hour funding rates, shape (n_paths, n_steps)
        index_price (Tensor): Underlying index prices, shape (n_paths, n_steps)
    """

    def __init__(
        self,
        cost: float = 0.0006,  # Lower cost for perpetual (taker fee)
        dt: float = 1 / 24 / 12,  # 5-minute bars
        leverage: float = 20.0,  # Maximum leverage
        data_loader: Optional[object] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Bitcoin perpetual instrument."""
        super().__init__(
            cost=cost, dt=dt, data_loader=data_loader, dtype=dtype, device=device
        )

        # Perpetual specific attributes
        self.instrument_type = "perpetual"
        self.leverage = leverage
        self.funding_interval = 8 / 24  # Funding every 8 hours

    def _load_data_for_simulation(self, time_horizon: float) -> pd.DataFrame:
        """Load perpetual futures data for simulation.

        Args:
            time_horizon: Time period to load data for

        Returns:
            DataFrame with perpetual data including funding rates
        """
        if self.data_loader is None:
            raise ValueError("No data_loader provided")

        # Load perpetual data with funding rates
        perpetual_df = self.data_loader.load_perpetual_data()

        if perpetual_df is None or perpetual_df.empty:
            # Try to access cached data
            perpetual_df = self.data_loader.perpetual_data

        if perpetual_df is None or perpetual_df.empty:
            raise ValueError("No perpetual data available in data_loader")

        # Ensure we have required columns
        required_cols = ["timestamp", "last_price"]
        if not all(col in perpetual_df.columns for col in required_cols):
            raise ValueError(f"Perpetual data must contain: {required_cols}")

        return perpetual_df

    def load_historical_data(self, data: pd.DataFrame, n_paths: int = 1) -> None:
        """Load historical data into buffers, including funding-specific data.

        Args:
            data: DataFrame with perpetual data
            n_paths: Number of paths
        """
        # Call parent to load basic price data
        super().load_historical_data(data, n_paths)

        # Load funding rate if available
        if "funding_8h" in data.columns:
            funding_rates = torch.tensor(
                data["funding_8h"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                funding_rates = funding_rates.repeat(n_paths, 1)
            self.register_buffer("_funding_rate", funding_rates)
        else:
            # Default to zero funding if not available
            spot = self.get_buffer("spot")
            self.register_buffer("_funding_rate", torch.zeros_like(spot))

        # Load index price if available
        if "index_price" in data.columns:
            index_prices = torch.tensor(
                data["index_price"].values, dtype=self.dtype, device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                index_prices = index_prices.repeat(n_paths, 1)
            self.register_buffer("index_price", index_prices)
        else:
            # Use spot price as index if not available
            self.register_buffer("index_price", self.get_buffer("spot").clone())

    @property
    def funding_rate(self) -> Tensor:
        """Get the 8-hour funding rate.

        Returns:
            Tensor of funding rates, shape (n_paths, n_steps)
        """
        if not hasattr(self, "_buffers") or "_funding_rate" not in self._buffers:
            raise ValueError("No funding rate data loaded. Call simulate() first.")
        return self.get_buffer("_funding_rate")

    @property
    def has_funding(self) -> bool:
        """Perpetual instruments have funding rates."""
        return True

    @property
    def max_leverage(self) -> float:
        """Maximum leverage available for perpetual trading."""
        return self.leverage

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
        # Positive position pays when funding > 0
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
        """String representation of BitcoinPerpetual."""
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"leverage={self.leverage}",
            "type='perpetual'",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinPerpetual({', '.join(params)})"
