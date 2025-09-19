"""
Bitcoin perpetual with historical data replay for backtesting.
"""
from typing import Optional, Tuple
import pandas as pd
import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar

from .bitcoin_perpetual_base import BitcoinPerpetualBase


class BitcoinPerpetualHistorical(BitcoinPerpetualBase):
    """Bitcoin perpetual using historical data for backtesting.

    This implementation loads real historical data for backtesting strategies.
    Unlike BitcoinPerpetualBrownian which generates synthetic paths for training,
    this uses actual market data to test performance on historical scenarios.

    Args:
        data_loader (CryptoDataLoader): Data loader for historical data.
        cost (float, default=0.0006): Transaction cost rate.
        dt (float, default=1/24/12): Time step (5 minutes).
        leverage (float, default=20.0): Maximum leverage.
        dtype (torch.dtype, optional): Tensor dtype.
        device (torch.device, optional): Tensor device.

    Examples:
        >>> from crypto.data.loader import CryptoDataLoader
        >>> loader = CryptoDataLoader("sample_data")
        >>> btc = BitcoinPerpetualHistorical(data_loader=loader)
        >>> btc.simulate(n_paths=1, time_horizon=30/365)
        >>> print(btc.spot.shape)
        torch.Size([1, 8640])  # 1 path, 30 days of 5-min data
        >>>
        >>> # Backtesting a strategy
        >>> from pfhedge.instruments import EuropeanOption
        >>> option = EuropeanOption(btc, strike=50000, maturity=30/365)
        >>> # Now can backtest hedging strategies on real data
    """

    def __init__(
        self,
        data_loader: object,  # Required for historical data
        cost: float = 0.0006,
        dt: float = 1 / 24 / 12,
        leverage: float = 20.0,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize historical Bitcoin perpetual."""
        super().__init__(
            cost=cost,
            dt=dt,
            leverage=leverage,
            dtype=dtype,
            device=device
        )

        if data_loader is None:
            raise ValueError("data_loader is required for BitcoinPerpetualHistorical")

        self.data_loader = data_loader

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Load historical data for backtesting.

        For backtesting, we typically use n_paths=1 to test on actual history.
        If n_paths > 1 is requested, we replicate the same historical path
        multiple times (useful for Monte Carlo with transaction costs).

        Args:
            n_paths: Number of paths (typically 1 for backtesting)
            time_horizon: Time period to load
            init_state: Not used for historical data (uses actual prices)

        Examples:
            >>> btc = BitcoinPerpetualHistorical(data_loader=loader)
            >>> btc.simulate(n_paths=1, time_horizon=30/365)
            >>> # Now have actual historical data for backtesting
            >>>
            >>> # Can also replicate for Monte Carlo with costs
            >>> btc.simulate(n_paths=100, time_horizon=30/365)
            >>> # 100 identical paths, useful for averaging over random trades
        """
        # Load perpetual data with funding rates
        perpetual_df = self.data_loader.load_perpetual_data()

        if perpetual_df is None or perpetual_df.empty:
            # Try cached data
            perpetual_df = self.data_loader.perpetual_data

        if perpetual_df is None or perpetual_df.empty:
            raise ValueError("No perpetual data available in data_loader")

        # Ensure required columns exist
        required_cols = ['timestamp', 'last_price']
        if not all(col in perpetual_df.columns for col in required_cols):
            raise ValueError(f"Perpetual data must contain: {required_cols}")

        # Calculate number of steps needed
        import math
        n_steps = math.ceil(time_horizon / self.dt + 1)

        # Limit data to required time horizon
        data = perpetual_df.head(n_steps).copy()
        actual_steps = len(data)

        if actual_steps < n_steps:
            import warnings
            warnings.warn(
                f"Requested {n_steps} steps but only {actual_steps} available in historical data. "
                f"Using all available data."
            )

        # Extract spot prices
        spot_prices = torch.tensor(
            data['last_price'].values,
            dtype=self.dtype,
            device=self.device
        ).unsqueeze(0)  # Shape: (1, n_steps)

        # Replicate paths if needed
        if n_paths > 1:
            spot_prices = spot_prices.repeat(n_paths, 1)

        self.register_buffer("spot", spot_prices)

        # Load bid/ask spreads
        if 'bid_price' in data.columns and 'ask_price' in data.columns:
            bid_prices = torch.tensor(
                data['bid_price'].values,
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(0)
            ask_prices = torch.tensor(
                data['ask_price'].values,
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(0)

            if n_paths > 1:
                bid_prices = bid_prices.repeat(n_paths, 1)
                ask_prices = ask_prices.repeat(n_paths, 1)
        else:
            # Estimate bid/ask from spot with typical spread
            spread = 0.0002  # 2 basis points
            bid_prices = spot_prices * (1 - spread / 2)
            ask_prices = spot_prices * (1 + spread / 2)

        self.register_buffer("bid", bid_prices)
        self.register_buffer("ask", ask_prices)
        self.register_buffer("mid", (bid_prices + ask_prices) / 2)

        # Load funding rates
        if 'funding_8h' in data.columns:
            funding_rates = torch.tensor(
                data['funding_8h'].values,
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                funding_rates = funding_rates.repeat(n_paths, 1)
        else:
            # Default to zero funding if not available
            funding_rates = torch.zeros_like(spot_prices)

        self.register_buffer("_funding_rate", funding_rates)

        # Load index price
        if 'index_price' in data.columns:
            index_prices = torch.tensor(
                data['index_price'].values,
                dtype=self.dtype,
                device=self.device
            ).unsqueeze(0)
            if n_paths > 1:
                index_prices = index_prices.repeat(n_paths, 1)
        else:
            # Use spot price as index if not available
            index_prices = spot_prices.clone()

        self.register_buffer("index_price", index_prices)

    @property
    def volatility(self) -> Tensor:
        """Returns historical realized volatility.

        Calculates rolling volatility from actual price movements.
        """
        if not hasattr(self, 'spot'):
            raise ValueError("No data loaded. Call simulate() first.")

        spot = self.get_buffer("spot")

        # Calculate returns
        returns = torch.log(spot[:, 1:] / spot[:, :-1])

        # Annualized volatility (assuming 5-min bars)
        # There are 288 5-min bars per day, 365 days per year
        periods_per_year = 288 * 365

        if returns.shape[1] > 0:
            # Use expanding window volatility
            vol_list = []
            for i in range(returns.shape[1]):
                if i == 0:
                    # Use a default volatility for the first period
                    vol = torch.full((returns.shape[0], 1), 0.8,
                                   dtype=self.dtype, device=self.device)
                else:
                    # Calculate volatility up to current point
                    hist_returns = returns[:, :i+1]
                    vol = torch.std(hist_returns, dim=1, keepdim=True) * (periods_per_year ** 0.5)
                vol_list.append(vol)

            # Add initial volatility for time 0
            initial_vol = torch.full((returns.shape[0], 1), 0.8,
                                   dtype=self.dtype, device=self.device)
            vol_list.insert(0, initial_vol)

            volatility = torch.cat(vol_list, dim=1)
        else:
            # No returns, use default
            volatility = torch.full_like(spot, 0.8)

        return volatility

    @property
    def variance(self) -> Tensor:
        """Returns historical realized variance."""
        return self.volatility ** 2

    def simulate_bootstrap(
        self,
        n_paths: int,
        time_horizon: float,
        window_size: Optional[int] = None,
    ) -> None:
        """Bootstrap simulation by randomly sampling historical windows.

        This method creates multiple paths by randomly selecting different
        starting points in the historical data. Useful for generating
        multiple scenarios from limited historical data.

        Args:
            n_paths: Number of bootstrap paths to generate
            time_horizon: Time period for each path
            window_size: Size of historical window to sample from
                        (if None, uses all available data)

        Examples:
            >>> btc = BitcoinPerpetualHistorical(data_loader=loader)
            >>> btc.simulate_bootstrap(n_paths=1000, time_horizon=5/365)
            >>> # Creates 1000 different 5-day paths from historical data
        """
        import math
        import random

        # Load all available data
        perpetual_df = self.data_loader.load_perpetual_data()

        if perpetual_df is None or perpetual_df.empty:
            perpetual_df = self.data_loader.perpetual_data

        if perpetual_df is None or perpetual_df.empty:
            raise ValueError("No perpetual data available")

        # Calculate steps needed per path
        n_steps = math.ceil(time_horizon / self.dt + 1)

        # Determine sampling window
        total_available = len(perpetual_df)
        if window_size is None:
            window_size = total_available
        else:
            window_size = min(window_size, total_available)

        if window_size < n_steps:
            raise ValueError(
                f"Window size ({window_size}) must be >= steps needed ({n_steps})"
            )

        # Generate bootstrap samples
        path_list = []
        bid_list = []
        ask_list = []
        funding_list = []
        index_list = []

        for _ in range(n_paths):
            # Random starting point
            max_start = window_size - n_steps
            start_idx = random.randint(0, max_start)
            end_idx = start_idx + n_steps

            # Extract window
            window_data = perpetual_df.iloc[start_idx:end_idx]

            # Extract prices
            spot = torch.tensor(
                window_data['last_price'].values,
                dtype=self.dtype,
                device=self.device
            )
            path_list.append(spot)

            # Bid/ask
            if 'bid_price' in window_data.columns:
                bid = torch.tensor(
                    window_data['bid_price'].values,
                    dtype=self.dtype,
                    device=self.device
                )
                ask = torch.tensor(
                    window_data['ask_price'].values,
                    dtype=self.dtype,
                    device=self.device
                )
            else:
                spread = 0.0002
                bid = spot * (1 - spread / 2)
                ask = spot * (1 + spread / 2)

            bid_list.append(bid)
            ask_list.append(ask)

            # Funding
            if 'funding_8h' in window_data.columns:
                funding = torch.tensor(
                    window_data['funding_8h'].values,
                    dtype=self.dtype,
                    device=self.device
                )
            else:
                funding = torch.zeros_like(spot)
            funding_list.append(funding)

            # Index
            if 'index_price' in window_data.columns:
                index = torch.tensor(
                    window_data['index_price'].values,
                    dtype=self.dtype,
                    device=self.device
                )
            else:
                index = spot.clone()
            index_list.append(index)

        # Stack all paths
        self.register_buffer("spot", torch.stack(path_list))
        self.register_buffer("bid", torch.stack(bid_list))
        self.register_buffer("ask", torch.stack(ask_list))
        self.register_buffer("mid", (self.bid + self.ask) / 2)
        self.register_buffer("_funding_rate", torch.stack(funding_list))
        self.register_buffer("index_price", torch.stack(index_list))

    def __repr__(self) -> str:
        """String representation."""
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"leverage={self.leverage}",
            "data_loader=...",
        ]
        if hasattr(self, 'dtype') and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, 'device') and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinPerpetualHistorical({', '.join(params)})"