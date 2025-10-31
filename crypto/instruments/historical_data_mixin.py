"""
Mixin for instruments that use historical data for backtesting.

This module provides shared functionality for bootstrap path generation
from historical data, used by both spot and perpetual historical instruments.
"""

from typing import Optional, Tuple, cast
import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar


class HistoricalDataMixin:
    """Mixin providing historical data bootstrap functionality.

    This mixin handles the common logic for:
    - Loading historical price data via data_loader
    - Generating bootstrap paths for backtesting
    - Creating bid/ask spreads based on historical prices

    Used by both BitcoinSpotHistorical and BitcoinPerpetualHistorical
    to share bootstrap logic without inheritance coupling.

    Required attributes (must be set by the class using this mixin):
        - data_loader: CryptoDataLoader instance
        - cost: Transaction cost rate
        - dt: Time step in years
        - dtype: Torch dtype
        - device: Torch device
    """

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Generate bootstrap paths from historical data.

        Creates multiple resampled paths from historical prices for
        Monte Carlo evaluation of hedging strategies.

        Args:
            n_paths: Number of bootstrap paths to generate
            time_horizon: Time period to simulate (in years)
            init_state: Initial price state

        Note:
            This calls data_loader.generate_bootstrap_paths() which
            handles the actual bootstrap resampling logic.
        """
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        # Delegate to data loader for bootstrap path generation
        # The data loader knows how to resample historical data
        paths = self.data_loader.generate_bootstrap_paths(
            n_paths=n_paths, time_horizon=time_horizon, init_state=init_state
        )

        # Register as spot buffer
        self.register_buffer("spot", paths)

        # Generate bid/ask spreads (same logic as Brownian simulations)
        base_spread = 0.0001  # 1 basis point
        volatility_adjustment = torch.abs(torch.randn_like(paths) * base_spread)
        spread = base_spread + volatility_adjustment

        bid = paths * (1 - spread / 2)
        ask = paths * (1 + spread / 2)

        self.register_buffer("bid", bid)
        self.register_buffer("ask", ask)
        self.register_buffer("mid", (bid + ask) / 2)
