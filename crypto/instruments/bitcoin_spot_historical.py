"""
Bitcoin spot with historical data for backtesting.

BitcoinSpotHistorical inherits from BitcoinPerpetualHistorical because they share
the same complex bootstrap path generation logic. The only difference is that spot
has no funding rates.

This is a pragmatic design choice: while philosophically spot and perpetual are peers,
for historical backtesting they share 99% of the code (bootstrap sampling, vol calculation,
bid/ask spreads, etc.). The alternative would be extracting 200+ lines of bootstrap logic
into a mixin, which adds complexity without clear benefit.
"""

from .bitcoin_perpetual_historical import BitcoinPerpetualHistorical


class BitcoinSpotHistorical(BitcoinPerpetualHistorical):
    """Bitcoin spot using historical price data for backtesting.

    Inherits all bootstrap functionality from BitcoinPerpetualHistorical but
    overrides funding-related properties to reflect that spot has no funding rates.

    Key differences from perpetual:
    - has_funding = False (no funding rate payments)
    - max_leverage = 1.0 (spot trading has no leverage)

    Everything else (bootstrap paths, volatility, bid/ask) works identically.

    Args:
        data_loader: DataLoader providing historical price data
        cost (float, default=0.001): Transaction cost rate (0.1%)
        dt (float, default=8/24/365): Time step (8 hours in years)
        constant_volatility (float, optional): Override volatility calculation
        volatility_window (int, default=20): Rolling window for realized vol
        dtype (torch.dtype, optional): Tensor dtype
        device (torch.device, optional): Tensor device

    Examples:
        >>> from crypto.data.loader import CryptoDataLoader
        >>> data_loader = CryptoDataLoader("historical_data/")
        >>> data_loader.load_spot_data()
        >>> btc = BitcoinSpotHistorical(data_loader=data_loader)
        >>> btc.simulate_bootstrap(n_paths=100, time_horizon=30/365)
        >>> print(btc.spot.shape)
        torch.Size([100, 91])  # 100 bootstrap paths
    """

    @property
    def has_funding(self) -> bool:
        """Spot instruments do not have funding rates."""
        return False

    @property
    def max_leverage(self) -> float:
        """Spot trading uses no leverage (1x)."""
        return 1.0

    def __repr__(self) -> str:
        """String representation."""
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"volatility_window={self.volatility_window}",
        ]
        if self.constant_volatility is not None:
            params.append(f"constant_volatility={self.constant_volatility}")
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinSpotHistorical({', '.join(params)})"
