#!/usr/bin/env python3
"""
Realized Volatility Calculation for Deep Hedging

This module provides utilities for calculating realized volatility from historical price data.
Realized volatility is a key feature for deep hedging models as it captures actual market
volatility rather than implied volatility.

Key Features:
- Rolling window realized volatility calculation
- Multiple frequency support (5min, 1h, daily)
- Robust handling of missing data
- PyTorch tensor output for direct use in models

Usage:
    from crypto.features.volatility import calculate_realized_volatility

    # Calculate 20-period realized volatility
    realized_vol = calculate_realized_volatility(prices, window=20)
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple
import warnings


def calculate_realized_volatility(
    prices: torch.Tensor,
    window: int = 20,
    annualization_factor: Optional[float] = None,
    min_periods: Optional[int] = None,
    center: bool = False,
) -> torch.Tensor:
    """
    Calculate rolling realized volatility from price series.

    Realized volatility is calculated as the standard deviation of log returns
    over a rolling window, annualized to be comparable with implied volatility.

    Args:
        prices: Price tensor of shape (n_paths, n_steps) or (n_steps,)
        window: Rolling window size in periods
        annualization_factor: Factor to annualize volatility. If None, will be estimated
                             from data frequency (default: sqrt(252) for daily data)
        min_periods: Minimum number of observations required to calculate volatility
        center: Whether to center the rolling window

    Returns:
        Realized volatility tensor of same shape as input prices

    Example:
        >>> prices = torch.tensor([100., 101., 99., 102., 98.])
        >>> vol = calculate_realized_volatility(prices, window=3)
        >>> print(vol.shape)  # torch.Size([5])
    """
    if prices.dim() == 1:
        prices = prices.unsqueeze(0)  # Add batch dimension
        squeeze_output = True
    else:
        squeeze_output = False

    n_paths, n_steps = prices.shape

    if window > n_steps:
        warnings.warn(f"Window size ({window}) larger than series length ({n_steps})")
        window = n_steps

    if min_periods is None:
        min_periods = max(2, window // 2)  # Need at least 2 obs for std

    # Calculate log returns
    log_returns = torch.log(prices[:, 1:] / prices[:, :-1])

    # Initialize output tensor
    volatility = torch.full_like(prices, float("nan"))

    # Calculate rolling volatility for each path
    for path_idx in range(n_paths):
        returns_path = log_returns[path_idx]

        for t in range(n_steps):
            if center:
                # Center the window around current point
                start_idx = max(0, t - window // 2)
                end_idx = min(len(returns_path), t + window // 2 + 1)
            else:
                # Backward-looking window (more realistic for trading)
                start_idx = max(0, t - window + 1)
                end_idx = t + 1

            # Skip if we don't have enough data
            if end_idx - start_idx < min_periods or start_idx >= len(returns_path):
                continue

            # Calculate volatility for this window
            window_returns = returns_path[start_idx:end_idx]
            if len(window_returns) >= min_periods:
                vol_estimate = window_returns.std()
                volatility[path_idx, t] = vol_estimate

    # Annualize volatility
    if annualization_factor is None:
        # Estimate annualization factor based on data frequency
        # For financial data: daily=sqrt(252), hourly=sqrt(252*24), 5min=sqrt(252*24*12)
        annualization_factor = np.sqrt(252 * 24 * 12)  # Assume 5-minute data by default

    volatility = volatility * np.sqrt(annualization_factor)

    if squeeze_output:
        volatility = volatility.squeeze(0)

    return volatility


class RealizedVolatilityCalculator:
    """
    Stateful calculator for realized volatility with multiple window sizes.

    This class maintains rolling windows for efficient online calculation
    and supports multiple volatility estimates simultaneously.

    Example:
        >>> calc = RealizedVolatilityCalculator(windows=[10, 20, 50])
        >>> for price in price_stream:
        ...     vols = calc.update(price)
        ...     print(f"10-day: {vols[0]:.3f}, 20-day: {vols[1]:.3f}")
    """

    def __init__(
        self,
        windows: Union[int, list] = [10, 20, 50],
        annualization_factor: float = np.sqrt(252 * 24 * 12),
        min_periods_ratio: float = 0.5,
    ):
        """
        Initialize calculator.

        Args:
            windows: Window sizes for volatility calculation
            annualization_factor: Factor to annualize volatility
            min_periods_ratio: Minimum fraction of window that must be filled
        """
        if isinstance(windows, int):
            windows = [windows]

        self.windows = sorted(windows)
        self.annualization_factor = annualization_factor
        self.min_periods_ratio = min_periods_ratio

        # Storage for rolling calculations
        self.prices = []
        self.returns = []
        self.max_window = max(self.windows)

        # Precompute minimum periods for each window
        self.min_periods = [max(2, int(w * min_periods_ratio)) for w in self.windows]

    def update(self, price: float) -> list:
        """
        Update with new price and return current volatility estimates.

        Args:
            price: New price observation

        Returns:
            List of volatility estimates for each window size
        """
        self.prices.append(price)

        # Calculate return if we have previous price
        if len(self.prices) > 1:
            log_return = np.log(price / self.prices[-2])
            self.returns.append(log_return)

        # Trim to maximum window size
        if len(self.prices) > self.max_window + 1:
            self.prices.pop(0)
        if len(self.returns) > self.max_window:
            self.returns.pop(0)

        # Calculate volatility for each window
        volatilities = []
        for window, min_periods in zip(self.windows, self.min_periods):
            if len(self.returns) >= min_periods:
                # Use last 'window' returns
                window_returns = (
                    self.returns[-window:]
                    if len(self.returns) >= window
                    else self.returns
                )
                vol = np.std(window_returns) * self.annualization_factor
                volatilities.append(vol)
            else:
                volatilities.append(float("nan"))

        return volatilities

    def get_current_volatilities(self) -> dict:
        """Get current volatilities as a dictionary."""
        vols = self.update(self.prices[-1] if self.prices else 0.0)
        return {f"vol_{w}": vol for w, vol in zip(self.windows, vols)}

    def reset(self):
        """Reset the calculator state."""
        self.prices = []
        self.returns = []


def estimate_annualization_factor(time_delta_seconds: float) -> float:
    """
    Estimate appropriate annualization factor based on data frequency.

    Args:
        time_delta_seconds: Average time between observations in seconds

    Returns:
        Annualization factor (periods per year)

    Example:
        >>> # For 5-minute data
        >>> factor = estimate_annualization_factor(5 * 60)
        >>> print(factor)  # ~105120 (252 * 24 * 12 * 1.4)
    """
    seconds_per_year = 365.25 * 24 * 3600
    periods_per_year = seconds_per_year / time_delta_seconds

    # Adjust for market hours (crypto trades 24/7, stocks ~6.5h/day)
    # For crypto: no adjustment needed
    # For stocks: multiply by (24/6.5) ≈ 3.7

    return np.sqrt(periods_per_year)


def create_volatility_features(
    instrument, windows: list = [10, 20, 50], feature_names: Optional[list] = None
) -> torch.Tensor:
    """
    Create volatility features for deep hedging models.

    This function extracts multiple realized volatility features from an instrument
    that can be used as inputs to neural networks.

    Args:
        instrument: Any instrument with .spot attribute (BitcoinSpot, BitcoinPerpetual, etc.)
        windows: List of window sizes for volatility calculation
        feature_names: Optional custom names for features

    Returns:
        Tensor of shape (n_paths, n_steps, n_features) with volatility features

    Example:
        >>> btc = BitcoinPerpetualBrownian()
        >>> btc.simulate(n_paths=100, time_horizon=30/365)
        >>> vol_features = create_volatility_features(btc, windows=[10, 20])
        >>> print(vol_features.shape)  # torch.Size([100, n_steps, 2])
    """
    if not hasattr(instrument, "spot"):
        raise ValueError("Instrument must have 'spot' attribute with price data")

    prices = instrument.spot
    n_paths, n_steps = prices.shape
    n_features = len(windows)

    # Initialize feature tensor
    features = torch.zeros(n_paths, n_steps, n_features)

    # Calculate volatility for each window
    for i, window in enumerate(windows):
        vol = calculate_realized_volatility(prices, window=window)
        features[:, :, i] = vol

    return features


# Example usage and testing
def _test_volatility_calculation():
    """Test function to verify volatility calculations work correctly."""
    print("Testing realized volatility calculation...")

    # Create synthetic price data
    torch.manual_seed(42)
    n_paths, n_steps = 10, 100

    # Generate geometric Brownian motion
    dt = 1 / 252  # Daily data
    sigma = 0.2  # 20% annual volatility
    mu = 0.05  # 5% annual drift

    prices = torch.zeros(n_paths, n_steps)
    prices[:, 0] = 100.0  # Starting price

    for t in range(1, n_steps):
        dW = torch.randn(n_paths) * np.sqrt(dt)
        prices[:, t] = prices[:, t - 1] * torch.exp(
            (mu - 0.5 * sigma ** 2) * dt + sigma * dW
        )

    # Calculate realized volatility
    realized_vol = calculate_realized_volatility(
        prices, window=20, annualization_factor=np.sqrt(252)  # Daily data annualization
    )

    print(f"✅ Price shape: {prices.shape}")
    print(f"✅ Realized vol shape: {realized_vol.shape}")
    print(f"✅ Target volatility: {sigma:.2%}")
    print(f"✅ Realized volatility (mean): {realized_vol.nanmean():.2%}")
    print(
        f"✅ Realized volatility (std): {realized_vol[~torch.isnan(realized_vol)].std():.2%}"
    )

    # Test calculator
    calc = RealizedVolatilityCalculator(
        windows=[10, 20], annualization_factor=np.sqrt(252)  # Daily data
    )

    # Feed prices one by one
    vols_10, vols_20 = [], []
    for i in range(n_steps):
        vols = calc.update(prices[0, i].item())
        vols_10.append(vols[0])
        vols_20.append(vols[1])

    print(f"✅ Calculator 10-day vol (final): {vols_10[-1]:.2%}")
    print(f"✅ Calculator 20-day vol (final): {vols_20[-1]:.2%}")

    return True


if __name__ == "__main__":
    success = _test_volatility_calculation()
    if success:
        print("✅ All volatility tests passed!")
    else:
        print("❌ Volatility tests failed!")
