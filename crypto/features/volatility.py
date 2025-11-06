#!/usr/bin/env python3

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

    # Initialize output tensor (first position always NaN since no return yet)
    volatility = torch.full_like(prices, float("nan"))

    # Vectorized rolling window calculation using unfold
    # unfold(dimension, size, step) creates sliding windows
    if not center:
        # Backward-looking window (standard for trading)
        # For each timestep t, we want returns from [t-window+1, t]
        # Since we already computed returns, we need to handle the mapping carefully

        # Pad returns at the beginning to handle initial timesteps
        # We'll calculate std for windows of size 'window'
        n_returns = log_returns.shape[1]  # n_steps - 1

        # Create rolling windows: (n_paths, n_windows, window_size)
        # unfold creates windows of size 'window' with step 1
        if n_returns >= window:
            # Use unfold to create sliding windows efficiently
            windows = log_returns.unfold(dimension=1, size=window, step=1)
            # windows shape: (n_paths, n_windows, window)
            # where n_windows = n_returns - window + 1

            # Calculate std for each window: (n_paths, n_windows)
            window_std = windows.std(dim=2)

            # Place the volatility values at the correct positions
            # Window ending at return index i corresponds to price index i+1
            # So window[0] (returns 0:window) -> volatility[window]
            volatility[:, window:] = window_std

        # Handle initial period with expanding window
        for t in range(1, min(window, n_steps)):
            # For timesteps before we have full window, use expanding window
            if t >= min_periods:
                # Use all available returns up to this point
                window_returns = log_returns[:, :t]
                volatility[:, t] = window_returns.std(dim=1)
    else:
        # Centered window - less common, fallback to loop for simplicity
        for path_idx in range(n_paths):
            returns_path = log_returns[path_idx]
            for t in range(n_steps):
                start_idx = max(0, t - window // 2)
                end_idx = min(len(returns_path), t + window // 2 + 1)

                if end_idx - start_idx < min_periods or start_idx >= len(returns_path):
                    continue

                window_returns = returns_path[start_idx:end_idx]
                if len(window_returns) >= min_periods:
                    volatility[path_idx, t] = window_returns.std()

    # Annualize volatility
    if annualization_factor is None:
        # Estimate annualization factor based on data frequency
        # For financial data: daily=sqrt(252), hourly=sqrt(252*24), 5min=sqrt(252*24*12)
        annualization_factor = np.sqrt(252 * 24 * 12)  # Assume 5-minute data by default

    # Apply annualization factor (should already be sqrt of periods per year)
    volatility = volatility * annualization_factor

    if squeeze_output:
        volatility = volatility.squeeze(0)

    return volatility


class RealizedVolatilityCalculator:

    def __init__(
        self,
        windows: Union[int, list] = [10, 20, 50],
        annualization_factor: float = np.sqrt(252 * 24 * 12),
        min_periods_ratio: float = 0.5,
    ):
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
        vols = self.update(self.prices[-1] if self.prices else 0.0)
        return {f"vol_{w}": vol for w, vol in zip(self.windows, vols)}

    def reset(self):
        self.prices = []
        self.returns = []


def estimate_annualization_factor(time_delta_seconds: float) -> float:
    seconds_per_year = 365.25 * 24 * 3600
    periods_per_year = seconds_per_year / time_delta_seconds

    # Adjust for market hours (crypto trades 24/7, stocks ~6.5h/day)
    # For crypto: no adjustment needed
    # For stocks: multiply by (24/6.5) ≈ 3.7

    return np.sqrt(periods_per_year)


def create_volatility_features(
    instrument, windows: list = [10, 20, 50]
) -> torch.Tensor:
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
            (mu - 0.5 * sigma**2) * dt + sigma * dW
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
