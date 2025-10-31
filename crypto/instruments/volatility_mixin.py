"""
Volatility calculation mixin for Bitcoin instruments.

This module provides a shared volatility calculation method that can be used by
multiple instrument types (Brownian simulations and historical data).
"""

import torch
from torch import Tensor
from typing import Optional


class VolatilityMixin:
    """Mixin class providing configurable volatility calculation.

    This mixin supports three volatility modes:

    1. **Constant Volatility** (volatility_window=0):
       - Returns fixed sigma value for all time steps
       - Use case: Match training with constant vol, or when volatility is known

    2. **Rolling Window Realized Volatility** (volatility_window>0):
       - Calculates standard deviation of log returns over last N periods
       - More realistic: adapts to recent market conditions
       - Use case: Training and backtesting with time-varying volatility

    3. **Override Constant Volatility** (constant_volatility parameter):
       - Forces constant value regardless of volatility_window
       - Use case: Backtesting with train/test consistency

    How it works:
    -------------
    Rolling realized volatility measures actual price fluctuations:

    1. Calculate log returns: r_t = log(S_t / S_{t-1})
    2. Take standard deviation over window: σ = std(r_{t-N:t})
    3. Annualize: σ_annual = σ * sqrt(periods_per_year)

    For window=20 with 8-hour bars:
    - Looks back 20 periods = 160 hours ≈ 6.7 days
    - Captures recent volatility regime
    - More responsive than expanding window

    Usage in Deep Hedging:
    ----------------------
    The volatility feature is fed to the neural network:

        hedge_position = NN(log_moneyness, time_to_maturity, volatility, prev_hedge)

    With rolling vol:
    - Model sees σ vary from ~0.5 (calm) to ~1.5 (volatile)
    - Learns to adapt hedge based on current volatility regime
    - More realistic than constant 0.8 volatility

    With constant vol (training):
    - Model sees σ = 0.8 always
    - Simpler, but less realistic
    - Can lead to train/test mismatch if backtest uses rolling vol

    Example:
    --------
    >>> class MyInstrument(VolatilityMixin):
    ...     def __init__(self, sigma=0.8, volatility_window=20):
    ...         self.sigma = sigma
    ...         self.volatility_window = volatility_window
    ...         self.dt = 8/24/365  # 8-hour bars
    ...         self.constant_volatility = None
    ...
    >>> instrument = MyInstrument()
    >>> instrument.register_buffer("spot", prices)
    >>> vol = instrument.calculate_volatility()  # Time-varying volatility
    """

    def calculate_volatility(
        self,
        constant_volatility: Optional[float] = None,
        volatility_window: Optional[int] = None,
        sigma: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Tensor:
        """Calculate volatility based on configuration.

        Priority order:
        1. constant_volatility parameter (if provided)
        2. Rolling window realized vol (if volatility_window > 0)
        3. Constant sigma fallback

        Args:
            constant_volatility: Override with constant value (highest priority)
            volatility_window: Window size for rolling vol (0 = use constant)
            sigma: Constant volatility fallback value
            dt: Time step in years for annualization

        Returns:
            Volatility tensor of same shape as spot prices

        Raises:
            ValueError: If spot buffer not initialized
        """
        # Get parameters from instance if not provided
        if constant_volatility is None:
            constant_volatility = getattr(self, "constant_volatility", None)
        if volatility_window is None:
            volatility_window = getattr(self, "volatility_window", 0)
        if sigma is None:
            sigma = getattr(self, "sigma", 0.8)
        if dt is None:
            dt = getattr(self, "dt", 8 / 24 / 365)

        # Get spot prices
        if not hasattr(self, "spot"):
            raise ValueError("No spot data. Call simulate() first.")
        spot = self.get_buffer("spot")

        # Priority 1: Override with constant volatility
        if constant_volatility is not None:
            return torch.full_like(spot, constant_volatility)

        # Priority 2: Calculate rolling window realized volatility
        if volatility_window > 0:
            from crypto.features.volatility import calculate_realized_volatility

            # Annualization factor: sqrt(periods per year)
            # Example: 8-hour bars -> 1095 periods/year -> sqrt(1095) ≈ 33
            annualization_factor = torch.sqrt(torch.tensor(1.0 / dt))

            realized_vol = calculate_realized_volatility(
                spot,
                window=volatility_window,
                annualization_factor=annualization_factor.item(),
            )

            # Fill NaN values at beginning with constant volatility
            # (First N periods don't have enough history for window)
            realized_vol = torch.where(
                torch.isnan(realized_vol),
                torch.full_like(realized_vol, sigma),
                realized_vol,
            )

            return realized_vol

        # Priority 3: Fallback to constant sigma
        return torch.full_like(spot, sigma)

    @property
    def volatility(self) -> Tensor:
        """Returns volatility using configured calculation method.

        This property should be overridden by subclasses to call
        calculate_volatility() with appropriate parameters, or
        subclasses can use calculate_volatility() directly.

        Returns:
            Volatility tensor
        """
        return self.calculate_volatility()
