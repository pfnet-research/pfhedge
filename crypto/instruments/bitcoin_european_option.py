#!/usr/bin/env python3
"""
Bitcoin European Option for Deep Hedging

This module provides a Bitcoin-specific European option that integrates with our
deep hedging framework. It extends PFHedge's EuropeanOption with:
- Bitcoin-specific volatility modeling
- Realized volatility features
- Transaction costs
- Integration with our Bitcoin instruments

Usage:
    from crypto.instruments import BitcoinEuropeanOption, BitcoinPerpetualBrownian

    # Create underlying
    btc = BitcoinPerpetualBrownian(sigma=0.8)
    btc.simulate(n_paths=1000, time_horizon=30/365)

    # Create option
    option = BitcoinEuropeanOption(btc, strike=50000, maturity=30/365)
    option.simulate()  # Inherits paths from underlying

    # Get payoffs for deep hedging
    payoffs = option.payoff()
"""

import torch
import numpy as np
from typing import Optional, Union, Tuple
import warnings

from pfhedge.instruments import EuropeanOption
from crypto.features.volatility import (
    calculate_realized_volatility,
    create_volatility_features,
)


class BitcoinEuropeanOption(EuropeanOption):
    """
    Bitcoin European Option with deep hedging features.

    Extends PFHedge's EuropeanOption with Bitcoin-specific functionality:
    - Realized volatility calculation from underlying price paths
    - Transaction costs for Bitcoin trading
    - Feature extraction for neural networks
    - Integration with Bitcoin instruments

    Args:
        underlier: Bitcoin instrument (BitcoinPerpetualBrownian, etc.)
        strike: Strike price of the option
        maturity: Time to maturity in years
        call: True for call option, False for put option
        cost: Transaction cost rate for option trading
    """

    def __init__(
        self,
        underlier,
        strike: float,
        maturity: float,
        call: bool = True,
        cost: float = 0.0,
    ):
        super().__init__(
            underlier=underlier, strike=strike, maturity=maturity, call=call
        )

        self.cost = cost
        self._realized_vol_cache = None
        self._vol_features_cache = None

    @property
    def transaction_cost(self) -> float:
        """Transaction cost rate for option trading."""
        return self.cost

    def payoff(self) -> torch.Tensor:
        """
        Calculate option payoff with transaction costs.

        Returns:
            Option payoffs after transaction costs
        """
        # Get base payoff from parent class
        base_payoff = super().payoff()

        # Apply transaction costs
        if self.cost > 0:
            # Transaction cost applied to gross payoff
            transaction_cost = self.cost * torch.clamp(base_payoff, min=0)
            net_payoff = base_payoff - transaction_cost
            return torch.clamp(net_payoff, min=0)  # Can't have negative payoff
        else:
            return base_payoff

    # Note: We don't override moneyness() or time_to_maturity()
    # Use parent EuropeanOption implementations which have correct signatures:
    # - moneyness(time_step=None, log=False)
    # - time_to_maturity(time_step=None)

    def realized_volatility(
        self, windows: Optional[list] = None, recalculate: bool = False
    ) -> torch.Tensor:
        """
        Calculate realized volatility from underlying price paths.

        Args:
            windows: List of window sizes for volatility calculation
            recalculate: Force recalculation even if cached

        Returns:
            Realized volatility tensor
        """
        if windows is None:
            windows = [10, 20]

        cache_key = tuple(windows)

        # Check cache
        if not recalculate and self._realized_vol_cache is not None:
            if cache_key in self._realized_vol_cache:
                return self._realized_vol_cache[cache_key]

        if not hasattr(self.underlier, "spot"):
            raise ValueError("Underlier must be simulated first")

        # Calculate realized volatility for each window
        vol_tensors = []
        for window in windows:
            vol = calculate_realized_volatility(
                self.underlier.spot,
                window=window,
                annualization_factor=np.sqrt(252 * 24 * 12),  # Crypto 24/7, 5-min data
            )
            vol_tensors.append(vol)

        # Stack into single tensor: (n_paths, n_steps, n_windows)
        realized_vol = torch.stack(vol_tensors, dim=-1)

        # Cache result
        if self._realized_vol_cache is None:
            self._realized_vol_cache = {}
        self._realized_vol_cache[cache_key] = realized_vol

        return realized_vol

    def deep_hedging_features(
        self,
        vol_windows: Optional[list] = None,
        include_time: bool = True,
        include_moneyness: bool = True,
        include_volatility: bool = True,
    ) -> dict:
        """
        Create feature set for deep hedging neural networks.

        Args:
            vol_windows: Volatility calculation windows
            include_time: Include time-to-maturity feature
            include_moneyness: Include log-moneyness feature
            include_volatility: Include realized volatility features

        Returns:
            Dictionary of features ready for neural network input
        """
        if not hasattr(self.underlier, "spot"):
            raise ValueError("Underlier must be simulated first")

        features = {}
        n_paths, n_steps = self.underlier.spot.shape

        if include_moneyness:
            # Use parent's moneyness() with log=True for log-moneyness
            features["log_moneyness"] = self.moneyness(log=True)

        if include_time:
            ttm = self.time_to_maturity()
            # Broadcast to match spot shape
            features["time_to_maturity"] = ttm.expand(n_paths, n_steps)

        if include_volatility:
            if vol_windows is None:
                vol_windows = [5, 10, 20]

            vol_features = self.realized_volatility(vol_windows)
            # Add each volatility window as separate feature
            for i, window in enumerate(vol_windows):
                features[f"realized_vol_{window}"] = vol_features[:, :, i]

        return features

    def black_scholes_delta(
        self, volatility: Optional[torch.Tensor] = None, risk_free_rate: float = 0.0
    ) -> torch.Tensor:
        """
        Calculate Black-Scholes delta for comparison with deep hedging.

        This computes the optimal hedge ratio at each time step for each simulated path.
        For example, if n_steps represents daily observations, delta[i, j] tells you
        how many units of the underlying to hold on day j for path i.

        Args:
            volatility: Volatility to use. If None, uses underlier's constant volatility.
            risk_free_rate: Risk-free rate

        Returns:
            Black-Scholes delta values of shape (n_paths, n_steps).
            Each delta[i, j] is the hedge ratio for path i at time step j.
        """
        if volatility is None:
            # Use underlier's constant volatility instead of realized vol
            # to avoid NaN issues with short time series
            volatility = self.underlier.volatility

        # Get spot prices: shape (n_paths, n_steps)
        spot = self.underlier.spot
        # Strike is scalar
        strike = self.strike
        # Time to maturity: shape (n_steps,)
        ttm = self.time_to_maturity()

        # Broadcast time to match spot shape
        n_paths, n_steps = spot.shape
        # ttm after expand: (n_paths, n_steps)
        ttm = ttm.expand(n_paths, n_steps)

        # Black-Scholes delta calculation
        # All tensors below have shape (n_paths, n_steps)

        # Handle edge cases: add small epsilon to avoid sqrt(0)
        # sqrt_ttm: (n_paths, n_steps)
        sqrt_ttm = torch.sqrt(ttm + 1e-8)
        # vol_sqrt_ttm: (n_paths, n_steps)
        vol_sqrt_ttm = volatility * sqrt_ttm

        # Avoid division by zero or very small numbers
        # vol_sqrt_ttm: (n_paths, n_steps)
        vol_sqrt_ttm = torch.clamp(vol_sqrt_ttm, min=1e-6)

        # Calculate d1 from Black-Scholes formula
        # d1: (n_paths, n_steps)
        d1 = (
            torch.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * ttm
        ) / vol_sqrt_ttm

        # Handle NaN values by replacing with 0
        # d1: (n_paths, n_steps)
        d1 = torch.where(torch.isnan(d1) | torch.isinf(d1), torch.zeros_like(d1), d1)

        # Clamp d1 to reasonable range to avoid overflow in normal.cdf
        # d1: (n_paths, n_steps)
        d1 = torch.clamp(d1, min=-10, max=10)

        from torch.distributions import Normal

        normal = Normal(0, 1)

        if self.call:
            # Delta for call option: N(d1)
            # delta: (n_paths, n_steps)
            delta = normal.cdf(d1)
        else:
            # Delta for put option: N(d1) - 1
            # delta: (n_paths, n_steps)
            delta = normal.cdf(d1) - 1

        return delta

    def summary(self) -> dict:
        """
        Generate summary statistics for the option.

        Returns:
            Dictionary with option summary statistics
        """
        if not hasattr(self.underlier, "spot"):
            return {
                "strike": self.strike,
                "maturity": self.maturity,
                "call": self.call,
                "cost": self.cost,
                "simulated": False,
            }

        # Calculate payoffs and statistics
        payoffs = self.payoff()
        spot_final = self.underlier.spot[:, -1]

        summary = {
            "strike": self.strike,
            "maturity": self.maturity,
            "call": self.call,
            "cost": self.cost,
            "simulated": True,
            "n_paths": self.underlier.spot.shape[0],
            "n_steps": self.underlier.spot.shape[1],
            "spot_initial": self.underlier.spot[:, 0].mean().item(),
            "spot_final_mean": spot_final.mean().item(),
            "spot_final_std": spot_final.std().item(),
            "payoff_mean": payoffs.mean().item(),
            "payoff_std": payoffs.std().item(),
            "payoff_max": payoffs.max().item(),
            "itm_ratio": (payoffs > 0).float().mean().item(),
        }

        # Add moneyness information
        if self.call:
            summary["otm_ratio"] = (spot_final < self.strike).float().mean().item()
            summary["atm_ratio"] = 1 - summary["itm_ratio"] - summary["otm_ratio"]
        else:
            summary["otm_ratio"] = (spot_final > self.strike).float().mean().item()
            summary["atm_ratio"] = 1 - summary["itm_ratio"] - summary["otm_ratio"]

        return summary


def create_bitcoin_option_from_config(
    config: dict, underlier_class=None
) -> Tuple[BitcoinEuropeanOption, dict]:
    """
    Create Bitcoin option and underlying from configuration.

    Args:
        config: Configuration dictionary with option parameters
        underlier_class: Class to use for underlying (default: BitcoinPerpetualBrownian)

    Returns:
        Tuple of (option, summary_dict)

    Example:
        config = {
            'strike': 50000,
            'maturity_days': 30,
            'call': True,
            'cost': 0.001,
            'sigma': 0.8,
            'mu': 0.0,
            'n_paths': 1000,
            'seed': 42
        }
        option, summary = create_bitcoin_option_from_config(config)
    """
    if underlier_class is None:
        from crypto.instruments import BitcoinPerpetualBrownian

        underlier_class = BitcoinPerpetualBrownian

    # Set random seed if provided
    if "seed" in config:
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])

    # Create underlying instrument
    underlier = underlier_class(
        sigma=config.get("sigma", 0.8),
        mu=config.get("mu", 0.0),
        cost=config.get("underlier_cost", 0.001),
        dt=config.get("dt", 8 / 24 / 365),  # Default 8-hour bars
    )

    # Simulate underlying
    maturity = config.get("maturity_days", 30) / 365
    underlier.simulate(n_paths=config.get("n_paths", 1000), time_horizon=maturity)

    # Create option
    option = BitcoinEuropeanOption(
        underlier=underlier,
        strike=config.get("strike", 50000),
        maturity=maturity,
        call=config.get("call", True),
        cost=config.get("cost", 0.0),
    )

    # Generate summary
    summary = option.summary()
    summary["config"] = config

    return option, summary


# Example usage and testing
def _test_bitcoin_european_option():
    """Test function to verify Bitcoin European option works correctly."""
    print("Testing Bitcoin European Option...")

    torch.manual_seed(42)

    # Create test configuration
    config = {
        "strike": 50000,
        "maturity_days": 30,
        "call": True,
        "cost": 0.001,
        "sigma": 0.8,
        "mu": 0.0,
        "n_paths": 100,
        "seed": 42,
    }

    # Create option
    option, summary = create_bitcoin_option_from_config(config)

    print(f"✅ Created Bitcoin European Option")
    print(f"✅ Strike: ${summary['strike']:.0f}")
    print(f"✅ Maturity: {summary['maturity']:.3f} years")
    print(f"✅ Paths: {summary['n_paths']}")
    print(f"✅ ITM ratio: {summary['itm_ratio']:.1%}")
    print(f"✅ Average payoff: ${summary['payoff_mean']:.2f}")

    # Test features
    features = option.deep_hedging_features()
    print(f"✅ Features created: {list(features.keys())}")

    # Test Black-Scholes delta
    bs_delta = option.black_scholes_delta()
    print(f"✅ BS Delta shape: {bs_delta.shape}")
    print(f"✅ BS Delta range: {bs_delta.min():.3f} - {bs_delta.max():.3f}")

    return True


if __name__ == "__main__":
    success = _test_bitcoin_european_option()
    if success:
        print("✅ All Bitcoin European Option tests passed!")
    else:
        print("❌ Bitcoin European Option tests failed!")
