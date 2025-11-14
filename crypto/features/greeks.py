import torch
import numpy as np
from typing import Optional
from scipy.stats import norm


def calculate_bs_greeks(
    spot: torch.Tensor,
    strike: float,
    time_to_expiry: torch.Tensor,
    volatility: float,
    call: bool = True,
) -> dict:
    """Calculate Black-Scholes Greeks for hedging context.

    Returns dict with: delta, gamma, vega, theta
    All returned as torch tensors matching input shapes.
    """
    # Convert to numpy for scipy calculations
    S = spot.cpu().numpy()
    K = strike
    T = time_to_expiry.cpu().numpy()
    sigma = volatility
    r = 0.0  # Risk-free rate (assume 0 for crypto)

    # Prevent division by zero
    T = np.maximum(T, 1e-8)

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate Greeks
    if call:
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1.0

    # Gamma (same for call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for call and put)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Divide by 100 for 1% vol change

    # Theta
    if call:
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
        ) / 365
    else:
        theta = (
            -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
        ) / 365

    # Convert back to torch tensors
    device = spot.device
    return {
        "delta": torch.from_numpy(delta).to(device).float(),
        "gamma": torch.from_numpy(gamma).to(device).float(),
        "vega": torch.from_numpy(vega).to(device).float(),
        "theta": torch.from_numpy(theta).to(device).float(),
    }


def add_greeks_to_features(features_list: list) -> list:
    """Add Greek features to the standard feature list.

    Usage:
        features = add_greeks_to_features(['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge'])
        # Returns: ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge', 'gamma', 'vega']
    """
    # Add gamma and vega as most informative Greeks for hedging
    # Delta is implicitly captured by log_moneyness + expiry_time
    # Theta is less useful for discrete rebalancing
    enhanced_features = features_list.copy()
    if "gamma" not in enhanced_features:
        enhanced_features.append("gamma")
    if "vega" not in enhanced_features:
        enhanced_features.append("vega")
    return enhanced_features
