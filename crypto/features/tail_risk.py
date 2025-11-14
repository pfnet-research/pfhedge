import torch
from typing import Optional


def calculate_volatility_skew(
    spots: torch.Tensor,
    window_short: int = 10,
    window_long: int = 30,
) -> torch.Tensor:
    """Calculate volatility skew as ratio of short-term to long-term volatility.

    Higher ratio indicates increasing volatility (regime change).

    Args:
        spots: Price tensor (n_paths, n_steps)
        window_short: Short-term window
        window_long: Long-term window

    Returns:
        Volatility skew ratio (n_paths, n_steps)
    """
    from .volatility import calculate_realized_volatility

    vol_short = calculate_realized_volatility(spots, window=window_short)
    vol_long = calculate_realized_volatility(spots, window=window_long)

    # Avoid division by zero
    vol_skew = vol_short / (vol_long + 1e-8)

    return vol_skew


def calculate_spot_momentum(
    spots: torch.Tensor,
    window: int = 20,
) -> torch.Tensor:
    """Calculate spot momentum as normalized distance from moving average.

    Momentum captures trend that often precedes reversals (tail events).

    Args:
        spots: Price tensor (n_paths, n_steps)
        window: Moving average window

    Returns:
        Normalized momentum (n_paths, n_steps)
    """
    if spots.dim() == 1:
        spots = spots.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    n_paths, n_steps = spots.shape
    momentum = torch.zeros_like(spots)

    # Calculate moving average using unfold
    if n_steps >= window:
        windows = spots.unfold(dimension=1, size=window, step=1)
        ma = windows.mean(dim=2)

        # Place MA values at correct positions
        # Normalized momentum: (spot - MA) / MA
        for t in range(window - 1, n_steps):
            ma_value = ma[:, t - window + 1]
            momentum[:, t] = (spots[:, t] - ma_value) / (ma_value + 1e-8)

    # Handle initial period with expanding window
    for t in range(1, min(window, n_steps)):
        ma_value = spots[:, : t + 1].mean(dim=1)
        momentum[:, t] = (spots[:, t] - ma_value) / (ma_value + 1e-8)

    if squeeze_output:
        momentum = momentum.squeeze(0)

    return momentum


def calculate_gamma_exposure(
    spots: torch.Tensor,
    strike: float,
    time_to_expiry: torch.Tensor,
    volatility: float,
) -> torch.Tensor:
    """Calculate gamma exposure: BS_gamma * spot^2 / 100.

    High gamma means rapid delta changes requiring frequent rehedging.
    This is a key tail risk indicator.

    Args:
        spots: Price tensor (n_paths, n_steps)
        strike: Strike price
        time_to_expiry: Time to expiry tensor (n_paths, n_steps)
        volatility: Volatility parameter

    Returns:
        Gamma exposure (n_paths, n_steps)
    """
    from ..features.greeks import calculate_bs_greeks

    # Calculate BS greeks
    greeks = calculate_bs_greeks(
        spot=spots,
        strike=strike,
        time_to_expiry=time_to_expiry,
        volatility=volatility,
    )

    gamma = greeks["gamma"]

    # Scale gamma by spot^2 / 100 for exposure measure
    gamma_exposure = gamma * spots**2 / 100

    return gamma_exposure


def calculate_distance_to_strike(
    spots: torch.Tensor,
    strike: float,
) -> torch.Tensor:
    """Calculate absolute distance to strike, normalized by strike.

    Indicates how far the option is from being ATM.

    Args:
        spots: Price tensor (n_paths, n_steps)
        strike: Strike price

    Returns:
        Normalized distance (n_paths, n_steps)
    """
    distance = torch.abs(spots - strike) / strike
    return distance


def create_tail_risk_features(
    spots: torch.Tensor,
    strike: float,
    time_to_expiry: torch.Tensor,
    volatility: float,
    feature_names: list = None,
) -> dict:
    """Create all tail risk features for deep hedging.

    Args:
        spots: Price tensor (n_paths, n_steps)
        strike: Strike price
        time_to_expiry: Time to expiry tensor
        volatility: Volatility parameter
        feature_names: List of features to compute (None = all)

    Returns:
        Dictionary mapping feature names to tensors
    """
    if feature_names is None:
        feature_names = [
            "volatility_skew",
            "spot_momentum",
            "gamma_exposure",
            "distance_to_strike",
        ]

    features = {}

    if "volatility_skew" in feature_names:
        features["volatility_skew"] = calculate_volatility_skew(spots)

    if "spot_momentum" in feature_names:
        features["spot_momentum"] = calculate_spot_momentum(spots)

    if "gamma_exposure" in feature_names:
        features["gamma_exposure"] = calculate_gamma_exposure(
            spots, strike, time_to_expiry, volatility
        )

    if "distance_to_strike" in feature_names:
        features["distance_to_strike"] = calculate_distance_to_strike(spots, strike)

    return features
