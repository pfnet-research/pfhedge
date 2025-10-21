"""Domain constants for crypto hedging.

This module centralizes all domain-specific default values to avoid drift
and ensure consistency across the codebase.

Constants:
    TRANSACTION_COSTS: Default transaction cost rates for different markets
    LEVERAGE_LIMITS: Maximum leverage allowed for different instruments
    FUNDING_INTERVALS: Funding payment intervals in years
    TIME_STEPS: Standard time steps for rebalancing
    RISK_MEASURES: Valid risk measures for optimization
"""

from typing import Dict

# ==============================================================================
# Transaction Costs
# ==============================================================================

# Default transaction cost rates (as decimal, e.g., 0.0005 = 0.05% = 5 basis points)
TRANSACTION_COST_PERPETUAL = 0.0006  # Perpetual futures maker/taker fee
TRANSACTION_COST_SPOT = 0.0010  # Spot market transaction cost
TRANSACTION_COST_DEFAULT = 0.0005  # Default for synthetic/simulation

# Dictionary for easy lookup
TRANSACTION_COSTS: Dict[str, float] = {
    "perpetual": TRANSACTION_COST_PERPETUAL,
    "spot": TRANSACTION_COST_SPOT,
    "default": TRANSACTION_COST_DEFAULT,
    "option": 0.0,  # Options themselves have no transaction cost (only underlier)
}


# ==============================================================================
# Leverage Limits
# ==============================================================================

# Maximum leverage allowed for different instruments
LEVERAGE_PERPETUAL = 20.0  # Typical maximum for perpetual futures
LEVERAGE_SPOT = 1.0  # No leverage for spot

LEVERAGE_LIMITS: Dict[str, float] = {
    "perpetual": LEVERAGE_PERPETUAL,
    "spot": LEVERAGE_SPOT,
}


# ==============================================================================
# Funding Rates & Intervals
# ==============================================================================

# Funding payment intervals (in years, for compatibility with PFHedge time units)
FUNDING_INTERVAL_8H = (8 / 24) / 365  # 8-hour funding (3x per day)
FUNDING_INTERVAL_1H = (1 / 24) / 365  # Hourly funding (for some exchanges)

FUNDING_INTERVALS: Dict[str, float] = {
    "8h": FUNDING_INTERVAL_8H,
    "1h": FUNDING_INTERVAL_1H,
    "default": FUNDING_INTERVAL_8H,  # Most exchanges use 8-hour funding
}

# Typical funding rate ranges (annualized)
FUNDING_RATE_TYPICAL = 0.01  # 1% annualized (~0.003% per 8h)
FUNDING_RATE_MAX = 0.1  # 10% annualized (~0.03% per 8h)


# ==============================================================================
# Time Steps (dt) - in hours
# ==============================================================================

# Standard rebalancing frequencies in hours
DT_HOURS_5MIN = 5 / 60  # 5-minute bars
DT_HOURS_1H = 1.0  # Hourly
DT_HOURS_8H = 8.0  # 8-hour (aligns with funding)
DT_HOURS_1D = 24.0  # Daily

# Default for training and backtesting
DT_HOURS_DEFAULT = DT_HOURS_8H

TIME_STEPS: Dict[str, float] = {
    "5min": DT_HOURS_5MIN,
    "1h": DT_HOURS_1H,
    "8h": DT_HOURS_8H,
    "1d": DT_HOURS_1D,
    "default": DT_HOURS_DEFAULT,
}


# ==============================================================================
# Risk Measures
# ==============================================================================

# Valid risk measures for deep hedging optimization
RISK_MEASURES = [
    "expected_shortfall",  # CVaR / Expected Shortfall
    "entropic",  # Entropic risk measure
    "variance",  # Variance (traditional mean-variance)
]

# Aliases for risk measures (mapped to canonical names)
RISK_MEASURE_ALIASES: Dict[str, str] = {
    "cvar": "expected_shortfall",
    "es": "expected_shortfall",
    "var": "variance",
}

# Default risk measure and parameter
RISK_MEASURE_DEFAULT = "expected_shortfall"
RISK_PARAM_DEFAULT = 0.9  # 90% CVaR (focus on worst 10%)


# ==============================================================================
# Model Architecture Defaults
# ==============================================================================

N_LAYERS_DEFAULT = 4  # Number of hidden layers
N_UNITS_DEFAULT = 128  # Number of units per layer

MODEL_ARCHITECTURE: Dict[str, int] = {
    "n_layers": N_LAYERS_DEFAULT,
    "n_units": N_UNITS_DEFAULT,
}


# ==============================================================================
# Training Defaults
# ==============================================================================

N_PATHS_TRAIN = 10000  # Training paths
N_PATHS_TEST = 200  # Test paths
N_EPOCHS_DEFAULT = 80  # Training epochs

SEED_TRAIN_DEFAULT = 42  # Training seed
SEED_TEST_DEFAULT = 888  # Test seed (different from training)

TRAINING_DEFAULTS: Dict[str, int] = {
    "n_paths": N_PATHS_TRAIN,
    "n_epochs": N_EPOCHS_DEFAULT,
    "test_n_paths": N_PATHS_TEST,
    "train_seed": SEED_TRAIN_DEFAULT,
    "test_seed": SEED_TEST_DEFAULT,
}


# ==============================================================================
# Backtest Defaults
# ==============================================================================

N_BOOTSTRAP_PATHS_DEFAULT = 100  # Number of bootstrap paths

BACKTEST_DEFAULTS: Dict[str, int] = {
    "n_bootstrap_paths": N_BOOTSTRAP_PATHS_DEFAULT,
}


# ==============================================================================
# Market Simulation Defaults
# ==============================================================================

# Default market parameters for Geometric Brownian Motion simulation
VOLATILITY_DEFAULT = 0.8  # 80% annualized (typical for crypto)
DRIFT_DEFAULT = 0.0  # Risk-neutral drift

MARKET_DEFAULTS: Dict[str, float] = {
    "volatility": VOLATILITY_DEFAULT,
    "drift": DRIFT_DEFAULT,
}


# ==============================================================================
# Helper Functions
# ==============================================================================


def dt_hours_to_years(dt_hours: float) -> float:
    """Convert time step from hours to years (for PFHedge compatibility).

    Args:
        dt_hours: Time step in hours

    Returns:
        Time step in years

    Examples:
        >>> dt_hours_to_years(8.0)
        0.0009132420091324201  # 8 hours / (24 * 365)
    """
    return dt_hours / 24 / 365


def dt_years_to_hours(dt_years: float) -> float:
    """Convert time step from years to hours.

    Args:
        dt_years: Time step in years

    Returns:
        Time step in hours

    Examples:
        >>> dt_years_to_hours(0.0009132420091324201)
        8.0
    """
    return dt_years * 24 * 365


def normalize_risk_measure(risk_measure: str) -> str:
    """Normalize risk measure name using aliases.

    Args:
        risk_measure: Risk measure name (may be alias)

    Returns:
        Canonical risk measure name

    Examples:
        >>> normalize_risk_measure("cvar")
        'expected_shortfall'
        >>> normalize_risk_measure("expected_shortfall")
        'expected_shortfall'
    """
    return RISK_MEASURE_ALIASES.get(risk_measure.lower(), risk_measure)


def validate_risk_measure(risk_measure: str) -> bool:
    """Check if risk measure is valid.

    Args:
        risk_measure: Risk measure name to validate

    Returns:
        True if valid, False otherwise

    Examples:
        >>> validate_risk_measure("expected_shortfall")
        True
        >>> validate_risk_measure("cvar")  # alias
        True
        >>> validate_risk_measure("invalid")
        False
    """
    normalized = normalize_risk_measure(risk_measure)
    return normalized in RISK_MEASURES
