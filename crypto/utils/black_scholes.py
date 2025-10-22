"""
Black-Scholes utilities for option pricing and implied volatility calculation.
"""

import numpy as np
from scipy import optimize
from scipy.stats import norm
from typing import Optional


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    option_type: str = "call",
) -> float:
    """
    Calculate Black-Scholes option price.

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate (default: 0 for crypto)
        option_type: "call" or "put"

    Returns:
        Option price
    """
    if time_to_expiry <= 0:
        # Option expired
        if option_type == "call":
            return max(spot - strike, 0)
        else:
            return max(strike - spot, 0)

    # Calculate d1 and d2
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)

    # Calculate option price
    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(
            -risk_free_rate * time_to_expiry
        ) * norm.cdf(d2)
    else:  # put
        price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(
            -d2
        ) - spot * norm.cdf(-d1)

    return price


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.0,
    option_type: str = "call",
) -> float:
    """
    Calculate Black-Scholes delta.

    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Annualized volatility
        risk_free_rate: Risk-free rate
        option_type: "call" or "put"

    Returns:
        Delta (hedge ratio)
    """
    if time_to_expiry <= 0:
        # Option expired
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0

    # Calculate d1
    d1 = (
        np.log(spot / strike)
        + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry
    ) / (volatility * np.sqrt(time_to_expiry))

    # Calculate delta
    if option_type == "call":
        delta = norm.cdf(d1)
    else:  # put
        delta = norm.cdf(d1) - 1

    return delta


def implied_volatility(
    premium: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float = 0.0,
    option_type: str = "call",
    initial_guess: float = 0.5,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> Optional[float]:
    """
    Calculate implied volatility from option premium using Newton-Raphson method.

    Args:
        premium: Observed option premium
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free rate (default: 0 for crypto)
        option_type: "call" or "put"
        initial_guess: Initial volatility guess
        max_iterations: Maximum iterations for convergence
        tolerance: Convergence tolerance

    Returns:
        Implied volatility, or None if cannot converge
    """
    # Handle edge cases
    if time_to_expiry <= 0:
        return None

    if premium <= 0:
        return None

    # Check if premium is within arbitrage bounds
    if option_type == "call":
        intrinsic = max(spot - strike, 0)
        if premium < intrinsic:
            return None
        if premium > spot:
            return None
    else:  # put
        intrinsic = max(strike - spot, 0)
        if premium < intrinsic:
            return None
        if premium > strike:
            return None

    def objective(vol):
        """Objective function: model_price - market_price."""
        try:
            model_price = black_scholes_price(
                spot, strike, time_to_expiry, vol, risk_free_rate, option_type
            )
            return model_price - premium
        except:
            return float("inf")

    def vega_func(vol):
        """Vega: derivative of price with respect to volatility."""
        try:
            # Calculate vega
            d1 = (
                np.log(spot / strike)
                + (risk_free_rate + 0.5 * vol ** 2) * time_to_expiry
            ) / (vol * np.sqrt(time_to_expiry))
            vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry)
            return vega
        except:
            return 1e-10  # Small number to avoid division by zero

    # Newton-Raphson iteration
    vol = initial_guess
    for i in range(max_iterations):
        price_diff = objective(vol)

        # Check convergence
        if abs(price_diff) < tolerance:
            return vol

        # Calculate vega for Newton step
        vega = vega_func(vol)

        # Avoid division by zero
        if abs(vega) < 1e-10:
            break

        # Newton step with dampening to avoid overshooting
        vol_new = vol - price_diff / vega

        # Ensure volatility stays positive and reasonable
        vol_new = max(0.01, min(vol_new, 5.0))  # Between 1% and 500%

        # Check if we're making progress
        if abs(vol_new - vol) < tolerance:
            return vol_new

        vol = vol_new

    # If Newton-Raphson fails, try scipy optimizer as fallback
    try:
        result = optimize.brentq(
            objective, 0.01, 5.0, xtol=tolerance, maxiter=max_iterations
        )
        return result
    except:
        pass

    return None  # Failed to converge


def implied_volatility_from_btc_premium(
    premium_btc: float,
    spot: float,
    strike: float,
    time_to_expiry: float,
    option_type: str = "call",
) -> Optional[float]:
    """
    Calculate implied volatility from BTC-denominated premium.

    For Deribit, premiums are quoted in BTC, so we need to convert.

    Args:
        premium_btc: Premium in BTC
        spot: Spot price in USD
        strike: Strike price in USD
        time_to_expiry: Time to expiry in years
        option_type: "call" or "put"

    Returns:
        Implied volatility
    """
    # Convert premium from BTC to USD terms
    # For BTC options on Deribit, 1 contract = 1 BTC
    premium_usd = premium_btc * spot

    # Now calculate IV using USD values
    return implied_volatility(
        premium=premium_usd,
        spot=spot,
        strike=strike,
        time_to_expiry=time_to_expiry,
        risk_free_rate=0.0,  # Zero for crypto
        option_type=option_type,
    )


def calculate_breakeven(
    strike: float, premium: float, option_type: str = "call"
) -> float:
    """
    Calculate breakeven price for an option position.

    Args:
        strike: Strike price
        premium: Premium paid/received
        option_type: "call" or "put"

    Returns:
        Breakeven price
    """
    if option_type == "call":
        # For calls, breakeven is strike + premium
        return strike + premium
    else:
        # For puts, breakeven is strike - premium
        return strike - premium


def calculate_option_pnl(
    spot_at_expiry: float,
    strike: float,
    premium: float,
    option_type: str = "call",
    position: str = "long",
) -> float:
    """
    Calculate P&L for an option position at expiry.

    Args:
        spot_at_expiry: Spot price at expiry
        strike: Strike price
        premium: Premium paid (if long) or received (if short)
        option_type: "call" or "put"
        position: "long" or "short"

    Returns:
        Total P&L including premium
    """
    # Calculate payoff
    if option_type == "call":
        payoff = max(spot_at_expiry - strike, 0)
    else:
        payoff = max(strike - spot_at_expiry, 0)

    # Calculate P&L based on position
    if position == "long":
        # Long position: payoff - premium paid
        pnl = payoff - premium
    else:
        # Short position: premium received - payoff
        pnl = premium - payoff

    return pnl
