"""Option price comparison utilities for backtesting framework.

This module provides functionality to:
1. Load real option data from Deribit
2. Find options matching backtest configuration
3. Compare model-implied prices with market prices
4. Analyze implied volatility differences
"""

import pandas as pd
import numpy as np
import torch
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import warnings

try:
    from scipy import stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from crypto.data.loader import CryptoDataLoader


class OptionMatcher:
    """Match backtest configuration to real market options.

    This class helps find options in historical data that match the
    backtest configuration parameters (strike, maturity, call/put).

    **Time Conventions:**
    - All timestamps are timezone-aware (UTC)
    - Maturity measured in calendar days (not business days)
    - Days to expiry calculated as: (expiration - timestamp).total_seconds() / 86400
    - Time to expiry in years uses: days / 365.25 (accounts for leap years)
    - No automatic filtering for stale quotes; use reference_date to exclude old data

    **Stale Quote Handling:**
    - Options data returned as-is without age filtering
    - Use `reference_date` parameter to exclude data before a specific time
    - For time series analysis, use `start_date` and `end_date` parameters
    - Check `timestamp` field in results to verify data freshness

    **Missing Fields:**
    - If price field not found, falls back in order: mid → mark → ask → bid
    - Missing implied volatility fields are filled with 0
    - Warnings issued for missing critical fields

    Args:
        data_loader: CryptoDataLoader instance with options data loaded

    Examples:
        >>> loader = CryptoDataLoader("sample_data")
        >>> loader.load_options_data()
        >>> matcher = OptionMatcher(loader)
        >>> matches = matcher.find_matching_options(
        ...     strike=50000,
        ...     maturity_days=7,
        ...     call=True,
        ...     tolerance_pct=0.05
        ... )
    """

    def __init__(self, data_loader: CryptoDataLoader):
        """Initialize option matcher.

        Args:
            data_loader: Loaded CryptoDataLoader instance
        """
        if data_loader.options_data is None:
            raise ValueError("data_loader must have options data loaded")

        self.data_loader = data_loader
        self.options_data = data_loader.options_data

    def find_matching_options(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        reference_date: Optional[datetime] = None,
        strike_tolerance_pct: float = 0.05,
        maturity_tolerance_days: float = 1.0,
    ) -> pd.DataFrame:
        """Find options matching specified criteria.

        Args:
            strike: Target strike price
            maturity_days: Target maturity in days
            call: True for call options, False for puts
            reference_date: Reference date for matching (default: earliest in data)
            strike_tolerance_pct: Allowable strike deviation as percentage (default: 5%)
            maturity_tolerance_days: Allowable maturity deviation in days (default: 1 day)

        Returns:
            DataFrame of matching options sorted by timestamp

        Examples:
            >>> # Find $50k strike, 7-day calls within 5% strike and 1 day maturity
            >>> matches = matcher.find_matching_options(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True
            ... )
            >>> print(f"Found {len(matches)} matching options")
        """
        df = self.options_data.copy()

        # Filter by option type
        option_type_str = "call" if call else "put"
        df = df[df["option_type"].str.lower() == option_type_str]

        if df.empty:
            warnings.warn(f"No {option_type_str} options found in data")
            return pd.DataFrame()

        # Filter by strike (within tolerance)
        strike_min = strike * (1 - strike_tolerance_pct)
        strike_max = strike * (1 + strike_tolerance_pct)
        df = df[(df["strike"] >= strike_min) & (df["strike"] <= strike_max)]

        if df.empty:
            warnings.warn(
                f"No options found with strike between {strike_min:.0f} and {strike_max:.0f}"
            )
            return pd.DataFrame()

        # Calculate time to expiry in days
        if "time_to_expiry" in df.columns:
            df["days_to_expiry"] = df["time_to_expiry"] * 365.25
        else:
            # Calculate from timestamp and expiration
            df["days_to_expiry"] = (
                df["expiration"] - df["timestamp"]
            ).dt.total_seconds() / (24 * 3600)

        # Filter by maturity (within tolerance)
        maturity_min = maturity_days - maturity_tolerance_days
        maturity_max = maturity_days + maturity_tolerance_days
        df = df[
            (df["days_to_expiry"] >= maturity_min)
            & (df["days_to_expiry"] <= maturity_max)
        ]

        if df.empty:
            warnings.warn(
                f"No options found with maturity between {maturity_min:.1f} and {maturity_max:.1f} days"
            )
            return pd.DataFrame()

        # Filter by reference date if provided
        if reference_date is not None:
            # Convert reference_date to timezone-aware if needed
            if reference_date.tzinfo is None:
                reference_date = reference_date.replace(tzinfo=datetime.timezone.utc)
            df = df[df["timestamp"] >= reference_date]

        # Sort by timestamp
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df

    def get_closest_match(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        reference_date: Optional[datetime] = None,
        strike_tolerance_pct: float = 0.10,
        maturity_tolerance_days: float = 2.0,
        strike_weight: float = 0.7,
        maturity_weight: float = 0.3,
    ) -> Optional[pd.Series]:
        """Get single closest matching option.

        Finds the option that best matches the specified criteria using a
        weighted distance metric. Distance is calculated using relative errors
        (unitless) for both strike and maturity.

        Args:
            strike: Target strike price
            maturity_days: Target maturity in days
            call: True for call options, False for puts
            reference_date: Reference date for matching (default: earliest in data)
            strike_tolerance_pct: Allowable strike deviation (default: 10%)
            maturity_tolerance_days: Allowable maturity deviation (default: 2 days)
            strike_weight: Weight for strike matching (default: 0.7)
                Higher values prioritize strike accuracy.
            maturity_weight: Weight for maturity matching (default: 0.3)
                Higher values prioritize maturity accuracy.

        Returns:
            Series with best matching option, or None if no match found

        Raises:
            ValueError: If strike_weight + maturity_weight != 1.0

        Examples:
            >>> # Default: prioritize strike (70%) over maturity (30%)
            >>> best_match = matcher.get_closest_match(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True
            ... )
            >>>
            >>> # Equal weighting
            >>> best_match = matcher.get_closest_match(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True,
            ...     strike_weight=0.5,
            ...     maturity_weight=0.5
            ... )
            >>>
            >>> # Prioritize maturity for time-sensitive analysis
            >>> best_match = matcher.get_closest_match(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True,
            ...     strike_weight=0.3,
            ...     maturity_weight=0.7
            ... )

        Note:
            Distance metric uses relative errors for unit consistency:
            - Strike distance: |K_match - K_target| / K_target (unitless)
            - Maturity distance: |T_match - T_target| / T_target (unitless)
            - Combined: strike_weight * strike_dist + maturity_weight * maturity_dist
        """
        # Validate weights
        if strike_weight < 0 or maturity_weight < 0:
            raise ValueError(
                f"Weights must be non-negative, got strike_weight={strike_weight}, "
                f"maturity_weight={maturity_weight}"
            )
        if not np.isclose(strike_weight + maturity_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got strike_weight={strike_weight}, "
                f"maturity_weight={maturity_weight} (sum={strike_weight + maturity_weight})"
            )

        matches = self.find_matching_options(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            reference_date=reference_date,
            strike_tolerance_pct=strike_tolerance_pct,
            maturity_tolerance_days=maturity_tolerance_days,
        )

        if matches.empty:
            return None

        # Calculate relative distance metrics for unit consistency
        # Both are unitless (relative errors)
        strike_distance = abs(matches["strike"] - strike) / strike

        # Convert maturity to years for better scaling
        maturity_years = maturity_days / 365.25
        matched_maturity_years = matches["days_to_expiry"] / 365.25
        maturity_distance = abs(matched_maturity_years - maturity_years) / max(
            maturity_years, 1e-6
        )

        # Apply configurable weights
        matches["match_distance"] = (
            strike_weight * strike_distance + maturity_weight * maturity_distance
        )

        # Get the best match
        best_idx = matches["match_distance"].idxmin()
        best_match = matches.loc[best_idx]

        return best_match

    def get_time_series(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        strike_tolerance_pct: float = 0.05,
        maturity_tolerance_days: float = 1.0,
    ) -> pd.DataFrame:
        """Get time series of matching options.

        Returns all options matching criteria within the specified date range,
        suitable for tracking option prices over time.

        Args:
            strike: Target strike price
            maturity_days: Target maturity in days
            call: True for call options, False for puts
            start_date: Start of time range (default: earliest in data)
            end_date: End of time range (default: latest in data)
            strike_tolerance_pct: Allowable strike deviation (default: 5%)
            maturity_tolerance_days: Allowable maturity deviation (default: 1 day)

        Returns:
            DataFrame with time series of matching options

        Examples:
            >>> time_series = matcher.get_time_series(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True,
            ...     start_date=datetime(2024, 1, 1),
            ...     end_date=datetime(2024, 1, 10)
            ... )
            >>> # Useful for tracking option price evolution
        """
        matches = self.find_matching_options(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            reference_date=start_date,
            strike_tolerance_pct=strike_tolerance_pct,
            maturity_tolerance_days=maturity_tolerance_days,
        )

        if matches.empty:
            return pd.DataFrame()

        # Filter by end date if provided
        if end_date is not None:
            # Convert end_date to timezone-aware if needed
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=datetime.timezone.utc)
            matches = matches[matches["timestamp"] <= end_date]

        return matches

    def summary(self) -> Dict:
        """Get summary statistics of loaded options data.

        Returns:
            Dictionary with summary statistics

        Examples:
            >>> summary = matcher.summary()
            >>> print(f"Total options: {summary['total_options']}")
            >>> print(f"Strikes available: {summary['strikes']}")
        """
        df = self.options_data

        summary = {
            "total_options": len(df),
            "calls": len(df[df["option_type"] == "call"]),
            "puts": len(df[df["option_type"] == "put"]),
            "strikes": sorted(df["strike"].unique().tolist()),
            "date_range": (df["timestamp"].min(), df["timestamp"].max()),
            "expiries": (
                sorted(df["expiration"].unique().tolist())
                if "expiration" in df.columns
                else []
            ),
        }

        return summary


class PriceComparator:
    """Compare model-implied option prices with market prices.

    This class calculates model-implied prices from deep hedging results
    and compares them with observed market prices.

    The model-implied price represents the expected cost of replicating
    the option payoff using the model's hedging strategy.

    Args:
        backtest_results: BacktestResults object from running backtest
        option_matcher: OptionMatcher instance with loaded options data

    Examples:
        >>> comparator = PriceComparator(results, matcher)
        >>> implied_price = comparator.calculate_model_implied_price()
        >>> comparison = comparator.compare_with_market(
        ...     strike=50000,
        ...     maturity_days=7,
        ...     call=True
        ... )
    """

    def __init__(self, backtest_results, option_matcher: OptionMatcher):
        """Initialize price comparator.

        Args:
            backtest_results: BacktestResults from backtest
            option_matcher: OptionMatcher instance
        """
        self.backtest_results = backtest_results
        self.option_matcher = option_matcher

        # Cache for model prices to avoid redundant computation
        # Key: (strike, maturity_days) - strike/maturity independent from model
        # Note: Model price doesn't depend on (K, T) in current implementation,
        # but this cache enables future extensions where it might (e.g., different
        # hedging strategies for different options)
        self._model_price_cache = None

    def calculate_model_implied_price(
        self,
        method: str = "mean_cost",
    ) -> float:
        """Calculate model-implied option price.

        The model-implied price represents what the model believes
        the fair price should be, based on its hedging performance.

        **IMPORTANT: PFHedge PnL Convention**

        PFHedge's `cum_pl()` returns:
            cum_pl = hedging_gains - transaction_costs - payoff

        For an option seller:
            Total PnL = premium + cum_pl

        Setting E[Total PnL] = 0 for fair pricing:
            0 = premium + E[cum_pl]
            premium = -E[cum_pl]

        This is the formula used by this method.

        Args:
            method: Pricing method to use:
                - 'mean_cost': Uses P = -E[cum_pl] (default and recommended)

        Returns:
            Model-implied option price in dollars

        Examples:
            >>> implied_price = comparator.calculate_model_implied_price()
            >>> print(f"Model fair value: ${implied_price:.2f}")

        Note:
            The returned price represents the premium an option seller would
            charge to achieve zero expected PnL when hedging with the model's
            strategy. A positive price means the option has value; a negative
            price would indicate the model believes hedging costs exceed payoff.

            **Caching**: Result is cached to avoid redundant computation when
            called multiple times (e.g., during volatility smile generation).
        """
        # Use cache if available
        if self._model_price_cache is not None:
            return self._model_price_cache

        results = self.backtest_results

        if method == "mean_cost":
            # PFHedge Convention: cum_pl = hedging_gains - transaction_costs - payoff
            # Get final cumulative PnL from deep hedge
            deep_pnl_final = results.deep_pnl[:, -1].cpu()

            # Calculate fair option premium using PFHedge convention
            # For seller: Total PnL = premium + cum_pl
            # Fair pricing: 0 = premium + E[cum_pl]
            # Therefore: premium = -E[cum_pl]
            mean_cum_pl = deep_pnl_final.mean().item()
            model_implied_price = -mean_cum_pl

            # Cache the result
            self._model_price_cache = model_implied_price

            return model_implied_price

        else:
            raise ValueError(f"Unknown pricing method: {method}")

    def get_market_price(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        price_type: str = "mid",
        reference_date: Optional[datetime] = None,
    ) -> Optional[float]:
        """Get market price for matching option.

        Args:
            strike: Option strike
            maturity_days: Option maturity in days
            call: True for call, False for put
            price_type: Price to use - 'bid', 'ask', 'mid', or 'mark' (default: 'mid')
            reference_date: Reference date for finding option

        Returns:
            Market price in dollars, or None if no match found

        Examples:
            >>> market_price = comparator.get_market_price(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True,
            ...     price_type="mid"
            ... )
        """
        # Find closest matching option
        match = self.option_matcher.get_closest_match(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            reference_date=reference_date,
        )

        if match is None:
            return None

        # Get price based on type
        price_column = f"{price_type}_price"
        if price_column not in match:
            warnings.warn(f"Price column {price_column} not found, trying alternatives")
            # Try alternative columns
            for alt in ["mid_price", "mark_price", "ask_price", "bid_price"]:
                if alt in match:
                    return float(match[alt])
            return None

        return float(match[price_column])

    def compare_with_market(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        price_type: str = "mid",
        reference_date: Optional[datetime] = None,
        include_spread_diagnostics: bool = True,
    ) -> Dict[str, float]:
        """Compare model-implied price with market price.

        Args:
            strike: Option strike
            maturity_days: Option maturity in days
            call: True for call, False for put
            price_type: Market price type to use (default: 'mid')
            reference_date: Reference date for finding option
            include_spread_diagnostics: Include bid-ask spread analysis (default: True)

        Returns:
            Dictionary with comparison metrics:
            - model_price: Model-implied price
            - market_price: Market price (based on price_type)
            - difference: model_price - market_price
            - difference_pct: Percentage difference
            - matched_strike: Actual strike of matched option
            - matched_maturity: Actual maturity of matched option

            If include_spread_diagnostics=True, also includes:
            - bid_price: Bid price (if available)
            - ask_price: Ask price (if available)
            - mid_price: Mid price (bid+ask)/2 (if available)
            - mark_price: Mark price (if available)
            - spread: Bid-ask spread (ask - bid)
            - spread_pct: Spread as percentage of mid price
            - spread_bps: Spread in basis points
            - mark_mid_diff: Mark - mid price difference
            - model_in_spread: True if model price within bid-ask spread

        Examples:
            >>> comparison = comparator.compare_with_market(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True
            ... )
            >>> print(f"Model: ${comparison['model_price']:.2f}")
            >>> print(f"Market: ${comparison['market_price']:.2f}")
            >>> print(f"Spread: {comparison['spread_pct']:.2f}%")
            >>> if comparison['model_in_spread']:
            ...     print("Model price is within bid-ask spread!")
        """
        # Calculate model-implied price
        model_price = self.calculate_model_implied_price()

        # Get market price
        market_price = self.get_market_price(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            price_type=price_type,
            reference_date=reference_date,
        )

        # Get matched option details
        match = self.option_matcher.get_closest_match(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            reference_date=reference_date,
        )

        result = {
            "model_price": model_price,
            "market_price": market_price,
            "matched_strike": float(match["strike"]) if match is not None else None,
            "matched_maturity": (
                float(match["days_to_expiry"]) if match is not None else None
            ),
        }

        # Calculate price difference
        if market_price is not None:
            result["difference"] = model_price - market_price
            result["difference_pct"] = (result["difference"] / market_price) * 100
        else:
            result["difference"] = None
            result["difference_pct"] = None

        # Add spread diagnostics if requested
        if include_spread_diagnostics and match is not None:
            # Extract price fields
            bid = (
                float(match["bid_price"])
                if "bid_price" in match and pd.notna(match["bid_price"])
                else None
            )
            ask = (
                float(match["ask_price"])
                if "ask_price" in match and pd.notna(match["ask_price"])
                else None
            )
            mid = (
                float(match["mid_price"])
                if "mid_price" in match and pd.notna(match["mid_price"])
                else None
            )
            mark = (
                float(match["mark_price"])
                if "mark_price" in match and pd.notna(match["mark_price"])
                else None
            )

            result["bid_price"] = bid
            result["ask_price"] = ask
            result["mid_price"] = mid
            result["mark_price"] = mark

            # Calculate spread metrics
            if bid is not None and ask is not None:
                spread = ask - bid
                result["spread"] = spread

                # Calculate mid if not provided
                if mid is None:
                    mid = (bid + ask) / 2
                    result["mid_price"] = mid

                if mid > 0:
                    result["spread_pct"] = (spread / mid) * 100
                    result["spread_bps"] = (spread / mid) * 10000
                else:
                    result["spread_pct"] = None
                    result["spread_bps"] = None

                # Check if model price is within bid-ask spread
                result["model_in_spread"] = bid <= model_price <= ask
            else:
                result["spread"] = None
                result["spread_pct"] = None
                result["spread_bps"] = None
                result["model_in_spread"] = None

            # Calculate mark-mid difference
            if mark is not None and mid is not None:
                result["mark_mid_diff"] = mark - mid
                if mid > 0:
                    result["mark_mid_diff_pct"] = (result["mark_mid_diff"] / mid) * 100
                else:
                    result["mark_mid_diff_pct"] = None
            else:
                result["mark_mid_diff"] = None
                result["mark_mid_diff_pct"] = None

        return result

    def calculate_model_price_confidence(
        self,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """Calculate confidence interval for model-implied price.

        Uses bootstrap standard error from path dispersion to estimate
        uncertainty in the model-implied price.

        Args:
            confidence_level: Confidence level for interval (default: 0.95 for 95%)

        Returns:
            Dictionary with:
            - mean: Model-implied price (point estimate)
            - std_error: Standard error estimate
            - std_dev: Sample standard deviation
            - lower: Lower confidence bound
            - upper: Upper confidence bound
            - confidence_level: Confidence level used
            - n_paths: Number of paths used
            - distribution: 't' for n < 30, 'normal' for n >= 30
            - critical_value: t-score or z-score used

        Raises:
            ImportError: If scipy is not installed (needed for confidence intervals)

        Examples:
            >>> confidence = comparator.calculate_model_price_confidence()
            >>> print(f"Price: ${confidence['mean']:.2f}")
            >>> print(f"95% CI: [${confidence['lower']:.2f}, ${confidence['upper']:.2f}]")
            >>> print(f"Sample size: {confidence['n_paths']}, Distribution: {confidence['distribution']}")
            >>>
            >>> # With 99% confidence
            >>> confidence = comparator.calculate_model_price_confidence(confidence_level=0.99)

        Note:
            **Statistical methodology**:
            - For n_paths < 30: Uses t-distribution (more conservative for small samples)
            - For n_paths >= 30: Uses normal distribution (asymptotically valid by CLT)
            - Standard error assumes paths are independent samples from the same distribution

            **Why t-distribution for small samples?**
            When n is small, estimating σ from sample introduces additional uncertainty.
            The t-distribution accounts for this by having heavier tails than the normal.
        """
        if not HAS_SCIPY:
            raise ImportError(
                "scipy is required for confidence intervals. "
                "Install with: pip install scipy"
            )

        # Get final PnL for all paths
        deep_pnl_final = self.backtest_results.deep_pnl[:, -1].cpu().numpy()

        # Calculate price for each path (using PFHedge convention: P = -cum_pl)
        prices_per_path = -deep_pnl_final

        # Calculate statistics
        mean_price = prices_per_path.mean()
        std_dev = prices_per_path.std(ddof=1)  # Sample standard deviation
        n_paths = len(prices_per_path)
        std_error = std_dev / np.sqrt(n_paths)

        # Choose distribution based on sample size
        # For small samples (n < 30), use t-distribution
        # For large samples (n >= 30), use normal approximation (CLT)
        if n_paths < 30:
            # Use t-distribution for small samples
            df = n_paths - 1  # degrees of freedom
            critical_value = scipy_stats.t.ppf((1 + confidence_level) / 2, df)
            distribution = "t"
        else:
            # Use normal approximation for large samples
            critical_value = scipy_stats.norm.ppf((1 + confidence_level) / 2)
            distribution = "normal"

        margin = critical_value * std_error

        return {
            "mean": float(mean_price),
            "std_error": float(std_error),
            "std_dev": float(std_dev),
            "lower": float(mean_price - margin),
            "upper": float(mean_price + margin),
            "confidence_level": confidence_level,
            "n_paths": n_paths,
            "distribution": distribution,
            "critical_value": float(critical_value),
        }

    def _black_scholes_price(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        volatility: float,
        risk_free_rate: float,
        call: bool,
    ) -> float:
        """Calculate Black-Scholes option price.

        Helper method for IV calculation.
        """
        if HAS_SCIPY:
            from scipy.stats import norm
        else:
            # Fallback: use numpy approximation of normal CDF
            def norm_cdf(x):
                return 0.5 * (1.0 + np.tanh(x * np.sqrt(2.0 / np.pi)))

            class NormApprox:
                @staticmethod
                def cdf(x):
                    return norm_cdf(x)

            norm = NormApprox()

        d1 = (
            np.log(spot / strike)
            + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
        ) / (volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)

        if call:
            price = spot * norm.cdf(d1) - strike * np.exp(
                -risk_free_rate * time_to_expiry
            ) * norm.cdf(d2)
        else:
            price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(
                -d2
            ) - spot * norm.cdf(-d1)

        return price

    def _bisection_iv(
        self,
        target_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        call: bool,
        tol: float = 1e-4,
        max_iter: int = 100,
    ) -> Tuple[Optional[float], Dict]:
        """Bisection method for IV calculation (fallback when scipy unavailable).

        Returns:
            Tuple of (iv, diagnostics)
        """
        vol_low, vol_high = 0.01, 5.0

        for i in range(max_iter):
            vol_mid = (vol_low + vol_high) / 2.0
            price_mid = self._black_scholes_price(
                spot, strike, time_to_expiry, vol_mid, risk_free_rate, call
            )

            error = price_mid - target_price

            if abs(error) < tol:
                return vol_mid, {
                    "status": "success",
                    "iterations": i + 1,
                    "method": "bisection",
                    "price_error": abs(error),
                }

            # Update bounds
            price_low = self._black_scholes_price(
                spot, strike, time_to_expiry, vol_low, risk_free_rate, call
            )
            if (price_low - target_price) * error < 0:
                vol_high = vol_mid
            else:
                vol_low = vol_mid

        # Failed to converge
        return None, {
            "status": "solver_failed",
            "iterations": max_iter,
            "method": "bisection",
            "price_error": abs(error) if "error" in locals() else None,
        }

    def calculate_model_implied_iv(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        spot_price: Optional[float] = None,
        risk_free_rate: float = 0.0,
        return_diagnostics: bool = False,
    ):
        """Calculate model-implied volatility from model price.

        Uses Black-Scholes formula to invert model price to implied volatility.
        This allows direct comparison with market IV.

        **Important Assumptions:**
        - **Risk-free rate r = 0.0 by default** (crypto assumption: no riskless benchmark)
          You can override this, but crypto markets typically don't have a risk-free rate
        - No dividends/continuous yield
        - European exercise only
        - Black-Scholes model applies (lognormal returns, constant volatility)

        **IV Inversion Method:**
        - **Primary**: Brent's method (scipy.optimize.brentq) - robust bracketed root-finding
        - **Fallback**: Simple bisection if scipy unavailable (still works, slightly slower)
        - **Search range**: [0.01, 5.0] (1% to 500% annualized, appropriate for crypto)
        - **Convergence tolerance**: 1e-6 for Brent, 1e-4 for bisection

        **Numerical Stability:**
        - Checks arbitrage bounds before solving
        - **Guards deep ITM/OTM cases** where vega → 0 (ill-conditioned inversion)
          For deep ITM calls (S >> K), BS price ≈ S - K*exp(-rT), insensitive to σ
          For deep OTM options, price ≈ 0, also insensitive to σ
        - Returns None with warning if solver fails or price out of bounds
        - **Negative model prices** return None (economically invalid - see Note below)

        Args:
            strike: Option strike price
            maturity_days: Option maturity in days
            call: True for call, False for put
            spot_price: Current spot price (default: use final spot from backtest)
            risk_free_rate: Risk-free interest rate (default: 0.0)
            return_diagnostics: Return dict with solver diagnostics (default: False)

        Returns:
            If return_diagnostics=False:
                Model-implied volatility (annualized), or None if cannot be calculated
            If return_diagnostics=True:
                Tuple of (iv, diagnostics_dict) where diagnostics contains:
                - status: 'success', 'out_of_bounds', 'solver_failed', 'negative_price', 'ill_conditioned'
                - iterations: Number of solver iterations (if available)
                - method: 'brentq' or 'bisection'
                - price_error: Final price residual

        Examples:
            >>> # Basic usage
            >>> model_iv = comparator.calculate_model_implied_iv(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True
            ... )
            >>> if model_iv is not None:
            ...     print(f"Model IV: {model_iv:.2%}")
            >>>
            >>> # With diagnostics
            >>> iv, diag = comparator.calculate_model_implied_iv(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True,
            ...     return_diagnostics=True
            ... )
            >>> if iv is not None:
            ...     print(f"IV: {iv:.2%}, Method: {diag['method']}, Iterations: {diag['iterations']}")

        Note:
            **Returns None if:**
            - **Negative model price**: Can occur if model's hedging strategy has very
              high expected costs that exceed the expected payoff. Economically invalid
              for option pricing (no arbitrage requires price ≥ 0).
            - **Price violates arbitrage bounds**: price < intrinsic or price > upper_bound
            - **Solver fails to converge**: Root-finding unsuccessful
            - **Deep ITM/OTM with vega ≈ 0**: Ill-conditioned inversion (price insensitive to σ)

            **Why negative model prices occur:**
            If E[cum_pl] > 0, then premium = -E[cum_pl] < 0. This means the model
            expects to lose money on the hedging strategy even before paying the premium.
            This is not a valid option price and cannot be inverted to an IV.
        """
        # Cache key for memoization
        cache_key = (strike, maturity_days, call, risk_free_rate)
        if not hasattr(self, "_iv_cache"):
            self._iv_cache = {}

        if cache_key in self._iv_cache and not return_diagnostics:
            return self._iv_cache[cache_key]

        # Get model price (cached separately)
        model_price = self.calculate_model_implied_price()

        # Initialize diagnostics
        diagnostics = {
            "status": None,
            "iterations": None,
            "method": None,
            "price_error": None,
        }

        # Check for negative price (economically invalid)
        if model_price < -1e-6:
            diagnostics["status"] = "negative_price"
            warnings.warn(
                f"Model price ${model_price:.2f} is negative. Cannot calculate IV. "
                f"This indicates the model expects hedging costs to exceed payoff."
            )
            result = (None, diagnostics) if return_diagnostics else None
            self._iv_cache[cache_key] = None
            return result

        # Get spot price
        if spot_price is None:
            spot_price = self.backtest_results.spots[:, -1].mean().item()

        # Convert maturity to years
        time_to_expiry = maturity_days / 365.25

        # Check arbitrage bounds
        intrinsic = max(spot_price - strike, 0) if call else max(strike - spot_price, 0)
        upper_bound = (
            spot_price if call else strike * np.exp(-risk_free_rate * time_to_expiry)
        )

        if model_price < intrinsic - 1e-6:
            diagnostics["status"] = "out_of_bounds"
            warnings.warn(
                f"Model price ${model_price:.2f} below intrinsic value ${intrinsic:.2f}. "
                f"Violates arbitrage bounds. Cannot calculate IV."
            )
            result = (None, diagnostics) if return_diagnostics else None
            self._iv_cache[cache_key] = None
            return result

        if model_price > upper_bound + 1e-6:
            diagnostics["status"] = "out_of_bounds"
            warnings.warn(
                f"Model price ${model_price:.2f} above upper bound ${upper_bound:.2f}. "
                f"Violates arbitrage bounds. Cannot calculate IV."
            )
            result = (None, diagnostics) if return_diagnostics else None
            self._iv_cache[cache_key] = None
            return result

        # Check for deep ITM/OTM (vega near zero, ill-conditioned)
        moneyness = spot_price / strike if call else strike / spot_price
        if moneyness > 2.0 or moneyness < 0.5:
            # Deep ITM if moneyness > 2, deep OTM if moneyness < 0.5
            # In these regions, vega is very small and inversion is ill-conditioned
            diagnostics["status"] = "ill_conditioned"
            warnings.warn(
                f"Option is deep {'ITM' if moneyness > 2.0 else 'OTM'} "
                f"(moneyness={moneyness:.2f}). IV inversion may be ill-conditioned. "
                f"Proceeding with caution."
            )
            # Don't return None yet, try to solve anyway

        # Try IV calculation
        if HAS_SCIPY:
            # Use Brent's method (primary)
            try:
                from scipy.optimize import brentq, OptimizeResult

                def price_diff(sigma):
                    return (
                        self._black_scholes_price(
                            spot_price,
                            strike,
                            time_to_expiry,
                            sigma,
                            risk_free_rate,
                            call,
                        )
                        - model_price
                    )

                # Use Brent's method with full output
                result_obj = brentq(price_diff, 0.01, 5.0, xtol=1e-6, full_output=True)

                if isinstance(result_obj, tuple):
                    implied_vol, info = result_obj
                    diagnostics["iterations"] = (
                        info.iterations if hasattr(info, "iterations") else None
                    )
                else:
                    implied_vol = result_obj
                    diagnostics["iterations"] = None

                diagnostics["method"] = "brentq"
                diagnostics["status"] = "success"

                # Calculate final error
                final_price = self._black_scholes_price(
                    spot_price,
                    strike,
                    time_to_expiry,
                    implied_vol,
                    risk_free_rate,
                    call,
                )
                diagnostics["price_error"] = abs(final_price - model_price)

                iv_result = float(implied_vol)
                self._iv_cache[cache_key] = iv_result

                return (iv_result, diagnostics) if return_diagnostics else iv_result

            except (ValueError, RuntimeError) as e:
                warnings.warn(f"Brent's method failed: {e}. Trying bisection fallback.")
                # Fall through to bisection

        # Fallback: Use bisection method
        iv_result, bisect_diag = self._bisection_iv(
            model_price, spot_price, strike, time_to_expiry, risk_free_rate, call
        )

        diagnostics.update(bisect_diag)

        if iv_result is None:
            warnings.warn(
                f"Failed to calculate implied volatility. "
                f"Method: {diagnostics['method']}, Status: {diagnostics['status']}"
            )
        else:
            self._iv_cache[cache_key] = iv_result

        return (iv_result, diagnostics) if return_diagnostics else iv_result

    def get_market_iv(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        iv_type: str = "mark",
        reference_date: Optional[datetime] = None,
        sanity_check: bool = True,
    ) -> Optional[float]:
        """Get market implied volatility for matching option.

        Extracts market IV from Deribit options data with sanity checks.

        Args:
            strike: Option strike
            maturity_days: Option maturity in days
            call: True for call, False for put
            iv_type: IV to use - 'bid', 'ask', or 'mark' (default: 'mark')
            reference_date: Reference date for finding option
            sanity_check: Perform sanity checks on IV values (default: True)

        Returns:
            Market implied volatility (annualized), or None if no match found

        Examples:
            >>> market_iv = comparator.get_market_iv(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True,
            ...     iv_type="mark"
            ... )
            >>> if market_iv is not None:
            ...     print(f"Market IV: {market_iv:.2%}")

        Note:
            **Sanity checks performed** (if sanity_check=True):
            - Filters out zero IVs (data glitches)
            - Ensures IV in [0.01, 5.0] range (1% to 500%)
            - Warns if bid/ask/mark IVs are wildly inconsistent (spread > 50% of mark)

            **Fallback order**: mark_iv → bid_iv → ask_iv
        """
        # Find closest matching option
        match = self.option_matcher.get_closest_match(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            reference_date=reference_date,
        )

        if match is None:
            return None

        # Get IV based on type with fallback
        iv_column = f"{iv_type}_iv"
        iv = None

        if iv_column in match and pd.notna(match[iv_column]):
            iv = float(match[iv_column])
        else:
            # Try fallback order
            for alt in ["mark_iv", "bid_iv", "ask_iv"]:
                if alt in match and pd.notna(match[alt]):
                    iv = float(match[alt])
                    break

        if iv is None:
            return None

        # Sanity checks
        if sanity_check:
            # Filter out zeros (data glitches)
            if abs(iv) < 1e-6:
                warnings.warn(
                    f"Market IV is zero for strike={strike}, maturity={maturity_days}. Likely data glitch."
                )
                return None

            # Ensure IV in reasonable range [0.01, 5.0]
            # NOTE: We drop out-of-bounds IVs rather than clamping them.
            # Clamping could hide data quality issues and introduce bias.
            # Safer to return None and let caller decide how to handle missing data.
            if iv < 0.01 or iv > 5.0:
                warnings.warn(
                    f"Market IV {iv:.2%} outside reasonable range [1%, 500%]. "
                    f"Strike={strike}, maturity={maturity_days}."
                )
                return None

            # Check for wildly inconsistent bid/ask/mark IVs
            bid_iv = (
                float(match["bid_iv"])
                if "bid_iv" in match and pd.notna(match["bid_iv"])
                else None
            )
            ask_iv = (
                float(match["ask_iv"])
                if "ask_iv" in match and pd.notna(match["ask_iv"])
                else None
            )
            mark_iv = (
                float(match["mark_iv"])
                if "mark_iv" in match and pd.notna(match["mark_iv"])
                else None
            )

            if bid_iv is not None and ask_iv is not None and mark_iv is not None:
                iv_spread = ask_iv - bid_iv
                iv_spread_pct = (iv_spread / mark_iv) * 100 if mark_iv > 0 else 0

                if iv_spread_pct > 50:
                    warnings.warn(
                        f"IV bid-ask spread is {iv_spread_pct:.1f}% of mark IV. "
                        f"Bid={bid_iv:.2%}, Ask={ask_iv:.2%}, Mark={mark_iv:.2%}. "
                        f"Market may be illiquid or data quality issue."
                    )

        return iv

    def compare_implied_volatility(
        self,
        strike: float,
        maturity_days: int,
        call: bool,
        spot_price: Optional[float] = None,
        risk_free_rate: float = 0.0,
        iv_type: str = "mark",
        reference_date: Optional[datetime] = None,
    ) -> Dict:
        """Compare model-implied IV with market IV.

        Args:
            strike: Option strike
            maturity_days: Option maturity in days
            call: True for call, False for put
            spot_price: Current spot price (default: use final spot from backtest)
            risk_free_rate: Risk-free interest rate (default: 0.0)
            iv_type: Market IV type to use (default: 'mark')
            reference_date: Reference date for finding option

        Returns:
            Dictionary with IV comparison metrics:
            - model_iv: Model-implied volatility
            - market_iv: Market implied volatility
            - iv_difference: model_iv - market_iv (absolute)
            - iv_difference_pct: Percentage difference relative to market_iv
            - model_price: Model-implied price
            - market_price: Market price (mid if available)
            - matched_strike: Actual strike of matched option
            - matched_maturity: Actual maturity of matched option (days)
            - matched_instrument: Instrument name (e.g., "BTC-25DEC23-50000-C") for debugging
            - moneyness: Strike/Spot ratio (defined as K/S, >1 is OTM call, <1 is ITM call)

        Examples:
            >>> iv_comp = comparator.compare_implied_volatility(
            ...     strike=50000,
            ...     maturity_days=7,
            ...     call=True
            ... )
            >>> print(f"Model IV: {iv_comp['model_iv']:.2%}")
            >>> print(f"Market IV: {iv_comp['market_iv']:.2%}")
            >>> print(f"Moneyness (K/S): {iv_comp['moneyness']:.3f}")

        Note:
            **Moneyness definition**: K/S (strike divided by spot)
            - K/S > 1: Out-of-the-money (OTM) call / In-the-money (ITM) put
            - K/S = 1: At-the-money (ATM)
            - K/S < 1: In-the-money (ITM) call / Out-of-the-money (OTM) put
        """
        # Get spot price
        if spot_price is None:
            spot_price = self.backtest_results.spots[:, -1].mean().item()

        # Calculate model IV
        model_iv = self.calculate_model_implied_iv(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            spot_price=spot_price,
            risk_free_rate=risk_free_rate,
        )

        # Get market IV
        market_iv = self.get_market_iv(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            iv_type=iv_type,
            reference_date=reference_date,
        )

        # Get model and market prices
        model_price = self.calculate_model_implied_price()
        market_price = self.get_market_price(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            price_type="mid",
            reference_date=reference_date,
        )

        # Get matched option details
        match = self.option_matcher.get_closest_match(
            strike=strike,
            maturity_days=maturity_days,
            call=call,
            reference_date=reference_date,
        )

        # Calculate moneyness as K/S (standard definition)
        moneyness = strike / spot_price

        result = {
            "model_iv": model_iv,
            "market_iv": market_iv,
            "model_price": model_price,
            "market_price": market_price,
            "moneyness": moneyness,
            "matched_strike": float(match["strike"]) if match is not None else None,
            "matched_maturity": (
                float(match["days_to_expiry"]) if match is not None else None
            ),
            "matched_instrument": (
                str(match["instrument"])
                if match is not None and "instrument" in match
                else None
            ),
        }

        # Calculate IV difference
        if model_iv is not None and market_iv is not None:
            result["iv_difference"] = model_iv - market_iv
            result["iv_difference_pct"] = (result["iv_difference"] / market_iv) * 100
        else:
            result["iv_difference"] = None
            result["iv_difference_pct"] = None

        return result

    def get_volatility_smile(
        self,
        strikes: List[float],
        maturity_days: int,
        call: bool,
        spot_price: Optional[float] = None,
        risk_free_rate: float = 0.0,
        iv_type: str = "mark",
        reference_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """Get volatility smile (IV across strikes) for comparison.

        Calculates both model and market IVs across multiple strikes to
        enable volatility smile/skew analysis. Ensures all points share
        the same maturity window.

        Args:
            strikes: List of strikes to analyze
            maturity_days: Option maturity in days (same for all strikes)
            call: True for call, False for put
            spot_price: Current spot price (default: use final spot from backtest)
            risk_free_rate: Risk-free interest rate (default: 0.0)
            iv_type: Market IV type to use (default: 'mark')
            reference_date: Reference date for finding options

        Returns:
            DataFrame with columns:
            - strike: Strike price
            - moneyness: K/S ratio (strike/spot, >1 is OTM call, <1 is ITM call)
            - model_iv: Model-implied volatility
            - market_iv: Market implied volatility
            - iv_difference: model_iv - market_iv
            - model_price: Model-implied price
            - market_price: Market price (mid)
            - matched_strike: Actual strike from data
            - matched_maturity: Actual maturity from data (days)
            - matched_instrument: Instrument name for debugging
            - skipped: True if this strike was skipped due to missing data

        Examples:
            >>> strikes = [45000, 47500, 50000, 52500, 55000]
            >>> smile = comparator.get_volatility_smile(
            ...     strikes=strikes,
            ...     maturity_days=7,
            ...     call=True
            ... )
            >>> # Filter out skipped points
            >>> valid_smile = smile[~smile['skipped']]
            >>> print(valid_smile[['strike', 'moneyness', 'model_iv', 'market_iv']])

        Note:
            **Moneyness**: K/S (strike/spot)
            - All strikes use the same maturity window (±2 days by default)
            - Skipped strikes are included in output with skipped=True
            - Returns model price and market price for cross-checks
        """
        # Get spot price
        if spot_price is None:
            spot_price = self.backtest_results.spots[:, -1].mean().item()

        results = []
        skipped_count = 0

        for strike in strikes:
            comparison = self.compare_implied_volatility(
                strike=strike,
                maturity_days=maturity_days,
                call=call,
                spot_price=spot_price,
                risk_free_rate=risk_free_rate,
                iv_type=iv_type,
                reference_date=reference_date,
            )

            # Check if this strike was skipped (no market data)
            skipped = comparison["market_iv"] is None
            if skipped:
                skipped_count += 1

            results.append(
                {
                    "strike": strike,
                    "moneyness": comparison["moneyness"],
                    "model_iv": comparison["model_iv"],
                    "market_iv": comparison["market_iv"],
                    "iv_difference": comparison["iv_difference"],
                    "model_price": comparison["model_price"],
                    "market_price": comparison["market_price"],
                    "matched_strike": comparison["matched_strike"],
                    "matched_maturity": comparison["matched_maturity"],
                    "matched_instrument": comparison["matched_instrument"],
                    "skipped": skipped,
                }
            )

        if skipped_count > 0:
            warnings.warn(
                f"Skipped {skipped_count}/{len(strikes)} strikes due to missing market data. "
                f"Filter with smile[~smile['skipped']] to get valid points only."
            )

        return pd.DataFrame(results)

    def summary(self) -> Dict:
        """Get summary of price comparison.

        Returns:
            Dictionary with comparison summary

        Examples:
            >>> summary = comparator.summary()
            >>> print(f"Model implied: ${summary['model_implied_price']:.2f}")
        """
        model_price = self.calculate_model_implied_price()

        summary = {
            "model_implied_price": model_price,
            "calculation_method": "mean_cost",
            "n_paths": self.backtest_results.n_paths,
            "n_steps": self.backtest_results.n_steps,
        }

        return summary
