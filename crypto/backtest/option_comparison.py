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
            "expiries": sorted(df["expiration"].unique().tolist())
            if "expiration" in df.columns
            else [],
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
        """
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
            "matched_maturity": float(match["days_to_expiry"])
            if match is not None
            else None,
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
            - lower: Lower confidence bound
            - upper: Upper confidence bound
            - confidence_level: Confidence level used
            - n_paths: Number of paths used

        Raises:
            ImportError: If scipy is not installed (needed for confidence intervals)

        Examples:
            >>> confidence = comparator.calculate_model_price_confidence()
            >>> print(f"Price: ${confidence['mean']:.2f}")
            >>> print(f"95% CI: [${confidence['lower']:.2f}, ${confidence['upper']:.2f}]")
            >>>
            >>> # With 99% confidence
            >>> confidence = comparator.calculate_model_price_confidence(confidence_level=0.99)

        Note:
            Standard error assumes paths are independent samples from the same
            distribution. Confidence intervals use normal approximation, which
            is valid for large sample sizes (typically n_paths > 30).
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

        # Calculate confidence interval using normal approximation
        # For large n, sample mean is approximately normal
        z_score = scipy_stats.norm.ppf((1 + confidence_level) / 2)
        margin = z_score * std_error

        return {
            "mean": float(mean_price),
            "std_error": float(std_error),
            "std_dev": float(std_dev),
            "lower": float(mean_price - margin),
            "upper": float(mean_price + margin),
            "confidence_level": confidence_level,
            "n_paths": n_paths,
            "z_score": float(z_score),
        }

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
