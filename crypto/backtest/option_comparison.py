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

    def __init__(self, data_loader: CryptoDataLoader):
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

    def __init__(self, backtest_results, option_matcher: OptionMatcher):
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
                from scipy.optimize import brentq

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
        model_price = self.calculate_model_implied_price()

        summary = {
            "model_implied_price": model_price,
            "calculation_method": "mean_cost",
            "n_paths": self.backtest_results.n_paths,
            "n_steps": self.backtest_results.n_steps,
        }

        return summary
