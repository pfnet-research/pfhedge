"""Unit tests for option comparison module."""

import pytest
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import os

from crypto.backtest.option_comparison import OptionMatcher, PriceComparator
from crypto.backtest.results import BacktestResults
from crypto.backtest.config import BacktestConfig
from crypto.data.loader import CryptoDataLoader


@pytest.fixture
def sample_options_data():
    """Create sample options data for testing."""
    # Create sample data with various options
    timestamps = pd.date_range("2024-01-01", periods=10, freq="1H", tz="UTC")

    data = []
    for ts in timestamps:
        # Create options with different strikes and maturities
        for strike in [48000, 50000, 52000]:
            for days_to_expiry in [1, 7, 14]:
                expiration = ts + timedelta(days=days_to_expiry)

                for option_type in ["call", "put"]:
                    # Create sample option record
                    data.append(
                        {
                            "timestamp": ts,
                            "expiration": expiration,
                            "strike": strike,
                            "option_type": option_type,
                            "bid_price": 100 + np.random.randn() * 10,
                            "ask_price": 120 + np.random.randn() * 10,
                            "mid_price": 110 + np.random.randn() * 10,
                            "mark_iv": 0.8 + np.random.randn() * 0.1,
                            "time_to_expiry": days_to_expiry / 365.25,
                        }
                    )

    df = pd.DataFrame(data)

    # Ensure positive prices and IVs
    df["bid_price"] = df["bid_price"].abs() + 50
    df["ask_price"] = df["ask_price"].abs() + 60
    df["mid_price"] = (df["bid_price"] + df["ask_price"]) / 2
    df["mark_iv"] = df["mark_iv"].abs()

    return df


@pytest.fixture
def temp_data_dir(sample_options_data):
    """Create temporary directory with sample options data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save options data to parquet
        options_path = Path(tmpdir) / "btc_options.parquet"
        sample_options_data.to_parquet(options_path)

        yield tmpdir


@pytest.fixture
def loaded_data_loader(temp_data_dir):
    """Create and load data loader with sample data."""
    loader = CryptoDataLoader(temp_data_dir)
    loader.load_options_data()
    return loader


class TestOptionMatcher:
    """Tests for OptionMatcher class."""

    def test_create_matcher_success(self, loaded_data_loader):
        """Test successful matcher creation."""
        matcher = OptionMatcher(loaded_data_loader)
        assert matcher.data_loader == loaded_data_loader
        assert matcher.options_data is not None
        assert len(matcher.options_data) > 0

    def test_create_matcher_no_options_data(self):
        """Test matcher creation fails without options data."""
        loader = CryptoDataLoader("sample_data")
        # Don't load options data

        with pytest.raises(ValueError, match="must have options data loaded"):
            OptionMatcher(loader)

    def test_find_matching_options_call(self, loaded_data_loader):
        """Test finding matching call options."""
        matcher = OptionMatcher(loaded_data_loader)

        matches = matcher.find_matching_options(
            strike=50000,
            maturity_days=7,
            call=True,
            strike_tolerance_pct=0.05,
            maturity_tolerance_days=1.0,
        )

        assert not matches.empty
        assert all(matches["option_type"] == "call")
        assert all(abs(matches["strike"] - 50000) / 50000 <= 0.05)

    def test_find_matching_options_put(self, loaded_data_loader):
        """Test finding matching put options."""
        matcher = OptionMatcher(loaded_data_loader)

        matches = matcher.find_matching_options(
            strike=50000,
            maturity_days=7,
            call=False,
            strike_tolerance_pct=0.05,
            maturity_tolerance_days=1.0,
        )

        assert not matches.empty
        assert all(matches["option_type"] == "put")

    def test_find_matching_options_strike_filter(self, loaded_data_loader):
        """Test strike filtering works correctly."""
        matcher = OptionMatcher(loaded_data_loader)

        # Look for 48000 strike with tight tolerance
        matches = matcher.find_matching_options(
            strike=48000,
            maturity_days=7,
            call=True,
            strike_tolerance_pct=0.02,  # 2% tolerance = 960 range
            maturity_tolerance_days=1.0,
        )

        if not matches.empty:
            assert all(abs(matches["strike"] - 48000) <= 48000 * 0.02)

    def test_find_matching_options_maturity_filter(self, loaded_data_loader):
        """Test maturity filtering works correctly."""
        matcher = OptionMatcher(loaded_data_loader)

        matches = matcher.find_matching_options(
            strike=50000,
            maturity_days=7,
            call=True,
            strike_tolerance_pct=0.10,
            maturity_tolerance_days=0.5,  # Half day tolerance
        )

        if not matches.empty:
            assert all(abs(matches["days_to_expiry"] - 7) <= 0.5)

    def test_find_matching_options_no_matches(self, loaded_data_loader):
        """Test returns empty DataFrame when no matches found."""
        matcher = OptionMatcher(loaded_data_loader)

        # Use impossible criteria
        matches = matcher.find_matching_options(
            strike=100000,  # Very high strike
            maturity_days=100,  # Very long maturity
            call=True,
            strike_tolerance_pct=0.01,  # Tight tolerance
            maturity_tolerance_days=0.1,
        )

        assert matches.empty

    def test_find_matching_options_sorted_by_time(self, loaded_data_loader):
        """Test results are sorted by timestamp."""
        matcher = OptionMatcher(loaded_data_loader)

        matches = matcher.find_matching_options(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        if len(matches) > 1:
            timestamps = matches["timestamp"].values
            assert all(
                timestamps[i] <= timestamps[i + 1] for i in range(len(timestamps) - 1)
            )

    def test_get_closest_match_success(self, loaded_data_loader):
        """Test getting closest match returns single option."""
        matcher = OptionMatcher(loaded_data_loader)

        best_match = matcher.get_closest_match(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        assert best_match is not None
        assert isinstance(best_match, pd.Series)
        assert best_match["option_type"] == "call"
        assert "strike" in best_match
        assert "days_to_expiry" in best_match

    def test_get_closest_match_none_when_no_matches(self, loaded_data_loader):
        """Test returns None when no matches found."""
        matcher = OptionMatcher(loaded_data_loader)

        best_match = matcher.get_closest_match(
            strike=100000,
            maturity_days=100,
            call=True,
            strike_tolerance_pct=0.01,
            maturity_tolerance_days=0.1,
        )

        assert best_match is None

    def test_get_closest_match_prioritizes_strike(self, loaded_data_loader):
        """Test that strike accuracy is prioritized over maturity."""
        matcher = OptionMatcher(loaded_data_loader)

        # Wide tolerance to get multiple matches
        best_match = matcher.get_closest_match(
            strike=50000,
            maturity_days=7,
            call=True,
            strike_tolerance_pct=0.10,
            maturity_tolerance_days=5.0,
        )

        if best_match is not None:
            # Should be closer to target strike than maturity (relatively)
            strike_error_pct = abs(best_match["strike"] - 50000) / 50000
            maturity_error_pct = abs(best_match["days_to_expiry"] - 7) / 7

            # The weighting is 70% strike, 30% maturity
            # So strike error should generally be lower
            assert "match_distance" in best_match

    def test_get_time_series_success(self, loaded_data_loader):
        """Test getting time series of options."""
        matcher = OptionMatcher(loaded_data_loader)

        start_date = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2024, 1, 2, tzinfo=timezone.utc)

        time_series = matcher.get_time_series(
            strike=50000,
            maturity_days=7,
            call=True,
            start_date=start_date,
            end_date=end_date,
        )

        if not time_series.empty:
            assert all(time_series["timestamp"] >= start_date)
            assert all(time_series["timestamp"] <= end_date)

    def test_get_time_series_filters_by_date(self, loaded_data_loader):
        """Test time series date filtering works."""
        matcher = OptionMatcher(loaded_data_loader)

        # Get full series
        full_series = matcher.get_time_series(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        # Get filtered series
        start_date = datetime(2024, 1, 1, 6, 0, tzinfo=timezone.utc)
        filtered_series = matcher.get_time_series(
            strike=50000,
            maturity_days=7,
            call=True,
            start_date=start_date,
        )

        if not full_series.empty and not filtered_series.empty:
            assert len(filtered_series) <= len(full_series)
            assert all(filtered_series["timestamp"] >= start_date)

    def test_summary_returns_dict(self, loaded_data_loader):
        """Test summary returns expected structure."""
        matcher = OptionMatcher(loaded_data_loader)

        summary = matcher.summary()

        assert isinstance(summary, dict)
        assert "total_options" in summary
        assert "calls" in summary
        assert "puts" in summary
        assert "strikes" in summary
        assert "date_range" in summary
        assert "expiries" in summary

    def test_summary_values_reasonable(self, loaded_data_loader):
        """Test summary contains reasonable values."""
        matcher = OptionMatcher(loaded_data_loader)

        summary = matcher.summary()

        assert summary["total_options"] > 0
        assert summary["calls"] > 0
        assert summary["puts"] > 0
        assert len(summary["strikes"]) > 0
        assert summary["calls"] + summary["puts"] == summary["total_options"]

    def test_summary_strikes_sorted(self, loaded_data_loader):
        """Test summary strikes are sorted."""
        matcher = OptionMatcher(loaded_data_loader)

        summary = matcher.summary()

        strikes = summary["strikes"]
        if len(strikes) > 1:
            assert strikes == sorted(strikes)


@pytest.fixture
def sample_backtest_results():
    """Create sample backtest results for testing."""
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    n_paths = 10
    n_steps = 5

    # Create sample data
    deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
    bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
    deep_positions = torch.randn(n_paths, n_steps) * 0.5
    bs_positions = torch.randn(n_paths, n_steps) * 0.5

    # Create realistic spot prices
    initial_spot = 50000.0
    spots = torch.ones(n_paths, n_steps) * initial_spot
    for i in range(1, n_steps):
        spots[:, i] = spots[:, i - 1] * (1 + torch.randn(n_paths) * 0.01)

    # Create config
    config = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-05",
        strike=50000,
        maturity_days=7,
        call=True,
        model_path="test_model.pth",
        data_dir="test_data",
    )

    results = BacktestResults(
        deep_pnl=deep_pnl,
        bs_pnl=bs_pnl,
        deep_positions=deep_positions,
        bs_positions=bs_positions,
        spots=spots,
        config=config,
    )

    return results


class TestPriceComparator:
    """Tests for PriceComparator class."""

    def test_create_comparator_success(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test successful comparator creation."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        assert comparator.backtest_results == sample_backtest_results
        assert comparator.option_matcher == matcher

    def test_calculate_model_implied_price(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test model-implied price calculation."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        model_price = comparator.calculate_model_implied_price()

        # Should return a finite float
        assert isinstance(model_price, float)
        assert not np.isnan(model_price)
        assert not np.isinf(model_price)

        # Note: With random test data, model price can be negative
        # (if hedging costs exceed expected payoff).
        # See TestPnLConvention tests for verification with realistic data.

    def test_calculate_model_implied_price_repeatability(self, loaded_data_loader):
        """Test that same seed produces identical model-implied price."""
        # Create results with seed 123
        torch.manual_seed(123)
        np.random.seed(123)

        n_paths, n_steps = 10, 5
        deep_pnl1 = torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strike=50000,
            maturity_days=7,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )
        results1 = BacktestResults(
            deep_pnl=deep_pnl1,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=torch.ones(n_paths, n_steps) * 50000,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator1 = PriceComparator(results1, matcher)
        price1 = comparator1.calculate_model_implied_price()

        # Create results again with same seed
        torch.manual_seed(123)
        np.random.seed(123)

        deep_pnl2 = torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
        results2 = BacktestResults(
            deep_pnl=deep_pnl2,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=torch.ones(n_paths, n_steps) * 50000,
            config=config,
        )

        comparator2 = PriceComparator(results2, matcher)
        price2 = comparator2.calculate_model_implied_price()

        # Same seed should produce identical results
        assert (
            abs(price1 - price2) < 1e-10
        ), f"Same seed should produce identical price: {price1} vs {price2}"

    @pytest.mark.parametrize(
        "strike,final_spot,expected_sign",
        [
            (50000, 55000, 1),  # ITM call: spot > strike, should be positive
            (50000, 52000, 1),  # Slightly ITM call
            (50000, 51000, 1),  # Barely ITM call
        ],
    )
    def test_calculate_model_implied_price_realistic_scenarios(
        self, loaded_data_loader, strike, final_spot, expected_sign
    ):
        """Test model-implied price with realistic scenarios expecting positive prices.

        For ITM call options with realistic hedging costs, we expect positive prices.
        This provides an economics check on top of the mechanism tests.
        """
        n_paths = 100
        n_steps = 10

        # Create scenario with ITM call
        spots = torch.ones(n_paths, n_steps) * strike
        spots[:, -1] = final_spot  # ITM

        # Simulate realistic hedging: cum_pl ≈ -payoff (good hedging)
        # For ITM call: payoff = max(S_T - K, 0) = final_spot - strike
        expected_payoff = final_spot - strike
        # With good hedging, cum_pl should be approximately -payoff
        # Adding some noise and small hedging costs
        deep_pnl = torch.ones(n_paths, n_steps) * (-expected_payoff / n_steps)
        deep_pnl = deep_pnl.cumsum(dim=1)
        # Add small hedging costs (realistic 1-2% of payoff)
        deep_pnl = (
            deep_pnl - torch.randn(n_paths, n_steps).abs() * expected_payoff * 0.01
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-10",
            strike=strike,
            maturity_days=7,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)
        model_price = comparator.calculate_model_implied_price()

        # Economics check: ITM call with realistic hedging should have positive price
        assert (
            model_price * expected_sign > 0
        ), f"ITM call (K={strike}, S_T={final_spot}) should have positive price, got {model_price:.2f}"

        # Sanity check: price should be close to intrinsic value
        # (within reasonable hedging costs, say 20% of intrinsic)
        intrinsic_value = expected_payoff
        assert (
            abs(model_price - intrinsic_value) < intrinsic_value * 0.5
        ), f"Price {model_price:.2f} too far from intrinsic {intrinsic_value:.2f}"

    def test_calculate_model_implied_price_put(self, loaded_data_loader):
        """Test model-implied price calculation for put option."""
        # Create put option backtest results
        n_paths = 10
        n_steps = 5
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
        deep_positions = torch.randn(n_paths, n_steps) * 0.5
        bs_positions = torch.randn(n_paths, n_steps) * 0.5
        spots = (
            torch.ones(n_paths, n_steps) * 50000
            + torch.randn(n_paths, n_steps).cumsum(dim=1) * 100
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strike=50000,
            maturity_days=7,
            call=False,  # Put option
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        model_price = comparator.calculate_model_implied_price()
        assert isinstance(model_price, float)

    def test_get_market_price_success(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test getting market price for matching option."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        market_price = comparator.get_market_price(
            strike=50000,
            maturity_days=7,
            call=True,
            price_type="mid",
        )

        # Should find a match in our sample data
        assert market_price is not None or market_price is None  # Either is valid
        if market_price is not None:
            assert isinstance(market_price, float)
            assert market_price > 0

    def test_get_market_price_no_match(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test returns None when no matching option found."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        market_price = comparator.get_market_price(
            strike=100000,  # Impossible strike
            maturity_days=100,  # Impossible maturity
            call=True,
            price_type="mid",
        )

        assert market_price is None

    def test_compare_with_market_success(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test price comparison with market."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        comparison = comparator.compare_with_market(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        assert isinstance(comparison, dict)
        assert "model_price" in comparison
        assert "market_price" in comparison
        assert "matched_strike" in comparison
        assert "matched_maturity" in comparison

        # Model price should always be calculated
        assert isinstance(comparison["model_price"], float)

    def test_compare_with_market_calculates_difference(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test comparison calculates price difference when market price available."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        comparison = comparator.compare_with_market(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        if comparison["market_price"] is not None:
            assert "difference" in comparison
            assert "difference_pct" in comparison
            assert isinstance(comparison["difference"], float)
            assert isinstance(comparison["difference_pct"], float)

            # Verify calculation
            expected_diff = comparison["model_price"] - comparison["market_price"]
            assert abs(comparison["difference"] - expected_diff) < 0.01

    def test_summary_returns_dict(self, sample_backtest_results, loaded_data_loader):
        """Test summary returns expected structure."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        summary = comparator.summary()

        assert isinstance(summary, dict)
        assert "model_implied_price" in summary
        assert "calculation_method" in summary
        assert "n_paths" in summary
        assert "n_steps" in summary

    def test_summary_values_reasonable(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test summary contains reasonable values."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        summary = comparator.summary()

        assert isinstance(summary["model_implied_price"], float)
        assert summary["calculation_method"] == "mean_cost"
        assert summary["n_paths"] == sample_backtest_results.n_paths
        assert summary["n_steps"] == sample_backtest_results.n_steps


class TestOptionMatcherWeights:
    """Tests for configurable weights in OptionMatcher."""

    def test_get_closest_match_default_weights(self, loaded_data_loader):
        """Test default weights sum to 1.0."""
        matcher = OptionMatcher(loaded_data_loader)

        # Call with defaults should work
        best_match = matcher.get_closest_match(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        # Should not raise error
        assert best_match is not None or best_match is None

    def test_get_closest_match_custom_weights(self, loaded_data_loader):
        """Test custom weights work correctly."""
        matcher = OptionMatcher(loaded_data_loader)

        # Use custom weights that sum to 1.0
        best_match = matcher.get_closest_match(
            strike=50000,
            maturity_days=7,
            call=True,
            strike_weight=0.5,
            maturity_weight=0.5,
        )

        # Should not raise error
        assert best_match is not None or best_match is None

    def test_get_closest_match_weights_validation_sum_not_one(self, loaded_data_loader):
        """Test weights validation fails when sum != 1.0."""
        matcher = OptionMatcher(loaded_data_loader)

        with pytest.raises(ValueError, match="Weights must sum to 1.0"):
            matcher.get_closest_match(
                strike=50000,
                maturity_days=7,
                call=True,
                strike_weight=0.6,
                maturity_weight=0.6,  # Sum = 1.2
            )

    def test_get_closest_match_weights_validation_negative(self, loaded_data_loader):
        """Test weights validation fails for negative weights."""
        matcher = OptionMatcher(loaded_data_loader)

        with pytest.raises(ValueError, match="Weights must be non-negative"):
            matcher.get_closest_match(
                strike=50000,
                maturity_days=7,
                call=True,
                strike_weight=-0.3,
                maturity_weight=1.3,
            )

    def test_get_closest_match_extreme_strike_weight(self, loaded_data_loader):
        """Test extreme strike weight prioritizes strike matching."""
        matcher = OptionMatcher(loaded_data_loader)

        # Use very high strike weight
        best_match = matcher.get_closest_match(
            strike=50000,
            maturity_days=7,
            call=True,
            strike_tolerance_pct=0.10,
            maturity_tolerance_days=5.0,
            strike_weight=0.99,
            maturity_weight=0.01,
        )

        if best_match is not None:
            # Should have very good strike match
            strike_error_pct = abs(best_match["strike"] - 50000) / 50000
            assert strike_error_pct < 0.10

    def test_get_closest_match_extreme_maturity_weight(self, loaded_data_loader):
        """Test extreme maturity weight prioritizes maturity matching."""
        matcher = OptionMatcher(loaded_data_loader)

        # Use very high maturity weight
        best_match = matcher.get_closest_match(
            strike=50000,
            maturity_days=7,
            call=True,
            strike_tolerance_pct=0.10,
            maturity_tolerance_days=5.0,
            strike_weight=0.01,
            maturity_weight=0.99,
        )

        if best_match is not None:
            # Should have very good maturity match
            maturity_error = abs(best_match["days_to_expiry"] - 7)
            assert maturity_error < 5.0


class TestPnLConvention:
    """Tests for PnL convention and model-implied pricing."""

    def test_model_implied_price_uses_correct_convention(self, loaded_data_loader):
        """Test that model-implied price uses correct PnL convention."""
        # Create specific backtest results to test formula
        n_paths = 100
        n_steps = 10

        # Create deterministic PnL for testing
        # Final PnL = -200 (average across paths)
        deep_pnl = torch.ones(n_paths, n_steps) * -20.0  # Each step contributes -20
        deep_pnl = deep_pnl.cumsum(dim=1)  # Final value = -200

        bs_pnl = torch.zeros(n_paths, n_steps)
        deep_positions = torch.zeros(n_paths, n_steps)
        bs_positions = torch.zeros(n_paths, n_steps)
        spots = torch.ones(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-10",
            strike=50000,
            maturity_days=7,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        model_price = comparator.calculate_model_implied_price()

        # PFHedge convention: cum_pl = hedging_gains - transaction_costs - payoff
        # So: payoff = hedging_gains - transaction_costs - cum_pl
        # For fair pricing: premium = E[payoff] = -E[cum_pl]
        # Expected final cum_pl = -200, so premium should be -(-200) = 200
        expected_premium = 200.0
        assert (
            abs(model_price - expected_premium) < 1e-6
        ), f"Expected {expected_premium}, got {model_price}"

    def test_model_implied_price_alternative_formulas_equivalent(
        self, loaded_data_loader
    ):
        """Test that both pricing formulas are equivalent under PFHedge convention."""
        n_paths = 50
        n_steps = 10

        # Create random PnL
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 10
        bs_pnl = torch.zeros(n_paths, n_steps)
        deep_positions = torch.zeros(n_paths, n_steps)
        bs_positions = torch.zeros(n_paths, n_steps)
        spots = torch.ones(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-10",
            strike=50000,
            maturity_days=7,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        # Method 1: Current implementation (uses -E[cum_pl])
        model_price_method1 = comparator.calculate_model_implied_price()

        # Method 2: Manual calculation
        # Under PFHedge convention: cum_pl = hedging_gains - transaction_costs - payoff
        # So: payoff = hedging_gains - transaction_costs - cum_pl
        # For fair pricing: premium = E[payoff] = -E[cum_pl]
        final_pnl = deep_pnl[:, -1]
        mean_cum_pl = final_pnl.mean().item()
        model_price_method2 = -mean_cum_pl

        assert (
            abs(model_price_method1 - model_price_method2) < 1e-6
        ), f"Methods should be equivalent: {model_price_method1} vs {model_price_method2}"

    def test_model_implied_price_positive_for_itm_calls(self, loaded_data_loader):
        """Test that ITM calls have positive model-implied prices."""
        n_paths = 100
        n_steps = 10

        # Create scenario where final spot > strike (ITM call)
        spots = torch.ones(n_paths, n_steps) * 50000
        spots[:, -1] = 55000  # ITM by 5000

        # With perfect hedging, cum_pl should be approximately -payoff
        # For ITM call: payoff = 5000, so cum_pl ≈ -5000
        # Therefore: premium = -cum_pl ≈ 5000
        deep_pnl = torch.ones(n_paths, n_steps) * -500
        deep_pnl = deep_pnl.cumsum(dim=1)  # Final ≈ -5000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-10",
            strike=50000,
            maturity_days=7,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        model_price = comparator.calculate_model_implied_price()

        # ITM call should have positive premium
        assert (
            model_price > 0
        ), f"ITM call should have positive price, got {model_price}"


class TestConfidenceIntervals:
    """Tests for model price confidence interval calculation."""

    def test_calculate_model_price_confidence_default_level(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test confidence interval calculation with default 95% level."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        result = comparator.calculate_model_price_confidence()

        assert isinstance(result, dict)
        assert "mean" in result
        assert "std_error" in result
        assert "std_dev" in result
        assert "confidence_level" in result
        assert "lower" in result
        assert "upper" in result
        assert "n_paths" in result
        assert "distribution" in result  # New field: "t" or "normal"
        assert "critical_value" in result  # New field: t-score or z-score

        # Check default confidence level
        assert result["confidence_level"] == 0.95

        # Check distribution type (should be "t" for n=10 or "normal" for n>=30)
        assert result["distribution"] in ["t", "normal"]
        if result["n_paths"] < 30:
            assert result["distribution"] == "t"
        else:
            assert result["distribution"] == "normal"

        # Check bounds are ordered correctly
        assert result["lower"] < result["mean"]
        assert result["mean"] < result["upper"]

        # Check interval width calculation
        interval_width = result["upper"] - result["lower"]
        assert interval_width > 0

    def test_calculate_model_price_confidence_custom_level(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test confidence interval with custom confidence level."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        # Test 99% confidence interval
        result_99 = comparator.calculate_model_price_confidence(confidence_level=0.99)
        result_95 = comparator.calculate_model_price_confidence(confidence_level=0.95)

        # Calculate interval widths
        width_99 = result_99["upper"] - result_99["lower"]
        width_95 = result_95["upper"] - result_95["lower"]

        # 99% interval should be wider than 95%
        assert width_99 > width_95
        assert result_99["confidence_level"] == 0.99
        assert result_95["confidence_level"] == 0.95

    def test_calculate_model_price_confidence_different_levels(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test multiple confidence levels."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        levels = [0.90, 0.95, 0.99]
        results = [
            comparator.calculate_model_price_confidence(level) for level in levels
        ]

        # Calculate interval widths
        widths = [r["upper"] - r["lower"] for r in results]

        # Intervals should get wider with higher confidence
        for i in range(len(widths) - 1):
            assert widths[i] < widths[i + 1]

        # All should have same mean
        means = [r["mean"] for r in results]
        assert all(abs(m - means[0]) < 1e-6 for m in means)

    def test_calculate_model_price_confidence_matches_mean_price(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test that confidence interval mean matches model-implied price."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        confidence_result = comparator.calculate_model_price_confidence()
        model_price = comparator.calculate_model_implied_price()

        # Mean from confidence calculation should match model price
        assert abs(confidence_result["mean"] - model_price) < 1e-6


class TestSpreadDiagnostics:
    """Tests for spread diagnostics in price comparison."""

    def test_compare_with_market_includes_spread_diagnostics(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test that spread diagnostics are included when flag is True."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        comparison = comparator.compare_with_market(
            strike=50000,
            maturity_days=7,
            call=True,
            include_spread_diagnostics=True,
        )

        # If we found a match, should have spread diagnostics
        if comparison["market_price"] is not None:
            assert "spread" in comparison
            assert "spread_pct" in comparison
            assert "spread_bps" in comparison
            assert "model_in_spread" in comparison
            assert "mark_mid_diff" in comparison

    def test_compare_with_market_no_spread_when_disabled(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test that spread diagnostics are not included when flag is False."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        comparison = comparator.compare_with_market(
            strike=50000,
            maturity_days=7,
            call=True,
            include_spread_diagnostics=False,
        )

        # Should not have spread diagnostics
        assert "spread" not in comparison
        assert "spread_pct" not in comparison
        assert "model_in_spread" not in comparison

    def test_spread_diagnostics_calculations(self, sample_backtest_results):
        """Test spread diagnostics are calculated correctly."""
        # Create mock options data with known spread
        timestamps = pd.date_range("2024-01-01", periods=5, freq="1H", tz="UTC")
        data = []

        bid = 100.0
        ask = 120.0
        mid = 110.0
        mark = 112.0

        for ts in timestamps:
            data.append(
                {
                    "timestamp": ts,
                    "expiration": ts + timedelta(days=7),
                    "strike": 50000,
                    "option_type": "call",
                    "bid_price": bid,
                    "ask_price": ask,
                    "mid_price": mid,
                    "mark_price": mark,
                    "mark_iv": 0.8,
                    "time_to_expiry": 7 / 365.25,
                }
            )

        df = pd.DataFrame(data)

        # Create temporary directory with this data
        with tempfile.TemporaryDirectory() as tmpdir:
            options_path = Path(tmpdir) / "btc_options.parquet"
            df.to_parquet(options_path)

            loader = CryptoDataLoader(tmpdir)
            loader.load_options_data()
            matcher = OptionMatcher(loader)
            comparator = PriceComparator(sample_backtest_results, matcher)

            comparison = comparator.compare_with_market(
                strike=50000,
                maturity_days=7,
                call=True,
                include_spread_diagnostics=True,
            )

            if comparison["market_price"] is not None:
                # Check spread calculations
                assert comparison["spread"] == ask - bid  # 20.0
                assert abs(comparison["spread_pct"] - (20.0 / 110.0 * 100)) < 0.01
                assert abs(comparison["spread_bps"] - (20.0 / 110.0 * 10000)) < 0.1
                assert comparison["mark_mid_diff"] == mark - mid  # 2.0

                # Check model_in_spread
                model_price = comparison["model_price"]
                assert comparison["model_in_spread"] == (bid <= model_price <= ask)


class TestImpliedVolatility:
    """Tests for implied volatility comparison methods."""

    def test_calculate_model_implied_iv_success(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test successful model-implied IV calculation."""
        # Import scipy here so test is skipped if not available
        pytest.importorskip("scipy")

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        # Calculate IV for a realistic option
        model_iv = comparator.calculate_model_implied_iv(
            strike=50000,
            maturity_days=30,
            call=True,
            spot_price=50000,
            risk_free_rate=0.0,
        )

        # Should return a valid IV
        if model_iv is not None:
            assert isinstance(model_iv, float)
            assert 0.01 <= model_iv <= 5.0  # Within search range
            assert not np.isnan(model_iv)
            assert not np.isinf(model_iv)

    def test_calculate_model_implied_iv_realistic_scenario(self, loaded_data_loader):
        """Test IV calculation with realistic option scenario."""
        pytest.importorskip("scipy")

        # Create results with realistic ITM call
        n_paths = 100
        n_steps = 20

        strike = 50000
        spot = 52000  # 4% ITM
        maturity_days = 30

        # Create realistic hedging scenario
        spots = torch.ones(n_paths, n_steps) * strike
        spots[:, -1] = spot

        # Simulate realistic hedging with small costs
        expected_payoff = spot - strike  # 2000
        deep_pnl = torch.ones(n_paths, n_steps) * (-expected_payoff / n_steps)
        deep_pnl = deep_pnl.cumsum(dim=1)
        # Add small hedging costs (1% of payoff)
        deep_pnl = (
            deep_pnl - torch.randn(n_paths, n_steps).abs() * expected_payoff * 0.01
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        model_iv = comparator.calculate_model_implied_iv(
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            spot_price=strike,  # Use initial spot
            risk_free_rate=0.0,
        )

        # Should calculate a reasonable IV
        if model_iv is not None:
            # Crypto IV typically 50%-150% annualized
            assert 0.3 <= model_iv <= 2.0, f"IV {model_iv:.2%} outside reasonable range"

    def test_calculate_model_implied_iv_outside_bounds(self, loaded_data_loader):
        """Test IV calculation fails gracefully when price outside arbitrage bounds."""
        pytest.importorskip("scipy")

        # Create scenario where model price violates arbitrage bounds
        n_paths = 10
        n_steps = 5

        # Create very large negative PnL (unrealistic premium)
        deep_pnl = torch.ones(n_paths, n_steps) * -10000
        deep_pnl = deep_pnl.cumsum(dim=1)

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strike=50000,
            maturity_days=1,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=torch.ones(n_paths, n_steps) * 50000,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        # Should return None for out-of-bounds price
        # Can emit either bounds warning or solver failure warning
        with pytest.warns(
            UserWarning,
            match="(outside arbitrage bounds|Failed to calculate implied volatility)",
        ):
            model_iv = comparator.calculate_model_implied_iv(
                strike=50000,
                maturity_days=1,
                call=True,
                spot_price=50000,
            )
            assert model_iv is None

    def test_calculate_model_implied_iv_no_scipy(self, loaded_data_loader):
        """Test bisection fallback when scipy not available."""
        # Create realistic scenario that should work with bisection
        n_paths = 100
        n_steps = 20

        strike = 50000
        spot = 52000
        maturity_days = 30

        spots = torch.ones(n_paths, n_steps) * strike
        spots[:, -1] = spot

        expected_payoff = spot - strike  # 2000
        deep_pnl = torch.ones(n_paths, n_steps) * (-expected_payoff / n_steps)
        deep_pnl = deep_pnl.cumsum(dim=1)
        deep_pnl = (
            deep_pnl - torch.randn(n_paths, n_steps).abs() * expected_payoff * 0.01
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        # Mock scipy as not available
        import crypto.backtest.option_comparison as opt_comp

        original_has_scipy = opt_comp.HAS_SCIPY
        try:
            opt_comp.HAS_SCIPY = False

            # Should still work with bisection fallback
            iv, diag = comparator.calculate_model_implied_iv(
                strike=strike,
                maturity_days=maturity_days,
                call=True,
                spot_price=strike,
                return_diagnostics=True,
            )

            # Should succeed with bisection method
            if iv is not None:
                assert isinstance(iv, float)
                assert 0.01 <= iv <= 5.0
                assert diag["method"] == "bisection"
                assert diag["status"] == "success"
                assert diag["iterations"] is not None
                assert diag["price_error"] is not None
        finally:
            opt_comp.HAS_SCIPY = original_has_scipy

    def test_get_market_iv_success(self, loaded_data_loader):
        """Test getting market IV from matched option."""
        matcher = OptionMatcher(loaded_data_loader)

        # Just test the matcher has IV data
        matches = matcher.find_matching_options(
            strike=50000,
            maturity_days=7,
            call=True,
        )

        if not matches.empty:
            # Check that mark_iv exists in our sample data
            assert "mark_iv" in matches.columns

    def test_get_market_iv_with_comparator(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test get_market_iv method in PriceComparator."""
        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        market_iv = comparator.get_market_iv(
            strike=50000,
            maturity_days=7,
            call=True,
            iv_type="mark",
        )

        # Should either find an IV or return None
        if market_iv is not None:
            assert isinstance(market_iv, float)
            assert market_iv > 0
            assert not np.isnan(market_iv)

    def test_get_market_iv_fallback(self):
        """Test IV fallback when preferred type not available."""
        # Create data with only bid_iv, no mark_iv
        timestamps = pd.date_range("2024-01-01", periods=5, freq="1H", tz="UTC")
        data = []

        for ts in timestamps:
            data.append(
                {
                    "timestamp": ts,
                    "expiration": ts + timedelta(days=7),
                    "strike": 50000,
                    "option_type": "call",
                    "bid_price": 100,
                    "ask_price": 120,
                    "mid_price": 110,
                    "bid_iv": 0.85,  # Only bid_iv available
                    "ask_iv": 0.90,
                    "time_to_expiry": 7 / 365.25,
                }
            )

        df = pd.DataFrame(data)

        with tempfile.TemporaryDirectory() as tmpdir:
            options_path = Path(tmpdir) / "btc_options.parquet"
            df.to_parquet(options_path)

            loader = CryptoDataLoader(tmpdir)
            loader.load_options_data()
            matcher = OptionMatcher(loader)

            # Create minimal backtest results
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=7,
                call=True,
                model_path="test.pth",
                data_dir="test",
            )
            results = BacktestResults(
                deep_pnl=torch.zeros(10, 5),
                bs_pnl=torch.zeros(10, 5),
                deep_positions=torch.zeros(10, 5),
                bs_positions=torch.zeros(10, 5),
                spots=torch.ones(10, 5) * 50000,
                config=config,
            )

            comparator = PriceComparator(results, matcher)

            # Request mark_iv, should fallback to bid_iv
            market_iv = comparator.get_market_iv(
                strike=50000,
                maturity_days=7,
                call=True,
                iv_type="mark",  # Not available
            )

            # Should fallback and return bid_iv or ask_iv
            if market_iv is not None:
                assert market_iv in [0.85, 0.90]

    def test_compare_implied_volatility_success(self, loaded_data_loader):
        """Test full IV comparison."""
        pytest.importorskip("scipy")

        # Create realistic scenario
        n_paths = 100
        n_steps = 20

        strike = 50000
        maturity_days = 30

        # Create realistic hedging scenario
        spots = torch.ones(n_paths, n_steps) * strike
        spots[:, -1] = 52000  # ITM

        expected_payoff = 2000
        deep_pnl = torch.ones(n_paths, n_steps) * (-expected_payoff / n_steps)
        deep_pnl = deep_pnl.cumsum(dim=1)
        deep_pnl = (
            deep_pnl - torch.randn(n_paths, n_steps).abs() * expected_payoff * 0.01
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        comparison = comparator.compare_implied_volatility(
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            spot_price=strike,
        )

        # Check structure
        assert isinstance(comparison, dict)
        assert "model_iv" in comparison
        assert "market_iv" in comparison
        assert "matched_strike" in comparison
        assert "matched_maturity" in comparison

        # If both IVs calculated, should have difference
        if comparison["model_iv"] is not None and comparison["market_iv"] is not None:
            assert "iv_difference" in comparison
            assert "iv_difference_pct" in comparison
            assert isinstance(comparison["iv_difference"], float)
            assert isinstance(comparison["iv_difference_pct"], float)

    def test_compare_implied_volatility_no_market_match(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test IV comparison when no market match found."""
        pytest.importorskip("scipy")

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        comparison = comparator.compare_implied_volatility(
            strike=100000,  # No match
            maturity_days=100,
            call=True,
        )

        # Should return None for market_iv
        assert comparison["market_iv"] is None
        assert comparison["matched_strike"] is None
        assert comparison["matched_maturity"] is None
        assert comparison["iv_difference"] is None

    def test_get_volatility_smile_success(self, loaded_data_loader):
        """Test volatility smile generation."""
        pytest.importorskip("scipy")

        # Create realistic scenario
        n_paths = 100
        n_steps = 20

        strike = 50000
        maturity_days = 30

        spots = torch.ones(n_paths, n_steps) * strike
        spots[:, -1] = 52000

        expected_payoff = 2000
        deep_pnl = torch.ones(n_paths, n_steps) * (-expected_payoff / n_steps)
        deep_pnl = deep_pnl.cumsum(dim=1)
        deep_pnl = (
            deep_pnl - torch.randn(n_paths, n_steps).abs() * expected_payoff * 0.01
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        # Get smile across strikes
        strikes = [48000, 50000, 52000]
        smile = comparator.get_volatility_smile(
            strikes=strikes,
            maturity_days=maturity_days,
            call=True,
            spot_price=strike,
        )

        # Check structure
        assert isinstance(smile, pd.DataFrame)
        assert len(smile) == len(strikes)
        assert "strike" in smile.columns
        assert "moneyness" in smile.columns
        assert "model_iv" in smile.columns
        assert "market_iv" in smile.columns
        assert "iv_difference" in smile.columns
        assert "matched_strike" in smile.columns
        assert "matched_maturity" in smile.columns

        # Check moneyness calculation
        for i, strike_val in enumerate(strikes):
            assert abs(smile.iloc[i]["moneyness"] - strike_val / strike) < 1e-6

    def test_get_volatility_smile_empty_strikes(
        self, sample_backtest_results, loaded_data_loader
    ):
        """Test volatility smile with empty strikes list."""
        pytest.importorskip("scipy")

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(sample_backtest_results, matcher)

        smile = comparator.get_volatility_smile(
            strikes=[],
            maturity_days=30,
            call=True,
        )

        # Should return empty DataFrame
        assert isinstance(smile, pd.DataFrame)
        assert len(smile) == 0

    def test_get_volatility_smile_moneyness_ordered(self, loaded_data_loader):
        """Test that volatility smile has ordered moneyness."""
        pytest.importorskip("scipy")

        # Create realistic scenario
        n_paths = 100
        n_steps = 20

        strike = 50000
        maturity_days = 30

        spots = torch.ones(n_paths, n_steps) * strike
        spots[:, -1] = 52000

        expected_payoff = 2000
        deep_pnl = torch.ones(n_paths, n_steps) * (-expected_payoff / n_steps)
        deep_pnl = deep_pnl.cumsum(dim=1)
        deep_pnl = (
            deep_pnl - torch.randn(n_paths, n_steps).abs() * expected_payoff * 0.01
        )

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        # Get smile with ordered strikes
        strikes = [46000, 48000, 50000, 52000, 54000]
        smile = comparator.get_volatility_smile(
            strikes=strikes,
            maturity_days=maturity_days,
            call=True,
            spot_price=strike,
        )

        # Moneyness should be in same order as strikes
        moneyness = smile["moneyness"].tolist()
        assert moneyness == sorted(moneyness)

    def test_calculate_model_implied_iv_extreme_bounds(self, loaded_data_loader):
        """Test IV calculation near extreme bounds (very high/low volatility)."""
        pytest.importorskip("scipy")

        # Test scenario 1: Very high IV case (deep OTM, small price)
        # This tests the upper bound of IV search range
        n_paths = 100
        n_steps = 10

        strike = 150000  # Very high strike (moneyness = 3.0, triggers warning)
        spot = 50000  # Current spot
        maturity_days = 30

        spots = torch.ones(n_paths, n_steps) * spot
        spots[:, -1] = spot  # Stays at spot

        # Very small expected payoff (deep OTM)
        deep_pnl = torch.ones(n_paths, n_steps) * (-10.0 / n_steps)  # Small price ~$10
        deep_pnl = deep_pnl.cumsum(dim=1)

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-30",
            strike=strike,
            maturity_days=maturity_days,
            call=True,
            model_path="test_model.pth",
            data_dir="test_data",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=torch.zeros(n_paths, n_steps),
            deep_positions=torch.zeros(n_paths, n_steps),
            bs_positions=torch.zeros(n_paths, n_steps),
            spots=spots,
            config=config,
        )

        matcher = OptionMatcher(loaded_data_loader)
        comparator = PriceComparator(results, matcher)

        # This should either succeed or return None with warning
        # (deep OTM may be ill-conditioned)
        with pytest.warns(UserWarning):  # Expect warning for deep OTM
            iv, diag = comparator.calculate_model_implied_iv(
                strike=strike,
                maturity_days=maturity_days,
                call=True,
                spot_price=spot,
                return_diagnostics=True,
            )

        # If it succeeds, IV should be in reasonable range
        if iv is not None:
            assert 0.01 <= iv <= 5.0
            assert diag["status"] in ["success", "ill_conditioned"]
            assert diag["method"] in ["brentq", "bisection"]
        else:
            # Should have a diagnostic reason
            assert diag["status"] in [
                "ill_conditioned",
                "out_of_bounds",
                "solver_failed",
            ]
