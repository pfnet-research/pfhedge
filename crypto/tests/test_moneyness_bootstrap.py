import pytest
import torch
import numpy as np
import pandas as pd
import random

from crypto.backtest.config import BacktestConfig
from crypto.instruments.bitcoin_perpetual_historical import BitcoinPerpetualHistorical


# Mock data loader for testing
class MockDataLoader:

    def __init__(self, n_records=500):
        # Create synthetic data with varying price levels (simulating different historical periods)
        dates = pd.date_range("2024-01-01", periods=n_records, freq="8h")

        # Different price regimes to test rescaling
        n_low = n_records // 3
        n_med = n_records // 3
        n_high = n_records - n_low - n_med

        prices = np.concatenate(
            [
                np.random.uniform(40000, 45000, n_low),  # Jan: Low prices
                np.random.uniform(65000, 70000, n_med),  # Mar: Medium prices
                np.random.uniform(105000, 110000, n_high),  # Oct: High prices
            ]
        )

        self.perpetual_data = pd.DataFrame(
            {
                "timestamp": dates,
                "last_price": prices,
                "funding_rate": np.random.normal(0.0001, 0.00005, n_records),
            }
        )


class TestRescalingInvariance:

    def test_log_returns_invariance(self):
        # Original prices
        spot_raw = torch.tensor([42000.0, 43500.0, 41200.0, 44100.0, 42800.0])

        # Rescale
        target_spot = 108000.0
        rescale_factor = target_spot / spot_raw[0]
        spot_rescaled = spot_raw * rescale_factor

        # Calculate returns
        returns_raw = torch.log(spot_raw[1:] / spot_raw[:-1])
        returns_rescaled = torch.log(spot_rescaled[1:] / spot_rescaled[:-1])

        # Should be identical within floating point precision
        torch.testing.assert_close(returns_raw, returns_rescaled, atol=1e-7, rtol=1e-6)

    def test_volatility_invariance(self):
        # Generate random price path
        torch.manual_seed(42)
        returns = torch.randn(100) * 0.02  # 2% daily vol
        spot_raw = torch.exp(returns.cumsum(0)) * 50000

        # Rescale
        rescale_factor = 2.5
        spot_rescaled = spot_raw * rescale_factor

        # Calculate volatilities
        vol_raw = torch.std(torch.log(spot_raw[1:] / spot_raw[:-1]))
        vol_rescaled = torch.std(torch.log(spot_rescaled[1:] / spot_rescaled[:-1]))

        # Should be identical (within floating point precision)
        assert abs(vol_raw - vol_rescaled) < 1e-8


class TestMoneynessConsistency:

    def test_normalize_spot_moneyness_equality(self):
        loader = MockDataLoader(n_records=500)

        underlier = BitcoinPerpetualHistorical(
            data_loader=loader,
            cost=0.0006,
            dt=8 / 24 / 365,
        )

        # Parameters
        strike = 110000
        target_moneyness = 0.98
        target_initial_spot = strike * target_moneyness
        n_paths = 100
        time_horizon = 10 / 365

        # Set seed for reproducibility
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Bootstrap with rescaling
        underlier.simulate_bootstrap(
            n_paths=n_paths,
            time_horizon=time_horizon,
            target_initial_spot=target_initial_spot,
            store_scale_factors=True,
        )

        # Check initial spots
        initial_spots = underlier.spot[:, 0]

        # All should equal target_initial_spot (within numerical tolerance)
        torch.testing.assert_close(
            initial_spots,
            torch.full_like(initial_spots, target_initial_spot),
            atol=1e-3,
            rtol=1e-6,
        )

        # Check log_moneyness
        log_moneyness = torch.log(initial_spots / strike)
        target_log_moneyness = np.log(target_moneyness)

        # All should be equal
        assert log_moneyness.std() < 1e-6
        assert abs(log_moneyness.mean() - target_log_moneyness) < 1e-6

    def test_initial_spot_mean_matches_target(self):
        loader = MockDataLoader(n_records=300)

        underlier = BitcoinPerpetualHistorical(
            data_loader=loader,
            cost=0.0006,
            dt=8 / 24 / 365,
        )

        target_initial_spot = 107800.0

        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)

        underlier.simulate_bootstrap(
            n_paths=50,
            time_horizon=5 / 365,
            target_initial_spot=target_initial_spot,
        )

        initial_spots = underlier.spot[:, 0]
        assert abs(initial_spots.mean().item() - target_initial_spot) < 1e-3


class TestFundingCostScaling:

    def test_funding_cost_scales_with_spot(self):
        position = 1.0
        funding_rate = 0.0001  # 0.01%

        spot_raw = 50000.0
        spot_rescaled = 100000.0

        # Funding cost = position * rate * spot
        cost_raw = position * funding_rate * spot_raw
        cost_rescaled = position * funding_rate * spot_rescaled

        # Should scale by exactly 2x
        assert abs(cost_rescaled / cost_raw - 2.0) < 1e-9


class TestBackwardCompatibility:

    def test_absolute_strike_no_rescaling(self):
        loader = MockDataLoader(n_records=300)

        underlier = BitcoinPerpetualHistorical(
            data_loader=loader,
            cost=0.0006,
            dt=8 / 24 / 365,
        )

        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)

        # Bootstrap without rescaling
        underlier.simulate_bootstrap(
            n_paths=50,
            time_horizon=10 / 365,
            target_initial_spot=None,  # No rescaling
            store_scale_factors=True,
        )

        # Verify scale factors are all 1.0
        if hasattr(underlier, "_bootstrap_scale_factors"):
            scale_factors = underlier._bootstrap_scale_factors
            torch.testing.assert_close(
                scale_factors,
                torch.ones_like(scale_factors),
                atol=1e-9,
                rtol=1e-9,
            )


class TestConfigValidation:

    def test_normalize_spot_requires_moneyness(self):
        with pytest.raises(
            ValueError, match="requires either 'initial_spot' or 'target_moneyness'"
        ):
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-11",
                strike=110000,
                maturity_days=10,
                model_path="dummy.pth",
                bootstrap_mode="normalize_spot",
                # Missing: initial_spot and target_moneyness
            )
            config.validate()

    def test_normalize_spot_with_initial_spot_works(self):
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-11",
            strike=110000,
            maturity_days=10,
            model_path="dummy.pth",
            bootstrap_mode="normalize_spot",
            initial_spot=108000,
        )
        config.validate()

        target_moneyness = config.effective_target_moneyness
        expected = 108000 / 110000
        assert abs(target_moneyness - expected) < 1e-9

    def test_normalize_spot_with_target_moneyness_works(self):
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-11",
            strike=110000,
            maturity_days=10,
            model_path="dummy.pth",
            bootstrap_mode="normalize_spot",
            target_moneyness=0.95,
        )
        config.validate()

        assert config.effective_target_moneyness == 0.95

    def test_absolute_strike_no_moneyness_required(self):
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-11",
            strike=110000,
            maturity_days=10,
            model_path="dummy.pth",
            bootstrap_mode="absolute_strike",
            # No initial_spot - should be fine
        )
        config.validate()
        assert config.effective_target_moneyness is None

    def test_invalid_initial_spot(self):
        with pytest.raises(ValueError, match="initial_spot must be positive"):
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-11",
                strike=110000,
                maturity_days=10,
                model_path="dummy.pth",
                bootstrap_mode="normalize_spot",
                initial_spot=-1000,
            )
            config.validate()

    def test_invalid_target_moneyness(self):
        with pytest.raises(ValueError, match="target_moneyness must be positive"):
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-11",
                strike=110000,
                maturity_days=10,
                model_path="dummy.pth",
                bootstrap_mode="normalize_spot",
                target_moneyness=0.0,
            )
            config.validate()

    def test_strike_zero_error(self):
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-11",
            strike=0,
            maturity_days=10,
            model_path="dummy.pth",
            initial_spot=108000,
        )

        with pytest.raises(
            ValueError, match="Cannot calculate moneyness with strike=0"
        ):
            _ = config.effective_target_moneyness


class TestEdgeCases:

    def test_single_window_rescaling(self):
        # Create minimal synthetic data (exactly n_steps long)
        n_steps = 30
        dates = pd.date_range("2024-01-01", periods=n_steps, freq="8h")

        loader = MockDataLoader.__new__(MockDataLoader)
        loader.perpetual_data = pd.DataFrame(
            {
                "timestamp": dates,
                "last_price": np.linspace(42000, 45000, n_steps),
                "funding_rate": np.zeros(n_steps),
            }
        )
        # Add perpetual_data_full attribute expected by simulate_bootstrap
        loader.perpetual_data_full = loader.perpetual_data.copy()

        underlier = BitcoinPerpetualHistorical(
            data_loader=loader,
            dt=8 / 24 / 365,
        )

        # Bootstrap 10 paths - but with only 1 unique window, it will cap to 1
        target_spot = 108000
        underlier.simulate_bootstrap(
            n_paths=10,
            time_horizon=(n_steps - 1) * 8 / 24 / 365,
            target_initial_spot=target_spot,
        )

        # With new auto-capping feature, only 1 path should be generated
        assert (
            underlier.spot.shape[0] == 1
        ), "Should cap to 1 path when only 1 unique window available"

        # That single path should have correct initial spot
        initial_spot = underlier.spot[0, 0]
        torch.testing.assert_close(
            initial_spot,
            torch.tensor(target_spot, dtype=initial_spot.dtype),
            atol=1e-3,
            rtol=1e-6,
        )

    def test_insufficient_data_error(self):
        # Create very short dataset
        dates = pd.date_range("2024-01-01", periods=10, freq="8h")

        loader = MockDataLoader.__new__(MockDataLoader)
        loader.perpetual_data = pd.DataFrame(
            {
                "timestamp": dates,
                "last_price": np.ones(10) * 50000,
            }
        )

        underlier = BitcoinPerpetualHistorical(
            data_loader=loader,
            dt=8 / 24 / 365,
        )

        # Try to bootstrap with more steps than available
        with pytest.raises(ValueError, match="Insufficient historical data"):
            underlier.simulate_bootstrap(
                n_paths=10,
                time_horizon=30 / 365,  # Requires ~91 steps, but only 10 available
            )


class TestSeedReproducibility:

    def test_same_seed_identical_paths(self):
        loader = MockDataLoader(n_records=500)

        def run_bootstrap(seed_val):
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)

            underlier = BitcoinPerpetualHistorical(
                data_loader=loader,
                dt=8 / 24 / 365,
            )
            underlier.simulate_bootstrap(
                n_paths=50,
                time_horizon=10 / 365,
                target_initial_spot=108000,
            )
            return underlier.spot.clone()

        # Run 1
        spots1 = run_bootstrap(42)

        # Run 2 with same seed
        spots2 = run_bootstrap(42)

        # Should be identical
        torch.testing.assert_close(spots1, spots2)

    def test_different_seed_different_paths(self):
        loader = MockDataLoader(n_records=500)

        def run_bootstrap(seed_val):
            random.seed(seed_val)
            np.random.seed(seed_val)
            torch.manual_seed(seed_val)

            underlier = BitcoinPerpetualHistorical(
                data_loader=loader,
                dt=8 / 24 / 365,
            )
            underlier.simulate_bootstrap(
                n_paths=50,
                time_horizon=10 / 365,
                target_initial_spot=108000,
            )
            return underlier.spot.clone()

        spots1 = run_bootstrap(42)
        spots2 = run_bootstrap(43)

        # Should be different
        assert not torch.allclose(spots1, spots2)
