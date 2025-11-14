import unittest
import logging
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from crypto.instruments import BitcoinPerpetualHistorical, BitcoinSpotHistorical


class MockDataLoader:
    """Mock data loader with controllable dataset size for testing path limits"""

    def __init__(self, n_points=200):
        dates = pd.date_range("2023-01-01", periods=n_points, freq="8h")
        prices = 50000 + np.random.randn(n_points).cumsum() * 100

        self.data = pd.DataFrame(
            {
                "timestamp": dates,
                "last_price": prices,
                "bid_price": prices - 10,
                "ask_price": prices + 10,
                "funding_8h": np.random.randn(n_points) * 0.0001,
                "index_price": prices + np.random.randn(n_points) * 5,
            }
        )

        # Store as attributes expected by simulate_bootstrap
        self.perpetual_data = self.data
        self.perpetual_data_full = self.data.copy()
        self.spot_data = self.data

    def load_perpetual_data(self):
        return self.data

    def load_spot_data(self):
        return self.data


class TestBootstrapPathOptimization(unittest.TestCase):
    """Test automatic path number optimization in bootstrap generation"""

    def setUp(self):
        self.mock_loader = MockDataLoader(n_points=200)
        # dt should be in years: 8 hours = 8/24/365 years
        self.underlier = BitcoinPerpetualHistorical(
            data_loader=self.mock_loader, dt=8.0 / 24.0 / 365.0, cost=0.0004
        )

    def test_auto_maximum_with_none(self):
        """Test that n_paths=None uses maximum unique paths"""
        n_steps = 100
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0

        # Calculate expected max unique paths
        expected_max = 200 - n_steps + 1  # 101

        with self.assertLogs(
            "crypto.instruments.bitcoin_perpetual_historical", level="INFO"
        ) as log_context:
            self.underlier.simulate_bootstrap(
                n_paths=None,
                time_horizon=time_horizon,
                target_initial_spot=None,
            )

            # Check log message
            self.assertTrue(
                any(
                    f"using max unique paths: {expected_max:,}" in msg.lower()
                    for msg in log_context.output
                ),
                "Expected log message about using max unique paths",
            )

        # Verify correct number of paths generated
        self.assertEqual(
            self.underlier.spot.shape[0],
            expected_max,
            f"Expected {expected_max} paths when n_paths=None",
        )

    def test_auto_maximum_with_zero(self):
        """Test that n_paths=0 uses maximum unique paths"""
        n_steps = 50
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0
        expected_max = 200 - n_steps + 1  # 151

        with self.assertLogs(
            "crypto.instruments.bitcoin_perpetual_historical", level="INFO"
        ) as log_context:
            self.underlier.simulate_bootstrap(
                n_paths=0,
                time_horizon=time_horizon,
                target_initial_spot=None,
            )

            self.assertTrue(
                any(
                    f"using max unique paths: {expected_max:,}" in msg.lower()
                    for msg in log_context.output
                ),
            )

        self.assertEqual(self.underlier.spot.shape[0], expected_max)

    def test_capping_when_exceeds_max(self):
        """Test that n_paths is capped when it exceeds max unique paths"""
        n_steps = 100
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0
        expected_max = 200 - n_steps + 1  # 101
        requested_paths = 5000

        with self.assertLogs(
            "crypto.instruments.bitcoin_perpetual_historical", level="WARNING"
        ) as log_context:
            self.underlier.simulate_bootstrap(
                n_paths=requested_paths,
                time_horizon=time_horizon,
                target_initial_spot=None,
            )

            # Check warning message
            self.assertTrue(
                any(
                    f"requested {requested_paths:,} paths exceeds" in msg.lower()
                    for msg in log_context.output
                ),
                "Expected warning about exceeding max paths",
            )
            self.assertTrue(
                any(
                    f"using {expected_max:,} paths instead" in msg.lower()
                    for msg in log_context.output
                ),
                "Expected message about using max paths",
            )

        # Verify capped to max
        self.assertEqual(
            self.underlier.spot.shape[0],
            expected_max,
            f"Expected paths capped to {expected_max}",
        )

    def test_normal_operation_within_limit(self):
        """Test that valid n_paths works normally without warnings"""
        n_steps = 100
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0
        requested_paths = 50  # Well within limit of 101

        # Should not log warning
        with self.assertLogs(
            "crypto.instruments.bitcoin_perpetual_historical", level="INFO"
        ) as log_context:
            self.underlier.simulate_bootstrap(
                n_paths=requested_paths,
                time_horizon=time_horizon,
                target_initial_spot=None,
            )

            # Should NOT have the auto-max or capping messages
            self.assertFalse(
                any(
                    "using max unique paths" in msg.lower()
                    for msg in log_context.output
                ),
            )
            self.assertFalse(
                any("exceeds maximum" in msg.lower() for msg in log_context.output),
            )

        # Verify correct number of paths
        self.assertEqual(self.underlier.spot.shape[0], requested_paths)

    def test_max_paths_calculation(self):
        """Test that max_unique_paths calculation is correct"""
        test_cases = [
            (200, 50, 151),  # 200 records, 50 steps -> 151 unique paths
            (200, 100, 101),  # 200 records, 100 steps -> 101 unique paths
            (200, 199, 2),  # 200 records, 199 steps -> 2 unique paths
            (200, 200, 1),  # 200 records, 200 steps -> 1 unique path
        ]

        for n_points, n_steps, expected_max in test_cases:
            with self.subTest(n_points=n_points, n_steps=n_steps):
                loader = MockDataLoader(n_points=n_points)
                underlier = BitcoinPerpetualHistorical(
                    data_loader=loader, dt=8.0 / 24.0 / 365.0, cost=0.0004
                )

                time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0

                with self.assertLogs(
                    "crypto.instruments.bitcoin_perpetual_historical", level="INFO"
                ):
                    underlier.simulate_bootstrap(
                        n_paths=None,
                        time_horizon=time_horizon,
                        target_initial_spot=None,
                    )

                self.assertEqual(
                    underlier.spot.shape[0],
                    expected_max,
                    f"Expected {expected_max} paths for {n_points} records and {n_steps} steps",
                )

    def test_spot_instrument_same_behavior(self):
        """Test that BitcoinSpotHistorical has same optimization"""
        underlier = BitcoinSpotHistorical(
            data_loader=self.mock_loader, dt=8.0 / 24.0 / 365.0, cost=0.0004
        )

        n_steps = 100
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0
        expected_max = 200 - n_steps + 1  # 101

        with self.assertLogs(
            "crypto.instruments.bitcoin_perpetual_historical", level="INFO"
        ):
            underlier.simulate_bootstrap(
                n_paths=None,
                time_horizon=time_horizon,
                target_initial_spot=None,
            )

        self.assertEqual(underlier.spot.shape[0], expected_max)

    def test_edge_case_single_path(self):
        """Test edge case where only 1 unique path is available"""
        loader = MockDataLoader(n_points=100)
        underlier = BitcoinPerpetualHistorical(
            data_loader=loader, dt=8.0 / 24.0 / 365.0, cost=0.0004
        )

        n_steps = 100  # Exactly matches data size
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0

        with self.assertLogs(
            "crypto.instruments.bitcoin_perpetual_historical", level="INFO"
        ):
            underlier.simulate_bootstrap(
                n_paths=None,
                time_horizon=time_horizon,
                target_initial_spot=None,
            )

        # Should get exactly 1 path
        self.assertEqual(underlier.spot.shape[0], 1)


class TestBackwardCompatibility(unittest.TestCase):
    """Ensure new feature doesn't break existing behavior"""

    def setUp(self):
        self.mock_loader = MockDataLoader(n_points=200)
        # dt should be in years: 8 hours = 8/24/365 years
        self.underlier = BitcoinPerpetualHistorical(
            data_loader=self.mock_loader, dt=8.0 / 24.0 / 365.0, cost=0.0004
        )

    def test_explicit_path_count_unchanged(self):
        """Test that explicit n_paths still works as before"""
        n_steps = 50
        time_horizon = (n_steps - 1) * (8.0 / 24.0) / 365.0

        for n_paths in [1, 10, 50, 100]:
            with self.subTest(n_paths=n_paths):
                self.underlier.simulate_bootstrap(
                    n_paths=n_paths,
                    time_horizon=time_horizon,
                    target_initial_spot=None,
                )

                self.assertEqual(
                    self.underlier.spot.shape[0],
                    n_paths,
                    f"Explicit n_paths={n_paths} should work as before",
                )


class TestConfigIntegration(unittest.TestCase):
    """Test that config-level auto (n_bootstrap_paths=0) works correctly"""

    def test_config_allows_zero_for_auto(self):
        """Test that BacktestConfig accepts 0 as auto"""
        from crypto.backtest.config import BacktestConfig

        # Should not raise ValueError
        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            strike=100000,
            maturity_days=30,
            model_path="dummy.pth",
            n_bootstrap_paths=0,  # Auto mode
        )

        self.assertEqual(config.n_bootstrap_paths, 0)

    def test_config_rejects_negative(self):
        """Test that BacktestConfig rejects negative values"""
        from crypto.backtest.config import BacktestConfig

        config = BacktestConfig(
            start_date="2023-01-01",
            end_date="2023-12-31",
            strike=100000,
            maturity_days=30,
            model_path="dummy.pth",
            n_bootstrap_paths=-1,
        )

        # Validation happens when validate() is called
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            config.validate()


if __name__ == "__main__":
    unittest.main()
