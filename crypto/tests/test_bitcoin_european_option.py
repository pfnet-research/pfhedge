#!/usr/bin/env python3
"""
Tests for Bitcoin European Option instrument.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import unittest
import torch
import numpy as np
from crypto.instruments import BitcoinPerpetualBrownian, BitcoinEuropeanOption, create_bitcoin_option_from_config


class TestBitcoinEuropeanOption(unittest.TestCase):
    """Test Bitcoin European Option functionality."""

    def setUp(self):
        """Set up test data."""
        torch.manual_seed(42)
        np.random.seed(42)

        # Create underlying
        self.btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.0, cost=0.001)
        self.btc.simulate(n_paths=10, time_horizon=30/365)

        # Create option
        self.option = BitcoinEuropeanOption(
            underlier=self.btc,
            strike=50000,
            maturity=30/365,
            call=True,
            cost=0.001
        )

    def test_initialization(self):
        """Test option initialization."""
        self.assertEqual(self.option.strike, 50000)
        self.assertAlmostEqual(self.option.maturity, 30/365, places=6)
        self.assertTrue(self.option.call)
        self.assertEqual(self.option.cost, 0.001)
        self.assertEqual(self.option.transaction_cost, 0.001)

    def test_payoff_calculation(self):
        """Test option payoff calculation."""
        payoffs = self.option.payoff()

        self.assertEqual(payoffs.shape, (self.btc.spot.shape[0],))
        self.assertTrue(torch.all(payoffs >= 0))  # Call payoffs never negative

        # Test that payoffs are reasonable
        final_prices = self.btc.spot[:, -1]
        expected_gross = torch.clamp(final_prices - self.option.strike, min=0)

        # With transaction costs, net payoff should be <= gross payoff
        self.assertTrue(torch.all(payoffs <= expected_gross))

    def test_payoff_without_transaction_costs(self):
        """Test payoff calculation without transaction costs."""
        option_no_cost = BitcoinEuropeanOption(
            underlier=self.btc,
            strike=50000,
            maturity=30/365,
            call=True,
            cost=0.0
        )

        payoffs = option_no_cost.payoff()
        final_prices = self.btc.spot[:, -1]
        expected = torch.clamp(final_prices - option_no_cost.strike, min=0)

        torch.testing.assert_close(payoffs, expected)

    def test_put_option(self):
        """Test put option payoff."""
        put_option = BitcoinEuropeanOption(
            underlier=self.btc,
            strike=50000,
            maturity=30/365,
            call=False,
            cost=0.0
        )

        payoffs = put_option.payoff()
        final_prices = self.btc.spot[:, -1]
        expected = torch.clamp(put_option.strike - final_prices, min=0)

        torch.testing.assert_close(payoffs, expected)

    def test_moneyness_calculation(self):
        """Test log-moneyness calculation."""
        moneyness = self.option.moneyness()

        expected_shape = self.btc.spot.shape
        self.assertEqual(moneyness.shape, expected_shape)

        # Test that moneyness is log(S/K)
        expected = torch.log(self.btc.spot / self.option.strike)
        torch.testing.assert_close(moneyness, expected)

    def test_time_to_maturity(self):
        """Test time to maturity calculation."""
        ttm = self.option.time_to_maturity()

        self.assertEqual(len(ttm), self.btc.spot.shape[1])
        self.assertTrue(torch.all(ttm >= 0))  # Never negative
        self.assertTrue(torch.all(ttm <= self.option.maturity))  # Never exceeds maturity

        # Should be decreasing
        for i in range(1, len(ttm)):
            self.assertLessEqual(ttm[i], ttm[i-1])

    def test_realized_volatility(self):
        """Test realized volatility calculation."""
        windows = [5, 10]
        realized_vol = self.option.realized_volatility(windows=windows)

        expected_shape = (self.btc.spot.shape[0], self.btc.spot.shape[1], len(windows))
        self.assertEqual(realized_vol.shape, expected_shape)

        # Check that volatilities are positive (where not NaN)
        valid_vol = realized_vol[~torch.isnan(realized_vol)]
        self.assertTrue(torch.all(valid_vol > 0))

    def test_deep_hedging_features(self):
        """Test deep hedging feature creation."""
        features = self.option.deep_hedging_features()

        expected_features = ['log_moneyness', 'time_to_maturity', 'realized_vol_5', 'realized_vol_10', 'realized_vol_20']
        self.assertEqual(set(features.keys()), set(expected_features))

        # Check shapes
        expected_shape = self.btc.spot.shape
        for feature_name, feature_tensor in features.items():
            self.assertEqual(feature_tensor.shape, expected_shape, f"Feature {feature_name} has wrong shape")

    def test_deep_hedging_features_selective(self):
        """Test selective feature creation."""
        features = self.option.deep_hedging_features(
            include_time=False,
            include_volatility=False
        )

        self.assertEqual(list(features.keys()), ['log_moneyness'])

    def test_black_scholes_delta(self):
        """Test Black-Scholes delta calculation."""
        delta = self.option.black_scholes_delta()

        self.assertEqual(delta.shape, self.btc.spot.shape)

        # Call delta should be between 0 and 1
        self.assertTrue(torch.all(delta >= 0))
        self.assertTrue(torch.all(delta <= 1))

        # No NaN or infinite values
        self.assertFalse(torch.any(torch.isnan(delta)))
        self.assertFalse(torch.any(torch.isinf(delta)))

    def test_black_scholes_delta_put(self):
        """Test Black-Scholes delta for put option."""
        put_option = BitcoinEuropeanOption(
            underlier=self.btc,
            strike=50000,
            maturity=30/365,
            call=False
        )

        delta = put_option.black_scholes_delta()

        # Put delta should be between -1 and 0
        self.assertTrue(torch.all(delta >= -1))
        self.assertTrue(torch.all(delta <= 0))

    def test_summary(self):
        """Test option summary generation."""
        summary = self.option.summary()

        required_keys = [
            'strike', 'maturity', 'call', 'cost', 'simulated',
            'n_paths', 'n_steps', 'payoff_mean', 'itm_ratio'
        ]

        for key in required_keys:
            self.assertIn(key, summary)

        self.assertTrue(summary['simulated'])
        self.assertEqual(summary['n_paths'], self.btc.spot.shape[0])
        self.assertEqual(summary['n_steps'], self.btc.spot.shape[1])
        self.assertGreaterEqual(summary['itm_ratio'], 0)
        self.assertLessEqual(summary['itm_ratio'], 1)

    def test_summary_without_simulation(self):
        """Test summary when underlying is not simulated."""
        btc_no_sim = BitcoinPerpetualBrownian()
        option_no_sim = BitcoinEuropeanOption(btc_no_sim, strike=50000, maturity=30/365)

        summary = option_no_sim.summary()
        self.assertFalse(summary['simulated'])
        self.assertIn('strike', summary)
        self.assertIn('maturity', summary)


class TestCreateBitcoinOptionFromConfig(unittest.TestCase):
    """Test configuration-based option creation."""

    def setUp(self):
        """Set up test configuration."""
        self.config = {
            'strike': 50000,
            'maturity_days': 30,
            'call': True,
            'cost': 0.001,
            'sigma': 0.8,
            'mu': 0.0,
            'n_paths': 50,
            'seed': 42
        }

    def test_create_from_config(self):
        """Test creating option from configuration."""
        option, summary = create_bitcoin_option_from_config(self.config)

        self.assertIsInstance(option, BitcoinEuropeanOption)
        self.assertEqual(option.strike, self.config['strike'])
        self.assertEqual(option.cost, self.config['cost'])
        self.assertTrue(option.call)

        # Check summary
        self.assertEqual(summary['n_paths'], self.config['n_paths'])
        self.assertTrue(summary['simulated'])
        self.assertEqual(summary['config'], self.config)

    def test_create_with_defaults(self):
        """Test creating option with default values."""
        minimal_config = {
            'strike': 45000,
            'maturity_days': 7
        }

        option, summary = create_bitcoin_option_from_config(minimal_config)

        self.assertEqual(option.strike, 45000)
        self.assertAlmostEqual(option.maturity, 7/365, places=6)
        self.assertTrue(option.call)  # Default
        self.assertEqual(option.cost, 0.0)  # Default

    def test_put_option_creation(self):
        """Test creating put option from config."""
        put_config = self.config.copy()
        put_config['call'] = False

        option, summary = create_bitcoin_option_from_config(put_config)

        self.assertFalse(option.call)
        self.assertFalse(summary['call'])


class TestIntegrationWithVisualization(unittest.TestCase):
    """Test integration with visualization utilities."""

    def setUp(self):
        """Set up test data."""
        torch.manual_seed(42)

        config = {
            'strike': 50000,
            'maturity_days': 14,
            'n_paths': 20,
            'seed': 42
        }

        self.option, self.summary = create_bitcoin_option_from_config(config)

    def test_option_analysis_compatibility(self):
        """Test that option works with our visualization utilities."""
        # Test that we can extract the data needed for option analysis

        # Should have underlying with spot prices
        self.assertTrue(hasattr(self.option.underlier, 'spot'))
        self.assertEqual(len(self.option.underlier.spot.shape), 2)

        # Should be able to calculate payoffs
        payoffs = self.option.payoff()
        self.assertGreater(len(payoffs), 0)

        # Should work with our analysis functions
        features = self.option.deep_hedging_features()
        self.assertIn('log_moneyness', features)
        self.assertIn('time_to_maturity', features)

    def test_hedging_performance_compatibility(self):
        """Test compatibility with hedging performance analysis."""
        # Calculate simple hedge
        hedge_ratio = 0.5
        initial_prices = self.option.underlier.spot[:, 0]
        final_prices = self.option.underlier.spot[:, -1]
        payoffs = self.option.payoff()

        hedge_pnl = hedge_ratio * (final_prices - initial_prices) - payoffs

        # Should have reasonable hedge PnL
        self.assertEqual(len(hedge_pnl), len(payoffs))
        self.assertFalse(torch.any(torch.isnan(hedge_pnl)))


if __name__ == "__main__":
    unittest.main()