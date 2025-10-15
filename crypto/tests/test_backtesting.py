"""Tests for backtesting framework."""

import pytest
import torch
import tempfile
import os
import pandas as pd
import numpy as np
from crypto.backtest.config import BacktestConfig
from crypto.backtest.backtester import Backtester
from crypto.backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_max_drawdown,
    calculate_cvar,
    calculate_var,
    calculate_win_rate,
    calculate_calmar_ratio,
    calculate_all_metrics,
    print_metrics,
)


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_create_basic_config(self):
        """Test creating a basic config with required parameters."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test_model.pth",
        )

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-01-31"
        assert config.strike == 50000
        assert config.maturity_days == 14
        assert config.model_path == "models/test_model.pth"

        # Check defaults
        assert config.call is True
        assert config.n_bootstrap_paths == 100
        assert config.transaction_cost == 0.0005
        assert config.dt_hours == 8.0
        assert config.data_dir == "sample_data"
        assert config.output_dir == "backtest_results"

    def test_create_config_with_custom_values(self):
        """Test creating config with custom values."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-02-01",
            strike=60000,
            maturity_days=30,
            model_path="models/custom.pth",
            call=False,
            n_bootstrap_paths=200,
            transaction_cost=0.001,
            dt_hours=4.0,
            data_dir="custom_data",
            output_dir="custom_results",
        )

        assert config.call is False
        assert config.n_bootstrap_paths == 200
        assert config.transaction_cost == 0.001
        assert config.dt_hours == 4.0
        assert config.data_dir == "custom_data"
        assert config.output_dir == "custom_results"

    def test_validate_valid_config(self):
        """Test validation passes for valid config."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        config.validate()  # Should not raise

    def test_validate_invalid_date_format(self):
        """Test validation fails for invalid date format."""
        config = BacktestConfig(
            start_date="01-01-2024",  # Wrong format
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        with pytest.raises(ValueError, match="must be in YYYY-MM-DD format"):
            config.validate()

    def test_validate_end_before_start(self):
        """Test validation fails when end_date is before start_date."""
        config = BacktestConfig(
            start_date="2024-01-31",
            end_date="2024-01-01",  # Before start
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        with pytest.raises(ValueError, match="must be after start_date"):
            config.validate()

    def test_validate_negative_strike(self):
        """Test validation fails for negative strike."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=-1000,  # Invalid
            maturity_days=14,
            model_path="models/test.pth",
        )

        with pytest.raises(ValueError, match="strike must be positive"):
            config.validate()

    def test_validate_zero_maturity(self):
        """Test validation fails for zero maturity."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=0,  # Invalid
            model_path="models/test.pth",
        )

        with pytest.raises(ValueError, match="maturity_days must be positive"):
            config.validate()

    def test_validate_negative_bootstrap_paths(self):
        """Test validation fails for negative n_bootstrap_paths."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            n_bootstrap_paths=-10,  # Invalid
        )

        with pytest.raises(ValueError, match="n_bootstrap_paths must be positive"):
            config.validate()

    def test_validate_negative_transaction_cost(self):
        """Test validation fails for negative transaction cost."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            transaction_cost=-0.001,  # Invalid
        )

        with pytest.raises(ValueError, match="transaction_cost must be non-negative"):
            config.validate()

    def test_validate_too_high_transaction_cost(self):
        """Test validation fails for unreasonably high transaction cost."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            transaction_cost=0.15,  # 15% seems wrong
        )

        with pytest.raises(ValueError, match="transaction_cost seems too high"):
            config.validate()

    def test_validate_zero_dt_hours(self):
        """Test validation fails for zero dt_hours."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=0,  # Invalid
        )

        with pytest.raises(ValueError, match="dt_hours must be positive"):
            config.validate()

    def test_validate_dt_hours_too_large(self):
        """Test validation fails for dt_hours > 24."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=48,  # Invalid
        )

        with pytest.raises(ValueError, match="dt_hours must be <= 24"):
            config.validate()

    def test_to_dict(self):
        """Test converting config to dictionary."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            call=False,
            n_bootstrap_paths=200,
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["start_date"] == "2024-01-01"
        assert config_dict["end_date"] == "2024-01-31"
        assert config_dict["strike"] == 50000
        assert config_dict["maturity_days"] == 14
        assert config_dict["model_path"] == "models/test.pth"
        assert config_dict["call"] is False
        assert config_dict["n_bootstrap_paths"] == 200

    def test_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            "start_date": "2024-01-01",
            "end_date": "2024-01-31",
            "strike": 50000,
            "maturity_days": 14,
            "model_path": "models/test.pth",
            "call": False,
            "n_bootstrap_paths": 200,
        }

        config = BacktestConfig.from_dict(config_dict)

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-01-31"
        assert config.strike == 50000
        assert config.maturity_days == 14
        assert config.model_path == "models/test.pth"
        assert config.call is False
        assert config.n_bootstrap_paths == 200

    def test_to_dict_from_dict_roundtrip(self):
        """Test that to_dict and from_dict are inverse operations."""
        original = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            transaction_cost=0.001,
            dt_hours=4.0,
        )

        config_dict = original.to_dict()
        reconstructed = BacktestConfig.from_dict(config_dict)

        assert reconstructed.start_date == original.start_date
        assert reconstructed.end_date == original.end_date
        assert reconstructed.strike == original.strike
        assert reconstructed.maturity_days == original.maturity_days
        assert reconstructed.model_path == original.model_path
        assert reconstructed.transaction_cost == original.transaction_cost
        assert reconstructed.dt_hours == original.dt_hours

    def test_dt_property(self):
        """Test dt property converts hours to years correctly."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=8.0,
        )

        expected_dt = 8.0 / 24 / 365
        assert abs(config.dt - expected_dt) < 1e-10

    def test_repr(self):
        """Test string representation."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        repr_str = repr(config)

        assert "2024-01-01" in repr_str
        assert "2024-01-31" in repr_str
        assert "50,000" in repr_str
        assert "14d" in repr_str
        assert "Call" in repr_str
        assert "models/test.pth" in repr_str

    def test_repr_put_option(self):
        """Test string representation for put option."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            call=False,
        )

        repr_str = repr(config)
        assert "Put" in repr_str


class TestMetrics:
    """Tests for metrics module."""

    def test_sharpe_ratio_basic(self):
        """Test basic Sharpe ratio calculation."""
        pnl = torch.tensor([100.0, 150.0, 80.0, 120.0, 110.0])
        sharpe = calculate_sharpe_ratio(pnl)

        # Manual calculation
        mean = pnl.mean().item()
        std = pnl.std().item()
        expected_sharpe = mean / std

        assert abs(sharpe - expected_sharpe) < 1e-6

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        pnl = torch.tensor([100.0, 100.0, 100.0, 100.0])
        sharpe = calculate_sharpe_ratio(pnl)
        assert sharpe == 0.0

    def test_sharpe_ratio_2d_input(self):
        """Test Sharpe ratio with 2D cumulative PnL."""
        cum_pnl = torch.tensor(
            [
                [0.0, 10.0, 20.0, 30.0, 100.0],
                [0.0, -5.0, 10.0, 20.0, 150.0],
                [0.0, 5.0, 15.0, 25.0, 80.0],
            ]
        )
        sharpe = calculate_sharpe_ratio(cum_pnl)

        # Should use final values
        final_pnl = cum_pnl[:, -1]
        expected_sharpe = calculate_sharpe_ratio(final_pnl)

        assert abs(sharpe - expected_sharpe) < 1e-6

    def test_sortino_ratio_basic(self):
        """Test basic Sortino ratio calculation."""
        pnl = torch.tensor([100.0, 150.0, -50.0, 120.0, -20.0])
        sortino = calculate_sortino_ratio(pnl)

        # Manual calculation
        mean = pnl.mean().item()
        downside = torch.clamp(pnl - 0.0, max=0.0)
        downside_dev = torch.sqrt(torch.mean(downside ** 2)).item()
        expected_sortino = mean / downside_dev

        assert abs(sortino - expected_sortino) < 1e-6

    def test_sortino_ratio_no_losses(self):
        """Test Sortino ratio with no losses."""
        pnl = torch.tensor([100.0, 150.0, 200.0, 120.0])
        sortino = calculate_sortino_ratio(pnl)
        assert sortino == 0.0  # No downside deviation

    def test_max_drawdown_simple(self):
        """Test max drawdown with known sequence."""
        cum_pnl = torch.tensor([[0.0, 10.0, 15.0, 8.0, 12.0, 5.0]])
        max_dd = calculate_max_drawdown(cum_pnl)

        # Max is 15, then drops to 5, so max drawdown is 10
        assert abs(max_dd - 10.0) < 1e-6

    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown when PnL only increases."""
        cum_pnl = torch.tensor([[0.0, 10.0, 20.0, 30.0, 40.0]])
        max_dd = calculate_max_drawdown(cum_pnl)
        assert max_dd == 0.0

    def test_max_drawdown_1d_input(self):
        """Test max drawdown with 1D input."""
        cum_pnl = torch.tensor([0.0, 10.0, 15.0, 8.0, 12.0, 5.0])
        max_dd = calculate_max_drawdown(cum_pnl)
        assert abs(max_dd - 10.0) < 1e-6

    def test_max_drawdown_multiple_paths(self):
        """Test max drawdown averages across paths."""
        cum_pnl = torch.tensor(
            [[0.0, 10.0, 15.0, 8.0], [0.0, 5.0, 10.0, 3.0]]  # max dd = 7  # max dd = 7
        )
        max_dd = calculate_max_drawdown(cum_pnl)
        assert abs(max_dd - 7.0) < 1e-6

    def test_cvar_basic(self):
        """Test CVaR calculation."""
        torch.manual_seed(42)
        pnl = torch.randn(1000) * 100
        cvar_5 = calculate_cvar(pnl, alpha=0.05)

        # CVaR should be in the left tail
        sorted_pnl = torch.sort(pnl)[0]
        n_tail = int(0.05 * 1000)
        expected_cvar = sorted_pnl[:n_tail].mean().item()

        assert abs(cvar_5 - expected_cvar) < 1e-4

    def test_cvar_alpha_levels(self):
        """Test CVaR at different alpha levels."""
        torch.manual_seed(42)
        pnl = torch.randn(1000) * 100

        cvar_1 = calculate_cvar(pnl, alpha=0.01)
        cvar_5 = calculate_cvar(pnl, alpha=0.05)
        cvar_10 = calculate_cvar(pnl, alpha=0.10)

        # Lower alpha (more extreme) should give more extreme CVaR
        assert cvar_1 < cvar_5 < cvar_10

    def test_var_basic(self):
        """Test VaR calculation."""
        torch.manual_seed(42)
        pnl = torch.randn(1000) * 100
        var_5 = calculate_var(pnl, alpha=0.05)

        # VaR should be the 5th percentile
        expected_var = torch.quantile(pnl, 0.05).item()

        assert abs(var_5 - expected_var) < 1e-4

    def test_var_alpha_levels(self):
        """Test VaR at different alpha levels."""
        torch.manual_seed(42)
        pnl = torch.randn(1000) * 100

        var_1 = calculate_var(pnl, alpha=0.01)
        var_5 = calculate_var(pnl, alpha=0.05)
        var_10 = calculate_var(pnl, alpha=0.10)

        # Lower alpha (more extreme) should give more extreme VaR
        assert var_1 < var_5 < var_10

    def test_win_rate_basic(self):
        """Test win rate calculation."""
        pnl = torch.tensor([100.0, -50.0, 30.0, 80.0, -20.0, 60.0])
        win_rate = calculate_win_rate(pnl)

        # 4 out of 6 are positive
        assert abs(win_rate - 4 / 6) < 1e-6

    def test_win_rate_all_wins(self):
        """Test win rate with all positive."""
        pnl = torch.tensor([100.0, 50.0, 30.0, 80.0])
        win_rate = calculate_win_rate(pnl)
        assert win_rate == 1.0

    def test_win_rate_all_losses(self):
        """Test win rate with all negative."""
        pnl = torch.tensor([-100.0, -50.0, -30.0, -80.0])
        win_rate = calculate_win_rate(pnl)
        assert win_rate == 0.0

    def test_win_rate_2d_input(self):
        """Test win rate with 2D cumulative PnL."""
        cum_pnl = torch.tensor(
            [
                [0.0, 10.0, 20.0, 100.0],
                [0.0, -5.0, -10.0, -50.0],
                [0.0, 5.0, 10.0, 30.0],
            ]
        )
        win_rate = calculate_win_rate(cum_pnl)

        # 2 out of 3 final values are positive
        assert abs(win_rate - 2 / 3) < 1e-6

    def test_calmar_ratio_basic(self):
        """Test Calmar ratio calculation."""
        cum_pnl = torch.tensor(
            [[0.0, 10.0, 15.0, 8.0, 20.0], [0.0, 5.0, 10.0, 3.0, 18.0]]
        )
        calmar = calculate_calmar_ratio(cum_pnl)

        # Mean final return is (20 + 18) / 2 = 19
        # Max drawdown is (15-8 + 10-3) / 2 = 7
        expected_calmar = 19.0 / 7.0

        assert abs(calmar - expected_calmar) < 1e-6

    def test_calmar_ratio_requires_2d(self):
        """Test Calmar ratio requires cumulative PnL."""
        pnl = torch.tensor([100.0, 150.0, 80.0])
        with pytest.raises(ValueError, match="requires cumulative PnL"):
            calculate_calmar_ratio(pnl)

    def test_calmar_ratio_zero_drawdown(self):
        """Test Calmar ratio with no drawdown."""
        cum_pnl = torch.tensor([[0.0, 10.0, 20.0, 30.0]])
        calmar = calculate_calmar_ratio(cum_pnl)
        assert calmar == 0.0

    def test_calculate_all_metrics_basic(self):
        """Test calculating all metrics at once."""
        pnl = torch.randn(1000) * 100
        cum_pnl = torch.randn(1000, 50).cumsum(dim=1)

        metrics = calculate_all_metrics(pnl, cum_pnl)

        # Check all expected keys exist
        assert "mean" in metrics
        assert "std" in metrics
        assert "min" in metrics
        assert "max" in metrics
        assert "median" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "cvar_95" in metrics
        assert "var_95" in metrics
        assert "win_rate" in metrics
        assert "max_drawdown" in metrics
        assert "calmar_ratio" in metrics

    def test_calculate_all_metrics_without_cumulative(self):
        """Test calculating metrics without cumulative PnL."""
        pnl = torch.randn(1000) * 100

        metrics = calculate_all_metrics(pnl)

        # Check basic metrics exist
        assert "mean" in metrics
        assert "sharpe_ratio" in metrics

        # Check cumulative-dependent metrics don't exist
        assert "max_drawdown" not in metrics
        assert "calmar_ratio" not in metrics

    def test_calculate_all_metrics_custom_alpha(self):
        """Test calculating metrics with custom alpha."""
        pnl = torch.randn(1000) * 100

        metrics = calculate_all_metrics(pnl, alpha_cvar=0.01, alpha_var=0.01)

        # Check correct alpha levels used
        assert "cvar_99" in metrics
        assert "var_99" in metrics

    def test_print_metrics_runs(self):
        """Test print_metrics doesn't crash."""
        pnl = torch.randn(100) * 100
        cum_pnl = torch.randn(100, 50).cumsum(dim=1)

        metrics = calculate_all_metrics(pnl, cum_pnl)

        # Should not raise
        import io
        import sys

        captured_output = io.StringIO()
        sys.stdout = captured_output

        try:
            print_metrics(metrics, name="Test Strategy")
            output = captured_output.getvalue()

            # Check some expected content
            assert "Test Strategy" in output
            assert "Sharpe Ratio" in output
            assert "Win Rate" in output
        finally:
            sys.stdout = sys.__stdout__


class TestBacktester:
    """Tests for Backtester class."""

    @staticmethod
    def create_dummy_checkpoint(path: str):
        """Helper to create a dummy model checkpoint for testing."""
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        # Create a simple model
        model = create_deep_hedger(
            n_layers=2,
            n_units=32,
            risk_measure="expected_shortfall",
            risk_param=0.5,
            features=["log_moneyness", "time_to_maturity", "volatility", "prev_hedge"],
        )

        # Save checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_layers": 2,
                "n_units": 32,
                "criterion": "expected_shortfall",
                "risk_param": 0.5,
                "features": [
                    "log_moneyness",
                    "time_to_maturity",
                    "volatility",
                    "prev_hedge",
                ],
            },
        }
        torch.save(checkpoint, path)

    def test_create_backtester(self):
        """Test creating a Backtester instance."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        backtester = Backtester(config)

        assert backtester.config == config
        assert backtester.model is None
        assert backtester.data_loader is None
        assert backtester.option is None

    def test_load_model_success(self):
        """Test successful model loading."""
        # Create temporary checkpoint
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create dummy checkpoint
            self.create_dummy_checkpoint(temp_path)

            # Create config and backtester
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            # Load model
            model = backtester.load_model()

            # Verify model loaded
            assert model is not None
            assert backtester.model is model
            assert not model.training  # Should be in eval mode

        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_file_not_found(self):
        """Test that load_model raises FileNotFoundError for missing file."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="nonexistent/model.pth",
        )
        backtester = Backtester(config)

        with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
            backtester.load_model()

    def test_load_model_missing_state_dict(self):
        """Test that load_model raises KeyError for missing model_state_dict."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint without model_state_dict
            checkpoint = {
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "criterion": "expected_shortfall",
                    "risk_param": 0.5,
                    "features": ["log_moneyness"],
                }
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            with pytest.raises(KeyError, match="model_state_dict"):
                backtester.load_model()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_missing_config(self):
        """Test that load_model raises KeyError for missing model_config."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint without model_config
            from crypto.strategies.deep_hedge_utils import create_deep_hedger

            model = create_deep_hedger(n_layers=2, n_units=32)

            checkpoint = {"model_state_dict": model.state_dict()}
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            with pytest.raises(KeyError, match="model_config"):
                backtester.load_model()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_missing_config_key(self):
        """Test that load_model raises KeyError for missing config keys."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint with incomplete model_config
            from crypto.strategies.deep_hedge_utils import create_deep_hedger

            model = create_deep_hedger(n_layers=2, n_units=32)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_layers": 2,
                    # Missing n_units, features, criterion, risk_param
                },
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            with pytest.raises(KeyError, match="Model config missing required key"):
                backtester.load_model()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_with_device(self):
        """Test loading model to specific device."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            self.create_dummy_checkpoint(temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            # Load to CPU explicitly
            model = backtester.load_model(device="cpu")
            assert model is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_backward_compat_risk_measure(self):
        """Test backward compatibility with 'risk_measure' key."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint with 'risk_measure' instead of 'criterion'
            from crypto.strategies.deep_hedge_utils import create_deep_hedger

            model = create_deep_hedger(n_layers=2, n_units=32)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "risk_measure": "expected_shortfall",  # Old name
                    "risk_param": 0.5,
                    "features": ["log_moneyness", "time_to_maturity"],
                },
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            # Should load successfully
            model = backtester.load_model()
            assert model is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_missing_features_fallback(self):
        """Test default features fallback when 'features' key missing."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint without 'features'
            from crypto.strategies.deep_hedge_utils import create_deep_hedger

            model = create_deep_hedger(n_layers=2, n_units=32)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "criterion": "expected_shortfall",
                    "risk_param": 0.5,
                    # Missing 'features' - should fallback to DEFAULT_FEATURES
                },
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            # Should load successfully with default features
            model = backtester.load_model()
            assert model is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_missing_criterion_and_risk_measure(self):
        """Test error when both 'criterion' and 'risk_measure' missing."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            from crypto.strategies.deep_hedge_utils import create_deep_hedger

            model = create_deep_hedger(n_layers=2, n_units=32)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "risk_param": 0.5,
                    "features": ["log_moneyness"]
                    # Missing both 'criterion' and 'risk_measure'
                },
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            with pytest.raises(KeyError, match="criterion.*risk_measure"):
                backtester.load_model()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_load_model_state_dict_mismatch(self):
        """Test error handling for state dict mismatch."""
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint with different architecture than config
            from crypto.strategies.deep_hedge_utils import create_deep_hedger

            # Create model with 3 layers
            model_3layers = create_deep_hedger(n_layers=3, n_units=32)

            # But save config for 2 layers (mismatch)
            checkpoint = {
                "model_state_dict": model_3layers.state_dict(),
                "model_config": {
                    "n_layers": 2,  # Mismatch!
                    "n_units": 32,
                    "criterion": "expected_shortfall",
                    "risk_param": 0.5,
                    "features": ["log_moneyness", "time_to_maturity"],
                },
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            with pytest.raises(RuntimeError, match="State dict mismatch"):
                backtester.load_model()

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @staticmethod
    def create_dummy_parquet_data(data_dir: str, n_days: int = 10):
        """Helper to create dummy parquet data files for testing.

        Args:
            data_dir: Directory to create data files in
            n_days: Number of days of data to generate
        """
        from datetime import datetime, timedelta

        os.makedirs(data_dir, exist_ok=True)

        # Create perpetual data spanning n_days with 5-minute intervals
        records_per_day = 288  # 24 * 60 / 5
        n_records = n_days * records_per_day
        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(minutes=5 * i) for i in range(n_records)]

        perpetual_data = {
            "timestamp": timestamps,
            "last_price": np.random.uniform(45000, 55000, n_records),
            "bid_price": np.random.uniform(44900, 54900, n_records),
            "ask_price": np.random.uniform(45100, 55100, n_records),
            "funding_8h": np.random.uniform(-0.001, 0.001, n_records),
            "index_price": np.random.uniform(45000, 55000, n_records),
        }
        perpetual_df = pd.DataFrame(perpetual_data)
        perpetual_path = os.path.join(data_dir, "btc_perpetual.parquet")
        perpetual_df.to_parquet(perpetual_path)

        # Create options data
        n_options = min(100, n_records // 10)
        options_data = {
            "timestamp": timestamps[:n_options],
            "strike": [50000] * n_options,
            "expiration": [datetime(2024, 1, 15).timestamp() * 1000] * n_options,
            "option_type": ["call"] * (n_options // 2) + ["put"] * (n_options // 2),
            "last_price": np.random.uniform(100, 5000, n_options),
            "bid_price": np.random.uniform(100, 4900, n_options),
            "ask_price": np.random.uniform(100, 5100, n_options),
            "mark_iv": np.random.uniform(0.5, 1.0, n_options),
        }
        options_df = pd.DataFrame(options_data)
        options_path = os.path.join(data_dir, "btc_options.parquet")
        options_df.to_parquet(options_path)

    def test_load_data_success(self):
        """Test successful data loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy parquet files
            self.create_dummy_parquet_data(temp_dir)

            # Create config and backtester
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            # Load data
            loader = backtester.load_data()

            # Verify loader returned and stored
            assert loader is not None
            assert backtester.data_loader is loader

            # Verify data was actually loaded
            assert loader.perpetual_data is not None
            assert not loader.perpetual_data.empty
            assert len(loader.perpetual_data) > 0

            # Verify data columns exist
            assert "timestamp" in loader.perpetual_data.columns
            assert "last_price" in loader.perpetual_data.columns

    def test_load_data_directory_not_found(self):
        """Test error when data directory doesn't exist."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            data_dir="nonexistent_directory_12345",
        )
        backtester = Backtester(config)

        with pytest.raises(FileNotFoundError, match="Data directory not found"):
            backtester.load_data()

    def test_load_data_no_perpetual_files(self):
        """Test error when no perpetual data files found."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create empty directory (no parquet files)
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            with pytest.raises(
                FileNotFoundError, match="No perpetual data found.*perpetual.*parquet"
            ):
                backtester.load_data()

    def test_load_data_missing_options_is_ok(self):
        """Test that missing options data is handled gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create only perpetual data (no options)
            import pandas as pd
            import numpy as np
            from datetime import datetime, timedelta

            n_records = 100
            start_time = datetime(2024, 1, 1)
            timestamps = [
                start_time + timedelta(minutes=5 * i) for i in range(n_records)
            ]

            perpetual_data = {
                "timestamp": timestamps,
                "last_price": np.random.uniform(45000, 55000, n_records),
                "bid_price": np.random.uniform(44900, 54900, n_records),
                "ask_price": np.random.uniform(45100, 55100, n_records),
            }
            perpetual_df = pd.DataFrame(perpetual_data)
            perpetual_path = os.path.join(temp_dir, "btc_perpetual.parquet")
            perpetual_df.to_parquet(perpetual_path)

            # Create config and backtester
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            # Should load successfully despite missing options data
            loader = backtester.load_data()

            assert loader is not None
            assert loader.perpetual_data is not None
            assert not loader.perpetual_data.empty

    def test_load_data_verifies_real_market_data(self):
        """Test that loaded data has expected properties of real market data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-10",  # Within generated range
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            loader = backtester.load_data()

            # Verify perpetual data properties
            perp_df = loader.perpetual_data
            assert len(perp_df) > 0

            # Verify timestamp column exists and is datetime
            assert "timestamp" in perp_df.columns
            assert pd.api.types.is_datetime64_any_dtype(perp_df["timestamp"])

            # Verify price columns exist and are numeric
            assert "last_price" in perp_df.columns
            assert pd.api.types.is_numeric_dtype(perp_df["last_price"])

            # Verify prices are in reasonable range (Bitcoin)
            assert perp_df["last_price"].min() > 1000  # Sanity check
            assert perp_df["last_price"].max() < 1000000  # Sanity check

            # Verify summary statistics are generated
            summary = loader.summary()
            assert "perpetual" in summary
            assert summary["perpetual"]["records"] > 0
            assert "date_range" in summary["perpetual"]
            assert "price_range" in summary["perpetual"]

    def test_load_data_with_resampling(self):
        """Test data loading with resampling to different frequencies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=5)

            # Test with integer hours (8 hours)
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
                dt_hours=8.0,  # Should resample to 8H
            )
            backtester = Backtester(config)
            loader = backtester.load_data()

            # Verify resampling occurred
            perp_df = loader.perpetual_data
            assert len(perp_df) > 0
            # With 8H resampling over 5 days: ~15 records (5 days * 24 hours / 8)
            assert len(perp_df) <= 20  # Allow some margin

    def test_load_data_with_non_integer_hours(self):
        """Test resampling with non-integer hours (converted to minutes)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=3)

            # Test with 0.5 hours = 30 minutes
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-03",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
                dt_hours=0.5,  # Should resample to 30T
            )
            backtester = Backtester(config)
            loader = backtester.load_data()

            perp_df = loader.perpetual_data
            assert len(perp_df) > 0
            # With 30T resampling over 3 days: ~97-144 records (depends on exact date boundaries)
            assert len(perp_df) >= 90  # Should have many records

    def test_load_data_date_filtering(self):
        """Test that date filtering works correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            # Filter to first 3 days only
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-03",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
                dt_hours=1.0,  # 1 hour resampling
            )
            backtester = Backtester(config)
            loader = backtester.load_data()

            perp_df = loader.perpetual_data
            assert len(perp_df) > 0

            # Verify date range
            min_date = perp_df["timestamp"].min()
            max_date = perp_df["timestamp"].max()

            assert min_date >= pd.to_datetime("2024-01-01")
            assert max_date <= pd.to_datetime("2024-01-04")  # Allow end of day

    def test_load_data_invalid_date_range(self):
        """Test error when requested date range has no data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=5)  # 2024-01-01 to 01-05

            # Request data from 2025 (out of range)
            config = BacktestConfig(
                start_date="2025-01-01",
                end_date="2025-01-31",
                strike=50000,
                maturity_days=14,
                model_path="models/test.pth",
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            with pytest.raises(ValueError, match="No data found in date range"):
                backtester.load_data()

    def test_create_bootstrap_option_success(self):
        """Test successful bootstrap option creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create data with enough records for bootstrap
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
                n_bootstrap_paths=10,
                dt_hours=1.0,
            )
            backtester = Backtester(config)

            # Load data first
            loader = backtester.load_data()

            # Create bootstrap option
            option = backtester.create_bootstrap_option(loader)

            # Verify option was created
            assert option is not None
            assert backtester.option is option

            # Verify paths were generated
            assert hasattr(option.underlier, "spot")
            assert option.underlier.spot.shape[0] == 10  # n_bootstrap_paths

            # Verify option properties
            assert option.strike == 50000
            assert option.call == True  # Default
            assert option.maturity == pytest.approx(3 / 365.0)

    def test_create_bootstrap_option_uses_self_data_loader(self):
        """Test that create_bootstrap_option can use self.data_loader."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
                n_bootstrap_paths=5,
            )
            backtester = Backtester(config)

            # Load data (stores in self.data_loader)
            backtester.load_data()

            # Create option without passing data_loader
            option = backtester.create_bootstrap_option()  # No argument

            # Should work and use self.data_loader
            assert option is not None
            assert option.underlier.spot.shape[0] == 5

    def test_create_bootstrap_option_no_data_loaded(self):
        """Test error when create_bootstrap_option called without loading data."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strike=50000,
            maturity_days=3,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        # Don't load data, call create_bootstrap_option directly
        with pytest.raises(ValueError, match="No data loaded"):
            backtester.create_bootstrap_option()

    def test_create_bootstrap_option_correct_time_steps(self):
        """Test that bootstrap paths have correct number of time steps."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            # 3 days maturity with 8-hour time steps
            # Expected steps: ceil(3/365 / (8/24/365)) + 1 = ceil(9) + 1 = 10
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-08",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
                n_bootstrap_paths=5,
                dt_hours=8.0,
            )
            backtester = Backtester(config)
            loader = backtester.load_data()
            option = backtester.create_bootstrap_option(loader)

            # Verify shape
            n_paths, n_steps = option.underlier.spot.shape
            assert n_paths == 5

            # Steps should match maturity / dt
            # 3 days / (8 hours) = 9 steps + 1 initial = 10
            assert n_steps >= 9  # At least 9 steps for 3 days

    def test_create_bootstrap_option_put_option(self):
        """Test creating a put option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=5)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-03",
                strike=50000,
                maturity_days=2,
                model_path="models/test.pth",
                data_dir=temp_dir,
                call=False,  # Put option
                n_bootstrap_paths=3,
            )
            backtester = Backtester(config)
            loader = backtester.load_data()
            option = backtester.create_bootstrap_option(loader)

            # Verify it's a put option
            assert option.call is False
            assert option.strike == 50000

    def test_run_deep_hedge_success(self):
        """Test successful deep hedge evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            # Create model checkpoint
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path=model_path,
                data_dir=temp_dir,
                n_bootstrap_paths=5,
            )
            backtester = Backtester(config)

            # Load model, data, create option
            model = backtester.load_model()
            loader = backtester.load_data()
            option = backtester.create_bootstrap_option(loader)

            # Run deep hedge
            pnl = backtester.run_deep_hedge(option, model)

            # Verify output
            assert pnl is not None
            assert pnl.shape[0] == 5  # n_bootstrap_paths
            assert pnl.shape[1] > 0  # n_steps
            assert isinstance(pnl, torch.Tensor)

    def test_run_deep_hedge_uses_stored_option_and_model(self):
        """Test that run_deep_hedge can use stored option and model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path=model_path,
                data_dir=temp_dir,
                n_bootstrap_paths=3,
            )
            backtester = Backtester(config)

            # Load and store
            backtester.load_model()
            backtester.load_data()
            backtester.create_bootstrap_option()

            # Run without passing arguments
            pnl = backtester.run_deep_hedge()  # Uses self.option and self.model

            # Verify
            assert pnl is not None
            assert pnl.shape[0] == 3

    def test_run_deep_hedge_no_model_loaded(self):
        """Test error when run_deep_hedge called without model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            # Create option but don't load model
            loader = backtester.load_data()
            option = backtester.create_bootstrap_option(loader)

            # Try to run deep hedge without model
            with pytest.raises(ValueError, match="No model loaded"):
                backtester.run_deep_hedge(option, None)

    def test_run_deep_hedge_no_option_created(self):
        """Test error when run_deep_hedge called without option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path=model_path,
                data_dir=temp_dir,
            )
            backtester = Backtester(config)

            # Load model but don't create option
            model = backtester.load_model()

            # Try to run deep hedge without option
            with pytest.raises(ValueError, match="No option provided"):
                backtester.run_deep_hedge(None, model)

    def test_run_deep_hedge_correct_pnl_shape(self):
        """Test that deep hedge PnL has correct shape matching option paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            n_paths = 7
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path=model_path,
                data_dir=temp_dir,
                n_bootstrap_paths=n_paths,
            )
            backtester = Backtester(config)

            backtester.load_model()
            backtester.load_data()
            backtester.create_bootstrap_option()

            pnl = backtester.run_deep_hedge()

            # Verify shape matches option
            option = backtester.option
            assert pnl.shape[0] == option.underlier.spot.shape[0]  # Same n_paths
            assert pnl.shape[1] == option.underlier.spot.shape[1]  # Same n_steps

    def test_run_bs_baseline_success(self):
        """Test successful BS baseline evaluation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create dummy data
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
                n_bootstrap_paths=5,
            )
            backtester = Backtester(config)

            # Load data and create option
            loader = backtester.load_data()
            option = backtester.create_bootstrap_option(loader)

            # Run BS baseline
            pnl = backtester.run_bs_baseline(option)

            # Verify output
            assert pnl is not None
            assert pnl.shape[0] == 5  # n_bootstrap_paths
            assert pnl.shape[1] > 0  # n_steps
            assert isinstance(pnl, torch.Tensor)

    def test_run_bs_baseline_uses_stored_option(self):
        """Test that run_bs_baseline can use stored option."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
                n_bootstrap_paths=3,
            )
            backtester = Backtester(config)

            # Load data and create option (stores in self.option)
            backtester.load_data()
            backtester.create_bootstrap_option()

            # Run without passing argument
            pnl = backtester.run_bs_baseline()  # Uses self.option

            # Verify
            assert pnl is not None
            assert pnl.shape[0] == 3

    def test_run_bs_baseline_no_option_created(self):
        """Test error when run_bs_baseline called without option."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-05",
            strike=50000,
            maturity_days=3,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        # Try to run BS baseline without option
        with pytest.raises(ValueError, match="No option provided"):
            backtester.run_bs_baseline(None)

    def test_run_bs_baseline_correct_pnl_shape(self):
        """Test that BS baseline PnL has correct shape matching option paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)

            n_paths = 7
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path="models/test.pth",
                data_dir=temp_dir,
                n_bootstrap_paths=n_paths,
            )
            backtester = Backtester(config)

            backtester.load_data()
            backtester.create_bootstrap_option()

            pnl = backtester.run_bs_baseline()

            # Verify shape matches option
            option = backtester.option
            assert pnl.shape[0] == option.underlier.spot.shape[0]  # Same n_paths
            assert pnl.shape[1] == option.underlier.spot.shape[1]  # Same n_steps

    def test_run_bs_baseline_matches_deep_hedge_shape(self):
        """Test that BS baseline and deep hedge produce same shape outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path=model_path,
                data_dir=temp_dir,
                n_bootstrap_paths=5,
            )
            backtester = Backtester(config)

            # Setup
            backtester.load_model()
            backtester.load_data()
            backtester.create_bootstrap_option()

            # Run both strategies
            deep_pnl = backtester.run_deep_hedge()
            bs_pnl = backtester.run_bs_baseline()

            # Verify they produce same shape outputs (for fair comparison)
            assert deep_pnl.shape == bs_pnl.shape

    def test_run_with_single_path(self):
        """Test that strategies work correctly with n_paths=1 (catches squeeze bugs)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=10)
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            # Use n_paths=1 to test shape handling
            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-05",
                strike=50000,
                maturity_days=3,
                model_path=model_path,
                data_dir=temp_dir,
                n_bootstrap_paths=1,  # Critical: test with single path
            )
            backtester = Backtester(config)

            # Setup
            backtester.load_model()
            backtester.load_data()
            backtester.create_bootstrap_option()

            # Run both strategies with n_paths=1
            deep_pnl = backtester.run_deep_hedge()
            bs_pnl = backtester.run_bs_baseline()

            # Verify shapes are preserved even with single path
            # Should be (1, n_steps), NOT (n_steps,)
            assert deep_pnl.ndim == 2, f"Expected 2D tensor, got {deep_pnl.ndim}D"
            assert (
                deep_pnl.shape[0] == 1
            ), f"Expected n_paths=1, got {deep_pnl.shape[0]}"
            assert bs_pnl.ndim == 2, f"Expected 2D tensor, got {bs_pnl.ndim}D"
            assert bs_pnl.shape[0] == 1, f"Expected n_paths=1, got {bs_pnl.shape[0]}"

            # Verify same shape for both strategies
            assert deep_pnl.shape == bs_pnl.shape

    def test_run_not_implemented(self):
        """Test that run raises NotImplementedError."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        with pytest.raises(NotImplementedError, match="Step 1.10"):
            backtester.run()

    def test_repr(self):
        """Test string representation."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        repr_str = repr(backtester)

        assert "Backtester" in repr_str
        assert "config=" in repr_str
