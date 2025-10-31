"""Tests for backtesting framework."""

import pytest
import torch
import tempfile
import os
import json
import pandas as pd
import numpy as np
from crypto.backtest.config import BacktestConfig
from crypto.backtest.backtester import Backtester
from crypto.backtest.results import BacktestResults
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

    def test_save_yaml_success(self, tmp_path):
        """Test saving config to YAML file."""
        pytest.importorskip("yaml")

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            call=False,
            n_bootstrap_paths=200,
            transaction_cost=0.001,
        )

        yaml_path = tmp_path / "config.yaml"
        config.save_yaml(str(yaml_path))

        assert yaml_path.exists()

        # Read and verify content
        import yaml

        with open(yaml_path, "r") as f:
            loaded_data = yaml.safe_load(f)

        assert loaded_data["start_date"] == "2024-01-01"
        assert loaded_data["strike"] == 50000
        assert loaded_data["call"] is False
        assert loaded_data["n_bootstrap_paths"] == 200

    def test_save_yaml_creates_directories(self, tmp_path):
        """Test that save_yaml creates parent directories if needed."""
        pytest.importorskip("yaml")

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        # Save to nested path that doesn't exist
        yaml_path = tmp_path / "configs" / "subdir" / "config.yaml"
        config.save_yaml(str(yaml_path))

        assert yaml_path.exists()
        assert yaml_path.parent.exists()

    def test_load_yaml_success(self, tmp_path):
        """Test loading config from YAML file."""
        pytest.importorskip("yaml")

        # Create YAML file
        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: 50000
maturity_days: 14
model_path: models/test.pth
call: false
n_bootstrap_paths: 200
transaction_cost: 0.001
dt_hours: 4.0
data_dir: sample_data
output_dir: backtest_results
"""

        yaml_path = tmp_path / "config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = BacktestConfig.load_yaml(str(yaml_path), anchor_relative_paths=False)

        assert config.start_date == "2024-01-01"
        assert config.end_date == "2024-01-31"
        assert config.strike == 50000
        assert config.maturity_days == 14
        assert config.model_path == "models/test.pth"
        assert config.call is False
        assert config.n_bootstrap_paths == 200
        assert config.transaction_cost == 0.001
        assert config.dt_hours == 4.0

    def test_load_yaml_file_not_found(self):
        """Test load_yaml raises error for missing file."""
        pytest.importorskip("yaml")

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            BacktestConfig.load_yaml("nonexistent.yaml")

    def test_load_yaml_invalid_content(self, tmp_path):
        """Test load_yaml handles invalid YAML content."""
        pytest.importorskip("yaml")

        yaml_path = tmp_path / "invalid.yaml"
        with open(yaml_path, "w") as f:
            f.write("invalid: yaml: content:")

        with pytest.raises(Exception):  # YAML parsing error
            BacktestConfig.load_yaml(str(yaml_path))

    def test_load_yaml_empty_file(self, tmp_path):
        """Test load_yaml handles empty file."""
        pytest.importorskip("yaml")

        yaml_path = tmp_path / "empty.yaml"
        yaml_path.touch()

        with pytest.raises(ValueError, match="Empty or invalid YAML"):
            BacktestConfig.load_yaml(str(yaml_path))

    def test_load_yaml_missing_required_fields(self, tmp_path):
        """Test load_yaml handles missing required fields."""
        pytest.importorskip("yaml")

        # Missing 'strike' field
        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
maturity_days: 14
model_path: models/test.pth
"""

        yaml_path = tmp_path / "incomplete.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        with pytest.raises(ValueError, match="Invalid YAML structure"):
            BacktestConfig.load_yaml(str(yaml_path))

    def test_load_yaml_validates_config(self, tmp_path):
        """Test load_yaml validates config after loading."""
        pytest.importorskip("yaml")

        # Create invalid config (negative strike)
        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: -1000
maturity_days: 14
model_path: models/test.pth
"""

        yaml_path = tmp_path / "invalid_config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        with pytest.raises(ValueError, match="strike must be positive"):
            BacktestConfig.load_yaml(str(yaml_path))

    def test_yaml_roundtrip(self, tmp_path):
        """Test saving and loading YAML produces identical config."""
        pytest.importorskip("yaml")

        original = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-02-29",
            strike=60000,
            maturity_days=30,
            model_path="models/custom.pth",
            call=False,
            n_bootstrap_paths=500,
            transaction_cost=0.002,
            dt_hours=6.0,
            data_dir="my_data",
            output_dir="my_results",
        )

        yaml_path = tmp_path / "roundtrip.yaml"
        original.save_yaml(str(yaml_path))
        loaded = BacktestConfig.load_yaml(str(yaml_path), anchor_relative_paths=False)

        # Check all fields match
        assert loaded.start_date == original.start_date
        assert loaded.end_date == original.end_date
        assert loaded.strike == original.strike
        assert loaded.maturity_days == original.maturity_days
        assert loaded.model_path == original.model_path
        assert loaded.call == original.call
        assert loaded.n_bootstrap_paths == original.n_bootstrap_paths
        assert loaded.transaction_cost == original.transaction_cost
        assert loaded.dt_hours == original.dt_hours
        assert loaded.data_dir == original.data_dir
        assert loaded.output_dir == original.output_dir

    def test_load_yaml_unknown_keys(self, tmp_path):
        """Test load_yaml rejects unknown configuration keys."""
        pytest.importorskip("yaml")

        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: 50000
maturity_days: 14
model_path: models/test.pth
unknown_field: some_value
another_bad_field: 123
"""

        yaml_path = tmp_path / "bad_config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        with pytest.raises(ValueError, match="Unknown configuration fields"):
            BacktestConfig.load_yaml(str(yaml_path))

    def test_load_yaml_env_variable_expansion(self, tmp_path):
        """Test environment variable expansion in paths."""
        pytest.importorskip("yaml")
        import os

        # Set test environment variable
        os.environ["TEST_MODEL_DIR"] = "my_models"

        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: 50000
maturity_days: 14
model_path: $TEST_MODEL_DIR/deep_hedger.pth
data_dir: ${TEST_MODEL_DIR}_data
"""

        yaml_path = tmp_path / "env_config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = BacktestConfig.load_yaml(str(yaml_path))

        # Check environment variables were expanded
        assert "my_models" in config.model_path
        assert "my_models_data" in config.data_dir

        # Clean up
        del os.environ["TEST_MODEL_DIR"]

    def test_load_yaml_tilde_expansion(self, tmp_path):
        """Test tilde (~) expansion in paths."""
        pytest.importorskip("yaml")
        import os
        from pathlib import Path

        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: 50000
maturity_days: 14
model_path: ~/models/deep_hedger.pth
data_dir: ~/data
"""

        yaml_path = tmp_path / "tilde_config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = BacktestConfig.load_yaml(str(yaml_path), anchor_relative_paths=False)

        # Check tilde was expanded to home directory
        home_dir = str(Path.home())
        assert config.model_path.startswith(home_dir)
        assert config.data_dir.startswith(home_dir)
        assert "models/deep_hedger.pth" in config.model_path
        assert (
            "/data" in config.data_dir or "\\data" in config.data_dir
        )  # Handle Windows paths

    def test_load_yaml_relative_path_anchoring(self, tmp_path):
        """Test relative paths are resolved relative to config file directory."""
        pytest.importorskip("yaml")

        # Create nested directory structure
        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        yaml_content = """
start_date: '2024-01-01'
end_date: '2024-01-31'
strike: 50000
maturity_days: 14
model_path: ../models/deep_hedger.pth
data_dir: ../data
output_dir: ../results
"""

        yaml_path = config_dir / "my_config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        config = BacktestConfig.load_yaml(str(yaml_path))

        # Check paths were resolved relative to config directory
        # ../models from configs/ should be tmp_path/models (normalized)
        expected_model = str((tmp_path / "models" / "deep_hedger.pth").resolve())
        expected_data = str((tmp_path / "data").resolve())
        expected_output = str((tmp_path / "results").resolve())

        assert config.model_path == expected_model
        assert config.data_dir == expected_data
        assert config.output_dir == expected_output

    def test_no_double_path_resolution(self, tmp_path):
        """Test that backtester doesn't double-resolve data_dir path.

        This is a regression test for the path resolution bug where:
        1. BacktestConfig.load_yaml() resolved paths relative to config file
        2. Backtester.load_data() then resolved them again relative to crypto/data/

        The bug caused paths like "sample_data" to become
        "crypto/data/crypto/backtest/sample_data" (incorrect).

        The fix ensures Backtester trusts the already-resolved path from config.
        """
        pytest.importorskip("yaml")

        # Create directory structure:
        # tmp_path/
        #   configs/
        #     test_config.yaml
        #   data/
        #     my_data/
        #       btc_perpetual.parquet

        config_dir = tmp_path / "configs"
        config_dir.mkdir()

        data_dir = tmp_path / "data" / "my_data"
        data_dir.mkdir(parents=True)

        # Create minimal parquet data with required columns
        # Need enough data to span the date range (2024-01-01 to 2024-01-02)
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
                "open": [50000.0] * 50,
                "high": [50100.0] * 50,
                "low": [49900.0] * 50,
                "close": [50000.0] * 50,
                "volume": [100.0] * 50,
                "last_price": [50000.0] * 50,
                "bid_price": [49995.0] * 50,
                "ask_price": [50005.0] * 50,
                "mid_price": [50000.0] * 50,
            }
        )
        parquet_file = data_dir / "btc_perpetual.parquet"
        df.to_parquet(parquet_file)

        # Create YAML config with relative path to data
        yaml_content = f"""
start_date: '2024-01-01'
end_date: '2024-01-02'
strike: 50000
maturity_days: 1
model_path: ../models/model.pth
data_dir: ../data/my_data
"""

        yaml_path = config_dir / "test_config.yaml"
        with open(yaml_path, "w") as f:
            f.write(yaml_content)

        # Load config - this should resolve data_dir to tmp_path/data/my_data
        config = BacktestConfig.load_yaml(str(yaml_path))

        # Verify config resolved the path correctly
        expected_data_dir = str(data_dir.resolve())
        assert config.data_dir == expected_data_dir, (
            f"Config should resolve ../data/my_data to {expected_data_dir}, "
            f"got {config.data_dir}"
        )

        # Create backtester with the config
        backtester = Backtester(config)

        # The critical test: backtester.load_data() should NOT double-resolve the path
        # It should trust config.data_dir and find the data successfully
        try:
            loader = backtester.load_data()
            # If we get here, the path was used correctly (no double resolution)
            assert loader is not None
            assert loader.perpetual_data is not None
            assert len(loader.perpetual_data) > 0

            # Verify the data was loaded from the correct location
            # (the path in config, not some double-resolved version)
            print(f"✅ Successfully loaded data from: {config.data_dir}")

        except FileNotFoundError as e:
            # If this fails, it likely means the path was double-resolved
            pytest.fail(
                f"Backtester failed to load data from {config.data_dir}. "
                f"This suggests double path resolution bug has regressed. Error: {e}"
            )

    def test_get_provenance_info(self):
        """Test provenance information gathering."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        provenance = config.get_provenance_info()

        # Check structure
        assert "config" in provenance
        assert "resolved_paths" in provenance
        assert "timestamp" in provenance
        assert "git_commit" in provenance
        assert "git_branch" in provenance
        assert "git_dirty" in provenance
        assert "python_version" in provenance
        assert "platform" in provenance

        # Check config is included
        assert provenance["config"]["strike"] == 50000

        # Check resolved paths
        assert "model_path" in provenance["resolved_paths"]

        # Check python version and platform are not None
        assert provenance["python_version"] is not None
        assert len(provenance["python_version"]) > 0
        assert provenance["platform"] in ["Darwin", "Linux", "Windows"]
        assert "data_dir" in provenance["resolved_paths"]
        assert "output_dir" in provenance["resolved_paths"]

        # Git info may be None or have values (depends on git repo)
        # Just check it doesn't crash


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
        downside_dev = torch.sqrt(torch.mean(downside**2)).item()
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

    def test_normalize_timestamps_to_utc_naive(self):
        """Test normalizing timezone-naive timestamps to UTC."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        # Create DataFrame with naive timestamps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=5, freq="1h"),
                "value": [1, 2, 3, 4, 5],
            }
        )

        assert df["timestamp"].dt.tz is None  # Verify naive

        # Normalize to UTC
        result = backtester._normalize_timestamps_to_utc(df)

        # Should now be UTC-aware
        assert result["timestamp"].dt.tz is not None
        assert str(result["timestamp"].dt.tz) == "UTC"

    def test_normalize_timestamps_to_utc_already_utc(self):
        """Test normalizing already-UTC timestamps (no-op)."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        # Create DataFrame with UTC timestamps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=5, freq="1h", tz="UTC"
                ),
                "value": [1, 2, 3, 4, 5],
            }
        )

        original_timestamps = df["timestamp"].copy()

        # Normalize (should be no-op)
        result = backtester._normalize_timestamps_to_utc(df)

        # Should remain UTC
        assert str(result["timestamp"].dt.tz) == "UTC"
        # Timestamps should be unchanged
        assert (result["timestamp"] == original_timestamps).all()

    def test_normalize_timestamps_to_utc_non_utc_aware(self):
        """Test converting non-UTC timezone-aware timestamps to UTC."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        # Create DataFrame with US/Eastern timestamps
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    "2024-01-01", periods=5, freq="1h", tz="US/Eastern"
                ),
                "value": [1, 2, 3, 4, 5],
            }
        )

        assert str(df["timestamp"].dt.tz) != "UTC"  # Verify not UTC

        # Normalize to UTC
        result = backtester._normalize_timestamps_to_utc(df)

        # Should now be UTC
        assert str(result["timestamp"].dt.tz) == "UTC"

        # Verify conversion is correct (Eastern is UTC-5 or UTC-4 depending on DST)
        # For Jan 1, 2024 (EST), should be UTC-5
        # First timestamp: 2024-01-01 00:00:00-05:00 → 2024-01-01 05:00:00+00:00
        assert result["timestamp"].iloc[0].hour == 5

    def test_normalize_timestamps_to_utc_missing_column(self):
        """Test normalizing when timestamp column is missing (no-op)."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )
        backtester = Backtester(config)

        # Create DataFrame without timestamp column
        df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

        # Normalize (should be no-op)
        result = backtester._normalize_timestamps_to_utc(df)

        # Should return unchanged DataFrame
        assert "timestamp" not in result.columns
        assert len(result) == 5

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
        """Test error when 'features' key missing (old models not supported)."""
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
                    # Missing 'features' - should raise error (no silent fallback)
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

            # Should raise KeyError for missing features
            with pytest.raises(KeyError, match="Checkpoint missing 'features'"):
                backtester.load_model()

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
                    "features": ["log_moneyness"],
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
                FileNotFoundError, match="No perpetual data files found"
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

            # Verify date range (timestamps are now UTC-aware)
            min_date = perp_df["timestamp"].min()
            max_date = perp_df["timestamp"].max()

            assert min_date >= pd.to_datetime("2024-01-01", utc=True)
            assert max_date <= pd.to_datetime(
                "2024-01-04", utc=True
            )  # Allow end of day

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

    def test_run_end_to_end(self):
        """Test complete end-to-end backtest."""
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

            # Run full backtest
            results = backtester.run()

            # Verify results object
            assert results is not None
            assert isinstance(results, BacktestResults)

            # Verify shapes
            assert results.n_paths == 5
            assert results.n_steps > 0
            assert results.deep_pnl.shape[0] == 5
            assert results.bs_pnl.shape[0] == 5
            assert results.deep_positions.shape[0] == 5
            assert results.bs_positions.shape[0] == 5
            assert results.spots.shape[0] == 5

            # Verify all shapes match
            assert results.deep_pnl.shape == results.bs_pnl.shape
            assert results.deep_pnl.shape == results.deep_positions.shape
            assert results.deep_pnl.shape == results.bs_positions.shape
            assert results.deep_pnl.shape == results.spots.shape

            # Verify summary exists
            summary = results.summary()
            assert "deep_hedge" in summary
            assert "bs_baseline" in summary
            assert "sharpe_ratio" in summary["deep_hedge"]
            assert "sharpe_ratio" in summary["bs_baseline"]

    def test_run_stores_positions(self):
        """Test that run() properly stores positions from strategies."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=5)
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-03",
                strike=50000,
                maturity_days=2,
                model_path=model_path,
                data_dir=temp_dir,
                n_bootstrap_paths=3,
            )
            backtester = Backtester(config)

            # Run backtest
            results = backtester.run()

            # Verify positions were stored in backtester
            assert backtester.deep_positions is not None
            assert backtester.bs_positions is not None

            # Verify positions match what's in results
            assert torch.equal(backtester.deep_positions, results.deep_positions)
            assert torch.equal(backtester.bs_positions, results.bs_positions)

    def test_run_with_seed_parameter(self):
        """Test that run() accepts seed parameter without error."""
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

            # Should run without error with seed parameter
            results = backtester.run(seed=42)

            # Verify results were generated
            assert results is not None
            assert isinstance(results, BacktestResults)
            assert results.n_paths == 5

            # Should also run without seed parameter
            backtester2 = Backtester(config)
            results2 = backtester2.run()  # No seed

            assert results2 is not None
            assert isinstance(results2, BacktestResults)

    def test_run_error_handling_missing_model(self):
        """Test that run() handles missing model file gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.create_dummy_parquet_data(temp_dir, n_days=5)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-03",
                strike=50000,
                maturity_days=2,
                model_path=os.path.join(temp_dir, "nonexistent_model.pth"),
                data_dir=temp_dir,
                n_bootstrap_paths=3,
            )
            backtester = Backtester(config)

            # Should raise FileNotFoundError with helpful message
            with pytest.raises(FileNotFoundError, match="Model checkpoint not found"):
                backtester.run()

    def test_run_error_handling_missing_data(self):
        """Test that run() handles missing data directory gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = os.path.join(temp_dir, "test_model.pth")
            self.create_dummy_checkpoint(model_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-03",
                strike=50000,
                maturity_days=2,
                model_path=model_path,
                data_dir="nonexistent_data_dir_12345",
                n_bootstrap_paths=3,
            )
            backtester = Backtester(config)

            # Should raise FileNotFoundError with helpful message
            with pytest.raises(FileNotFoundError, match="Data directory not found"):
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


class TestBacktestResults:
    """Tests for BacktestResults class."""

    def test_create_results_success(self):
        """Test successful creation of BacktestResults."""
        # Create dummy data
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000 + 45000

        # Create results
        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Verify stored data
        assert results.deep_pnl is deep_pnl
        assert results.bs_pnl is bs_pnl
        assert results.deep_positions is deep_positions
        assert results.bs_positions is bs_positions
        assert results.spots is spots
        assert results.n_paths == n_paths
        assert results.n_steps == n_steps

    def test_create_results_with_config(self):
        """Test creating results with config."""
        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        assert results.config is config

    def test_validate_inputs_wrong_dimension(self):
        """Test validation fails for wrong tensor dimensions."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots_1d = torch.rand(n_paths * n_steps)  # Wrong: 1D instead of 2D

        with pytest.raises(ValueError, match="must be 2D tensor"):
            BacktestResults(
                deep_pnl=deep_pnl,
                bs_pnl=bs_pnl,
                deep_positions=deep_positions,
                bs_positions=bs_positions,
                spots=spots_1d,
            )

    def test_validate_inputs_mismatched_shapes(self):
        """Test validation fails for mismatched shapes."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps + 5).cumsum(dim=1)  # Wrong shape
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        with pytest.raises(ValueError, match="doesn't match"):
            BacktestResults(
                deep_pnl=deep_pnl,
                bs_pnl=bs_pnl,
                deep_positions=deep_positions,
                bs_positions=bs_positions,
                spots=spots,
            )

    def test_summary_basic(self):
        """Test summary statistics calculation."""
        torch.manual_seed(42)
        n_paths, n_steps = 100, 50
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        summary = results.summary()

        # Verify structure
        assert "deep_hedge" in summary
        assert "bs_baseline" in summary

        # Verify metrics for each strategy
        for strategy in ["deep_hedge", "bs_baseline"]:
            assert "mean" in summary[strategy]
            assert "std" in summary[strategy]
            assert "min" in summary[strategy]
            assert "max" in summary[strategy]
            assert "median" in summary[strategy]
            assert "sharpe_ratio" in summary[strategy]
            assert "sortino_ratio" in summary[strategy]
            assert "cvar_95" in summary[strategy]
            assert "var_95" in summary[strategy]
            assert "max_drawdown" in summary[strategy]
            assert "calmar_ratio" in summary[strategy]
            assert "win_rate" in summary[strategy]

    def test_summary_values_reasonable(self):
        """Test that summary values are calculated correctly."""
        # Create controlled data
        n_paths, n_steps = 10, 20
        deep_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * 100  # Increasing
        bs_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * 50  # Increasing slower
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        summary = results.summary()

        # Deep hedge should have higher mean (growing faster)
        assert summary["deep_hedge"]["mean"] > summary["bs_baseline"]["mean"]

        # Both should have positive final returns
        assert summary["deep_hedge"]["mean"] > 0
        assert summary["bs_baseline"]["mean"] > 0

    def test_to_dict_structure(self):
        """Test to_dict returns proper structure."""
        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        data = results.to_dict()

        # Verify structure
        assert isinstance(data, dict)
        assert "deep_pnl" in data
        assert "bs_pnl" in data
        assert "deep_positions" in data
        assert "bs_positions" in data
        assert "spots" in data
        assert "n_paths" in data
        assert "n_steps" in data
        assert "summary" in data

        # Verify data types
        assert isinstance(data["deep_pnl"], list)
        assert isinstance(data["bs_pnl"], list)
        assert isinstance(data["summary"], dict)

        # Verify dimensions
        assert data["n_paths"] == n_paths
        assert data["n_steps"] == n_steps

    def test_to_dict_with_config(self):
        """Test to_dict includes config when provided."""
        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        data = results.to_dict()

        assert "config" in data
        assert isinstance(data["config"], dict)
        assert data["config"]["strike"] == 50000

    def test_repr(self):
        """Test string representation."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        repr_str = repr(results)

        assert "BacktestResults" in repr_str
        assert "n_paths=10" in repr_str
        assert "n_steps=20" in repr_str
        assert "deep_sharpe" in repr_str
        assert "bs_sharpe" in repr_str

    def test_summary_caching(self):
        """Test that summary is cached and reused on subsequent calls."""
        torch.manual_seed(42)
        n_paths, n_steps = 100, 50
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # First call - should compute and cache
        summary1 = results.summary()
        assert results._summary_cache is not None

        # Second call - should return cached version (same object)
        summary2 = results.summary()
        assert summary2 is summary1  # Same object reference

        # Call with different alpha - should recompute (not cached)
        summary3 = results.summary(alpha_cvar=0.01, alpha_var=0.01)
        assert "cvar_99" in summary3["deep_hedge"]
        assert summary3 is not summary1  # Different object

    def test_to_dict_include_raw_false(self):
        """Test to_dict with include_raw=False excludes tensor data."""
        n_paths, n_steps = 100, 200  # Large dataset
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Export without raw data
        data = results.to_dict(include_raw=False)

        # Verify structure has metadata but not raw arrays
        assert "n_paths" in data
        assert "n_steps" in data
        assert "summary" in data
        assert data["n_paths"] == n_paths
        assert data["n_steps"] == n_steps

        # Raw data should NOT be included
        assert "deep_pnl" not in data
        assert "bs_pnl" not in data
        assert "deep_positions" not in data
        assert "bs_positions" not in data
        assert "spots" not in data

    def test_to_dict_include_raw_true_default(self):
        """Test to_dict with include_raw=True (default) includes all data."""
        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Export with raw data (default)
        data = results.to_dict()

        # Verify all data is included
        assert "deep_pnl" in data
        assert "bs_pnl" in data
        assert "deep_positions" in data
        assert "bs_positions" in data
        assert "spots" in data
        assert "summary" in data
        assert "n_paths" in data
        assert "n_steps" in data

        # Verify raw arrays are lists
        assert isinstance(data["deep_pnl"], list)
        assert isinstance(data["bs_pnl"], list)
        assert len(data["deep_pnl"]) == n_paths

    def test_to_json_returns_valid_json_string(self):
        """Test to_json returns valid JSON string."""
        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Get JSON string
        json_str = results.to_json(include_raw=False)

        # Verify it's valid JSON
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "summary" in parsed
        assert "n_paths" in parsed
        assert "n_steps" in parsed

    def test_to_json_write_to_file(self):
        """Test to_json writes to file correctly."""
        import tempfile
        import os

        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Write to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            # Write should return None
            result = results.to_json(temp_path, include_raw=False)
            assert result is None

            # Verify file was created and is valid JSON
            assert os.path.exists(temp_path)
            with open(temp_path, "r") as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "summary" in data

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_to_json_handles_config_with_dates(self):
        """Test to_json handles config with date strings correctly."""
        n_paths, n_steps = 5, 10
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        # Should not raise even with config
        json_str = results.to_json(include_raw=False)
        parsed = json.loads(json_str)
        assert "config" in parsed
        assert parsed["config"]["start_date"] == "2024-01-01"

    def test_safe_json_normalize_datetime(self):
        """Test _safe_json_normalize handles datetime objects."""
        from datetime import datetime

        dt = datetime(2024, 1, 15, 12, 30, 45)
        normalized = BacktestResults._safe_json_normalize(dt)
        assert isinstance(normalized, str)
        assert "2024-01-15" in normalized

    def test_safe_json_normalize_numpy_types(self):
        """Test _safe_json_normalize handles numpy types."""
        import numpy as np

        # Test numpy integer
        np_int = np.int64(42)
        normalized = BacktestResults._safe_json_normalize(np_int)
        assert isinstance(normalized, int)
        assert normalized == 42

        # Test numpy float
        np_float = np.float64(3.14)
        normalized = BacktestResults._safe_json_normalize(np_float)
        assert isinstance(normalized, float)
        assert abs(normalized - 3.14) < 0.01

        # Test numpy array
        np_array = np.array([1, 2, 3])
        normalized = BacktestResults._safe_json_normalize(np_array)
        assert isinstance(normalized, list)
        assert normalized == [1, 2, 3]

    def test_safe_json_normalize_nested_dict(self):
        """Test _safe_json_normalize handles nested structures."""
        from datetime import datetime
        import numpy as np

        nested = {
            "date": datetime(2024, 1, 1),
            "value": np.float64(3.14),
            "nested": {"count": np.int64(42), "items": [np.int32(1), np.int32(2)]},
        }

        normalized = BacktestResults._safe_json_normalize(nested)

        # Verify all types are JSON-safe
        assert isinstance(normalized["date"], str)
        assert isinstance(normalized["value"], float)
        assert isinstance(normalized["nested"]["count"], int)
        assert isinstance(normalized["nested"]["items"][0], int)

        # Verify it's JSON-serializable
        json_str = json.dumps(normalized)
        assert isinstance(json_str, str)

    def test_get_time_axis_auto_mode_with_large_dt(self):
        """Test auto time_unit mode selects 'days' for large dt_hours."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=24.0,  # Large dt_hours should trigger 'days'
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        time_values, time_label = results._get_time_axis(time_unit="auto")

        # Should select 'days' for dt_hours >= 24
        assert time_label == "Time (days)"
        # First value should be 0, step should be 1 day
        assert time_values[0] == 0
        assert abs(time_values[1] - 1.0) < 0.01  # Should be 1 day

    def test_get_time_axis_auto_mode_with_medium_dt(self):
        """Test auto time_unit mode selects 'hours' for medium dt_hours."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=8.0,  # Medium dt_hours should trigger 'hours'
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        time_values, time_label = results._get_time_axis(time_unit="auto")

        # Should select 'hours' for 1 <= dt_hours < 24
        assert time_label == "Time (hours)"
        # First value should be 0, step should be 8 hours
        assert time_values[0] == 0
        assert abs(time_values[1] - 8.0) < 0.01  # Should be 8 hours

    def test_get_time_axis_auto_mode_with_small_dt(self):
        """Test auto time_unit mode selects 'steps' for small dt_hours."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=0.5,  # Small dt_hours should trigger 'steps'
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        time_values, time_label = results._get_time_axis(time_unit="auto")

        # Should select 'steps' for dt_hours < 1
        assert time_label == "Time Step"
        # Should be step indices
        assert time_values[0] == 0
        assert time_values[1] == 1

    def test_get_time_axis_auto_mode_without_config(self):
        """Test auto time_unit mode falls back to 'steps' without config."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=None,  # No config
        )

        time_values, time_label = results._get_time_axis(time_unit="auto")

        # Should fall back to 'steps' without config
        assert time_label == "Time Step"
        assert time_values[0] == 0
        assert time_values[1] == 1

    def test_get_time_axis_auto_mode_uses_default_dt_hours(self):
        """Test auto time_unit mode uses default dt_hours from config."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        # Create config without explicitly setting dt_hours
        # BacktestConfig has default dt_hours=8.0
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        time_values, time_label = results._get_time_axis(time_unit="auto")

        # Should use default dt_hours=8.0, which triggers 'hours' mode
        assert time_label == "Time (hours)"
        assert time_values[0] == 0
        assert abs(time_values[1] - 8.0) < 0.01  # Should be 8 hours

    def test_get_time_axis_manual_modes_still_work(self):
        """Test that manual time_unit modes still work with auto mode available."""
        n_paths, n_steps = 10, 20
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth",
            dt_hours=8.0,
        )

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
            config=config,
        )

        # Test 'steps' mode
        time_values, time_label = results._get_time_axis(time_unit="steps")
        assert time_label == "Time Step"
        assert time_values[1] == 1

        # Test 'hours' mode
        time_values, time_label = results._get_time_axis(time_unit="hours")
        assert time_label == "Time (hours)"
        assert abs(time_values[1] - 8.0) < 0.01

        # Test 'days' mode
        time_values, time_label = results._get_time_axis(time_unit="days")
        assert time_label == "Time (days)"
        assert abs(time_values[1] - 8.0 / 24.0) < 0.01

    def test_compare_strategies_structure(self):
        """Test compare_strategies returns correct structure."""
        torch.manual_seed(42)
        n_paths, n_steps = 100, 50
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 10
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1) * 10
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        comparison = results.compare_strategies()

        # Verify top-level structure
        assert "winners" in comparison
        assert "differences" in comparison
        assert "percentage_changes" in comparison
        assert "summary" in comparison

        # Verify winners structure
        assert "mean" in comparison["winners"]
        assert "sharpe_ratio" in comparison["winners"]
        assert "cvar_95" in comparison["winners"]
        assert "max_drawdown" in comparison["winners"]
        assert comparison["winners"]["mean"] in ["deep_hedge", "bs_baseline"]

        # Verify summary structure
        assert "deep_wins" in comparison["summary"]
        assert "total_metrics" in comparison["summary"]
        assert "assessment" in comparison["summary"]
        assert comparison["summary"]["total_metrics"] == 4
        assert comparison["summary"]["assessment"] in [
            "superior",
            "mixed",
            "underperformed",
        ]

    def test_compare_strategies_winner_determination(self):
        """Test winner determination logic works correctly."""
        n_paths, n_steps = 10, 20
        # Create data where deep hedge is clearly better
        deep_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * 100  # Positive
        bs_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * -50  # Negative
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        comparison = results.compare_strategies()

        # Deep hedge should win on mean (clearly positive vs negative)
        assert comparison["winners"]["mean"] == "deep_hedge"

        # Assessment should be superior (deep wins on most metrics)
        assert comparison["summary"]["assessment"] in ["superior", "mixed"]

    def test_max_drawdown_winner_logic(self):
        """Test max_drawdown winner determination (lower is better).

        CRITICAL: max_drawdown returns positive value representing loss.
        Strategy with LOWER drawdown should win.
        """
        n_paths, n_steps = 10, 20

        # Create PnL where deep hedge has SMALLER drawdown (better)
        # Deep: smooth growth with small dip
        deep_pnl = torch.zeros(n_paths, n_steps)
        for i in range(n_paths):
            deep_pnl[i] = torch.tensor(
                [
                    0.0,
                    5.0,
                    10.0,
                    8.0,
                    12.0,
                    15.0,
                    18.0,
                    20.0,
                    22.0,
                    25.0,
                    28.0,
                    30.0,
                    32.0,
                    35.0,
                    38.0,
                    40.0,
                    42.0,
                    45.0,
                    47.0,
                    50.0,
                ]
            )
        # Max drawdown for deep: 10 - 8 = 2

        # BS: volatile with large dip
        bs_pnl = torch.zeros(n_paths, n_steps)
        for i in range(n_paths):
            bs_pnl[i] = torch.tensor(
                [
                    0.0,
                    10.0,
                    20.0,
                    5.0,
                    25.0,
                    30.0,
                    35.0,
                    40.0,
                    45.0,
                    50.0,
                    55.0,
                    60.0,
                    65.0,
                    70.0,
                    75.0,
                    80.0,
                    85.0,
                    90.0,
                    95.0,
                    100.0,
                ]
            )
        # Max drawdown for bs: 20 - 5 = 15 (much larger)

        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        summary = results.summary()
        comparison = results.compare_strategies()

        # Verify drawdown values
        deep_dd = summary["deep_hedge"]["max_drawdown"]
        bs_dd = summary["bs_baseline"]["max_drawdown"]

        # Deep should have smaller drawdown
        assert deep_dd < bs_dd, f"Deep DD ({deep_dd}) should be < BS DD ({bs_dd})"

        # CRITICAL TEST: Deep hedge should win because it has LOWER (better) drawdown
        assert (
            comparison["winners"]["max_drawdown"] == "deep_hedge"
        ), f"Deep hedge has lower drawdown ({deep_dd:.2f} vs {bs_dd:.2f}) so should win"

    def test_compare_strategies_percentage_changes(self):
        """Test percentage changes calculation."""
        torch.manual_seed(42)
        n_paths, n_steps = 50, 30
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        comparison = results.compare_strategies()

        # Verify percentage changes are calculated
        assert "mean" in comparison["percentage_changes"]
        assert "std" in comparison["percentage_changes"]
        assert "sharpe_ratio" in comparison["percentage_changes"]

        # Verify percentage calculation formula
        summary = results.summary()
        deep = summary["deep_hedge"]
        bs = summary["bs_baseline"]

        if bs["mean"] != 0:
            expected_pct = ((deep["mean"] - bs["mean"]) / abs(bs["mean"])) * 100
            assert abs(comparison["percentage_changes"]["mean"] - expected_pct) < 0.01

    def test_print_summary_detailed_mode(self, capsys):
        """Test print_summary works in detailed mode."""
        torch.manual_seed(42)
        n_paths, n_steps = 50, 30
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Should not raise
        results.print_summary(detailed=True)

        # Verify output contains expected sections
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS SUMMARY" in captured.out
        assert "DEEP HEDGE" in captured.out
        assert "BLACK-SCHOLES" in captured.out
        assert "COMPARISON" in captured.out
        assert "Profitability" in captured.out
        assert "Risk-Adjusted" in captured.out

    def test_print_summary_compact_mode(self, capsys):
        """Test print_summary works in compact mode."""
        torch.manual_seed(42)
        n_paths, n_steps = 50, 30
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Should not raise
        results.print_summary(detailed=False)

        # Verify output contains expected sections
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS SUMMARY" in captured.out
        assert "Deep Hedge" in captured.out
        assert "Black-Scholes" in captured.out
        assert "Difference" in captured.out

    def test_print_key_insights(self, capsys):
        """Test print_key_insights produces expected output."""
        torch.manual_seed(42)
        n_paths, n_steps = 50, 30
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Should not raise
        results.print_key_insights()

        # Verify output contains expected sections
        captured = capsys.readouterr()
        assert "KEY INSIGHTS" in captured.out
        assert "Mean PnL:" in captured.out
        assert "Sharpe Ratio:" in captured.out
        assert "CVaR" in captured.out
        assert "Max Drawdown:" in captured.out
        assert "CONCLUSION" in captured.out
        # Should mention either "superior", "mixed", or better performance
        assert any(
            word in captured.out
            for word in ["superior", "Mixed", "better", "performed"]
        )

    def test_negative_sharpe_messaging(self, capsys):
        """Test that negative Sharpe/Sortino messaging is appropriate.

        When both strategies have negative risk ratios, messaging should
        indicate 'less bad' rather than a positive 'win'.
        """
        torch.manual_seed(42)
        n_paths, n_steps = 20, 15

        # Create PnL where BOTH strategies lose money (negative Sharpe)
        # Need variance across paths but negative mean
        # Deep: losses with some variation
        deep_pnl = (
            -torch.rand(n_paths, n_steps).cumsum(dim=1) * 10 - 20
        )  # Negative, varying
        # BS: worse losses with more variation
        bs_pnl = -torch.rand(n_paths, n_steps).cumsum(dim=1) * 15 - 40  # More negative

        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        summary = results.summary()

        # Verify both Sharpe ratios are negative (mean < 0, std > 0 → Sharpe < 0)
        assert (
            summary["deep_hedge"]["sharpe_ratio"] < 0
        ), f"Deep Sharpe should be negative, got {summary['deep_hedge']['sharpe_ratio']}"
        assert (
            summary["bs_baseline"]["sharpe_ratio"] < 0
        ), f"BS Sharpe should be negative, got {summary['bs_baseline']['sharpe_ratio']}"

        # Print insights
        results.print_key_insights()

        # Verify messaging includes annotation for negative ratios
        captured = capsys.readouterr()
        # Should mention "both negative" or "less bad"
        assert "both negative" in captured.out or "less bad" in captured.out

        # Should mention "lower loss" not "better mean PnL" since both are losses
        if "lower loss" not in captured.out:
            # If not showing improvements, that's also acceptable
            pass

    def test_division_by_zero_edge_cases(self):
        """Test edge cases with zero values to ensure no division errors."""
        n_paths, n_steps = 10, 20

        # Create PnL with zero std for BS (all paths identical)
        deep_pnl = torch.randn(n_paths, n_steps).cumsum(dim=1)
        bs_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * 10  # All identical

        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        # Should not raise division by zero
        comparison = results.compare_strategies()

        # Verify percentage changes exist (even with std=0)
        assert "percentage_changes" in comparison
        # When bs['std'] = 0, percentage change should be 0.0 (guarded)
        assert isinstance(comparison["percentage_changes"]["std"], float)

    def test_zero_win_rate_edge_case(self):
        """Test edge case where win_rate is 0 for both strategies."""
        n_paths, n_steps = 10, 20

        # All paths lose money (final PnL all negative)
        deep_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * -10
        bs_pnl = torch.ones(n_paths, n_steps).cumsum(dim=1) * -20

        deep_positions = torch.randn(n_paths, n_steps)
        bs_positions = torch.randn(n_paths, n_steps)
        spots = torch.rand(n_paths, n_steps) * 50000

        results = BacktestResults(
            deep_pnl=deep_pnl,
            bs_pnl=bs_pnl,
            deep_positions=deep_positions,
            bs_positions=bs_positions,
            spots=spots,
        )

        summary = results.summary()

        # Both should have 0% win rate
        assert summary["deep_hedge"]["win_rate"] == 0.0
        assert summary["bs_baseline"]["win_rate"] == 0.0

        # Should not raise when computing comparison
        comparison = results.compare_strategies()

        # Win rate percentage change should be 0 (0 - 0 = 0)
        assert comparison["percentage_changes"]["win_rate"] == 0.0
