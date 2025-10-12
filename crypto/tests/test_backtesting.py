"""Tests for backtesting framework."""

import pytest
from crypto.backtest.config import BacktestConfig


class TestBacktestConfig:
    """Tests for BacktestConfig."""

    def test_create_basic_config(self):
        """Test creating a basic config with required parameters."""
        config = BacktestConfig(
            start_date="2024-01-01",
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test_model.pth"
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
            output_dir="custom_results"
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
            model_path="models/test.pth"
        )
        config.validate()  # Should not raise

    def test_validate_invalid_date_format(self):
        """Test validation fails for invalid date format."""
        config = BacktestConfig(
            start_date="01-01-2024",  # Wrong format
            end_date="2024-01-31",
            strike=50000,
            maturity_days=14,
            model_path="models/test.pth"
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
            model_path="models/test.pth"
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
            model_path="models/test.pth"
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
            model_path="models/test.pth"
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
            n_bootstrap_paths=-10  # Invalid
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
            transaction_cost=-0.001  # Invalid
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
            transaction_cost=0.15  # 15% seems wrong
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
            dt_hours=0  # Invalid
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
            dt_hours=48  # Invalid
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
            n_bootstrap_paths=200
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
            "n_bootstrap_paths": 200
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
            dt_hours=4.0
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
            dt_hours=8.0
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
            model_path="models/test.pth"
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
            call=False
        )

        repr_str = repr(config)
        assert "Put" in repr_str
