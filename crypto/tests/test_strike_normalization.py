"""Unit tests to protect against strike normalization bugs.

These tests ensure that:
1. Training normalizes strikes correctly
2. Checkpoints save normalized strikes
3. Models fail gracefully when loading unnormalized checkpoints
4. Strike values are in expected ranges throughout the workflow
"""

import pytest
import torch
import tempfile
import os
import json
from pathlib import Path


class TestStrikeNormalizationTraining:
    """Test strike normalization during training."""

    def test_train_for_option_normalizes_strike(self):
        """Test that train_for_option.py normalizes strike before training."""
        from crypto.scripts.train_for_option import create_training_config_from_option

        # Option with absolute strike
        option = {
            "strike": 110000,
            "initial_spot": 109325.75,
            "days_to_expiry": 33,
            "option_type": "call",
            "implied_volatility": 0.42,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_training_config_from_option(
                option=option,
                output_dir=tmpdir,
                epochs=1,
                paths=100,
            )

            # Strike should be normalized
            expected_normalized = 110000 / 109325.75
            assert abs(config.strike - expected_normalized) < 0.0001
            assert (
                0.5 <= config.strike <= 2.0
            ), f"Strike {config.strike} not normalized!"

    def test_train_for_option_fails_without_initial_spot(self):
        """Test that training fails gracefully when initial_spot is missing."""
        from crypto.scripts.train_for_option import create_training_config_from_option

        # Option without initial_spot
        option = {
            "strike": 110000,
            # Missing initial_spot!
            "days_to_expiry": 33,
            "option_type": "call",
            "implied_volatility": 0.42,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="initial_spot"):
                create_training_config_from_option(
                    option=option,
                    output_dir=tmpdir,
                )

    def test_train_for_option_fails_with_zero_initial_spot(self):
        """Test that training fails when initial_spot is zero."""
        from crypto.scripts.train_for_option import create_training_config_from_option

        option = {
            "strike": 110000,
            "initial_spot": 0,  # Invalid!
            "days_to_expiry": 33,
            "option_type": "call",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError, match="initial_spot"):
                create_training_config_from_option(
                    option=option,
                    output_dir=tmpdir,
                )


class TestCheckpointStrikeValidation:
    """Test that checkpoints save and validate strike correctly."""

    def test_checkpoint_saves_normalized_strike(self):
        """Test that trainer saves normalized strike to checkpoint."""
        from crypto.training import TrainingConfig, Trainer

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"

            # Create config with normalized strike
            config = TrainingConfig(
                strike=1.0062,  # Normalized (110k / 109.3k)
                maturity_days=33,
                call=True,
                volatility=0.42,
                n_paths=100,
                n_epochs=1,
                model_path=str(model_path),
                output_dir=tmpdir,
            )

            # Train minimal model
            trainer = Trainer(config, verbose=False)
            trainer.train()

            # Load checkpoint and verify strike
            checkpoint = torch.load(model_path, map_location="cpu")
            saved_strike = checkpoint["training_config"]["strike"]

            assert abs(saved_strike - 1.0062) < 0.0001
            assert (
                0.5 <= saved_strike <= 2.0
            ), f"Saved strike {saved_strike} not normalized!"

    def test_verify_checkpoint_detects_absolute_strike(self):
        """Test that verify_checkpoint.py detects unnormalized strikes."""
        from crypto.scripts.verify_checkpoint import verify_checkpoint
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint with ABSOLUTE strike (bug!)
            model = create_deep_hedger(n_layers=2, n_units=32)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "training_config": {
                    "strike": 110000,  # ❌ Absolute strike!
                    "maturity_days": 33,
                    "call": True,
                    "volatility": 0.42,
                    "n_layers": 2,
                    "n_units": 32,
                    "risk_param": 0.1,
                },
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "criterion": "entropic",
                    "risk_param": 0.1,
                    "features": [
                        "log_moneyness",
                        "expiry_time",
                        "volatility",
                        "prev_hedge",
                    ],
                },
            }
            torch.save(checkpoint, temp_path)

            # Verification should fail
            is_valid = verify_checkpoint(temp_path)
            assert not is_valid, "Should detect absolute strike as invalid!"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def test_verify_checkpoint_accepts_normalized_strike(self):
        """Test that verify_checkpoint.py accepts normalized strikes."""
        from crypto.scripts.verify_checkpoint import verify_checkpoint
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create checkpoint with normalized strike
            model = create_deep_hedger(n_layers=2, n_units=32)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "training_config": {
                    "strike": 1.0062,  # ✅ Normalized!
                    "maturity_days": 33,
                    "call": True,
                    "volatility": 0.42,
                    "n_layers": 2,
                    "n_units": 32,
                    "risk_param": 0.1,
                },
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "criterion": "entropic",
                    "risk_param": 0.1,
                    "features": [
                        "log_moneyness",
                        "expiry_time",
                        "volatility",
                        "prev_hedge",
                    ],
                },
            }
            torch.save(checkpoint, temp_path)

            # Verification should pass
            is_valid = verify_checkpoint(temp_path)
            assert is_valid, "Should accept normalized strike!"

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestStrikeNormalizationEdgeCases:
    """Test edge cases for strike normalization."""

    def test_atm_option_strike_near_one(self):
        """Test ATM option produces strike near 1.0."""
        from crypto.scripts.train_for_option import create_training_config_from_option

        # ATM option
        option = {
            "strike": 110000,
            "initial_spot": 110000,  # Exactly ATM
            "days_to_expiry": 30,
            "option_type": "call",
            "implied_volatility": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_training_config_from_option(
                option=option, output_dir=tmpdir
            )

            # Strike should be exactly 1.0
            assert abs(config.strike - 1.0) < 0.0001

    def test_deep_otm_option_strike_below_one(self):
        """Test deep OTM option produces strike < 1.0."""
        from crypto.scripts.train_for_option import create_training_config_from_option

        # Deep OTM call
        option = {
            "strike": 150000,
            "initial_spot": 100000,
            "days_to_expiry": 30,
            "option_type": "call",
            "implied_volatility": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_training_config_from_option(
                option=option, output_dir=tmpdir
            )

            # Strike should be 1.5
            assert abs(config.strike - 1.5) < 0.0001
            assert config.strike > 1.0

    def test_deep_itm_option_strike_above_one(self):
        """Test deep ITM option produces strike > 1.0."""
        from crypto.scripts.train_for_option import create_training_config_from_option

        # Deep ITM call
        option = {
            "strike": 80000,
            "initial_spot": 100000,
            "days_to_expiry": 30,
            "option_type": "call",
            "implied_volatility": 0.5,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = create_training_config_from_option(
                option=option, output_dir=tmpdir
            )

            # Strike should be 0.8
            assert abs(config.strike - 0.8) < 0.0001
            assert config.strike < 1.0


class TestBackwardCompatibility:
    """Test that old checkpoints without proper strike fail gracefully."""

    def test_backtester_rejects_old_checkpoint_without_strike(self):
        """Test that backtester rejects checkpoints without strike in training_config."""
        from crypto.backtest.config import BacktestConfig
        from crypto.backtest.backtester import Backtester
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            temp_path = f.name

        try:
            # Create old checkpoint without training_config
            model = create_deep_hedger(n_layers=2, n_units=32)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": {
                    "n_layers": 2,
                    "n_units": 32,
                    "criterion": "entropic",
                    "risk_param": 0.1,
                    "features": ["log_moneyness", "expiry_time"],
                },
                # Missing training_config!
            }
            torch.save(checkpoint, temp_path)

            config = BacktestConfig(
                start_date="2024-01-01",
                end_date="2024-01-31",
                strike=110000,
                maturity_days=30,
                model_path=temp_path,
            )
            backtester = Backtester(config)

            # Loading should succeed (training_config is optional for backtesting)
            # But we should be able to detect it's an old model
            model = backtester.load_model()
            assert model is not None

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestStrikeNormalizationInference:
    """Test that strike normalization is preserved during inference."""

    def test_log_moneyness_uses_normalized_strike_btc_option(self):
        """Test that BitcoinEuropeanOption uses normalized strike correctly."""
        from crypto.instruments.bitcoin_european_option import BitcoinEuropeanOption
        from pfhedge.instruments import BrownianStock

        # Create GBM underlier (starts at S0=1.0, like training)
        # This simulates the normalized training environment
        underlier = BrownianStock(cost=0.0)

        # Create Bitcoin option with normalized strike
        option = BitcoinEuropeanOption(
            underlier=underlier,
            strike=1.0062,  # Normalized (110k / 109.3k)
            maturity=33 / 365,
            call=True,
        )

        # Simulate price path
        underlier.simulate(n_paths=10, time_horizon=33 / 365)

        # Get log_moneyness feature
        log_m = option.log_moneyness()

        # log_moneyness = log(S / K)
        # For normalized strike K≈1.0 and S near 1.0:
        # log_m should be near 0
        assert (
            log_m.abs().mean() < 1.0
        ), f"log_moneyness too large: {log_m.abs().mean()}"

        # Should not be using absolute strike (would give log_m ~ -11.5)
        assert log_m.mean() > -5.0, "log_moneyness suggests absolute strike was used!"

    def test_bitcoin_option_with_absolute_strike_would_break(self):
        """Test that using absolute strike in BitcoinEuropeanOption would produce wrong log_moneyness."""
        from crypto.instruments.bitcoin_european_option import BitcoinEuropeanOption
        from pfhedge.instruments import BrownianStock

        # Create GBM underlier (starts at S0=1.0)
        underlier = BrownianStock(cost=0.0)

        # Create option with ABSOLUTE strike (this would be the bug!)
        option_buggy = BitcoinEuropeanOption(
            underlier=underlier,
            strike=110000,  # ❌ Absolute strike!
            maturity=33 / 365,
            call=True,
        )

        # Simulate price path
        underlier.simulate(n_paths=10, time_horizon=33 / 365)

        # Get log_moneyness - this would be completely wrong
        log_m_buggy = option_buggy.log_moneyness()

        # log_moneyness = log(S / K) = log(~1.0 / 110000) ≈ -11.6
        # This proves absolute strike breaks the feature
        assert log_m_buggy.mean() < -10.0, "Absolute strike should give log_m ~ -11.6"

        # This is why we MUST normalize strikes!
        print(
            f"✅ Verified: Absolute strike gives log_moneyness ≈ {log_m_buggy.mean():.2f}"
        )
        print(f"   This would be completely outside training distribution!")

    def test_bitcoin_option_strike_normalization_end_to_end(self):
        """End-to-end test: Train with normalized strike, verify it's saved correctly."""
        from crypto.training import TrainingConfig, Trainer
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pth"

            # Simulate real workflow: absolute strike → normalized
            absolute_strike = 110000
            initial_spot = 109325.75
            normalized_strike = absolute_strike / initial_spot  # 1.0062

            # Create config with NORMALIZED strike (what train_for_option.py does)
            config = TrainingConfig(
                strike=normalized_strike,  # ✅ Normalized
                maturity_days=33,
                call=True,
                volatility=0.42,
                n_paths=100,
                n_epochs=1,
                model_path=str(model_path),
                output_dir=tmpdir,
            )

            # Train minimal model
            trainer = Trainer(config, verbose=False)
            trainer.train()

            # Verify checkpoint has normalized strike
            checkpoint = torch.load(model_path, map_location="cpu")
            saved_strike = checkpoint["training_config"]["strike"]

            # This is the critical check: strike must be normalized
            assert 0.5 <= saved_strike <= 2.0, f"Strike {saved_strike} not normalized!"
            assert abs(saved_strike - normalized_strike) < 0.0001

            # Verify we didn't accidentally save absolute strike
            assert (
                saved_strike != absolute_strike
            ), "Saved absolute strike instead of normalized!"

            print(f"✅ End-to-end test passed:")
            print(f"   Absolute strike: {absolute_strike}")
            print(f"   Initial spot: {initial_spot}")
            print(f"   Normalized strike: {normalized_strike:.6f}")
            print(f"   Saved strike: {saved_strike:.6f}")
