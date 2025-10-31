"""Tests for validation frequency optimization."""

import pytest
import torch
from pfhedge.instruments import BrownianStock, EuropeanOption
from pfhedge.nn import Hedger, MultiLayerPerceptron


class TestValidationFrequency:
    """Test suite for validation_freq parameter."""

    def test_validation_freq_default(self, device: str = "cpu"):
        """Verify default behavior: validate every epoch."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        n_epochs = 5
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=n_epochs,
            verbose=False,
            validation_freq=1,  # Default: validate every epoch
        )

        # Should have one loss value per epoch
        assert len(history) == n_epochs

    def test_validation_freq_every_2_epochs(self, device: str = "cpu"):
        """Verify validation happens every 2 epochs."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        n_epochs = 10
        validation_freq = 2
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=n_epochs,
            verbose=False,
            validation_freq=validation_freq,
        )

        # Should validate at epochs 2, 4, 6, 8, 10
        # = 5 validation points
        expected_validations = (n_epochs + validation_freq - 1) // validation_freq
        assert len(history) == expected_validations

    def test_validation_freq_every_5_epochs(self, device: str = "cpu"):
        """Verify validation happens every 5 epochs plus final epoch."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        n_epochs = 12
        validation_freq = 5
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=n_epochs,
            verbose=False,
            validation_freq=validation_freq,
        )

        # Should validate at epochs 5, 10, 12 (final)
        # = 3 validation points
        assert len(history) == 3

    def test_validation_freq_larger_than_epochs(self, device: str = "cpu"):
        """Verify validation happens only on final epoch when freq > n_epochs."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        n_epochs = 5
        validation_freq = 100  # Much larger than n_epochs
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=n_epochs,
            verbose=False,
            validation_freq=validation_freq,
        )

        # Should only validate on final epoch (epoch 5)
        assert len(history) == 1

    def test_validation_freq_zero_validates_final_only(self, device: str = "cpu"):
        """Verify validation_freq=0 skips intermediate validation but validates final epoch."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=10,
            verbose=False,
            validation_freq=0,  # Skip intermediate validation, only validate final
        )

        # Should only validate final epoch
        assert len(history) == 1

    def test_validation_freq_respects_validation_flag(self, device: str = "cpu"):
        """Verify validation_freq doesn't override validation=False."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=10,
            verbose=False,
            validation=False,  # Disable validation via flag
            validation_freq=2,  # This should be ignored
        )

        # Should return None when validation=False
        assert history is None

    def test_validation_freq_final_epoch_always_validated(self, device: str = "cpu"):
        """Verify final epoch is always validated regardless of freq."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        # Use n_epochs=11 with validation_freq=5
        # Should validate at: 5, 10, 11 (final)
        n_epochs = 11
        validation_freq = 5
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=n_epochs,
            verbose=False,
            validation_freq=validation_freq,
        )

        # Should have 3 validation points
        assert len(history) == 3

    @pytest.mark.gpu
    def test_validation_freq_every_2_epochs_gpu(self):
        """Verify validation frequency works on GPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        self.test_validation_freq_every_2_epochs(device="cuda")

    def test_validation_freq_with_amp(self, device: str = "cpu"):
        """Verify validation_freq works with AMP enabled."""
        torch.manual_seed(42)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        n_epochs = 8
        validation_freq = 3
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=n_epochs,
            verbose=False,
            validation_freq=validation_freq,
            use_amp=True,  # Enable AMP (will fall back to FP32 on CPU)
        )

        # Should validate at epochs 3, 6, 8 (final)
        assert len(history) == 3
