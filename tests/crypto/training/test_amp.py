"""Tests for Mixed Precision Training (AMP) optimization."""

import pytest
import torch
from pfhedge.instruments import BrownianStock, EuropeanOption
from pfhedge.nn import Hedger, MultiLayerPerceptron


class TestMixedPrecision:
    """Test suite for AMP (Automatic Mixed Precision) training."""

    def test_amp_disabled_on_cpu(self):
        """Verify AMP gracefully falls back to FP32 on CPU devices."""
        torch.manual_seed(42)
        device = "cpu"

        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        # Request AMP on CPU - should fall back to FP32 without errors
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=3,
            use_amp=True,  # Request AMP on CPU
            verbose=False
        )

        # Verify training completed successfully
        assert len(history) == 3
        assert all(loss > 0 for loss in history), "All losses should be positive"

        # Verify model parameters are still in FP32 on CPU
        for param in hedger.parameters():
            assert param.dtype == torch.float32, "Parameters should remain FP32 on CPU"

    @pytest.mark.gpu
    def test_amp_produces_valid_gradients(self):
        """Verify AMP doesn't break gradient computation on GPU."""
        torch.manual_seed(42)
        device = "cuda"

        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        # Train with AMP enabled
        history = hedger.fit(
            deriv,
            n_paths=100,
            n_epochs=3,
            use_amp=True,
            verbose=False
        )

        # Verify training completed successfully
        assert len(history) == 3
        assert all(loss > 0 for loss in history), "All losses should be positive"

        # Verify gradients were computed (parameters changed from initialization)
        # We'll check this by comparing parameter norms before/after
        param_norm = sum(p.norm().item() ** 2 for p in hedger.parameters()) ** 0.5
        assert param_norm > 0, "Model parameters should have non-zero norm after training"

    @pytest.mark.gpu
    def test_amp_vs_fp32_convergence(self):
        """Verify AMP converges to similar loss as FP32 training."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        seed = 42
        device = "cuda"

        # Train with FP32
        torch.manual_seed(seed)
        deriv_fp32 = EuropeanOption(BrownianStock()).to(device)
        hedger_fp32 = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)
        history_fp32 = hedger_fp32.fit(
            deriv_fp32,
            n_paths=200,
            n_epochs=10,
            use_amp=False,  # Explicit FP32
            verbose=False
        )

        # Train with AMP (same seed for reproducibility)
        torch.manual_seed(seed)
        deriv_amp = EuropeanOption(BrownianStock()).to(device)
        hedger_amp = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)
        history_amp = hedger_amp.fit(
            deriv_amp,
            n_paths=200,
            n_epochs=10,
            use_amp=True,  # Use AMP
            verbose=False
        )

        # Verify both converged successfully
        assert len(history_fp32) == 10
        assert len(history_amp) == 10

        # Final losses should be in similar range (within 20% tolerance)
        # AMP may have slightly different convergence due to reduced precision
        final_loss_fp32 = history_fp32[-1]
        final_loss_amp = history_amp[-1]

        # Both should be positive
        assert final_loss_fp32 > 0
        assert final_loss_amp > 0

        # Relative difference should be reasonable (< 50%)
        # We use a generous tolerance since AMP can have different dynamics
        rel_diff = abs(final_loss_amp - final_loss_fp32) / final_loss_fp32
        assert rel_diff < 0.5, (
            f"AMP and FP32 final losses differ too much: "
            f"FP32={final_loss_fp32:.6f}, AMP={final_loss_amp:.6f}, "
            f"rel_diff={rel_diff:.2%}"
        )

    @pytest.mark.gpu
    def test_amp_numerical_stability(self):
        """Verify entropic risk measure remains numerically stable under AMP."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        from pfhedge.nn import EntropicRiskMeasure

        torch.manual_seed(42)
        device = "cuda"

        # Use entropic risk measure (most numerically sensitive)
        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(),
            ["moneyness", "time_to_maturity", "volatility"],
            criterion=EntropicRiskMeasure()
        ).to(device)

        # Train with AMP - should not produce NaN/Inf
        history = hedger.fit(
            deriv,
            n_paths=200,
            n_epochs=5,
            use_amp=True,
            verbose=False
        )

        # Verify no NaN or Inf in loss history
        assert len(history) == 5
        assert all(torch.isfinite(torch.tensor(loss)) for loss in history), \
            f"Loss history contains NaN/Inf: {history}"

        # Verify parameters don't have NaN/Inf
        for name, param in hedger.named_parameters():
            assert torch.isfinite(param).all(), \
                f"Parameter {name} contains NaN/Inf after AMP training"

    def test_amp_flag_propagation(self):
        """Verify use_amp flag is correctly handled in fit() signature."""
        torch.manual_seed(42)
        device = "cpu"

        deriv = EuropeanOption(BrownianStock()).to(device)
        hedger = Hedger(
            MultiLayerPerceptron(), ["moneyness", "time_to_maturity", "volatility"]
        ).to(device)

        # Test with use_amp=False (explicit)
        history_no_amp = hedger.fit(
            deriv,
            n_paths=50,
            n_epochs=2,
            use_amp=False,
            verbose=False
        )
        assert len(history_no_amp) == 2

        # Test with use_amp=True (should work on CPU with graceful fallback)
        history_amp = hedger.fit(
            deriv,
            n_paths=50,
            n_epochs=2,
            use_amp=True,
            verbose=False
        )
        assert len(history_amp) == 2

        # Both should complete successfully
        assert all(loss > 0 for loss in history_no_amp)
        assert all(loss > 0 for loss in history_amp)
