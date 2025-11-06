import pytest
from crypto.training import TrainingConfig


class TestNormalizeRiskMeasure:

    def test_normalize_cvar_to_expected_shortfall(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="cvar",
        )
        # Before validation
        assert config.risk_measure == "cvar"

        # After validation, should be normalized
        config.validate()
        assert config.risk_measure == "expected_shortfall"

    def test_normalize_es_to_expected_shortfall(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="es",
        )
        config.validate()
        assert config.risk_measure == "expected_shortfall"

    def test_normalize_uppercase_cvar(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="CVaR",
        )
        config.validate()
        assert config.risk_measure == "expected_shortfall"

    def test_normalize_mixed_case_es(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="ES",
        )
        config.validate()
        assert config.risk_measure == "expected_shortfall"

    def test_no_normalization_for_canonical_value(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="expected_shortfall",
        )
        config.validate()
        assert config.risk_measure == "expected_shortfall"

    def test_no_normalization_for_variance(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="variance",
        )
        config.validate()
        assert config.risk_measure == "variance"

    def test_no_normalization_for_entropic(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="entropic",
        )
        config.validate()
        assert config.risk_measure == "entropic"

    def test_invalid_risk_measure_raises_error(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="invalid_measure",
        )
        with pytest.raises(ValueError, match="risk_measure must be one of"):
            config.validate()


class TestDeviceValidation:

    def test_cpu_device_valid(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            device="cpu",
        )
        config.validate()  # Should not raise
        assert config.device == "cpu"

    def test_cuda_device_valid(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            device="cuda",
        )
        config.validate()  # Should not raise
        assert config.device == "cuda"

    def test_cuda_with_index_valid(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            device="cuda:0",
        )
        config.validate()  # Should not raise
        assert config.device == "cuda:0"

    def test_cuda_with_multiple_gpus(self):
        for i in range(4):
            config = TrainingConfig(
                strike=50000,
                maturity_days=14,
                model_path="test.pth",
                device=f"cuda:{i}",
            )
            config.validate()  # Should not raise
            assert config.device == f"cuda:{i}"

    def test_mps_device_valid(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            device="mps",
        )
        config.validate()  # Should not raise
        assert config.device == "mps"

    def test_invalid_device_raises_error(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            device="invalid_device",
        )
        with pytest.raises(ValueError, match="device must be one of"):
            config.validate()

    def test_invalid_device_with_colon_raises_error(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            device="tpu:0",
        )
        with pytest.raises(ValueError, match="device must be one of"):
            config.validate()


class TestTrainingConfigIntegration:

    def test_full_config_with_normalization(self):
        config = TrainingConfig(
            strike=60000,
            maturity_days=7,
            model_path="test.pth",
            call=False,
            volatility=1.2,
            risk_measure="cvar",  # Should be normalized
            device="cuda:1",
            n_epochs=100,
        )

        # Before validation
        assert config.risk_measure == "cvar"

        # Validate
        config.validate()

        # After validation
        assert config.risk_measure == "expected_shortfall"
        assert config.device == "cuda:1"
        assert config.call is False
        assert config.strike == 60000

    def test_to_dict_after_normalization(self):
        config = TrainingConfig(
            strike=50000,
            maturity_days=14,
            model_path="test.pth",
            risk_measure="cvar",
        )
        config.validate()

        config_dict = config.to_dict()

        # Should contain normalized value
        assert config_dict["risk_measure"] == "expected_shortfall"
        assert config_dict["device"] == "cpu"

    def test_from_dict_with_alias(self):
        config_dict = {
            "strike": 50000,
            "maturity_days": 14,
            "model_path": "test.pth",
            "risk_measure": "es",  # Alias
        }

        config = TrainingConfig.from_dict(config_dict)
        config.validate()

        # Should be normalized
        assert config.risk_measure == "expected_shortfall"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
