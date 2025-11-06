import pytest
import torch
import torch.nn as nn

from crypto.strategies import (
    create_deep_hedger,
    LongShortTermMemory,
    GatedRecurrentUnit,
)
from pfhedge.nn import Hedger, MultiLayerPerceptron


class TestLongShortTermMemory:

    def test_initialization_with_features(self):
        model = LongShortTermMemory(
            in_features=4, hidden_size=64, num_layers=2, dropout=0.2
        )
        assert model.in_features == 4
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout == 0.2

    def test_initialization_minimal(self):
        model = LongShortTermMemory(in_features=4, hidden_size=32, num_layers=1)
        assert model.in_features == 4
        assert model.hidden_size == 32
        assert model.num_layers == 1

    def test_forward_2d_input(self):
        model = LongShortTermMemory(in_features=4, hidden_size=64, num_layers=2)
        x = torch.randn(100, 4)  # (batch_size, features)
        output = model(x)
        # Should add time dimension: (100, 1, 1)
        assert output.shape == (100, 1, 1)

    def test_forward_3d_input(self):
        model = LongShortTermMemory(in_features=4, hidden_size=64, num_layers=2)
        x = torch.randn(100, 42, 4)  # (batch_size, time_steps, features)
        output = model(x)
        assert output.shape == (100, 42, 1)  # (batch_size, time_steps, out_features)

    def test_different_input_sizes(self):
        for n_features in [2, 4, 8]:
            model = LongShortTermMemory(
                in_features=n_features, hidden_size=32, num_layers=1
            )
            x = torch.randn(10, 5, n_features)
            output = model(x)
            assert output.shape == (10, 5, 1)

    def test_single_layer_no_dropout(self):
        model = LongShortTermMemory(
            in_features=4, hidden_size=32, num_layers=1, dropout=0.2
        )
        # Dropout should be 0 for single layer
        assert model.lstm.dropout == 0.0

    def test_multi_layer_with_dropout(self):
        model = LongShortTermMemory(
            in_features=4, hidden_size=32, num_layers=2, dropout=0.3
        )
        assert model.lstm.dropout == 0.3

    def test_output_shape_consistency(self):
        model = LongShortTermMemory(in_features=4, hidden_size=64, num_layers=2)
        x1 = torch.randn(50, 10, 4)
        x2 = torch.randn(50, 10, 4)
        output1 = model(x1)
        output2 = model(x2)
        assert output1.shape == output2.shape == (50, 10, 1)

    def test_bidirectional(self):
        model = LongShortTermMemory(
            in_features=4, hidden_size=64, num_layers=2, bidirectional=True
        )
        x = torch.randn(100, 42, 4)
        output = model(x)
        assert output.shape == (100, 42, 1)
        # FC layer should have 128 inputs (64*2 for bidirectional)
        assert model.fc.in_features == 128

    def test_repr(self):
        model = LongShortTermMemory(in_features=4, hidden_size=64, num_layers=2)
        repr_str = repr(model)
        assert "LongShortTermMemory" in repr_str
        assert "hidden_size=64" in repr_str
        assert "num_layers=2" in repr_str


class TestGatedRecurrentUnit:

    def test_initialization_with_features(self):
        model = GatedRecurrentUnit(
            in_features=4, hidden_size=64, num_layers=2, dropout=0.2
        )
        assert model.in_features == 4
        assert model.hidden_size == 64
        assert model.num_layers == 2
        assert model.dropout == 0.2

    def test_initialization_minimal(self):
        model = GatedRecurrentUnit(in_features=4, hidden_size=32, num_layers=1)
        assert model.in_features == 4
        assert model.hidden_size == 32

    def test_forward_2d_input(self):
        model = GatedRecurrentUnit(in_features=4, hidden_size=64, num_layers=2)
        x = torch.randn(100, 4)
        output = model(x)
        assert output.shape == (100, 1, 1)

    def test_forward_3d_input(self):
        model = GatedRecurrentUnit(in_features=4, hidden_size=64, num_layers=2)
        x = torch.randn(100, 42, 4)
        output = model(x)
        assert output.shape == (100, 42, 1)

    def test_different_input_sizes(self):
        for n_features in [2, 4, 8]:
            model = GatedRecurrentUnit(
                in_features=n_features, hidden_size=32, num_layers=1
            )
            x = torch.randn(10, 5, n_features)
            output = model(x)
            assert output.shape == (10, 5, 1)

    def test_single_layer_no_dropout(self):
        model = GatedRecurrentUnit(
            in_features=4, hidden_size=32, num_layers=1, dropout=0.2
        )
        assert model.gru.dropout == 0.0

    def test_multi_layer_with_dropout(self):
        model = GatedRecurrentUnit(
            in_features=4, hidden_size=32, num_layers=2, dropout=0.3
        )
        assert model.gru.dropout == 0.3

    def test_bidirectional(self):
        model = GatedRecurrentUnit(
            in_features=4, hidden_size=64, num_layers=2, bidirectional=True
        )
        x = torch.randn(100, 42, 4)
        output = model(x)
        assert output.shape == (100, 42, 1)
        assert model.fc.in_features == 128

    def test_repr(self):
        model = GatedRecurrentUnit(in_features=4, hidden_size=64, num_layers=2)
        repr_str = repr(model)
        assert "GatedRecurrentUnit" in repr_str
        assert "hidden_size=64" in repr_str


class TestCreateDeepHedger:

    def test_create_mlp_default(self):
        hedger = create_deep_hedger()
        assert isinstance(hedger, Hedger)
        assert isinstance(hedger.model, MultiLayerPerceptron)

    def test_create_mlp_explicit(self):
        hedger = create_deep_hedger(
            model_type="mlp", n_layers=4, n_units=128, risk_measure="entropic"
        )
        assert isinstance(hedger.model, MultiLayerPerceptron)

    def test_create_lstm(self):
        hedger = create_deep_hedger(model_type="lstm", n_layers=2, n_units=64)
        assert isinstance(hedger, Hedger)
        assert isinstance(hedger.model, LongShortTermMemory)
        assert hedger.model.num_layers == 2
        assert hedger.model.hidden_size == 64

    def test_create_gru(self):
        hedger = create_deep_hedger(model_type="gru", n_layers=2, n_units=64)
        assert isinstance(hedger, Hedger)
        assert isinstance(hedger.model, GatedRecurrentUnit)
        assert hedger.model.num_layers == 2
        assert hedger.model.hidden_size == 64

    def test_model_type_case_insensitive(self):
        hedger1 = create_deep_hedger(model_type="LSTM")
        hedger2 = create_deep_hedger(model_type="lstm")
        hedger3 = create_deep_hedger(model_type="LsTm")
        assert isinstance(hedger1.model, LongShortTermMemory)
        assert isinstance(hedger2.model, LongShortTermMemory)
        assert isinstance(hedger3.model, LongShortTermMemory)

    def test_invalid_model_type(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            create_deep_hedger(model_type="transformer")

    def test_invalid_model_type_shows_available(self):
        with pytest.raises(ValueError, match="Available options"):
            create_deep_hedger(model_type="invalid")

    def test_risk_measure_expected_shortfall(self):
        hedger = create_deep_hedger(risk_measure="expected_shortfall", risk_param=0.95)
        assert hedger.criterion.__class__.__name__ == "ExpectedShortfall"

    def test_risk_measure_entropic(self):
        hedger = create_deep_hedger(risk_measure="entropic", risk_param=0.1)
        assert hedger.criterion.__class__.__name__ == "EntropicRiskMeasure"

    def test_risk_measure_case_insensitive(self):
        hedger1 = create_deep_hedger(risk_measure="ENTROPIC")
        hedger2 = create_deep_hedger(risk_measure="entropic")
        assert hedger1.criterion.__class__.__name__ == "EntropicRiskMeasure"
        assert hedger2.criterion.__class__.__name__ == "EntropicRiskMeasure"

    def test_invalid_risk_measure(self):
        with pytest.raises(ValueError, match="Unsupported risk_measure"):
            create_deep_hedger(risk_measure="invalid")

    def test_invalid_risk_measure_shows_available(self):
        with pytest.raises(ValueError, match="Available options"):
            create_deep_hedger(risk_measure="unknown")

    def test_custom_features(self):
        # Use features that pfhedge recognizes
        custom_features = ["log_moneyness", "volatility"]
        hedger = create_deep_hedger(features=custom_features)
        # Hedger converts feature names to Feature objects
        assert len(hedger.inputs) == len(custom_features)

    def test_default_features(self):
        hedger = create_deep_hedger()
        assert len(hedger.inputs) > 0
        # Check if any reasonable feature is in inputs
        assert any(
            feat in str(hedger.inputs) for feat in ["moneyness", "expiry", "volatility"]
        )

    def test_mlp_with_list_units(self):
        hedger = create_deep_hedger(
            model_type="mlp", n_layers=4, n_units=[128, 64, 32, 16]
        )
        assert isinstance(hedger.model, MultiLayerPerceptron)

    def test_lstm_with_list_units_uses_first(self):
        hedger = create_deep_hedger(
            model_type="lstm", n_layers=2, n_units=[128, 64, 32]
        )
        assert hedger.model.hidden_size == 128

    def test_gru_with_list_units_uses_first(self):
        hedger = create_deep_hedger(model_type="gru", n_layers=2, n_units=[64, 32])
        assert hedger.model.hidden_size == 64

    def test_all_model_types_with_all_risk_measures(self):
        model_types = ["mlp", "lstm", "gru"]
        risk_measures_params = [
            ("expected_shortfall", 0.95),
            ("entropic", 0.1),
            ("entropic_loss", 0.1),
            ("quadratic_cvar", 2.0),  # lam must be >= 1
        ]

        for model_type in model_types:
            for risk_measure, risk_param in risk_measures_params:
                hedger = create_deep_hedger(
                    model_type=model_type,
                    risk_measure=risk_measure,
                    risk_param=risk_param,
                    n_layers=2,
                    n_units=32,
                )
                assert isinstance(hedger, Hedger)


class TestModelRegistry:

    def test_registry_contains_all_models(self):
        from crypto.strategies.deep_hedge_utils import _MODEL_REGISTRY

        assert "mlp" in _MODEL_REGISTRY
        assert "lstm" in _MODEL_REGISTRY
        assert "gru" in _MODEL_REGISTRY

    def test_criterion_registry_contains_all_measures(self):
        from crypto.strategies.deep_hedge_utils import _CRITERION_REGISTRY

        assert "expected_shortfall" in _CRITERION_REGISTRY
        assert "entropic" in _CRITERION_REGISTRY
        assert "entropic_loss" in _CRITERION_REGISTRY
        assert "quadratic_cvar" in _CRITERION_REGISTRY

    def test_model_creators_are_callable(self):
        from crypto.strategies.deep_hedge_utils import _MODEL_REGISTRY

        for name, creator in _MODEL_REGISTRY.items():
            assert callable(creator), f"Model creator '{name}' is not callable"

    def test_criterion_creators_are_callable(self):
        from crypto.strategies.deep_hedge_utils import _CRITERION_REGISTRY

        for name, creator in _CRITERION_REGISTRY.items():
            assert callable(creator), f"Criterion creator '{name}' is not callable"


class TestModelIntegration:

    def test_lstm_hedger_forward_pass(self):
        hedger = create_deep_hedger(model_type="lstm", n_layers=2, n_units=64)
        # Simulate option data: (n_paths, n_steps, n_features)
        x = torch.randn(100, 42, 4)
        output = hedger.model(x)
        assert output.shape == (100, 42, 1)

    def test_gru_hedger_forward_pass(self):
        hedger = create_deep_hedger(model_type="gru", n_layers=2, n_units=64)
        x = torch.randn(100, 42, 4)
        output = hedger.model(x)
        assert output.shape == (100, 42, 1)

    def test_models_are_trainable(self):
        for model_type in ["mlp", "lstm", "gru"]:
            hedger = create_deep_hedger(model_type=model_type, n_layers=2, n_units=32)
            params = list(hedger.parameters())
            assert len(params) > 0, f"{model_type} has no parameters"
            assert all(
                p.requires_grad for p in params
            ), f"{model_type} has non-trainable parameters"

    def test_models_produce_different_outputs(self):
        torch.manual_seed(42)
        x = torch.randn(50, 20, 4)

        mlp_hedger = create_deep_hedger(model_type="mlp", n_layers=2, n_units=64)
        lstm_hedger = create_deep_hedger(model_type="lstm", n_layers=2, n_units=64)
        gru_hedger = create_deep_hedger(model_type="gru", n_layers=2, n_units=64)

        with torch.no_grad():
            mlp_out = mlp_hedger.model(x)
            lstm_out = lstm_hedger.model(x)
            gru_out = gru_hedger.model(x)

        # Outputs should be different (different architectures)
        assert not torch.allclose(mlp_out, lstm_out)
        assert not torch.allclose(lstm_out, gru_out)
        assert not torch.allclose(mlp_out, gru_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
