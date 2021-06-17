import pytest
import torch

from pfhedge.nn import MultiLayerPerceptron


class TestMultiLayerPerceptron:
    """
    pfhedge.nn.MultiLayerPerceptron
    """

    @pytest.mark.parametrize("out_features", [1, 2])
    def test_out_features(self, out_features):
        m = MultiLayerPerceptron(out_features=out_features)
        assert m[-2].out_features == out_features

    @pytest.mark.parametrize("n_layers", [1, 4, 10])
    def test_n_layers(self, n_layers):
        m = MultiLayerPerceptron(n_layers=n_layers)
        assert len(m) == 2 * (n_layers + 1)

    @pytest.mark.parametrize("n_units", [2, 8, 32])
    @pytest.mark.parametrize("in_features", [1, 2])
    @pytest.mark.parametrize("out_features", [1, 2])
    def test_n_units(self, n_units, in_features, out_features):
        n_layers = 4
        m = MultiLayerPerceptron(
            n_layers=n_layers, n_units=n_units, out_features=out_features
        )
        _ = m(torch.empty((1, in_features)))

        for i in range(n_layers + 1):
            linear = m[2 * i]
            in_expect = in_features if i == 0 else n_units
            out_expect = out_features if i == n_layers else n_units
            assert linear.in_features == in_expect
            assert linear.out_features == out_expect

    @pytest.mark.parametrize("activation", [torch.nn.ELU(), torch.nn.CELU()])
    @pytest.mark.parametrize("out_activation", [torch.nn.ELU(), torch.nn.CELU()])
    def test_activation(self, activation, out_activation):
        n_layers = 4
        m = MultiLayerPerceptron(
            n_layers=n_layers, activation=activation, out_activation=out_activation
        )

        for i in range(n_layers + 1):
            expect = out_activation if i == n_layers else activation
            activ = m[2 * i + 1]
            assert isinstance(activ, expect.__class__)

    def test_shape(self):
        N = 10
        H_in = 11
        M_1 = 12
        M_2 = 13
        H_out = 14

        input = torch.empty((N, H_in))
        m = MultiLayerPerceptron(H_in, H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.empty((N, M_1, H_in))
        m = MultiLayerPerceptron(H_in, H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.empty((N, M_1, M_2, H_in))
        m = MultiLayerPerceptron(H_in, H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))

    def test_shape_lazy(self):
        N = 10
        H_in = 11
        M_1 = 12
        M_2 = 13
        H_out = 14

        input = torch.empty((N, H_in))
        m = MultiLayerPerceptron(out_features=H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.empty((N, M_1, H_in))
        m = MultiLayerPerceptron(out_features=H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.empty((N, M_1, M_2, H_in))
        m = MultiLayerPerceptron(out_features=H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))
