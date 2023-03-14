from typing import Optional
from typing import Union

import pytest
import torch

from pfhedge.nn import MultiLayerPerceptron


class TestMultiLayerPerceptron:
    """
    pfhedge.nn.MultiLayerPerceptron
    """

    @pytest.mark.parametrize("out_features", [1, 2])
    def test_out_features(
        self, out_features, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        m = MultiLayerPerceptron(out_features=out_features).to(device)
        assert m[-2].out_features == out_features

    @pytest.mark.gpu
    @pytest.mark.parametrize("out_features", [1, 2])
    def test_out_features_gpu(self, out_features):
        self.test_out_features(out_features, device="cuda")

    @pytest.mark.parametrize("n_layers", [1, 4, 10])
    def test_n_layers(
        self, n_layers, device: Optional[Union[str, torch.device]] = "cpu"
    ):
        m = MultiLayerPerceptron(n_layers=n_layers).to(device)
        assert len(m) == 2 * (n_layers + 1)

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_layers", [1, 4, 10])
    def test_n_layers_gpu(self, n_layers):
        self.test_n_layers(n_layers, device="cuda")

    @pytest.mark.parametrize("n_units", [2, 8, 32])
    @pytest.mark.parametrize("in_features", [1, 2])
    @pytest.mark.parametrize("out_features", [1, 2])
    def test_n_units(
        self,
        n_units,
        in_features,
        out_features,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        n_layers = 4
        m = MultiLayerPerceptron(
            n_layers=n_layers, n_units=n_units, out_features=out_features
        ).to(device)
        _ = m(torch.zeros((1, in_features)).to(device))

        for i in range(n_layers + 1):
            linear = m[2 * i]
            in_expect = in_features if i == 0 else n_units
            out_expect = out_features if i == n_layers else n_units
            assert linear.in_features == in_expect
            assert linear.out_features == out_expect

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_units", [2, 8, 32])
    @pytest.mark.parametrize("in_features", [1, 2])
    @pytest.mark.parametrize("out_features", [1, 2])
    def test_n_units_gpu(self, n_units, in_features, out_features):
        self.test_n_units(n_units, in_features, out_features, device="cuda")

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

        input = torch.zeros((N, H_in))
        m = MultiLayerPerceptron(H_in, H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.zeros((N, M_1, H_in))
        m = MultiLayerPerceptron(H_in, H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.zeros((N, M_1, M_2, H_in))
        m = MultiLayerPerceptron(H_in, H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))

    def test_shape_lazy(self):
        N = 10
        H_in = 11
        M_1 = 12
        M_2 = 13
        H_out = 14

        input = torch.zeros((N, H_in))
        m = MultiLayerPerceptron(out_features=H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.zeros((N, M_1, H_in))
        m = MultiLayerPerceptron(out_features=H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.zeros((N, M_1, M_2, H_in))
        m = MultiLayerPerceptron(out_features=H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))
