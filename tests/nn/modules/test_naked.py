import pytest
import torch

from pfhedge.nn import Naked


class TestNaked:
    """
    pfhedge.nn.Naked
    """

    @pytest.mark.parametrize("n_paths", [1, 10])
    @pytest.mark.parametrize("n_features", [1, 10])
    def test(self, n_paths, n_features, device: str = "cpu"):
        m = Naked().to(device)
        input = torch.zeros((n_paths, n_features)).to(device)
        assert torch.equal(m(input), torch.zeros((n_paths, 1)).to(device))

    @pytest.mark.gpu
    @pytest.mark.parametrize("n_paths", [1, 10])
    @pytest.mark.parametrize("n_features", [1, 10])
    def test_gpu(self, n_paths, n_features):
        self.test(n_paths, n_features, device="cuda")

    def test_shape(self, device: str = "cpu"):
        N = 11
        H_in = 12
        M_1 = 13
        M_2 = 14
        H_out = 15

        input = torch.zeros((N, H_in)).to(device)
        m = Naked(H_out).to(device)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.zeros((N, M_1, H_in)).to(device)
        m = Naked(H_out).to(device)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.zeros((N, M_1, M_2, H_in)).to(device)
        m = Naked(H_out).to(device)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))

    @pytest.mark.gpu
    def test_shape_gpu(self):
        self.test_shape(device="cuda")
