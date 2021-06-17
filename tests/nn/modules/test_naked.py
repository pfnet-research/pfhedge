import pytest
import torch

from pfhedge.nn import Naked


class TestNaked:
    """
    pfhedge.nn.Naked
    """

    @pytest.mark.parametrize("n_paths", [1, 10])
    @pytest.mark.parametrize("n_features", [1, 10])
    def test(self, n_paths, n_features):
        m = Naked()
        input = torch.empty((n_paths, n_features))
        assert torch.equal(m(input), torch.zeros((n_paths, 1)))

    def test_shape(self):
        N = 11
        H_in = 12
        M_1 = 13
        M_2 = 14
        H_out = 15

        input = torch.empty((N, H_in))
        m = Naked(H_out)
        assert m(input).size() == torch.Size((N, H_out))

        input = torch.empty((N, M_1, H_in))
        m = Naked(H_out)
        assert m(input).size() == torch.Size((N, M_1, H_out))

        input = torch.empty((N, M_1, M_2, H_in))
        m = Naked(H_out)
        assert m(input).size() == torch.Size((N, M_1, M_2, H_out))
