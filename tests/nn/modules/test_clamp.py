import torch

from pfhedge.nn import Clamp
from pfhedge.nn import LeakyClamp


class TestLeakyClamp:
    """
    pfhedge.nn.LeakyClamp
    """

    def test_out(self):
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0])

        result = LeakyClamp(0.1)(x, 0, 1)
        expect = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1])
        assert torch.allclose(result, expect)

        result = LeakyClamp(0.01)(x, 0, 0)
        expect = 0.01 * x
        assert torch.allclose(result, expect)

        result = LeakyClamp(1.0)(x, 0, 1)
        expect = x
        assert torch.allclose(result, expect)

        result = LeakyClamp(0.01)(x, 1, 0)
        expect = torch.tensor(0.5)
        assert torch.allclose(result, expect)

        result = LeakyClamp(0.0)(x, 0, 1)
        expect = Clamp()(x, 0, 1)
        assert torch.allclose(result, expect)

    def test_repr(self):
        assert repr(LeakyClamp(0.1)) == "LeakyClamp(clamped_slope=0.1)"
