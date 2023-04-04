import pytest
import torch
from torch.testing import assert_close

from pfhedge.nn import Clamp
from pfhedge.nn import LeakyClamp


class TestLeakyClamp:
    def test_output(self, device: str = "cpu"):
        input = torch.tensor([-1.0, 0.0, 0.5, 1.0, 2.0]).to(device)

        result = LeakyClamp(0.1)(input, 0, 1)
        expect = torch.tensor([-0.1, 0.0, 0.5, 1.0, 1.1]).to(device)
        assert_close(result, expect)

        result = LeakyClamp(0.01)(input, 0, 0)
        expect = 0.01 * input
        assert_close(result, expect)

        result = LeakyClamp(1.0)(input, 0, 1)
        expect = input
        assert_close(result, expect)

        result = LeakyClamp(0.01)(input, 1, 0)
        expect = torch.full_like(input, 0.5)
        assert_close(result, expect)

        result = LeakyClamp(0.0)(input, 0, 1)
        expect = Clamp()(input, 0, 1)
        assert_close(result, expect)

    @pytest.mark.gpu
    def test_output_gpu(self):
        self.test_output(device="cuda")

    def test_repr(self):
        expect = "LeakyClamp(clamped_slope=0.1000)"
        assert repr(LeakyClamp(0.1)) == expect
