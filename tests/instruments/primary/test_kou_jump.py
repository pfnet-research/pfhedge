import pytest
import torch

from pfhedge.instruments import KouJumpStock
from tests.instruments.primary.test_merton_jump import (
    TestMertonJumpStock as BaseJumpStockTest,
)


class TestKouJumpStock(BaseJumpStockTest):
    cls = KouJumpStock

    def test_repr(self):
        s = KouJumpStock(cost=1e-4)
        expect = "KouJumpStock(\
sigma=0.2000, mu=0., cost=1.0000e-04, dt=0.0040, jump_per_year=68., jump_mean_up=0.0200, jump_mean_down=0.0500, jump_up_prob=0.5000)"
        assert repr(s) == expect
