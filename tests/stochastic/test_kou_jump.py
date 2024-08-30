from math import sqrt

import pytest
import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_kou_jump
from pfhedge.stochastic.engine import RandnSobolBoxMuller
from tests.stochastic.test_merton_jump import (
    TestGenerateMertonJumpStock as BaseGenerateJumpStockTest,
)


class TestGenerateKouJumpStock(BaseGenerateJumpStockTest):
    func = staticmethod(generate_kou_jump)

    def test_generate_brownian_mean_no_jump(self, device: str = "cpu"):
        # kou jump has no std
        return

    def test_generate_brownian_mean_no_jump_std(self, device: str = "cpu"):
        # kou jump has no std
        return

    def test_generate_jump_nosigma2(self, device: str = "cpu"):
        # kou jump has no std
        return

    def test_generate_jump_std2(self, device: str = "cpu"):
        # kou jump has no std
        return

    # addtional tests for Kou jumo model params
    def test_kou_jump_mean_up(self):
        n_paths = 10000
        n_steps = 250
        jump_mean_up = 1.1
        with (
            pytest.raises(
                ValueError, match="jump_mean_up must be postive and smaller than 1"
            )
        ):
            generate_kou_jump(n_paths, n_steps, jump_mean_up=jump_mean_up)

    def test_kou_jump_mean_down(self):
        n_paths = 10000
        n_steps = 250
        jump_mean_down = 0.0
        with pytest.raises(ValueError, match="jump_mean_down must be postive"):
            generate_kou_jump(n_paths, n_steps, jump_mean_down=jump_mean_down)

    def test_kou_up_jump_prob(self):
        n_paths = 10000
        n_steps = 250
        jump_up_prob = 1.1
        with pytest.raises(ValueError, match="jump prob must be in 0 and 1 incl"):
            generate_kou_jump(n_paths, n_steps, jump_up_prob=jump_up_prob)
