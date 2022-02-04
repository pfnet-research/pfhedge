import torch
from torch.testing import assert_close

from pfhedge.stochastic import randn_antithetic


def test_randn_antithetic():
    torch.manual_seed(42)
    output = randn_antithetic((200, 100))
    assert_close(output.mean(0), torch.zeros_like(output[0]))
