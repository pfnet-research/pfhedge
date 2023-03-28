import pytest
import torch
from torch.testing import assert_close

from pfhedge.stochastic import randn_antithetic


def test_randn_antithetic(device: str = "cpu"):
    torch.manual_seed(42)
    device = torch.device(device) if device else None
    output = randn_antithetic(200, 100, device=device)
    assert_close(output.mean(0), torch.zeros_like(output[0]))

    with pytest.raises(ValueError):
        # not supported
        output = randn_antithetic((200, 100), dim=1)


@pytest.mark.gpu
def test_randn_antithetic_gpu():
    test_randn_antithetic(device="cuda")
