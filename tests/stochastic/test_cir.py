import torch

from pfhedge.stochastic import generate_cir


def test_dtype():
    output = generate_cir(2, 3, dtype=torch.float32)
    assert output.dtype == torch.float32

    output = generate_cir(2, 3, dtype=torch.float64)
    assert output.dtype == torch.float64
