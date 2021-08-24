import torch
from torch.testing import assert_close

from pfhedge.stochastic import generate_heston


def test_generate_heston_repr():
    torch.manual_seed(42)
    output = generate_heston(2, 5)
    expect = """\
HestonTuple(
  spot=
  tensor([[1.0000, 0.9941, 0.9905, 0.9846, 0.9706],
          [1.0000, 1.0031, 0.9800, 0.9785, 0.9735]])
  variance=
  tensor([[0.0400, 0.0408, 0.0411, 0.0417, 0.0422],
          [0.0400, 0.0395, 0.0452, 0.0434, 0.0446]])
)"""
    assert repr(output) == expect


def test_generate_heston_volatility():
    torch.manual_seed(42)

    output = generate_heston(100, 250)
    assert_close(output.volatility, output.variance.sqrt())
