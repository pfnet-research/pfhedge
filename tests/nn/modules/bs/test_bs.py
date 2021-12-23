import torch

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import LookbackOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import BSEuropeanOption
from pfhedge.nn import BSLookbackOption


class TestBlackScholes:
    def test_init(self):
        derivative = EuropeanOption(BrownianStock())
        m = BlackScholes(derivative)
        assert m.__class__ == BSEuropeanOption
        assert m.strike == 1.0
        assert m.call

        derivative = EuropeanOption(BrownianStock(), strike=2.0, call=False)
        m = BlackScholes(derivative)
        assert m.__class__ == BSEuropeanOption
        assert m.strike == 2.0
        assert not m.call

        derivative = LookbackOption(BrownianStock())
        m = BlackScholes(derivative)
        assert m.__class__ == BSLookbackOption
        assert m.strike == 1.0
        assert m.call

        # not implemented
        # derivative = LookbackOption(BrownianStock(), strike=2.0, call=False)
        # m = BS(derivative)
        # assert m.__class__ == BSLookbackOption
        # assert m.strike == 2.0
        # assert not m.call

    def test_shape(self):
        torch.distributions.Distribution.set_default_validate_args(False)

        deriv = EuropeanOption(BrownianStock())
        m = BlackScholes(deriv)

        N = 10
        H_in = len(m.inputs())
        M_1 = 12
        M_2 = 13

        input = torch.empty((N, H_in))
        assert m(input).size() == torch.Size((N, 1))

        input = torch.empty((N, M_1, H_in))
        assert m(input).size() == torch.Size((N, M_1, 1))

        input = torch.empty((N, M_1, M_2, H_in))
        assert m(input).size() == torch.Size((N, M_1, M_2, 1))
