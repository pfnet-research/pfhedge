# Example to compute greeks using autogreek

import sys

import torch

sys.path.append("..")

import pfhedge.autogreek as autogreek
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import Hedger
from pfhedge.nn import WhalleyWilmott

if __name__ == "__main__":
    torch.manual_seed(42)
    # If we go with float32, autograd becomes nan because:
    # width in WhalleyWilmott ~ (gamma) ** 1/3, gamma is small to become zero,
    # grad of width diverges (resulting in nan) at zero
    torch.set_default_dtype(torch.float64)

    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    model = WhalleyWilmott(derivative)
    hedger = Hedger(model, inputs=model.inputs())

    def pricer(spot):
        return hedger.price(
            derivative, n_paths=10000, init_state=(spot,), enable_grad=True
        )

    delta = autogreek.delta(pricer, spot=torch.tensor(1.0))
    print("Delta:", delta)
    gamma = autogreek.gamma(pricer, spot=torch.tensor(1.0))
    print("Gamma:", gamma)
