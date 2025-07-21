#自动微分计算希腊字母
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
    # If we go with float32, autograd becomes NaN because:
    # width in WhalleyWilmott ~ (gamma) ** 1/3, gamma is small to become zero,
    # grad of width diverges (resulting in nan) at zero
    torch.set_default_dtype(torch.float64)
    #设为float64精度，避免计算中出现NaN

    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    model = WhalleyWilmott(derivative)
    hedger = Hedger(model, inputs=model.inputs()) #type: ignore

    def pricer(spot):
        return hedger.price(derivative, init_state=(spot,), enable_grad=True)

    #一阶敏感度Delta
    delta = autogreek.delta(pricer, spot=torch.tensor(1.0))
    print("Delta:", delta)

    #二阶敏感度Gamma
    gamma = autogreek.gamma(pricer, spot=torch.tensor(1.0))
    print("Gamma:", gamma)
