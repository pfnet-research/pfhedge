import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")

import pfhedge.autogreek as autogreek
from pfhedge.nn import BSEuropeanOption
from pfhedge.nn.functional import svi_variance

if __name__ == "__main__":
    a, b, rho, m, sigma = 0.02, 0.10, -0.40, 0.00, 0.20
    k = torch.linspace(-0.10, 0.10, 100)
    v = svi_variance(k, a=a, b=b, rho=rho, m=m, sigma=sigma)

    plt.figure()
    plt.plot(k.numpy(), v.numpy())
    plt.xlabel("Log strike")
    plt.xlabel("Variance")
    plt.savefig("output/svi_variance.pdf")
    print("Saved figure to output/svi_variance.pdf")

    bs = BSEuropeanOption()
    t = torch.full_like(k, 0.1)
    delta_sticky_strike = bs.delta(-k, t, v.sqrt())

    def price_sticky_delta(log_moneyness):
        v = svi_variance(-log_moneyness, a=a, b=b, rho=rho, m=m, sigma=sigma)
        return bs.price(log_moneyness, t, v.sqrt())

    log_moneyness = -k.requires_grad_()
    delta_sticky_delta = autogreek.delta(
        price_sticky_delta, log_moneyness=log_moneyness
    )
    k.detach_()

    plt.figure()
    plt.plot(-k.numpy(), delta_sticky_strike.numpy(), label="Delta for sticky strike")
    plt.plot(-k.numpy(), delta_sticky_delta.numpy(), label="Delta for sticky delta")
    plt.xlabel("Log strike")
    plt.xlabel("Delta")
    plt.legend()
    plt.savefig("output/svi_delta.pdf")
    print("Saved figure to output/svi_delta.pdf")
