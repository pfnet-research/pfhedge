import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")

from pfhedge.nn.functional import svi_variance

if __name__ == "__main__":
    a, b, rho, m, sigma = 0.03, 0.10, 0.10, 0.00, 0.10
    k = torch.linspace(-0.10, 0.10, 100)
    v = svi_variance(k, a=a, b=b, rho=rho, m=m, sigma=sigma)

    plt.plot(k.numpy(), v.numpy())
    plt.savefig("output/svi.pdf")
