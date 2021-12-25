import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
from pfhedge.nn import BSLookbackOption
from pfhedge.stochastic import RandnSobolBoxMuller
from pfhedge.stochastic import generate_brownian


def main():
    N_PATHS = 100
    N_STEPS = 10

    torch.manual_seed(42)
    b_randn = generate_brownian(N_PATHS, N_STEPS)

    engine = RandnSobolBoxMuller(scramble=True, seed=42)
    b_sobol = generate_brownian(N_PATHS, N_STEPS, engine=engine)

    plt.figure()
    plt.plot(b_randn.numpy().T, label="randn")
    plt.savefig("./output/brownian-randn.png")

    plt.figure()
    plt.plot(b_sobol.numpy().T, label="sobol")
    plt.savefig("./output/brownian-sobol.png")


if __name__ == "__main__":
    main()
