import sys
from math import exp

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import HestonStock
from pfhedge.nn import BlackScholes

LOG_MONEYNESS_MIN = -0.1
LOG_MONEYNESS_MAX = 0.1
LOG_MONEYNESS_STEPS = 20
LOG_MONEYNESS_RANGE = torch.linspace(
    LOG_MONEYNESS_MIN, LOG_MONEYNESS_MAX, LOG_MONEYNESS_STEPS
)


def compute_iv(log_moneyness: float, rho: float) -> float:
    torch.manual_seed(42)

    d = EuropeanOption(HestonStock(rho=rho))
    spot = exp(log_moneyness) * d.strike
    d.simulate(n_paths=int(1e5), init_state=(spot, d.ul().theta))
    p = d.payoff().mean(0)

    return BlackScholes(d).implied_volatility(log_moneyness, d.maturity, p).item()


def main():
    plt.figure()
    for rho in [-0.7, 0.0, 0.7]:
        y = [compute_iv(s, rho) for s in LOG_MONEYNESS_RANGE]
        plt.plot(LOG_MONEYNESS_RANGE, y, label=f"rho={rho}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Implied volatility")
    plt.legend()
    plt.savefig("output/heston-iv.png")


if __name__ == "__main__":
    main()
