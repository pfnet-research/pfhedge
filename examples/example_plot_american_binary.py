import sys

import matplotlib.pyplot as plt
import torch

sys.path.append("..")
from pfhedge.nn import BSAmericanBinaryOption

LOG_MONEYNESS_MIN = -0.2
LOG_MONEYNESS_MAX = 0.2
LOG_MONEYNESS_STEPS = 1000
S = torch.linspace(LOG_MONEYNESS_MIN, LOG_MONEYNESS_MAX, LOG_MONEYNESS_STEPS)
VOLATILITY_RANGE = [0.1, 0.2, 0.3]
VOLATILITY_DEFAULT = 0.1
TIME_TO_MATURITY_RANGE = [0.1, 0.2, 0.3]
TIME_TO_MATURITY_DEFAULT = 0.1


def plot_delta() -> None:
    m = BSAmericanBinaryOption()

    plt.figure()
    for volatility in VOLATILITY_RANGE:
        s = S
        t = torch.full_like(s, TIME_TO_MATURITY_DEFAULT)
        v = torch.full_like(s, volatility)
        y = m.delta(s, s, t, v)
        plt.plot(s, y, label=f"Volatility={volatility:.1f}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Delta")
    plt.legend()
    plt.savefig("./output/american-binary-delta-volatility.png")

    plt.figure()
    for time_to_maturity in TIME_TO_MATURITY_RANGE:
        s = S
        t = torch.full_like(s, time_to_maturity)
        v = torch.full_like(s, TIME_TO_MATURITY_DEFAULT)
        y = m.delta(s, s, t, v)
        plt.plot(s, y, label=f"Time to maturity={time_to_maturity:.1f}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Delta")
    plt.legend()
    plt.savefig("./output/american-binary-delta-time.png")


def plot_gamma() -> None:
    m = BSAmericanBinaryOption()

    plt.figure()
    for volatility in VOLATILITY_RANGE:
        s = S
        t = torch.full_like(s, TIME_TO_MATURITY_DEFAULT)
        v = torch.full_like(s, volatility)
        y = m.gamma(s, s, t, v)
        plt.plot(s, y, label=f"Volatility={volatility:.1f}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Gamma")
    plt.legend()
    plt.savefig("./output/american-binary-gamma-volatility.png")

    plt.figure()
    for time_to_maturity in TIME_TO_MATURITY_RANGE:
        s = S
        t = torch.full_like(s, time_to_maturity)
        v = torch.full_like(s, TIME_TO_MATURITY_DEFAULT)
        y = m.gamma(s, s, t, v)
        plt.plot(s, y, label=f"Time to maturity={time_to_maturity:.1f}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Gamma")
    plt.legend()
    plt.savefig("./output/american-binary-gamma-time.png")


def plot_price() -> None:
    m = BSAmericanBinaryOption()

    plt.figure()
    for volatility in VOLATILITY_RANGE:
        s = S
        t = torch.full_like(s, TIME_TO_MATURITY_DEFAULT)
        v = torch.full_like(s, volatility)
        y = m.price(s, s, t, v)
        plt.plot(s, y, label=f"Volatility={volatility:.1f}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("./output/american-binary-price-volatility.png")

    plt.figure()
    for time_to_maturity in TIME_TO_MATURITY_RANGE:
        s = S
        t = torch.full_like(s, time_to_maturity)
        v = torch.full_like(s, TIME_TO_MATURITY_DEFAULT)
        y = m.price(s, s, t, v)
        plt.plot(s, y, label=f"Time to maturity={time_to_maturity:.1f}")
    plt.xlabel("Log moneyness")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("./output/american-binary-price-time.png")


def main():
    plot_delta()
    plot_gamma()
    plot_price()


if __name__ == "__main__":
    main()
