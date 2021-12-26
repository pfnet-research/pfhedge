import sys
import torch
import matplotlib.pyplot as plt

sys.path.append("..")

from pfhedge.nn import BlackScholes
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption

if __name__ == "__main__":
    options_list = []
    strikes_list = []
    for call in (True, False):
        for strike in torch.arange(70, 180, 10):
            option = EuropeanOption(BrownianStock(), call=call, strike=strike)
            options_list.append(option)
            strikes_list.append(strike)

    spot = torch.linspace(50, 200, 100)
    t = options_list[0].maturity
    v = options_list[0].ul().sigma

    plt.figure()
    total_vega = torch.zeros_like(spot)
    for option, strike in zip(options_list, strikes_list):
        lm = (spot / strike).log()
        vega = BlackScholes(option).vega(lm, t, v) / (strike ** 2)
        total_vega += vega
        plt.plot(spot.numpy(), vega.numpy())
    plt.plot(spot.numpy(), total_vega.numpy(), color="k", lw=2)
    plt.savefig("./output/options-vega.png")
