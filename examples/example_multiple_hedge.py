# Example to use multiple hedging instruments.

import sys

import torch
from torch import Tensor

sys.path.append("..")

from pfhedge.instruments import HestonStock
from pfhedge.instruments import EuropeanOption
from pfhedge.instruments import VarianceSwap
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron

if __name__ == "__main__":
    torch.manual_seed(42)

    def pricer(varswap) -> Tensor:
        return varswap.ul().variance

    stock = HestonStock()
    option = EuropeanOption(stock)
    varswap = VarianceSwap(stock)
    varswap.list(pricer)

    # Fit and price: stock
    torch.manual_seed(42)
    model = MultiLayerPerceptron()
    hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"])
    hedger.fit(option, n_paths=10000, n_epochs=200)
    price = hedger.price(option, n_paths=10000, n_times=10)
    print(f"Price={price:.5e}")

    # Fit and price: stock and variance swap
    torch.manual_seed(42)
    model = MultiLayerPerceptron(out_features=2)
    hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility"])
    hedger.fit(option, hedge=[stock, varswap], n_paths=10000, n_epochs=200)
    price = hedger.price(option, hedge=[stock, varswap], n_paths=10000, n_times=10)
    print(f"Price={price:.5e}")
