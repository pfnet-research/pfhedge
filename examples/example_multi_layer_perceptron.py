# Example to use a multi-layer perceptron as a hedging model

import sys

import torch

sys.path.append("..")

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron

if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = MultiLayerPerceptron()
    hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility", "prev_hedge"])

    # Fit and price
    hedger.fit(derivative, n_paths=10000, n_epochs=200)
    price = hedger.price(derivative, n_paths=10000)
    print(f"Price={price:.5e}")
