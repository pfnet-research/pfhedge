# Example to use an output of a module as a feature by `ModuleOutput`.
# Here we use Black-Scholes' delta as a feature.

import sys

import torch

sys.path.append("..")

from pfhedge.features import ExpiryTime
from pfhedge.features import LogMoneyness
from pfhedge.features import ModuleOutput
from pfhedge.features import Volatility
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron

if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    deriv = EuropeanOption(BrownianStock(cost=1e-4))

    # bs is a module that outputs Black-Scholes' delta
    bs = BlackScholes(deriv)
    delta = ModuleOutput(bs, inputs=[LogMoneyness(), ExpiryTime(), Volatility()])

    # Create your hedger
    # Here `delta` is a feature that outputs Black-Scholes' delta
    model = MultiLayerPerceptron()
    hedger = Hedger(model, [delta, "prev_hedge"])

    # Fit and price
    hedger.fit(deriv, n_paths=10000, n_epochs=200)
    price = hedger.price(deriv, n_paths=10000)
    print(f"Price={price:.5e}")
