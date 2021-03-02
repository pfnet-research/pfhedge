# Example to use Black-Scholes' delta-hedging strategy as a hedging model

import sys

import torch

sys.path.append("..")

from pfhedge import Hedger  # noqa: E402
from pfhedge.instruments import BrownianStock  # noqa: E402
from pfhedge.instruments import EuropeanOption  # noqa: E402
from pfhedge.nn import BlackScholes  # noqa: E402

if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    deriv = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = BlackScholes(deriv)
    hedger = Hedger(model, model.features())

    # Fit and price
    price = hedger.price(deriv, n_paths=10000)
    print(f"Price={price:.5e}")
