# Example to use Whalley-Wilmott's asymptotically optimal strategy
# for small cost as a hedging model

import sys

import torch

sys.path.append("..")
from pfhedge import Hedger
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import WhalleyWilmott

if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    deriv = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = WhalleyWilmott(deriv)
    hedger = Hedger(model, model.inputs())

    # Fit and price
    price = hedger.price(deriv, n_paths=10000)
    print(f"Price={price:.5e}")
