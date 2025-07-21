#BS模型：在无交易成本假设下，连续动态调整持仓以保持Delta中性

import sys

import torch

sys.path.append("..")

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Hedger

if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = BlackScholes(derivative)
    hedger = Hedger(model, model.inputs()) #type: ignore

    # Fit and price
    price = hedger.price(derivative, n_paths=10000)
    print(f"Price={price:.5e}")
