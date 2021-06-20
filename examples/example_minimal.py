# Here is a minimal example to try out Deep Hedging.

import sys

sys.path.append("..")

if __name__ == "__main__":
    from pfhedge.instruments import BrownianStock
    from pfhedge.instruments import EuropeanOption
    from pfhedge.nn import Hedger
    from pfhedge.nn import MultiLayerPerceptron

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = MultiLayerPerceptron()
    hedger = Hedger(
        model, inputs=["log_moneyness", "expiry_time", "volatility", "prev_hedge"]
    )

    # Fit and price
    hedger.fit(derivative)
    price = hedger.price(derivative)
