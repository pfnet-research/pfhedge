from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn import WhalleyWilmott


def test_net():
    derivative = EuropeanOption(BrownianStock(cost=1e-4))
    model = MultiLayerPerceptron()
    hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility", "prev_hedge"])
    _ = hedger.fit(derivative, n_paths=100, n_epochs=10)
    _ = hedger.price(derivative)


def test_bs():
    derivative = EuropeanOption(BrownianStock(cost=1e-4))
    model = BlackScholes(derivative)
    hedger = Hedger(model, model.inputs())
    _ = hedger.price(derivative)


def test_ww():
    derivative = EuropeanOption(BrownianStock(cost=1e-4))
    model = WhalleyWilmott(derivative)
    hedger = Hedger(model, model.inputs())
    _ = hedger.price(derivative)
