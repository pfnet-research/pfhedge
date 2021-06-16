from pfhedge import Hedger
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn import WhalleyWilmott


def test_net():
    liability = EuropeanOption(BrownianStock(cost=1e-4))
    model = MultiLayerPerceptron()
    hedger = Hedger(model, ["log_moneyness", "expiry_time", "volatility", "prev_hedge"])
    _ = hedger.fit(liability, n_paths=100, n_epochs=10)
    _ = hedger.price(liability)


def test_bs():
    liability = EuropeanOption(BrownianStock(cost=1e-4))
    model = BlackScholes(liability)
    hedger = Hedger(model, model.inputs())
    _ = hedger.price(liability)


def test_ww():
    liability = EuropeanOption(BrownianStock(cost=1e-4))
    model = WhalleyWilmott(liability)
    hedger = Hedger(model, model.inputs())
    _ = hedger.price(liability)
