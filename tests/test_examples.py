import pytest

from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron
from pfhedge.nn import WhalleyWilmott


def test_net(device: str = "cpu"):
    derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(device)
    model = MultiLayerPerceptron().to(device)
    hedger = Hedger(
        model, ["log_moneyness", "time_to_maturity", "volatility", "prev_hedge"]
    ).to(device)
    _ = hedger.fit(derivative, n_paths=100, n_epochs=10)
    _ = hedger.price(derivative)

    _ = hedger.fit(
        derivative, n_paths=100, n_epochs=10, tqdm_kwargs={"desc": "description"}
    )


@pytest.mark.gpu
def test_net_gpu():
    test_net(device="cuda")


def test_bs(device: str = "cpu"):
    derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(device)
    model = BlackScholes(derivative).to(device)
    hedger = Hedger(model, model.inputs()).to(device)
    _ = hedger.price(derivative)


@pytest.mark.gpu
def test_bs_gpu():
    test_bs(device="cuda")


def test_ww(device: str = "cpu"):
    derivative = EuropeanOption(BrownianStock(cost=1e-4)).to(device)
    model = WhalleyWilmott(derivative).to(device)
    hedger = Hedger(model, model.inputs()).to(device)
    _ = hedger.price(derivative)


@pytest.mark.gpu
def test_ww_gpu():
    test_ww(device="cuda")
