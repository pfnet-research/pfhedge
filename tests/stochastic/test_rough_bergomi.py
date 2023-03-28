import pytest
import torch
from torch.testing import assert_close

from pfhedge.stochastic.rough_bergomi import generate_rough_bergomi


def test_generate_heston_repr():
    torch.manual_seed(42)
    output = generate_rough_bergomi(2, 5)
    expect = """\
SpotVarianceTuple(
  spot=
  tensor([[1.0000, 0.9807, 0.9563, 0.9540, 0.9570],
          [1.0000, 1.0147, 1.0097, 1.0107, 1.0164]])
  variance=
  tensor([[0.0400, 0.3130, 0.0105, 0.0164, 0.0068],
          [0.0400, 0.0396, 0.0049, 0.0064, 0.0149]])
)"""
    assert repr(output) == expect


def test_generate_heston_volatility(device: str = "cpu"):
    torch.manual_seed(42)

    output = generate_rough_bergomi(100, 250, device=device)
    assert_close(output.volatility, output.variance.sqrt())


@pytest.mark.gpu
def test_generate_heston_volatility_gpu():
    test_generate_heston_volatility(device="cuda")


@pytest.mark.skipif(True, reason="for development")
def test_generate_rough_bergomi() -> None:

    import numpy as np
    from scipy.optimize import brentq  # type: ignore # mypy ignore
    from scipy.stats import norm  # type: ignore # mypy ignore

    torch.manual_seed(42)

    # referring the original implementation
    # https://github.com/ryanmccrickerd/rough_bergomi/blob/master/notebooks/rbergomi.ipynb
    def bs(F, K, V, o="call"):
        """
        Returns the Black call price for given forward, strike and integrated variance.
        referring: https://github.com/ryanmccrickerd/rough_bergomi/blob/master/rbergomi/utils.py#L29-L45
        """
        # Set appropriate weight for option token o
        w = 1
        if o == "put":
            w = -1
        elif o == "otm":
            w = 2 * (K > 1.0) - 1

        sv = np.sqrt(V)
        d1 = np.log(F / K) / sv + 0.5 * sv
        d2 = d1 - sv
        P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
        return P

    def bsinv(P, F, K, t, o="call"):
        """
        Returns implied Black vol from given call price, forward, strike and time to maturity.
        referring: https://github.com/ryanmccrickerd/rough_bergomi/blob/master/rbergomi/utils.py#L29-L45
        """
        # Set appropriate weight for option token o
        w = 1
        if o == "put":
            w = -1
        elif o == "otm":
            w = 2 * (K > 1.0) - 1

        # Ensure at least instrinsic value
        P = np.maximum(P, np.maximum(w * (F - K), 0))

        def error(s):
            return bs(F, K, s ** 2 * t, o) - P

        s = brentq(error, 1e-9, 1e9)
        return s

    vec_bsinv = np.vectorize(bsinv)

    price, variance = generate_rough_bergomi(
        n_paths=30000,
        n_steps=100,
        alpha=-0.43,
        rho=-0.9,
        eta=1.9,
        xi=0.235 ** 2,
        dt=1 / 100,
        dtype=torch.float64,
    )
    k = np.arange(-0.5, 0.51, 0.01)

    ST = price.numpy()[:, -1][:, np.newaxis]
    K = np.exp(k)[np.newaxis, :]
    call_payoffs = np.maximum(ST - K, 0)
    call_prices = np.mean(call_payoffs, axis=0)[:, np.newaxis]
    implied_vols = vec_bsinv(call_prices, 1.0, np.transpose(K), 1.0)
    # import matplotlib.pyplot as plt
    # plt.plot(k, implied_vols)
    # plt.show()
