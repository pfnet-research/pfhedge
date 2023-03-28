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

    output = generate_rough_bergomi(100, 250, device=torch.device(device))
    assert_close(output.volatility, output.variance.sqrt())


@pytest.mark.gpu
def test_generate_heston_volatility_gpu():
    test_generate_heston_volatility(device="cuda")


def test_generate_rough_bergomi(device: str = "cpu") -> None:

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
        device=torch.device(device),
    )
    k = np.arange(-0.5, 0.51, 0.01)

    ST = price.numpy()[:, -1][:, np.newaxis]
    K = np.exp(k)[np.newaxis, :]
    call_payoffs = np.maximum(ST - K, 0)
    call_prices = np.mean(call_payoffs, axis=0)[:, np.newaxis]
    implied_vols = vec_bsinv(call_prices, 1.0, np.transpose(K), 1.0)
    iv_expected = np.array(
        [
            [0.32451138],
            [0.3221199],
            [0.31975228],
            [0.31737825],
            [0.31499795],
            [0.31260074],
            [0.31017555],
            [0.30774781],
            [0.30533804],
            [0.30290662],
            [0.3004768],
            [0.29804816],
            [0.29561211],
            [0.2931631],
            [0.29070142],
            [0.28823201],
            [0.28574193],
            [0.28323994],
            [0.28073102],
            [0.27822864],
            [0.27572974],
            [0.27322318],
            [0.27069789],
            [0.26815457],
            [0.26558445],
            [0.26299189],
            [0.26037686],
            [0.25774189],
            [0.25510366],
            [0.25246651],
            [0.24982423],
            [0.24717092],
            [0.24452846],
            [0.241885],
            [0.23924862],
            [0.23660132],
            [0.23395069],
            [0.23128472],
            [0.22862045],
            [0.22593397],
            [0.22325611],
            [0.22056891],
            [0.21786946],
            [0.21516296],
            [0.21245403],
            [0.20972242],
            [0.20699638],
            [0.20428346],
            [0.20157387],
            [0.19886236],
            [0.19617408],
            [0.19352421],
            [0.19087771],
            [0.188256],
            [0.18565397],
            [0.18310724],
            [0.18061363],
            [0.1781444],
            [0.1757258],
            [0.1733737],
            [0.17108018],
            [0.16883135],
            [0.16666179],
            [0.16458221],
            [0.1627026],
            [0.16098221],
            [0.1593937],
            [0.15796407],
            [0.15664978],
            [0.15548396],
            [0.15445759],
            [0.15359673],
            [0.15286472],
            [0.15225903],
            [0.15181109],
            [0.1514724],
            [0.15130644],
            [0.15129705],
            [0.15139689],
            [0.15159441],
            [0.15189465],
            [0.15234949],
            [0.15295635],
            [0.15362932],
            [0.15438601],
            [0.15532046],
            [0.15632668],
            [0.15741523],
            [0.15860614],
            [0.15982247],
            [0.16114744],
            [0.16264323],
            [0.16411025],
            [0.16563626],
            [0.16728338],
            [0.16895123],
            [0.17055043],
            [0.17215541],
            [0.17372374],
            [0.175215],
            [0.17673888],
        ]
    )
    assert_close(implied_vols, iv_expected)


@pytest.mark.gpu
def test_generate_rough_bergomi_gpu():
    test_generate_rough_bergomi(device="cuda")
