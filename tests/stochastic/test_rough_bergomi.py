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
        n_paths=100000,
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

    ST = price.cpu().numpy()[:, -1][:, np.newaxis]
    K = np.exp(k)[np.newaxis, :]
    call_payoffs = np.maximum(ST - K, 0)
    call_prices = np.mean(call_payoffs, axis=0)[:, np.newaxis]
    implied_vols = vec_bsinv(call_prices, 1.0, np.transpose(K), 1.0)
    # This should be the same values as the original notebook's original implied volatility has:
    # https://github.com/ryanmccrickerd/rough_bergomi/blob/master/notebooks/rbergomi.ipynb
    # Test herein changed N=1000000
    iv_expected = np.array(
        [
            [0.32329572],
            [0.321043],
            [0.31877741],
            [0.31649498],
            [0.31420173],
            [0.31189863],
            [0.30958379],
            [0.30726057],
            [0.30492994],
            [0.30259006],
            [0.30023852],
            [0.29787897],
            [0.295506],
            [0.29312565],
            [0.29073352],
            [0.28832833],
            [0.28591352],
            [0.28348824],
            [0.28104991],
            [0.27859909],
            [0.2761357],
            [0.27366018],
            [0.27117303],
            [0.26867288],
            [0.26616332],
            [0.2636462],
            [0.26111824],
            [0.25857965],
            [0.25603089],
            [0.25346889],
            [0.25089654],
            [0.24831597],
            [0.24572482],
            [0.24312431],
            [0.2405125],
            [0.23788922],
            [0.23525203],
            [0.23260567],
            [0.22995252],
            [0.22728888],
            [0.22461481],
            [0.22192968],
            [0.2192341],
            [0.21652712],
            [0.21381209],
            [0.2110976],
            [0.20838205],
            [0.20566693],
            [0.20295357],
            [0.20024574],
            [0.19754241],
            [0.19484656],
            [0.19216448],
            [0.18950412],
            [0.18686785],
            [0.18426568],
            [0.18170093],
            [0.17918579],
            [0.17672581],
            [0.17433524],
            [0.17202907],
            [0.16981844],
            [0.16772246],
            [0.16575174],
            [0.16391222],
            [0.16222634],
            [0.16069428],
            [0.15931391],
            [0.15808875],
            [0.15701571],
            [0.15610465],
            [0.15535841],
            [0.15476975],
            [0.15432322],
            [0.15400821],
            [0.15382531],
            [0.15376506],
            [0.15381003],
            [0.15396721],
            [0.15422858],
            [0.15458409],
            [0.15502357],
            [0.15553685],
            [0.15610412],
            [0.15673336],
            [0.15743268],
            [0.15821416],
            [0.15907241],
            [0.15999277],
            [0.1609493],
            [0.1619364],
            [0.16295763],
            [0.16401009],
            [0.16509084],
            [0.16620102],
            [0.1673327],
            [0.16850196],
            [0.16969727],
            [0.1709054],
            [0.1721337],
            [0.17338067],
        ]
    )
    assert_close(implied_vols, iv_expected, atol=1e-2, rtol=1e-2)


@pytest.mark.gpu
def test_generate_rough_bergomi_gpu():
    test_generate_rough_bergomi(device="cuda")
