import numpy as np
import pytest
import torch

from pfhedge.stochastic.rbergomi import generate_rbergomi


# @pytest.mark.skipif(True, reason="for development")
def test_generate_rbergomi() -> None:

    torch.manual_seed(42)
    from scipy.optimize import brentq  # type: ignore # mypy ignore
    from scipy.stats import norm  # type: ignore # mypy ignore

    # https://github.com/ryanmccrickerd/rough_bergomi/blob/master/notebooks/rbergomi.ipynb
    def bs(F, K, V, o="call"):
        """
        Returns the Black call price for given forward, strike and integrated
        variance.
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
        Returns implied Black vol from given call price, forward, strike and time
        to maturity.
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
            return bs(F, K, s**2 * t, o) - P

        s = brentq(error, 1e-9, 1e9)
        return s

    vec_bsinv = np.vectorize(bsinv)

    price, variance = generate_rbergomi(
        n_paths=30000,
        n_steps=100,
        alpha=-0.43,
        rho=-0.9,
        eta=1.9,
        xi=0.235**2,
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
