# Example to use a user-defined Module as a hedging model
# Here we show an example of No-Transaction Band Network,
# which is proposed in Imaki et al. 21.

import sys

import torch
import torch.nn.functional as fn

sys.path.append("..")
from pfhedge import Hedger  # noqa: E402
from pfhedge.instruments import BrownianStock  # noqa: E402
from pfhedge.instruments import EuropeanOption  # noqa: E402
from pfhedge.nn import BlackScholes  # noqa: E402
from pfhedge.nn import Clamp  # noqa: E402
from pfhedge.nn import MultiLayerPerceptron  # noqa: E402


class NoTransactionBandNet(torch.nn.Module):
    """
    Initialize the no-transaction band network.
    The `forward` method returns the next hedge ratio.

    Parameters
    ----------
    - derivative : Derivative
        The derivative to hedge.

    Shape
    -----
    Input : (N, H_in)
        See `features()` for input features.
        Here, `H_in` is the number of input features.
    Output : (N, 1)
        The hedge ratio at the next time step.

    Examples
    --------
    >>> from pfhedge.instruments import BrownianStock
    >>> from pfhedge.instruments import EuropeanOption
    >>> from pfhedge.instruments import LookbackOption

    >>> deriv = EuropeanOption(BrownianStock(cost=1e-4))
    >>> m = NoTransactionBandNet(deriv)
    >>> m.features()
    ['log_moneyness', 'expiry_time', 'volatility', 'prev_hedge']
    >>> input = torch.tensor([
    ...     [-0.05, 0.1, 0.2, 0.5],
    ...     [-0.01, 0.1, 0.2, 0.5],
    ...     [ 0.00, 0.1, 0.2, 0.5],
    ...     [ 0.01, 0.1, 0.2, 0.5],
    ...     [ 0.05, 0.1, 0.2, 0.5]])
    >>> m(input)
    tensor([[0.2232],
            [0.4489],
            [0.5000],
            [0.5111],
            [0.7310]], grad_fn=<SWhereBackward>)
    """

    def __init__(self, derivative):
        super().__init__()

        self.bs = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def features(self):
        return self.bs.features() + ["prev_hedge"]

    def forward(self, x):
        prev_hedge = x[:, [-1]]

        delta = self.bs(x[:, :-1]).reshape(-1, 1)
        width = self.mlp(x[:, :-1])

        lower = delta - fn.leaky_relu(width[:, [0]])
        upper = delta + fn.leaky_relu(width[:, [1]])

        return self.clamp(prev_hedge, min_value=lower, max_value=upper)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    deriv = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = NoTransactionBandNet(deriv)
    hedger = Hedger(model, model.features())

    # Fit and price
    hedger.fit(deriv, n_paths=10000, n_epochs=200)
    price = hedger.price(deriv, n_paths=10000)
    print(f"Price={price:.5e}")
