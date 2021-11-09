# Example to use a user-defined Module as a hedging model
# Here we show an example of No-Transaction Band Network,
# which is proposed in Imaki et al. 21.

import sys

import torch
import torch.nn.functional as fn
from torch import Tensor
from torch.nn import Module

sys.path.append("..")
from pfhedge.instruments import BrownianStock
from pfhedge.instruments import EuropeanOption
from pfhedge.nn import BlackScholes
from pfhedge.nn import Clamp
from pfhedge.nn import Hedger
from pfhedge.nn import MultiLayerPerceptron


class NoTransactionBandNet(Module):
    """Initialize a no-transaction band network.

    The `forward` method returns the next hedge ratio.

    Args:
        derivative (pfhedge.instruments.BaseDerivative): The derivative to hedge.

    Shape:
        - Input: :math:`(N, H_{\\text{in}})`, where :math:`(N, H_{\\text{in}})` is the
        number of input features. See `inputs()` for the names of input features.
        - Output: :math:`(N, 1)`.

    Examples:

        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption

        >>> derivative = EuropeanOption(BrownianStock(cost=1e-4))
        >>> m = NoTransactionBandNet(derivative)
        >>> m.inputs()
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

        self.delta = BlackScholes(derivative)
        self.mlp = MultiLayerPerceptron(out_features=2)
        self.clamp = Clamp()

    def inputs(self):
        return self.delta.inputs() + ["prev_hedge"]

    def forward(self, input: Tensor) -> Tensor:
        prev_hedge = input[..., [-1]]

        delta = self.delta(input[..., :-1])
        width = self.mlp(input[..., :-1])

        min = delta - fn.leaky_relu(width[..., [0]])
        max = delta + fn.leaky_relu(width[..., [1]])

        return self.clamp(prev_hedge, min=min, max=max)


if __name__ == "__main__":
    torch.manual_seed(42)

    # Prepare a derivative to hedge
    derivative = EuropeanOption(BrownianStock(cost=1e-4))

    # Create your hedger
    model = NoTransactionBandNet(derivative)
    hedger = Hedger(model, model.inputs())

    # Fit and price
    hedger.fit(derivative, n_paths=10000, n_epochs=200)
    price = hedger.price(derivative, n_paths=10000)
    print(f"Price={price:.5e}")
