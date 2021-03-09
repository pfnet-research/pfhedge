from copy import deepcopy

import torch
from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import ReLU


class MultiLayerPerceptron(torch.nn.ModuleList):
    """
    A feed-forward neural network.

    Number of input features is lazily determined.

    Parameters
    ----------
    - out_features : int, default 1
        Size of each output sample.
    - n_layers : int, default 4
        Number of hidden layers.
    - n_units : int, default 32
        Number of units in each hidden layer.
    - activation : `torch.nn.Module`, default `torch.nn.ReLU()`
        Activation module of the hidden layers.
    - out_activation : `torch.nn.Module`, default `torch.nn.Identity()`
        Activation module of the output layer.

    Shape
    -----
    - Input : (N, *, H_in)
        where where * means any number of
        additional dimensions and `H_in = in_features`.
    - Output : (N, *, H_out)
        where all but the last dimension
        are the same shape as the input and `H_out = out_features`.

    Examples
    --------
    >>> _ = torch.manual_seed(42)
    >>> m = MultiLayerPerceptron()
    >>> _ = m(torch.empty((1, 2)))  # lazily determine number of input features
    >>> m
    MultiLayerPerceptron(
      (0): Linear(in_features=2, out_features=32, bias=True)
      (1): ReLU()
      (2): Linear(in_features=32, out_features=32, bias=True)
      (3): ReLU()
      (4): Linear(in_features=32, out_features=32, bias=True)
      (5): ReLU()
      (6): Linear(in_features=32, out_features=32, bias=True)
      (7): ReLU()
      (8): Linear(in_features=32, out_features=1, bias=True)
      (9): Identity()
    )
    >>> m(torch.randn((3, 2)))
    tensor([[...],
            [...],
            [...]], grad_fn=<AddmmBackward>)
    """

    def __init__(
        self,
        out_features=1,
        n_layers=4,
        n_units=32,
        activation=ReLU(),
        out_activation=Identity(),
    ):
        super().__init__()

        for _ in range(n_layers):
            self.append(LazyLinear(n_units))
            self.append(deepcopy(activation))
        self.append(LazyLinear(out_features))
        self.append(deepcopy(out_activation))

    def forward(self, input):
        x = input
        for layer in self:
            x = layer(x)
        return x
