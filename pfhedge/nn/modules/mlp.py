from copy import deepcopy

import torch
from torch.nn import Identity
from torch.nn import LazyLinear
from torch.nn import Module
from torch.nn import ReLU
from torch.nn import Sequential


class MultiLayerPerceptron(Sequential):
    """A feed-forward neural network.

    Number of input features is lazily determined.

    Args:
        out_features (int, default=1): Size of each output sample.
        n_layers (int, default=4): Number of hidden layers.
        n_units (int or tuple[int], default=32): Number of units in each hidden layer.
            If `tuple[int]`, it specifies different number of units for each layer.
        activation (torch.nn.Module, default=torch.nn.ReLU()):
            Activation module of the hidden layers.
        out_activation (torch.nn.Module, default=torch.nn.Identity()):
            Activation module of the output layer.

    Shape:
        - Input: :math:`(N, *, H_{\\text{in}})`, where where * means any number of
          additional dimensions and :math:`H_{\\text{in}})` is the number of input
          features.
        - Output: :math:`(N, *, H_{\\text{out}})`, where all but the last dimension
          are the same shape as the input and :math:`H_{\\text{in}})` is
          `out_features`.

    Examples:

        >>> from pfhedge.nn import MultiLayerPerceptron
        >>> m = MultiLayerPerceptron()
        >>> _ = m(torch.empty((1, 2)))  # lazily determine the number of input features
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
        >>> input = torch.randn((3, 2))
        >>> m(input)
        tensor([[...],
                [...],
                [...]], grad_fn=<AddmmBackward>)

        Specify different number of layers for each layer:

        >>> m = MultiLayerPerceptron(n_layers=2, n_units=(16, 32))
        >>> _ = m(torch.empty(1, 1))
        >>> m
        MultiLayerPerceptron(
          (0): Linear(in_features=1, out_features=16, bias=True)
          (1): ReLU()
          (2): Linear(in_features=16, out_features=32, bias=True)
          (3): ReLU()
          (4): Linear(in_features=32, out_features=1, bias=True)
          (5): Identity()
        )
    """

    def __init__(
        self,
        out_features: int = 1,
        n_layers: int = 4,
        n_units=32,
        activation: Module = ReLU(),
        out_activation: Module = Identity(),
    ):
        n_units = (n_units,) * n_layers if isinstance(n_units, int) else n_units

        layers = []
        for i in range(n_layers):
            layers.append(LazyLinear(n_units[i]))
            layers.append(deepcopy(activation))
        layers.append(LazyLinear(out_features))
        layers.append(deepcopy(out_activation))

        super().__init__(*layers)
