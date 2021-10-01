import copy
from typing import List
from typing import Optional
from typing import TypeVar
from typing import Union

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge.instruments import Derivative

from ._base import Feature
from ._getter import get_feature

T = TypeVar("T", bound="FeatureList")


class FeatureList(Feature):
    """Holds features in a list.

    Args:
        features (list[str | Features]): A list of features.

    Examples:

        >>> from pfhedge.features import FeatureList
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>>
        >>> derivative = EuropeanOption(BrownianStock(), maturity=5/250)
        >>> derivative.simulate(n_paths=2)
        >>> f = FeatureList(["moneyness", "volatility", "empty"]).of(derivative)
        >>> len(f)
        3
        >>> f.get(0).size()
        torch.Size([2, 1, 3])
    """

    def __init__(self, features: List[Union[str, Feature]]):
        self.features = list(map(get_feature, features))

    def __len__(self):
        return len(self.features)

    def get(self, time_step: Optional[int]) -> Tensor:
        # Return size: (N, T, F)
        return torch.cat([f.get(time_step) for f in self.features], dim=-1)

    def __repr__(self):
        return str(list(map(str, self.features)))

    def of(self: T, derivative: Derivative, hedger: Optional[Module] = None) -> T:
        output = copy.copy(self)
        output.features = [f.of(derivative, hedger) for f in self.features]
        return output

    def is_state_dependent(self) -> bool:
        return any(map(lambda f: f.is_state_dependent(), self.features))


class ModuleOutput(Feature, Module):
    """The feature computed as an output of a :class:`torch.nn.Module`.

    Args:
        module (torch.nn.Module): Module to compute the value of the feature.
            The input and output shapes should be
            :math:`(N, *, H_{\\math{in}}) -> (N, *, H_{\\math{out}})` where
            :math:`N` is the number of simulated paths of the underlying instrument,
            :math:`H_{\\math{in}}` is the number of input features,
            :math:`H_{\\math{out}}` is the number of output features, and
            :math:`*` means any number of additional dimensions.
        inputs (list[Feature]): The input features to the module.

    Examples:

        >>> from torch.nn import Linear
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.simulate(n_paths=3)
        >>>
        >>> m = Linear(2, 1)
        >>> f = ModuleOutput(m, inputs=["moneyness", "expiry_time"]).of(derivative)
        >>> f.get(0).size()
        torch.Size([3, 1, 1])
        >>> f
        ModuleOutput(
          inputs=['moneyness', 'expiry_time']
          (module): Linear(in_features=2, out_features=1, bias=True)
        )

        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> _ = torch.manual_seed(42)
        >>> derivative = EuropeanOption(BrownianStock())
        >>> derivative.simulate(n_paths=3)
        >>> m = BlackScholes(derivative)
        >>> f = ModuleOutput(m, ["log_moneyness", "expiry_time", "volatility"])
        >>> f = f.of(derivative)
        >>> f.get(0).size()
        torch.Size([3, 1, 1])
    """

    module: Module
    inputs: FeatureList

    def __init__(self, module: Module, inputs: List[Union[str, Feature]]) -> None:
        super(Module, self).__init__()
        super(Feature, self).__init__()

        self.add_module("module", module)
        self.inputs = FeatureList(inputs)

    def extra_repr(self) -> str:
        return "inputs=" + str(self.inputs)

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input)

    def get(self, time_step: Optional[int]) -> Tensor:
        return self(self.inputs.get(time_step))

    def of(self, derivative=None, hedger=None):
        self.inputs = self.inputs.of(derivative, hedger)
        return self

    def is_state_dependent(self):
        return self.inputs.is_state_dependent()
