from collections import OrderedDict
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Tuple
from typing import Type

from torch import Tensor
from torch.nn import Module

from pfhedge.instruments import Derivative


class BlackScholesModuleFactory:

    _modules: Dict[str, Type[Module]]

    # singleton
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super().__new__(cls)
            cls._instance._modules = OrderedDict()
        return cls._instance

    def register_module(self, name: str, cls: Type[Module]) -> None:
        self._modules[name] = cls

    def named_modules(self) -> Iterator[Tuple[str, Type[Module]]]:
        for name, module in self._modules.items():
            if module is not None:
                yield name, module

    def names(self) -> Iterator[str]:
        for name, _ in self.named_modules():
            yield name

    def features(self) -> Iterator[Type[Module]]:
        for _, module in self.named_modules():
            yield module

    def get_class(self, name: str) -> Type[Module]:
        return self._modules[name]

    def get_class_from_derivative(self, derivative: Derivative) -> Type[Module]:
        return self.get_class(derivative.__class__.__name__).from_derivative(derivative)  # type: ignore


class BlackScholes(Module):
    """Creates Black-Scholes formula module from a derivative.

    The ``forward`` method returns the Black-Scholes delta.

    Args:
        derivative (:class:`BaseDerivative`): The derivative to get
            the Black-Scholes formula.

    Shape:
        - input : :math:`(N, *, H_{\\mathrm{in}})` where
          :math:`*` means any number of additional dimensions and
          :math:`H_{\\mathrm{in}}` is the number of input features.
          See :meth:`inputs` for the names of the input features.
        - output : :math:`(N, *, 1)`,
          all but the last dimension are the same shape as the input.

    Examples:
        One can instantiate Black-Scholes module by using a derivative.
        For example, one can instantiate :class:`BSEuropeanOption` using
        a :class:`pfhedge.instruments.EuropeanOption`.
        The ``forward`` method returns delta of the derivative.

        >>> import torch
        >>> from pfhedge.instruments import BrownianStock
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import BlackScholes
        >>>
        >>> derivative = EuropeanOption(BrownianStock(), strike=1.1)
        >>> m = BlackScholes(derivative)
        >>> m
        BSEuropeanOption(strike=1.1000)

        Instantiating :class:`BSLookbackOption` using a
        :class:`pfhedge.instruments.LookbackOption`.

        >>> from pfhedge.instruments import LookbackOption
        >>>
        >>> derivative = LookbackOption(BrownianStock(), strike=1.03)
        >>> m = BlackScholes(derivative)
        >>> m
        BSLookbackOption(strike=1.0300)
    """

    inputs: Callable[..., List[str]]  # inputs(self) -> List[str]
    price: Callable[..., Tensor]  # price(self, ...) -> Tensor
    delta: Callable[..., Tensor]  # delta(self, ...) -> Tensor
    gamma: Callable[..., Tensor]  # gamma(self, ...) -> Tensor
    vega: Callable[..., Tensor]  # vega(self, ...) -> Tensor
    theta: Callable[..., Tensor]  # theta(self, ...) -> Tensor

    def __new__(cls, derivative):
        return BlackScholesModuleFactory().get_class_from_derivative(derivative)
