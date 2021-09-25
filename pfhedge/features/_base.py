import copy
from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import TypeVar

from torch import Tensor
from torch.nn import Module

from pfhedge.instruments import Derivative

T = TypeVar("T", bound="Feature")


class Feature(ABC):
    """Base class for all features.

    All features should subclass this class.
    """

    derivative: Derivative
    hedger: Optional[Module]

    def __init__(self):
        self.register_hedger(None)

    @abstractmethod
    def get(self, time_step: Optional[int]) -> Tensor:
        """Return feature tensor.

        Returned tensor should have a shape :math:`(N, 1)` where
        :math:`N` is the number of simulated paths.

        Args:
            time_step (int): The index of the time step to get the feature.

        Shape:
            - Output: :math:`(N, T, F=1)` where
              :math:`N` is the number of paths,
              :math:`T` is the number of time steps, and
              :math:`F` is the number of feature size.
              If ``time_step`` is given, the shape is :math:`(N, 1, F)`.

        Returns:
            torch.Tensor
        """

    def of(self: T, derivative: Derivative, hedger: Optional[Module] = None) -> T:
        """Set ``derivative`` and ``hedger`` to the attributes of ``self``.

        Args:
            derivative (Derivative, optional): The derivative to compute features.
            hedger (Hedger, optional): The hedger to compute features.

        Returns:
            self
        """
        output = copy.copy(self)
        output.register_derivative(derivative)
        output.register_hedger(hedger)
        return output

    def register_derivative(self, derivative: Derivative) -> None:
        setattr(self, "derivative", derivative)

    def register_hedger(self, hedger: Optional[Module]) -> None:
        setattr(self, "hedger", hedger)

    def _get_name(self) -> str:
        return self.__class__.__name__

    def is_state_dependent(self) -> bool:
        # If a feature uses the state of a hedger, it is state dependent.
        return getattr(self, "hedger") is not None

    # TODO(simaki) Remove later
    def __getitem__(self, time_step: Optional[int]) -> Tensor:
        # raise DeprecationWarning("Use `<feature>.get(time_step)` instead")
        return self.get(time_step)


class StateIndependentFeature(Feature):
    # Features that does not use the state of the hedger.

    derivative: Derivative
    hedger: None

    def of(
        self: "StateIndependentFeature",
        derivative: Derivative,
        hedger: Optional[Module] = None,
    ) -> "StateIndependentFeature":
        return super().of(derivative=derivative, hedger=None)
