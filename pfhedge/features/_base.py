from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import TypeVar

from torch import Tensor

T = TypeVar("T")


class Feature(ABC):
    """Base class for all features.

    All features should subclass this class.
    """

    def __init__(self):
        self.derivative = None
        self.hedger = None

    @abstractmethod
    def __getitem__(self, i: Optional[int]) -> Tensor:
        """Return feature tensor.

        Returned tensor should have a shape :math:`(N, 1)`, where :math:`N` is
        the number of simulated paths.

        Args:
            i (int): The index of the time step to get the feature.

        Returns:
            torch.Tensor
        """

    def register_derivative(self, derivative) -> None:
        setattr(self, "derivative", derivative)

    def register_hedger(self, hedger) -> None:
        setattr(self, "hedger", hedger)

    def of(self: T, derivative=None, hedger=None) -> T:
        """Set ``derivative`` and ``hedger`` to the attributes of ``self``.

        Args:
            derivative (Derivative, optional): The derivative to compute features.
            hedger (Hedger, optional): The hedger to compute features.

        Returns:
            self
        """
        if not hasattr(self, "derivative") or derivative is not None:
            self.derivative = derivative
        if not hasattr(self, "hedger") or derivative is not None:
            self.hedger = hedger
        return self

    def _get_name(self) -> str:
        return self.__class__.__name__

    def is_state_dependent(self) -> bool:
        # If a feature is dependent on the the state of the hedger
        if not hasattr(self, "hedger"):
            raise ValueError
        return self.hedger is not None
