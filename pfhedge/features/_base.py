from abc import ABC
from abc import abstractmethod
from typing import TypeVar

from torch import Tensor

T = TypeVar("T")


class Feature(ABC):
    """Base class for all features.

    All features should subclass this class.
    """

    @abstractmethod
    def __getitem__(self, i: int) -> Tensor:
        """Return feature tensor.

        Returned tensor should have a shape :math:`(N, 1)`, where :math:`N` is
        the number of simulated paths.

        Args:
            i (int): The index of the time step to get the feature.

        Returns:
            torch.Tensor
        """

    def of(self: T, derivative=None, hedger=None) -> T:
        """Set `derivative` and `hedger` to the attributes of `self`.

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
