import abc

import torch


class Feature(abc.ABC):
    """Base class for all features.

    All features should subclass this class.
    """

    @abc.abstractmethod
    def __getitem__(self, i) -> torch.Tensor:
        """Return feature tensor.

        Args:
            i (int): The index of the time step to get the feature.

        Returns:
            torch.Tensor with shape :math:`(N, 1)`, where :math:`N` is the number
            of simulated price paths.
        """

    def of(self, derivative=None, hedger=None):
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
