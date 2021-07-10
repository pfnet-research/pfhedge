from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import TypeVar

import torch
from torch import Tensor

T = TypeVar("T")


class Instrument(ABC):
    """Base class for all financial instruments."""

    @abstractmethod
    def simulate(self, n_paths: int, time_horizon: float, **kwargs) -> None:
        """Simulate time series associated with the instrument itself
        (for a primary instrument) or its underlier (for a derivative)
        and add them as buffers.

        Args:
            n_paths (int): The number of paths to simulate.
            time_horizon (float): The period of time to simulate the price.

        Returns:
            None
        """

    @abstractmethod
    def to(self: T, *args, **kwargs) -> T:
        """Performs dtype and/or device conversion of the buffers associated to
        the instument.

        A `torch.dtype` and `torch.device` are inferred from the arguments of
        `self.to(*args, **kwargs)`.

        Args:
            dtype (torch.dtype): Desired floating point type of the floating point
                values of simulated time series.
            device (torch.device): Desired device of the values of simulated time
                series.

        Returns:
            self
        """

    def float(self: T) -> T:
        """`self.float()` is equivalent to `self.to(torch.float32)`. See :func:`to()`.

        Returns:
            self
        """
        return self.to(torch.float32)

    @property
    def dinfo(self) -> list:
        """Returns list of strings that tell `dtype` and `device` of `self`.

        Intended to be used in `__repr__`.

        If `dtype` (`device`) is the one specified in default type,
        `dinfo` will not have the information of it.

        Returns:
            list[str]
        """
        # Implementation here refers to the function `_str_intern` in
        # `pytorch/_tensor_str.py`.

        dinfo = []

        dtype = getattr(self, "dtype", None)
        if dtype is not None:
            if dtype != torch.get_default_dtype():
                dinfo.append("dtype=" + str(dtype))

        # A general logic here is we only print device when it doesn't match
        # the device specified in default tensor type.
        device = getattr(self, "device", None)
        if device is not None:
            if device.type != torch._C._get_default_device() or (
                device.type == "cuda" and torch.cuda.current_device() != device.index
            ):
                dinfo.append("device='" + str(device) + "'")

        return dinfo
