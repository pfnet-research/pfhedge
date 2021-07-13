from abc import ABC
from abc import abstractmethod
from typing import Optional
from typing import TypeVar

import torch

T = TypeVar("T", bound="Instrument")


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

    def cpu(self: T) -> T:
        """Returns a copy of this object in CPU memory.

        If this object is already in CPU memory and on the correct device,
        then no copy is performed and the original object is returned.
        """
        return self.to(torch.device("cpu"))

    def cuda(self: T, device: Optional[int] = None) -> T:
        """Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device,
        then no copy is performed and the original object is returned.

        Args:
            device (torch.device): The destination GPU device.
                Defaults to the current CUDA device.
        """
        return self.to(
            torch.device("cuda:{}".format(device) if device is not None else "cuda")
        )

    def double(self: T) -> T:
        """`self.double()` is equivalent to `self.to(torch.float64)`.
        See :func:`to()`.
        """
        return self.to(torch.float64)

    def float(self: T) -> T:
        """`self.float()` is equivalent to `self.to(torch.float32)`.
        See :func:`to()`.
        """
        return self.to(torch.float32)

    def half(self: T) -> T:
        """`self.half()` is equivalent to `self.to(torch.float16)`.
        See :func:`to()`.
        """
        return self.to(torch.float16)

    def bfloat16(self: T) -> T:
        """`self.bfloat16()` is equivalent to `self.to(torch.bfloat16)`.
        See :func:`to()`.
        """
        return self.to(torch.bfloat16)

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
