from abc import ABC
from abc import abstractmethod
from typing import List
from typing import Optional
from typing import TypeVar
from typing import no_type_check

import torch
from torch import Tensor

T = TypeVar("T", bound="Instrument")


class Instrument(ABC):
    """Base class for all financial instruments."""

    cost: float

    @property
    @abstractmethod
    def spot(self) -> Tensor:
        """Returns the spot price of self."""

    @abstractmethod
    @no_type_check
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
        """Moves and/or casts the buffers of the instrument.

        This can be called as

        .. function:: to(device=None, dtype=None)
        .. function:: to(tensor)

        Its signature is similar to :meth:`torch.nn.Module.to`.
        It only accepts floating point dtypes.
        See :ref:`instrument-attributes-doc` for details.

        .. note::
            This method modifies the instrument in-place.

        .. seealso::
            - :meth:`float()`: Cast to :class:`torch.float32`.
            - :meth:`double()`: Cast to :class:`torch.float64`.
            - :meth:`half()`: Cast to :class:`torch.float16`.
            - :meth:`bfloat16()`: Cast to :class:`torch.bfloat16`.
            - :meth:`cuda()`: Move to CUDA memory.
            - :meth:`cpu()`: Move to CPU memory.

        Args:
            dtype (torch.dtype): The desired floating point dtype of
                the buffers in this instrument.
            device (torch.device): The desired device of the buffers
                in this instrument.
            tensor (torch.Tensor): Tensor whose dtype and device are the desired
                dtype and device for all buffers in this module.

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
        return self.to(torch.device(f"cuda:{device}" if device is not None else "cuda"))

    def double(self: T) -> T:
        """It is equivalent to ``self.to(torch.float64)``.
        See :func:`to()`.
        """
        return self.to(torch.float64)

    def float(self: T) -> T:
        """It is equivalent to ``self.to(torch.float32)``.
        See :func:`to()`.
        """
        return self.to(torch.float32)

    def half(self: T) -> T:
        """It is equivalent to ``self.to(torch.float16)``.
        See :func:`to()`.
        """
        return self.to(torch.float16)

    def bfloat16(self: T) -> T:
        """It is equivalent to ``self.to(torch.bfloat16)``.
        See :func:`to()`.
        """
        return self.to(torch.bfloat16)

    def extra_repr(self) -> str:
        return ""

    def _get_name(self) -> str:
        return self.__class__.__name__

    def _dinfo(self) -> List[str]:
        # Returns list of strings that tell ``dtype`` and ``device`` of self.
        # Intended to be used in :func:`__repr__`.
        # If ``dtype`` (``device``) is the one specified in default type,
        # ``dinfo`` will not have the information of it.
        # Implementation here refers to the function _str_intern in
        # pytorch/_tensor_str.py.
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
