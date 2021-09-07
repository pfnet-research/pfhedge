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
        .. function:: to(instrument)

        Its signature is similar to :meth:`torch.nn.Module.to`.
        It only accepts floating point dtypes.
        See :ref:`instrument-attributes-doc` for details.

        Note:
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
            device (torch.device): The desired device of
                the buffers in this instrument.
            tensor (torch.Tensor): Tensor whose dtype and device are
                the desired dtype and device of
                the buffers in this instrument.
            instrument (Instrument): Instrument whose dtype and device are
                the desired dtype and device of
                the buffers in this instrument.

        Returns:
            self
        """

    def cpu(self: T) -> T:
        """Moves all buffers of this instrument and its underlier to the CPU.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self
        """
        return self.to(torch.device("cpu"))

    def cuda(self: T, device: Optional[int] = None) -> T:
        """Moves all buffers of this instrument and its underlier to the GPU.

        Note:
            This method modifies the instrument in-place.

        Args:
            device (int, optional): If specified,
                all buffers will be copied to that device.

        Returns:
            self
        """
        return self.to(torch.device(f"cuda:{device}" if device is not None else "cuda"))

    def double(self: T) -> T:
        """Casts all floating point parameters and buffers to
        ``torch.float64`` datatype.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self
        """
        return self.to(torch.float64)

    def float(self: T) -> T:
        """Casts all floating point parameters and buffers to
        ``torch.float32`` datatype.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self
        """
        return self.to(torch.float32)

    def half(self: T) -> T:
        """Casts all floating point parameters and buffers to
        ``torch.float16`` datatype.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self
        """
        return self.to(torch.float16)

    def bfloat16(self: T) -> T:
        """Casts all floating point parameters and buffers to
        ``torch.bfloat16`` datatype.

        Note:
            This method modifies the instrument in-place.

        Returns:
            self
        """
        return self.to(torch.bfloat16)

    def extra_repr(self) -> str:
        """Set the extra representation of the instrument.

        To print customized extra information,
        you should re-implement this method in your own instruments.
        Both single-line and multi-line strings are acceptable.
        """
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
