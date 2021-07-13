from abc import abstractmethod
from collections import OrderedDict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

import torch
from torch import Tensor

from ..base import Instrument

T = TypeVar("T", bound="Primary")


class Primary(Instrument):
    """Base class for all primary instruments.

    A primary instrument is a basic financial instrument which is traded on a market
    and therefore the price is accessible as the market price.
    Examples include stocks, bonds, commodities, and currencies.

    Derivatives are issued based on primary instruments
    (See :class:`Derivative` for details).

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.
    """

    spot: torch.Tensor
    dtype: torch.dtype
    device: torch.device

    def __init__(self):
        self._buffers = OrderedDict()

    @property
    def default_init_state(self):
        """Returns the default initial state of simulation."""

    @abstractmethod
    def simulate(
        self, n_paths: int, time_horizon: float, init_state: Optional[tuple] = None
    ) -> None:
        """Simulate time series associated with the instrument and add them as buffers.

        Args:
            n_paths (int): The number of paths to simulate.
            time_horizon (float): The period of time to simulate the price.
            init_state (tuple, optional): The initial state of the instrument.
                If `None` (default), sensible default value is used.
        """

    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """Adds a buffer to the module.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then
                operations that run on buffers, such as :attr:`cuda`, are ignored.
                If ``None``, the buffer is **not** included in the module's
                :attr:`state_dict`.
        """
        # Implementation here refers to `torch.nn.Module.register_buffer`.
        if "_buffers" not in self.__dict__:
            raise AttributeError("cannot assign buffer before Primary.__init__() call")
        elif not isinstance(name, torch._six.string_classes):
            raise TypeError(
                "buffer name should be a string. " "Got {}".format(torch.typename(name))
            )
        elif "." in name:
            raise KeyError('buffer name can\'t contain "."')
        elif name == "":
            raise KeyError('buffer name can\'t be empty string ""')
        elif hasattr(self, name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "cannot assign '{}' object to buffer '{}' "
                "(torch Tensor or None required)".format(torch.typename(tensor), name)
            )
        else:
            self._buffers[name] = tensor

    def named_buffers(self) -> Iterator[Tuple[str, Tensor]]:
        """Returns an iterator over module buffers, yielding both the
        name of the buffer as well as the buffer itself.

        Yields:
            (string, torch.Tensor): Tuple containing the name and buffer
        """
        for name, buffer in self._buffers.items():
            yield name, buffer

    def buffers(self) -> Iterator[Tensor]:
        r"""Returns an iterator over module buffers.

        Yields:
            torch.Tensor: module buffer
        """
        for _, buffer in self.named_buffers():
            yield buffer

    def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, name)
        )

    def to(self: T, *args, **kwargs) -> T:
        device, dtype, *_ = torch._C._nn._parse_to(*args, **kwargs)

        if dtype is not None and not dtype.is_floating_point:
            raise TypeError(
                f"Instrument.to only accepts floating point "
                f"dtypes, but got desired dtype={dtype}"
            )

        if not hasattr(self, "dtype") or dtype is not None:
            self.dtype = dtype
        if not hasattr(self, "device") or device is not None:
            self.device = device

        for name, buffer in self.named_buffers():
            self.register_buffer(name, buffer.to(*args, **kwargs))

        return self


# Assign docstrings so they appear in Sphinx documentation
Primary.to.__doc__ = Instrument.to.__doc__
Primary.cpu = Instrument.cpu
Primary.cpu.__doc__ = Instrument.cpu.__doc__
Primary.cuda = Instrument.cuda
Primary.cuda.__doc__ = Instrument.cuda.__doc__
Primary.double = Instrument.double
Primary.double.__doc__ = Instrument.double.__doc__
Primary.float = Instrument.float
Primary.float.__doc__ = Instrument.float.__doc__
Primary.half = Instrument.half
Primary.half.__doc__ = Instrument.half.__doc__
