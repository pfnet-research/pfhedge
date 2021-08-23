from abc import abstractmethod
from collections import OrderedDict
from typing import Dict
from typing import Iterator
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union
from typing import no_type_check

import torch
from torch import Tensor
from torch.nn import Module

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring

from ..base import Instrument

T = TypeVar("T", bound="Primary")
TensorOrFloat = Union[float, Tensor]


class Primary(Instrument):
    """Base class for all primary instruments.

    A primary instrument is a basic financial instrument which is traded on a market
    and therefore the price is accessible as the market price.
    Examples include stocks, bonds, commodities, and currencies.

    Derivatives are issued based on primary instruments
    (See :class:`Derivative` for details).

    Buffers:
        - spot (:class:`torch.Tensor`): The spot price of the instrument.

    Attributes:
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.
    """

    dtype: torch.dtype
    device: torch.device
    dt: float
    _buffers: Dict[str, Optional[Tensor]]
    spot: Tensor
    cost: float

    def __init__(self) -> None:
        super().__init__()
        self._buffers = OrderedDict()
        self.register_buffer("spot", None)

    @property
    def default_init_state(self) -> Tuple[TensorOrFloat, ...]:
        """Returns the default initial state of simulation."""

    # TODO(simaki): Remove @no_type_check once BrownianStock and HestonStock get
    #   unified signatures.
    @abstractmethod
    @no_type_check
    def simulate(
        self,
        n_paths: int,
        time_horizon: float,
        init_state: Optional[Tuple[TensorOrFloat, ...]] = None,
    ) -> None:
        """Simulate time series associated with the instrument and add them as buffers.

        The shapes of the registered buffers should be ``(n_paths, n_steps)``
        where ``n_steps`` is the minimum integer that satisfies
        ``n_steps * self.dt >= time_horizon``.

        Args:
            n_paths (int): The number of paths to simulate.
            time_horizon (float): The period of time to simulate the price.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the instrument.
                If ``None`` (default), it uses the default value
                (See :func:`default_init_state`).
        """

    def register_buffer(self, name: str, tensor: Optional[Tensor]) -> None:
        """Adds a buffer to the module.

        Buffers can be accessed as attributes using given names.

        Args:
            name (string): name of the buffer. The buffer can be accessed
                from this module using the given name
            tensor (Tensor or None): buffer to be registered. If ``None``, then
                operations that run on buffers, such as :attr:`cuda`, are ignored.
        """
        # Implementation here refers to torch.nn.Module.register_buffer.
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
            if buffer is not None:
                yield name, buffer

    def buffers(self) -> Iterator[Tensor]:
        r"""Returns an iterator over module buffers.

        Yields:
            torch.Tensor: module buffer
        """
        for _, buffer in self.named_buffers():
            yield buffer

    def __getattr__(self, name: str) -> Tensor:
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
                "Instrument.to only accepts floating point "
                "dtypes, but got desired dtype=" + str(dtype)
            )

        if not hasattr(self, "dtype") or dtype is not None:
            self.dtype = dtype
        if not hasattr(self, "device") or device is not None:
            self.device = device

        for name, buffer in self.named_buffers():
            self.register_buffer(name, buffer.to(*args, **kwargs))

        return self

    def __repr__(self) -> str:
        extra_repr = self.extra_repr()
        dinfo = ", ".join(self._dinfo())
        main_str = self._get_name() + "("
        if extra_repr and dinfo:
            extra_repr += ", "
        main_str += extra_repr + dinfo + ")"
        return main_str


# Assign docstrings so they appear in Sphinx documentation
set_docstring(Primary, "to", Instrument.to)
set_attr_and_docstring(Primary, "cpu", Instrument.cpu)
set_attr_and_docstring(Primary, "cuda", Instrument.cuda)
set_attr_and_docstring(Primary, "double", Instrument.double)
set_attr_and_docstring(Primary, "float", Instrument.float)
set_attr_and_docstring(Primary, "half", Instrument.half)
