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

        # If the buffers have been already simulated, move it
        if hasattr(self, "spot"):
            self.spot = self.spot.to(*args, **kwargs)

        return self


class Derivative(Instrument):
    """Base class for all derivatives.

    A derivative is a financial instrument whose payoff is contingent on
    a primary instrument (or a set of primary instruments).
    A (over-the-counter) derivative is not traded on the market and therefore the price
    is not directly accessible.
    Examples include options and swaps.

    A derivative relies on primary assets (See :class:`Primary` for details), such as
    stocks, bonds, commodities, and currencies.

    Attributes:
        underlier (:class:`Primary`): The underlying asset on which the derivative's
            payoff relies.
        dtype (torch.dtype): The dtype with which the simulated time-series are
            represented.
        device (torch.device): The device where the simulated time-series are.
    """

    underlier: Primary
    maturity: float

    @property
    def dtype(self) -> torch.dtype:
        return self.underlier.dtype

    @property
    def device(self) -> torch.device:
        return self.underlier.device

    def simulate(
        self, n_paths: int = 1, init_state: Optional[tuple] = None, **kwargs
    ) -> None:
        """Simulate time series associated with the underlier.

        Args:
            n_paths (int): The number of paths to simulate.
            init_state (tuple, optional): The initial state of the underlying
                instrument. If `None` (default), sensible default values are used.
            **kwargs: Other parameters passed to `self.underlier.simulate()`.
        """
        self.underlier.simulate(
            n_paths=n_paths, time_horizon=self.maturity, init_state=init_state, **kwargs
        )

    def to(self: T, *args, **kwargs) -> T:
        self.underlier.to(*args, **kwargs)
        return self

    @abstractmethod
    def payoff(self) -> Tensor:
        """Returns the payoffs of the derivative.

        Shape:
            - Output: :math:`(N)` where :math:`N` stands for the number of simulated
              paths.

        Returns:
            torch.Tensor
        """


Primary.to.__doc__ = Instrument.to.__doc__
Derivative.to.__doc__ = Instrument.to.__doc__
