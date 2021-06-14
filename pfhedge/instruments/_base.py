import abc

import torch
from torch import Tensor


class Instrument(abc.ABC):
    """Base class for all financial instruments."""

    @abc.abstractmethod
    def simulate(self, time_horizon, n_paths=1, init_price=1.0) -> None:
        """Simulate time series of prices of itself (for a primary instrument)
        or its underlier (for a derivative).
        """

    @abc.abstractmethod
    def to(self, *args, **kwargs):
        """Performs dtype and/or device conversion of the time series of prices.

        Parameters:
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
        # Implementation here refers to the function `_str_intern` in `pytorch/_tensor_str.py`.

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
        prices (torch.Tensor): The prices of the instrument.
            This attribute is supposed to be set by a method `simulate()`.
            Shape is :math:`(T, N)` where :math:`T` is the number of time steps
            and :math:`N` is the number of simulated paths.
    """

    @abc.abstractmethod
    def simulate(self, time_horizon, n_paths=1, init_price=1.0, **kwargs) -> None:
        """Simulate time series of prices and set an attribute `prices`.

        Args:
            time_horizon (float): The period of time to simulate the price.
            n_paths (int, default=1): The number of paths to simulate.
            init_price (float, default=1.0): The initial value of the prices.
        """

    def to(self, *args, **kwargs):
        """Performs dtype and/or device conversion of the time series of the prices.

        Args:
            dtype (torch.dtype): Desired floating point type of the floating point
                values of simulated time series.
            device (torch.device): Desired device of the values of simulated time
                series.

        Returns:
            self
        """
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

        # If prices have been already simulated, move it
        if hasattr(self, "prices"):
            self.prices = self.prices.to(*args, **kwargs)

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
    """

    underlier: Primary
    maturity: float

    @property
    def dtype(self) -> torch.dtype:
        return self.underlier.dtype

    @property
    def device(self) -> torch.device:
        return self.underlier.device

    def simulate(self, n_paths=1, init_price=1.0, **kwargs) -> None:
        """
        Simulates time series of the underlier's prices.

        Args:
            n_paths (int): The number of paths to simulate.
            init_price (float): The initial value of the prices.
        """
        self.underlier.simulate(
            time_horizon=self.maturity, n_paths=n_paths, init_price=init_price, **kwargs
        )

    def to(self, *args, **kwargs):
        """Performs dtype and/or device conversion of the time series of the prices.

        Args:
            dtype (torch.dtype): Desired floating point type of the floating point
                values of simulated time series.
            device (torch.device): Desired device of the values of simulated time
                series.

        Returns:
            self
        """
        self.underlier.to(*args, **kwargs)
        return self

    @abc.abstractmethod
    def payoff(self) -> Tensor:
        """Returns the payoffs of the derivative.
        The payoffs is computed based on the prices of `underlier`.

        Returns:
            torch.Tensor
        """
