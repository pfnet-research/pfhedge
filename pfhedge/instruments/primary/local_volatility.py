from math import ceil
from typing import Callable
from typing import Optional
from typing import Tuple
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import LocalVolatilityFunction
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_local_volatility_process

from .base import BasePrimary


class LocalVolatilityStock(BasePrimary):
    r"""A stock of which spot prices follow the local volatility model.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_local_volatility_process`:
          The stochastic process.

    Args:
        sigma_fn (callable): The local volatility function.
            Its signature is ``sigma_fn(time: Tensor, spot: Tensor) -> Tensor``.
        cost (float, default=0.0): The transaction cost rate.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.device, optional): Desired device of returned tensor.
            Default: If None, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

    Buffers:
        - spot (:class:`torch.Tensor`): The spot prices of the instrument.
          This attribute is set by a method :meth:`simulate()`.
          The shape is :math:`(N, T)` where
          :math:`N` is the number of simulated paths and
          :math:`T` is the number of time steps.
    """

    spot: Tensor
    volatility: Tensor

    def __init__(
        self,
        sigma_fn: LocalVolatilityFunction,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.sigma_fn = sigma_fn
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def variance(self) -> Tensor:
        """Returns the volatility of self.

        It is a tensor filled with the square of ``self.sigma``.
        """
        return self.volatility.square()

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar]] = None,
    ) -> None:
        """Simulate the spot price and add it as a buffer named ``spot``.

        The shape of the spot is :math:`(N, T)`, where :math:`N` is the number of
        simulated paths and :math:`T` is the number of time steps.
        The number of time steps is determinded from ``dt`` and ``time_horizon``.

        Args:
            n_paths (int, default=1): The number of paths to simulate.
            time_horizon (float, default=20/250): The period of time to simulate
                the price.
            init_state (tuple[torch.Tensor | float], optional): The initial state of
                the instrument.
                This is specified by a tuple :math:`(S(0),)` where
                :math:`S(0)` is the initial value of the spot price.
                If ``None`` (default), it uses the default value
                (See :attr:`default_init_state`).
                It also accepts a :class:`float` or a :class:`torch.Tensor`.

        Examples:
            >>> from pfhedge.instruments import LocalVolatilityStock
            ...
            >>> def sigma_fn(time: Tensor, spot: Tensor) -> Tensor:
            ...     a, b, sigma = 0.0001, 0.0004, 0.10
            ...     sqrt_term = (spot.log().square() + sigma ** 2).sqrt()
            ...     return ((a + b * sqrt_term) / time.clamp(min=1/250)).sqrt()
            ...
            >>> _ = torch.manual_seed(42)
            >>> stock = LocalVolatilityStock(sigma_fn)
            >>> stock.simulate(n_paths=2, time_horizon=5 / 250)
            >>> stock.spot
            tensor([[1.0000, 1.0040, 1.0055, 1.0075, 1.0091, 1.0024],
                    [1.0000, 1.0261, 1.0183, 1.0223, 1.0242, 1.0274]])
            >>> stock.volatility
            tensor([[0.1871, 0.1871, 0.1323, 0.1081, 0.0936, 0.0837],
                    [0.1871, 0.1880, 0.1326, 0.1084, 0.0939, 0.0841]])
        """
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        output = generate_local_volatility_process(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            sigma_fn=self.sigma_fn,
            init_state=init_state,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", output.spot)
        self.register_buffer("volatility", output.volatility)

    def extra_repr(self) -> str:
        params = []
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(
    LocalVolatilityStock, "default_init_state", BasePrimary.default_init_state
)
_set_attr_and_docstring(LocalVolatilityStock, "to", BasePrimary.to)
