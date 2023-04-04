from math import ceil
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_rough_bergomi

from .base import BasePrimary


class RoughBergomiStock(BasePrimary):
    r"""A stock of which spot price and variance follow rough Bergomi (rBergomi) process.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_rough_bergomi`:
          The stochastic process.

    Args:
        alpha (float, default=-0.4): The parameter :math:`\\alpha`.
        rho (float, default=-0.9): The parameter :math:`\\rho`.
        eta (float, default=1.9): The parameter :math:`\\eta`.
        xi (float, default=0.04): The parameter :math:`\\xi`.
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
        - spot (:class:`torch.Tensor`): The spot price of the instrument.
          This attribute is set by a method :meth:`simulate()`.
          The shape is :math:`(N, T)` where
          :math:`N` is the number of simulated paths and
          :math:`T` is the number of time steps.
        - variance (:class:`torch.Tensor`): The variance of the instrument.
          Note that this is different from the realized variance of the spot price.
          This attribute is set by a method :meth:`simulate()`.
          The shape is :math:`(N, T)`.

    Examples:
        >>> from pfhedge.instruments import RoughBergomiStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = RoughBergomiStock()
        >>> stock.simulate(n_paths=2, time_horizon=5/250)
        >>> stock.spot
        tensor([[1.0000, 0.9741, 0.9351, 0.9429, 0.9386, 0.9284],
                [1.0000, 1.0100, 1.0127, 1.0148, 1.0201, 1.0148]])
        >>> stock.variance
        tensor([[0.0400, 0.3130, 0.0107, 0.0279, 0.1336, 0.0170],
                [0.0400, 0.0175, 0.0164, 0.0274, 0.0099, 0.0196]])
        >>> stock.volatility
        tensor([[0.2000, 0.5595, 0.1034, 0.1670, 0.3656, 0.1304],
                [0.2000, 0.1324, 0.1282, 0.1655, 0.0993, 0.1402]])
    """

    spot: Tensor
    variance: Tensor

    def __init__(
        self,
        alpha: float = -0.4,
        rho: float = -0.9,
        eta: float = 1.9,
        xi: float = 0.04,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.rho = rho
        self.eta = eta
        self.xi = xi
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0, self.xi)

    @property
    def volatility(self) -> Tensor:
        """An alias for ``self.variance.sqrt()``."""
        return self.get_buffer("variance").clamp(min=0.0).sqrt()

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Simulate the spot price and add it as a buffer named ``spot``.

        The shape of the spot is :math:`(N, T)`, where
        :math:`N` is the number of simulated paths and
        :math:`T` is the number of time steps.
        The number of time steps is determinded from ``dt`` and ``time_horizon``.

        Args:
            n_paths (int, default=1): The number of paths to simulate.
            time_horizon (float, default=20/250): The period of time to simulate
                the price.
            init_state (tuple[torch.Tensor | float], optional): The initial
                state of the instrument.
                This is specified by a tuple :math:`(S(0), V(0))` where
                :math:`S(0)` and :math:`V(0)` are the initial values of
                spot and variance, respectively.
                If ``None`` (default), it uses the default value
                (See :attr:`default_init_state`).
        """
        if init_state is None:
            init_state = self.default_init_state

        output = generate_rough_bergomi(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            alpha=self.alpha,
            rho=self.rho,
            eta=self.eta,
            xi=self.xi,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", output.spot)
        self.register_buffer("variance", output.variance)

    def extra_repr(self) -> str:
        params = [
            "alpha=" + _format_float(self.alpha),
            "rho=" + _format_float(self.rho),
            "eta=" + _format_float(self.eta),
            "xi=" + _format_float(self.xi),
        ]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(RoughBergomiStock, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(RoughBergomiStock, "to", BasePrimary.to)
