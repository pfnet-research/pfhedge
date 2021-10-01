from math import ceil
from typing import Optional
from typing import Tuple
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_geometric_brownian

from .base import Primary


class BrownianStock(Primary):
    r"""A stock of which spot prices follow the geometric Brownian motion.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_geometric_brownian`:
          The stochastic process.

    Args:
        sigma (float, default=0.2): The parameter :math:`\sigma`,
            which stands for the volatility of the spot price.
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

    Examples:
        >>> from pfhedge.instruments import BrownianStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = BrownianStock()
        >>> stock.simulate(n_paths=2, time_horizon=5 / 250)
        >>> stock.spot
        tensor([[1.0000, 1.0016, 1.0044, 1.0073, 0.9930, 0.9906],
                [1.0000, 0.9919, 0.9976, 1.0009, 1.0076, 1.0179]])

        Using custom ``dtype`` and ``device``.

        >>> stock = BrownianStock()
        >>> stock.to(dtype=torch.float64, device="cuda:0")
        BrownianStock(..., dtype=torch.float64, device='cuda:0')
    """

    def __init__(
        self,
        sigma: float = 0.2,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.sigma = sigma
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def volatility(self) -> Tensor:
        """Returns the volatility of self.

        It is a tensor filled with ``self.sigma``.
        """
        return torch.full_like(self.spot, self.sigma)

    @property
    def variance(self) -> Tensor:
        """Returns the volatility of self.

        It is a tensor filled with the square of ``self.sigma``.
        """
        return torch.full_like(self.spot, self.sigma ** 2)

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
                :math:`spot` is the initial value of the spot price.
                If ``None`` (default), it uses the default value
                (See :attr:`default_init_state`).
                It also accepts a :class:`float` or a :class:`torch.Tensor`.

        Examples:

            >>> _ = torch.manual_seed(42)
            >>> stock = BrownianStock()
            >>> stock.simulate(n_paths=2, time_horizon=5 / 250, init_state=(2.0,))
            >>> stock.spot
            tensor([[2.0000, 2.0031, 2.0089, 2.0146, 1.9860, 1.9812],
                    [2.0000, 1.9838, 1.9952, 2.0018, 2.0153, 2.0358]])
        """
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        spot = generate_geometric_brownian(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            sigma=self.sigma,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)

    def extra_repr(self) -> str:
        params = ["sigma=" + _format_float(self.sigma)]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(BrownianStock, "default_init_state", Primary.default_init_state)
_set_attr_and_docstring(BrownianStock, "to", Primary.to)
