from math import ceil
from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.doc import set_attr_and_docstring
from pfhedge._utils.doc import set_docstring

from ...stochastic import generate_geometric_brownian
from .base import Primary

TensorOrFloat = Union[Tensor, float]


class BrownianStock(Primary):
    """A stock of which spot prices follow the geometric Brownian motion.

    The drift of the spot prices is assumed to be vanishing.

    See :func:`pfhedge.stochastic.generate_geometric_brownian`
    for details of the process.

    Args:
        volatility (float, default=0.2): The volatility of the price.
        cost (float, default=0.0): The transaction cost rate.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.device, optional): Desired device of returned tensor.
            Default: If None, uses a global default
            (see ``torch.set_default_tensor_type()``).
        device (torch.device, optional): Desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see ``torch.set_default_tensor_type()``).
            ``device`` will be the CPU for CPU tensor types and
            the current CUDA device for CUDA tensor types.

    Buffers:
        - ``spot`` (``torch.Tensor``): The spot prices of the instrument.
          This attribute is set by a method :func:`simulate()`.
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
        volatility: float = 0.2,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.volatility = volatility
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrFloat]] = None,
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
                This is specified by ``(spot,)``, where ``spot`` is the initial value
                of the stock price.
                If ``None`` (default), it uses the default value
                (See :func:`default_init_state`).
                It also accepts a ``float`` or a ``torch.Tensor``.

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
            volatility=self.volatility,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)

    def __repr__(self) -> str:
        params = [f"volatility={self.volatility:.2e}"]
        if self.cost != 0.0:
            params.append(f"cost={self.cost:.2e}")
        params.append(f"dt={self.dt:.2e}")
        params += self.dinfo
        return self.__class__.__name__ + "(" + ", ".join(params) + ")"


# Assign docstrings so they appear in Sphinx documentation
set_docstring(BrownianStock, "default_init_state", Primary.default_init_state)
set_attr_and_docstring(BrownianStock, "to", Primary.to)
