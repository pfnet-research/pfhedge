from math import ceil
from typing import Callable
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_merton_jump

from .base import BasePrimary


class MertonJumpStock(BasePrimary):
    r"""A stock of which spot price and variance follow Merton Jump Diffusion process.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_merton_jump`:
          The stochastic process.

    Args:
        mu (float, default=0.0): The parameter :math:`\mu`.
        sigma (float, default=0.2): The parameter :math:`\sigma`.
        jump_per_year (float, default=1.0): The frequency of jumps in one year.
        jump_mean (float, default=0.0): The mean of jumnp sizes.
        jump_std (float, default=0.3): The deviation of jump sizes.
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
        engine (callable, default=torch.randn): The desired generator of random numbers
            from a standard normal distribution.
            A function call ``engine(size, dtype=None, device=None)``
            should return a tensor filled with random numbers
            from a standard normal distribution.

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
        >>> from pfhedge.instruments import MertonJumpStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = MertonJumpStock()
        >>> stock.simulate(n_paths=2, time_horizon=5/250)
        >>> stock.spot
        tensor([[1.0000, 1.0100, 1.0135, 1.0141, 1.0208, 1.0176],
                [1.0000, 1.0066, 0.9911, 1.0002, 1.0018, 1.0127]])
        >>> stock.variance
        tensor([[0.0400, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400],
                [0.0400, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400]])
        >>> stock.volatility
        tensor([[0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000]])
    """

    spot: Tensor

    def __init__(
        self,
        mu: float = 0.0,
        sigma: float = 0.2,
        jump_per_year: float = 68,
        jump_mean: float = 0.0,
        jump_std: float = 0.01,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        engine: Callable[..., Tensor] = torch.randn,
    ) -> None:
        super().__init__()

        self.mu = mu
        self.sigma = sigma
        self.jump_per_year = jump_per_year
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.cost = cost
        self.dt = dt
        self.engine = engine

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def volatility(self) -> Tensor:
        """Returns the volatility of self.

        It is a tensor filled with ``self.sigma``.
        """
        return torch.full_like(self.get_buffer("spot"), self.sigma)

    @property
    def variance(self) -> Tensor:
        """Returns the volatility of self.

        It is a tensor filled with the square of ``self.sigma``.
        """
        return torch.full_like(self.get_buffer("spot"), self.sigma ** 2)

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

        output = generate_merton_jump(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            sigma=self.sigma,
            mu=self.mu,
            jump_per_year=self.jump_per_year,
            jump_mean=self.jump_mean,
            jump_std=self.jump_std,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
            engine=self.engine,
        )

        self.register_buffer("spot", output)

    def extra_repr(self) -> str:
        params = [
            "mu=" + _format_float(self.mu),
            "sigma=" + _format_float(self.sigma),
            "jump_per_year=" + _format_float(self.jump_per_year),
            "jump_mean=" + _format_float(self.jump_mean),
            "jump_std=" + _format_float(self.jump_std),
        ]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(MertonJumpStock, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(MertonJumpStock, "to", BasePrimary.to)
