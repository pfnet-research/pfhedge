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
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_kou_jump

from .base import BasePrimary


class KouJumpStock(BasePrimary):
    r"""A stock of which spot prices follow the Kou's jump diffusion.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_kou_jump`:
          The stochastic process.

    Args:
        sigma (float, default=0.2): The parameter :math:`\sigma`,
            which stands for the volatility of the spot price.
        mu (float, default=0.0): The parameter :math:`\mu`,
            which stands for the drift of the spot price.
        jump_per_year (float, optional): Jump poisson process annual
            lambda: Average number of annual jumps. Defaults to 1.0.
        jump_mean_up (float, optional): Mu for the up jumps:
            Instaneous value. Defaults to 0.02.
            This has to be positive and smaller than 1.
        jump_mean_down (float, optional): Mu for the down jumps:
            Instaneous value. Defaults to 0.05.
            This has to be larger than 0.
        jump_up_prob (float, optional): Given a jump occurs,
            this is conditional prob for up jump.
            Down jump occurs with prob 1-jump_up_prob.
            Has to be in [0,1].
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
            Only to be used for the normal component,
            jupms uses poisson distribution.

    Buffers:
        - spot (:class:`torch.Tensor`): The spot prices of the instrument.
          This attribute is set by a method :meth:`simulate()`.
          The shape is :math:`(N, T)` where
          :math:`N` is the number of simulated paths and
          :math:`T` is the number of time steps.

    Examples:
        >>> from pfhedge.instruments import KouJumpStock
        >>>
        >>> _ = torch.manual_seed(42)
        >>> stock = KouJumpStock(jump_per_year = 10.0)
        >>> stock.simulate(n_paths=2, time_horizon=5 / 250)
        >>> stock.spot
        tensor([[1.0000, 1.0021, 1.0055, 1.0089, 0.9952, 0.9933],
                [1.0000, 0.9924, 0.9987, 1.0025, 1.0098, 1.0207]])
        >>> stock.variance
        tensor([[0.0400, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400],
                [0.0400, 0.0400, 0.0400, 0.0400, 0.0400, 0.0400]])
        >>> stock.volatility
        tensor([[0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000],
                [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.2000]])
    """

    def __init__(
        self,
        sigma: float = 0.2,
        mu: float = 0.0,
        jump_per_year: float = 68.0,
        jump_mean_up: float = 0.02,
        jump_mean_down: float = 0.05,
        jump_up_prob: float = 0.5,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        engine: Callable[..., Tensor] = torch.randn,
    ) -> None:
        super().__init__()

        self.sigma = sigma
        self.mu = mu
        self.jump_per_year = jump_per_year
        self.jump_mean_up = jump_mean_up
        self.jump_mean_down = jump_mean_down
        self.jump_up_prob = jump_up_prob
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
        """
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        spot = generate_kou_jump(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            sigma=self.sigma,
            mu=self.mu,
            jump_per_year=self.jump_per_year,
            jump_mean_up=self.jump_mean_up,
            jump_mean_down=self.jump_mean_down,
            jump_up_prob=self.jump_up_prob,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
            engine=self.engine,
        )

        self.register_buffer("spot", spot)

    def extra_repr(self) -> str:
        params = ["sigma=" + _format_float(self.sigma)]
        params.append("mu=" + _format_float(self.mu))
        params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        params.append("jump_per_year=" + _format_float(self.jump_per_year))
        params.append("jump_mean_up=" + _format_float(self.jump_mean_up))
        params.append("jump_mean_down=" + _format_float(self.jump_mean_down))
        params.append("jump_up_prob=" + _format_float(self.jump_up_prob))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(KouJumpStock, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(KouJumpStock, "to", BasePrimary.to)
