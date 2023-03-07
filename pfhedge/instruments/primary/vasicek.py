from math import ceil
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor

from pfhedge._utils.doc import _set_attr_and_docstring
from pfhedge._utils.doc import _set_docstring
from pfhedge._utils.str import _format_float
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.stochastic import generate_vasicek

from .base import BasePrimary


class VasicekRate(BasePrimary):
    r"""A rate which follow the Vasicek model.

    .. seealso::
        - :func:`pfhedge.stochastic.generate_vasicek`:
          The stochastic process.

    Args:
        kappa (float, default=1.0): The parameter :math:`\kappa`.
        theta (float, default=0.04): The parameter :math:`\theta`.
        sigma (float, default=0.04): The parameter :math:`\sigma`.
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
        - spot (:class:`torch.Tensor`): The spot rate of the instrument.
          This attribute is set by a method :meth:`simulate()`.
          The shape is :math:`(N, T)` where
          :math:`N` is the number of simulated paths and
          :math:`T` is the number of time steps.

    Examples:
        >>> from pfhedge.instruments import VasicekRate
        ...
        >>> _ = torch.manual_seed(42)
        >>> rate = VasicekRate()
        >>> rate.simulate(n_paths=2, time_horizon=5/250)
        >>> rate.spot
        tensor([[0.0400, 0.0409, 0.0412, 0.0418, 0.0423, 0.0395],
                [0.0400, 0.0456, 0.0439, 0.0451, 0.0457, 0.0471]])
    """

    def __init__(
        self,
        kappa: float = 1.0,
        theta: float = 0.04,
        sigma: float = 0.04,
        cost: float = 0.0,
        dt: float = 1 / 250,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (self.theta,)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        if init_state is None:
            init_state = self.default_init_state

        spot = generate_vasicek(
            n_paths=n_paths,
            n_steps=ceil(time_horizon / self.dt + 1),
            init_state=init_state,
            kappa=self.kappa,
            theta=self.theta,
            sigma=self.sigma,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)

    def extra_repr(self) -> str:
        params = [
            "kappa=" + _format_float(self.kappa),
            "theta=" + _format_float(self.theta),
            "sigma=" + _format_float(self.sigma),
        ]
        if self.cost != 0.0:
            params.append("cost=" + _format_float(self.cost))
        params.append("dt=" + _format_float(self.dt))
        return ", ".join(params)


# Assign docstrings so they appear in Sphinx documentation
_set_docstring(VasicekRate, "default_init_state", BasePrimary.default_init_state)
_set_attr_and_docstring(VasicekRate, "to", BasePrimary.to)
