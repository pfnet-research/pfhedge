from collections import namedtuple
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from torch import Tensor

from pfhedge._utils.str import _addindent
from pfhedge._utils.typing import LocalVolatilityFunction
from pfhedge._utils.typing import TensorOrScalar

from ._utils import cast_state


class LocalVolatilityTuple(namedtuple("LocalVolatilityTuple", ["spot", "volatility"])):

    __module__ = "pfhedge.stochastic"

    def __repr__(self) -> str:
        items_str_list = []
        for field, tensor in self._asdict().items():

            items_str_list.append(field + "=\n" + str(tensor))
        items_str = _addindent("\n".join(items_str_list), 2)
        return self.__class__.__name__ + "(\n" + items_str + "\n)"

    @property
    def variance(self) -> Tensor:
        return self.volatility.square()


def generate_local_volatility_process(
    n_paths: int,
    n_steps: int,
    sigma_fn: LocalVolatilityFunction,
    init_state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar] = (1.0,),
    dt: float = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> LocalVolatilityTuple:
    r"""Returns time series following the local volatility model.

    The time evolution of the process is given by:

    .. math::
        dS(t) = \sigma_{\mathrm{LV}}(t, S(t)) S(t) dW(t) ,

    where :math:`\sigma_{\mathrm{LV}}` is the local volatility function.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], default=(0.0,)): The initial state of
            the time series.
            This is specified by a tuple :math:`(S(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
        sigma_fn (callable): The local volatility function.
            Its signature is ``sigma_fn(time: Tensor, spot: Tensor) -> Tensor``.
        dt (float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: If ``None``, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Shape:
        - Output: :math:`(N, T)` where
          :math:`N` is the number of paths and
          :math:`T` is the number of time steps.

    Returns:
        (torch.Tensor, torch.Tensor): A namedtuple ``(spot, volatility)``.

    Examples:
        >>> from pfhedge.stochastic import generate_local_volatility_process
        ...
        >>> def sigma_fn(time: Tensor, spot: Tensor) -> Tensor:
        ...     a, b, sigma = 0.0001, 0.0004, 0.1000
        ...     sqrt_term = (spot.log().square() + sigma ** 2).sqrt()
        ...     return ((a + b * sqrt_term) / time.clamp(min=1/250)).sqrt()
        ...
        >>> _ = torch.manual_seed(42)
        >>> spot, volatility = generate_local_volatility_process(2, 5, sigma_fn)
        >>> spot
        tensor([[1.0000, 1.0040, 1.0055, 1.0075, 1.0091],
                [1.0000, 0.9978, 1.0239, 1.0184, 1.0216]])
        >>> volatility
        tensor([[0.1871, 0.1871, 0.1323, 0.1081, 0.0936],
                [0.1871, 0.1871, 0.1328, 0.1083, 0.0938]])
    """
    init_state = cast_state(init_state, dtype=dtype, device=device)

    spot = torch.empty(*(n_paths, n_steps), dtype=dtype, device=device)  # type: ignore
    spot[:, 0] = init_state[0]
    volatility = torch.empty_like(spot)

    time = dt * torch.arange(n_steps).to(spot)
    dw = torch.randn_like(spot) * torch.as_tensor(dt).sqrt()

    for i_step in range(n_steps):
        sigma = sigma_fn(time[i_step], spot[:, i_step])
        volatility[:, i_step] = sigma
        if i_step != n_steps - 1:
            spot[:, i_step + 1] = spot[:, i_step] * (1 + sigma * dw[:, i_step])

    return LocalVolatilityTuple(spot, volatility)
