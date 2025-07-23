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
        #自定义局部波动率函数 此处使用Heston模型
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



import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
import matplotlib as mpl

def get_spline(data):  # 用来事先计算好4个期限的样条函数值，后续就不用反复计算了
    spline = []
    for m in data["maturity"].unique():
        moneyness = data.loc[data["maturity"] == m]["y"]
        sample_volatility = data.loc[data["maturity"] == m]["w"]
        cs_k = CubicSpline(x = moneyness, y = sample_volatility, extrapolate=True)
        spline.append(cs_k)
    return spline

spline = get_spline(option)

def get_total_v(data, spline, y, t):
    total_v = [float(cs(y)) for cs in spline]
    f = interp1d(x=data["maturity"].unique(), y=total_v, kind="linear", fill_value="extrapolate")
    v = float(f(t))
    return v

def diff(data, spline, y, t):
    yt = get_total_v(data, spline, y, t)
    y_up = get_total_v(data, spline, y*(1+0.001), t)
    y_down = get_total_v(data, spline, y*(1-0.001), t)
    t_up = get_total_v(data, spline, y, t*(1+0.001))

    dw_dt = (t_up - yt)/(t*0.001)
    dw_dy = (y_up - y_down)/(y*0.001*2)
    dw_dy2 = (y_up + y_down - 2*yt)/(y*0.001)**2
    return dw_dt, dw_dy, dw_dy2

def local_v(data, spline, y, t):
    w = get_total_v(data, spline, y, t)
    dw_dt, dw_dy, dw_dy2 = diff(data, spline, y, t)
    numetator = dw_dt
    denonimator = 1 - y/w*dw_dy + 0.25*(-0.25 - 1/w + y**2/w**2) * (dw_dy**2) + 0.5*dw_dy2
    local_variance = numetator / denonimator
    if local_variance < 0:  # 若存在套利机会，很可能会出现算出的结果为负数，这里简单处理一下
        local_variance = 1e-8
    return np.sqrt(local_variance)  # 公式计算的是方差，我们返回标准差，即波动率

