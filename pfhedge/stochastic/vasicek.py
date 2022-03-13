from typing import Optional
from typing import Tuple
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar


def generate_vasicek(
    n_paths: int,
    n_steps: int,
    init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    kappa: TensorOrScalar = 1.0,
    theta: TensorOrScalar = 0.04,
    sigma: TensorOrScalar = 0.04,
    dt: TensorOrScalar = 1 / 250,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tensor:
    r"""Returns time series following Vasicek model.

    The time evolution of the process is given by:

    .. math::

        dX(t) = \kappa (\theta - X(t)) dt + \sigma dW(t) .

    References:
        - Gillespie, D.T., 1996.
          Exact numerical simulation of the Ornstein-Uhlenbeck process and its integral.
          Physical review E, 54(2), p.2084.

    Args:
        n_paths (int): The number of simulated paths.
        n_steps (int): The number of time steps.
        init_state (tuple[torch.Tensor | float], optional): The initial state of
            the time series.
            This is specified by a tuple :math:`(X(0),)`.
            It also accepts a :class:`torch.Tensor` or a :class:`float`.
            If ``None`` (default), it uses :math:`(\theta, )`.
        kappa (torch.Tensor or float, default=1.0): The parameter :math:`\kappa`.
        theta (torch.Tensor or float, default=0.04): The parameter :math:`\theta`.
        sigma (torch.Tensor or float, default=0.04): The parameter :math:`\sigma`.
        dt (torch.Tensor or float, default=1/250): The intervals of the time steps.
        dtype (torch.dtype, optional): The desired data type of returned tensor.
            Default: If ``None``, uses a global default
            (see :func:`torch.set_default_tensor_type()`).
        device (torch.device, optional): The desired device of returned tensor.
            Default: if None, uses the current device for the default tensor type
            (see :func:`torch.set_default_tensor_type()`).
            ``device`` will be the CPU for CPU tensor types and the current CUDA device
            for CUDA tensor types.

    Shape:
        - Output: :math:`(N, T)` where
          :math:`N` is the number of paths and
          :math:`T` is the number of time steps.

    Returns:
        torch.Tensor

    Examples:
        >>> from pfhedge.stochastic import generate_vasicek
        ...
        >>> _ = torch.manual_seed(42)
        >>> generate_vasicek(2, 5)
        tensor([[0.0400, 0.0409, 0.0412, 0.0418, 0.0423],
                [0.0400, 0.0395, 0.0451, 0.0435, 0.0446]])
    """
    if init_state is None:
        init_state = (theta,)

    # Accept Union[float, Tensor] as well because making a tuple with a single element
    # is troublesome
    if isinstance(init_state, (float, Tensor)):
        init_state = (torch.as_tensor(init_state),)

    if init_state[0] != 0:
        new_init_state = (init_state[0] - theta,)
        return theta + generate_vasicek(
            n_paths=n_paths,
            n_steps=n_steps,
            init_state=new_init_state,
            kappa=kappa,
            theta=0.0,
            sigma=sigma,
            dt=dt,
            dtype=dtype,
            device=device,
        )

    # Cast to init_state: Tuple[Tensor, ...] with desired dtype and device
    init_state = cast(Tuple[Tensor, ...], tuple(map(torch.as_tensor, init_state)))
    init_state = tuple(map(lambda t: t.to(dtype=dtype, device=device), init_state))

    output = torch.empty(*(n_paths, n_steps), dtype=dtype, device=device)
    output[:, 0] = init_state[0]

    # Cast to Tensor with desired dtype and device
    kappa, theta, sigma, dt = map(torch.as_tensor, (kappa, theta, sigma, dt))
    kappa, theta, sigma, dt = map(lambda t: t.to(output), (kappa, theta, sigma, dt))

    randn = torch.randn_like(output)

    # Compute \mu: Equation (3.3)
    mu = (-kappa * dt).exp()
    for i_step in range(n_steps - 1):
        # Compute \sigma_X: Equation (3.4)
        vola = sigma * ((1 - mu.square()) / 2 / kappa).sqrt()
        output[:, i_step + 1] = mu * output[:, i_step] + vola * randn[:, i_step]

    return output
