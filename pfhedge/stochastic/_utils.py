from typing import Optional
from typing import Tuple
from typing import Union
from typing import cast

import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar


def cast_state(
    state: Union[Tuple[TensorOrScalar, ...], TensorOrScalar],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Tensor, ...]:
    """Cast ``init_state`` to a tuple of tensors.

    Args:
        init_state (torch.Tensor | float | tuple[(torch.Tensor | float), ...]):
            The initial state.
        dtype (torch.dtype, optional): The desired dtype.
        device (torch.device, optional): The desired device.

    Returns:
        tuple[torch.Tensor, ...]
    """
    if isinstance(state, (Tensor, float, int)):
        state_tuple: Tuple[TensorOrScalar, ...] = (state,)
    else:
        state_tuple = state

    # Cast to init_state: Tuple[Tensor, ...] with desired dtype and device
    state_tensor_tuple: Tuple[Tensor, ...] = tuple(map(torch.as_tensor, state_tuple))
    state_tensor_tuple = tuple(map(lambda t: t.to(device, dtype), state_tensor_tuple))

    return state_tensor_tuple
