from typing import Callable
from typing import Union

from torch import Tensor

TensorOrScalar = Union[Tensor, float, int]
LocalVolatilityFunction = Callable[[Tensor, Tensor], Tensor]
