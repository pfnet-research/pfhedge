import math
from typing import Optional
from typing import Tuple

import torch
from torch import Tensor
from torch.quasirandom import SobolEngine


def _box_muller(input0: Tensor, input1: Tensor) -> Tuple[Tensor, Tensor]:
    EPSILON = 1e-10
    radius = (-2 * input0.clamp(min=EPSILON).log()).sqrt()
    z0 = radius * (2 * math.pi * input1).cos()
    z1 = radius * (2 * math.pi * input1).sin()
    return z0, z1


def _get_numel(size: Tuple[int, ...]):
    out = 1
    for dim in size:
        out *= dim
    return out


class RandnSobolBoxMuller:
    """Generator of random numbers from a standard normal distribution
    using Sobol sequence and Box-Muller transformation.

    Args:
        scramble (bool, optional): Setting this to ``True`` will produce
            scrambled Sobol sequences. Scrambling is capable of producing
            better Sobol sequences. Default: ``False``.
        seed (int, optional): This is the seed for the scrambling.
            The seed of the random number generator is set to this,
            if specified. Otherwise, it uses a random seed.
            Default: ``None``.
    """

    def __init__(self, scramble: bool = False, seed: Optional[int] = None):
        self.scramble = scramble
        self.seed = seed

    def __call__(
        self,
        *size: Tuple[int, ...],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        numel = _get_numel(*size)
        output = self._generate_1d(numel, dtype=dtype, device=device)
        output.resize_(*size)
        return output

    def _generate_1d(
        self,
        n: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Tensor:
        engine = SobolEngine(2, scramble=self.scramble, seed=self.seed)
        rand = engine.draw(n // 2 + 1).to(dtype=dtype, device=device)
        z0, z1 = _box_muller(rand[:, 0], rand[:, 1])
        return torch.cat((z0, z1), dim=0)[:n]
