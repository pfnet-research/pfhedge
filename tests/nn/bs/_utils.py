from torch import Tensor


def compute_delta(module, input: Tensor) -> Tensor:
    return module.delta(*(input[..., i] for i in range(input.size(-1))))


def compute_gamma(module, input: Tensor) -> Tensor:
    return module.gamma(*(input[..., i] for i in range(input.size(-1))))


def compute_price(module, input: Tensor) -> Tensor:
    return module.price(*(input[..., i] for i in range(input.size(-1))))
