from torch.nn import Module
from torch.nn.parameter import is_lazy


def has_lazy(module: Module) -> bool:
    return any(map(is_lazy, module.parameters()))
