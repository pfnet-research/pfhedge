from torch.nn import Module
from torch.nn.parameter import is_lazy


def has_lazy(module: Module) -> bool:
    """Returns `True` if a module has any `UninitializedParameter`.

    Args:
        module (torch.nn.Module):

    Returns:
        bool
    """
    for t in module.parameters():
        if is_lazy(t):
            return True
    return False
