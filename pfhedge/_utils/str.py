import torch


def _format_float(value: float) -> str:
    """
    >>> _format_float(1)
    '1'
    >>> _format_float(1.0)
    '1.'
    >>> _format_float(1e-4)
    '1.0000e-04'
    """
    # format a float following PyTorch printoptions
    # see `torch.set_printoptions` for details
    tensor = torch.tensor([value])
    return torch._tensor_str._Formatter(tensor).format(value)
