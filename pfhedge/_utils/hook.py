def save_prev_output(module, input, output) -> None:
    """
    Save previous output as an attribute `prev`.

    Parameters
    ----------
    - module : torch.nn.Module
        The module.
    - input : Tensor
        The input tensor.
    - output : Tensor
        The output tensor.

    Examples
    --------
    >>> import torch
    >>> _ = torch.manual_seed(42)
    >>> m = torch.nn.Linear(3, 2)
    >>> hook = m.register_forward_hook(save_prev_output)
    >>> x = torch.randn(1, 3)
    >>> m(x)
    tensor([[-1.1647,  0.0244]], grad_fn=<AddmmBackward>)
    >>> m.prev
    tensor([[-1.1647,  0.0244]], grad_fn=<AddmmBackward>)
    """
    module.prev = output
