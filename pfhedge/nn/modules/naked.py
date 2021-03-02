import torch


class Naked(torch.nn.Module):
    """
    Returns a tensor filled with the scalar value zero.

    Parameters
    ----------
    - out_features : int, default 1
        Size of each output sample.

    Shape
    -----
    - Input : (N, *, H_in)
        where where * means any number of
        additional dimensions and `H_in = in_features`.
    - Output : (N, *, H_out)
        where all but the last dimension
        are the same shape as the input and `H_out = out_features`.

    Examples
    --------
    >>> m = Naked()
    >>> input = torch.empty((2, 3))
    >>> m(input)
    tensor([[0.],
            [0.]])
    """

    def __init__(self, out_features=1):
        super().__init__()
        self.out_features = out_features

    def forward(self, input):
        return torch.cat(
            [torch.zeros_like(input[..., :1]) for _ in range(self.out_features)], -1
        )
