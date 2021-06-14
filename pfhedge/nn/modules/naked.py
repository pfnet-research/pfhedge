import torch


class Naked(torch.nn.Module):
    """Returns a tensor filled with the scalar value zero.

    Args:
        out_features (int, default=1): Size of each output sample.

    Shape:
        - Input: :math:`(N, *, H_{\\text{in}})`, where where * means any number of
          additional dimensions and :math:`H_{\\text{in}})` is the number of input
          features.
        - Output: :math:`(N, *, H_{\\text{out}})`, where all but the last dimension
          are the same shape as the input and :math:`H_{\\text{in}})` is
          `out_features`.

    Examples:

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
