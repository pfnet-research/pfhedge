import torch
from torch import Tensor


def ensemble_mean(function, n_times: int = 1, *args, **kwargs) -> Tensor:
    """Compute ensemble mean from function.

    Args:
        function (callable[..., torch.Tensor]): Function to evaluate.
        n_times (int, default=1): Number of times to evaluate.
        *args, **kwargs
            Arguments passed to the function.

    Returns:
        torch.Tensor

    Examples:

        >>> function = lambda: torch.tensor([1.0, 2.0])
        >>> ensemble_mean(function, 5)
        tensor([1., 2.])

        >>> _ = torch.manual_seed(42)
        >>> function = lambda: torch.randn(2)
        >>> ensemble_mean(function, 5)
        tensor([ 0.4236, -0.0396])
    """
    return torch.stack([function(*args, **kwargs) for _ in range(n_times)]).mean(dim=0)
