import torch


def ensemble_mean(function, n_times=1, *args, **kwargs) -> torch.Tensor:
    """
    Compute ensemble mean from function.

    Parameters
    ----------
    - function : callable[..., Tensor], shape (*)
        Function to evaluate.
    - n_times : int, default 1
        Number of times to evaluate.
    - *args, **kwargs
        Arguments passed to the function.

    Returns
    -------
    ensemble_mean : Tensor, shape (*)

    Examples
    --------
    >>> function = lambda: torch.tensor([1.0, 2.0])
    >>> ensemble_mean(function, 5)
    tensor([1., 2.])

    >>> _ = torch.manual_seed(42)
    >>> function = lambda: torch.randn(2)
    >>> ensemble_mean(function, 5)
    tensor([ 0.4236, -0.0396])
    """
    return torch.stack([function(*args, **kwargs) for _ in range(n_times)]).mean(dim=0)
