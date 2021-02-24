from ..stochastic import generate_geometric_brownian
from ._base import Primary


class BrownianStock(Primary):
    """
    A stock of which prices follow the geometric Brownian motion.
    The drift of the prices is assumed to be vanishing.

    Parameters
    ----------
    - volatility : float, default 0.20
        The volatility of the price.
    - cost : float, default 0.0
        The transaction cost rate.
    - dt : float, default 1.0 / 365
        The intervals of the time steps.
    - dtype : torch.device, optional
        Desired device of returned tensor.
        Default: If None, uses a global default (see `torch.set_default_tensor_type()`).
    - device : torch.device, optional
        Desired device of returned tensor.
        Default: if None, uses the current device for the default tensor type
        (see `torch.set_default_tensor_type()`).
        `device` will be the CPU for CPU tensor types and
        the current CUDA device for CUDA tensor types.

    Attributes
    ----------
    - prices : Tensor, shape (N_STEPS, N_PATHS)
        The prices of the instrument.
        This attribute is supposed to set by a method `simulate()`.
        Here, `N_STEPS` is the number of time steps and
        `N_PATHS` is the number of simulated paths.

    Examples
    --------
    >>> import torch

    >>> _ = torch.manual_seed(42)
    >>> stock = BrownianStock(volatility=0.20)
    >>> stock.simulate(time_horizon=5 / 365, n_paths=2)
    >>> stock.prices
    tensor([[1.0000, 1.0000],
            [1.0024, 1.0024],
            [0.9906, 1.0004],
            [1.0137, 0.9936],
            [1.0186, 0.9964]])
    """

    def __init__(self, volatility=0.2, cost=0.0, dt=1.0 / 365, dtype=None, device=None):
        super().__init__()

        self.volatility = volatility
        self.cost = cost
        self.dt = dt

        self.to(dtype=dtype, device=device)

    def __repr__(self):
        params = [f"volatility={self.volatility:.2e}"]

        if self.cost != 0.0:
            params.append(f"cost={self.cost:.2e}")
        params.append(f"dt={self.dt:.2e}")
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device={self.device}")

        return self.__class__.__name__ + f"({', '.join(params)})"

    def simulate(self, time_horizon, n_paths=1, init_price=1.0) -> None:
        """
        Simulate time series of prices and set an attribute `prices`.

        Parameters
        ----------
        - time_horizon : float
            The period of time to simulate the price.
        - n_paths : int, default 1
            The number of paths to simulate.
        - init_price : float, default 1.0
            The initial value of the prices.
        """
        n_steps = int(time_horizon / self.dt)
        self.prices = generate_geometric_brownian(
            n_steps=n_steps,
            n_paths=n_paths,
            init_value=init_price,
            volatility=self.volatility,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )
