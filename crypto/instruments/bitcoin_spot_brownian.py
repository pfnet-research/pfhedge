"""
Bitcoin spot with Brownian motion simulation for deep hedging training.
"""

from math import ceil
from typing import Optional, Tuple, cast
import torch
from torch import Tensor

from pfhedge.stochastic import generate_geometric_brownian
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.instruments import BasePrimary
from .volatility_mixin import VolatilityMixin


class BitcoinSpotBrownian(VolatilityMixin, BasePrimary):
    """Bitcoin spot with geometric Brownian motion simulation.

    This implementation generates synthetic price paths using geometric
    Brownian motion for spot Bitcoin trading. Simpler than perpetual futures
    as it has no funding rates or leverage.

    Key differences from perpetual:
    - No funding rates (no periodic payments)
    - No leverage (full capital required)
    - Higher transaction costs (typical spot exchange fees)
    - Simpler P&L calculation

    Args:
        sigma (float, default=0.8): Volatility parameter (annualized).
            Default is higher than stocks due to crypto volatility.
        mu (float, default=0.0): Drift parameter (annualized).
        cost (float, default=0.001): Transaction cost rate (0.1%).
            Higher than perpetual due to spot market structure.
        dt (float, default=8/24/365): Time step (8 hours).
            Consistent with perpetual for fair comparison.
        dtype (torch.dtype, optional): Tensor dtype.
        device (torch.device, optional): Tensor device.

    Examples:
        >>> btc = BitcoinSpotBrownian(sigma=0.8, mu=0.1)
        >>> btc.simulate(n_paths=1000, time_horizon=30/365)
        >>> print(btc.spot.shape)
        torch.Size([1000, 91])  # 1000 paths, ~91 time steps (8-hour bars)
        >>>
        >>> # Training deep hedging with spot
        >>> from pfhedge.instruments import EuropeanOption
        >>> from pfhedge.nn import Hedger, MLP
        >>>
        >>> option = EuropeanOption(btc, strike=50000, maturity=30/365)
        >>> hedger = Hedger(
        ...     MLP(3, 1),
        ...     inputs=["log_moneyness", "time_to_maturity", "volatility"]
        ... )
        >>> hedger.fit(option, n_paths=10000, n_epochs=100)
    """

    def __init__(
        self,
        sigma: float = 0.8,  # Higher volatility for crypto
        mu: float = 0.0,
        cost: float = 0.001,  # 0.1% - typical spot fees
        dt: float = 8 / 24 / 365,  # 8-hour bars
        volatility_window: int = 0,  # 0 = constant vol, >0 = rolling realized vol
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Brownian Bitcoin spot."""
        super().__init__()

        self.cost = cost
        self.dt = dt
        self.sigma = sigma
        self.mu = mu
        self.volatility_window = volatility_window

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        """Default initial state for simulation.

        Returns (1.0,) to match normalized pricing convention used in training.
        When training with normalized strikes (K/S0), we start at S0=1.0.
        """
        return (1.0,)

    @property
    def is_listed(self) -> bool:
        """Bitcoin spot is always listed."""
        return True

    @property
    def volatility(self) -> Tensor:
        """Returns the volatility of the instrument.

        Uses VolatilityMixin to calculate volatility based on configuration.
        See VolatilityMixin documentation for details.
        """
        return self.calculate_volatility()

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the instrument.

        Returns a tensor filled with sigma squared.
        """
        return torch.full_like(self.get_buffer("spot"), self.sigma**2)

    @property
    def has_funding(self) -> bool:
        """Spot instruments do not have funding rates."""
        return False

    @property
    def max_leverage(self) -> float:
        """Spot trading uses no leverage (1x)."""
        return 1.0

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Simulate Bitcoin spot paths using geometric Brownian motion.

        This generates multiple synthetic price paths for Monte Carlo
        training of deep hedging models.

        Args:
            n_paths: Number of paths to simulate (can be large for training)
            time_horizon: Time period to simulate (in years)
            init_state: Initial price state, default (50000.0,)

        Examples:
            >>> btc = BitcoinSpotBrownian(sigma=0.8)
            >>> btc.simulate(n_paths=10000, time_horizon=30/365)
            >>> # Now have 10,000 different scenarios for training
        """
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        n_steps = ceil(time_horizon / self.dt + 1)

        # Generate spot price paths using geometric Brownian motion
        spot = generate_geometric_brownian(
            n_paths=n_paths,
            n_steps=n_steps,
            init_state=init_state,
            sigma=self.sigma,
            mu=self.mu,
            dt=self.dt,
            dtype=self.dtype,
            device=self.device,
        )

        self.register_buffer("spot", spot)

        # Generate bid/ask spreads
        # Spread increases with volatility
        base_spread = 0.0001  # 1 basis point
        volatility_adjustment = torch.abs(torch.randn_like(spot) * base_spread)
        spread = base_spread + volatility_adjustment

        bid = spot * (1 - spread / 2)
        ask = spot * (1 + spread / 2)

        self.register_buffer("bid", bid)
        self.register_buffer("ask", ask)
        self.register_buffer("mid", (bid + ask) / 2)

    def simulate_with_historical_parameters(
        self,
        n_paths: int,
        time_horizon: float,
        historical_data: Optional[dict] = None,
    ) -> None:
        """Simulate using parameters calibrated from historical data.

        Args:
            n_paths: Number of paths to simulate
            time_horizon: Time period to simulate
            historical_data: Dict with historical statistics
                - 'volatility': Historical volatility
                - 'drift': Historical drift
                - 'init_price': Starting price

        Examples:
            >>> # Calibrate from historical data
            >>> hist_params = {
            ...     'volatility': 0.75,
            ...     'drift': 0.15,
            ...     'init_price': 55000
            ... }
            >>> btc = BitcoinSpotBrownian()
            >>> btc.simulate_with_historical_parameters(
            ...     n_paths=10000,
            ...     time_horizon=30/365,
            ...     historical_data=hist_params
            ... )
        """
        if historical_data:
            # Override parameters with historical values
            if "volatility" in historical_data:
                self.sigma = historical_data["volatility"]
            if "drift" in historical_data:
                self.mu = historical_data["drift"]

            init_state = (historical_data.get("init_price", 50000.0),)
        else:
            init_state = None

        # Run simulation with calibrated parameters
        self.simulate(n_paths, time_horizon, init_state)

    def __repr__(self) -> str:
        """String representation."""
        params = [
            f"sigma={self.sigma}",
            f"mu={self.mu}",
            f"cost={self.cost}",
            f"dt={self.dt}",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinSpotBrownian({', '.join(params)})"
