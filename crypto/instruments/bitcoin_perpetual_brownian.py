"""
Bitcoin perpetual with Brownian motion simulation for deep hedging training.
"""

from math import ceil
from typing import Optional, Tuple, cast
import torch
from torch import Tensor
import numpy as np

from pfhedge.stochastic import generate_geometric_brownian
from pfhedge._utils.typing import TensorOrScalar

from .bitcoin_perpetual_base import BitcoinPerpetualBase
from .volatility_mixin import VolatilityMixin


class BitcoinPerpetualBrownian(VolatilityMixin, BitcoinPerpetualBase):
    """Bitcoin perpetual with geometric Brownian motion simulation.

    This implementation generates synthetic price paths using geometric
    Brownian motion, suitable for deep hedging training where we need
    many different scenarios.

    Args:
        sigma (float, default=0.8): Volatility parameter (annualized).
            Default is higher than stocks due to crypto volatility.
        mu (float, default=0.0): Drift parameter (annualized).
        funding_mean (float, default=0.0001): Mean funding rate (8-hour).
        funding_std (float, default=0.0002): Std dev of funding rate.
        cost (float, default=0.0006): Transaction cost rate.
        dt (float, default=1/24/12): Time step (5 minutes).
        leverage (float, default=20.0): Maximum leverage.
        volatility_window (int, default=0): Rolling window for realized volatility.
            If 0, uses constant volatility. If >0, calculates rolling realized vol.
        dtype (torch.dtype, optional): Tensor dtype.
        device (torch.device, optional): Tensor device.

    Examples:
        >>> btc = BitcoinPerpetualBrownian(sigma=0.8, mu=0.1)
        >>> btc.simulate(n_paths=1000, time_horizon=30/365)
        >>> print(btc.spot.shape)
        torch.Size([1000, 145])  # 1000 paths, ~145 time steps
        >>>
        >>> # Training deep hedging
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
        funding_mean: float = 0.0001,  # Average funding rate
        funding_std: float = 0.0002,  # Funding volatility
        cost: float = 0.0006,
        dt: float = 8 / 24 / 365,  # 8-hour bars (matches funding interval)
        leverage: float = 20.0,
        volatility_window: int = 0,  # 0 = constant vol, >0 = rolling realized vol
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize Brownian Bitcoin perpetual."""
        super().__init__(
            cost=cost, dt=dt, leverage=leverage, dtype=dtype, device=device
        )

        self.sigma = sigma
        self.mu = mu
        self.funding_mean = funding_mean
        self.funding_std = funding_std
        self.volatility_window = volatility_window

    @property
    def volatility(self) -> Tensor:
        """Returns the volatility of the instrument.

        Uses VolatilityMixin to calculate volatility based on configuration:
        - volatility_window=0: Returns constant sigma
        - volatility_window>0: Calculates rolling realized volatility

        See VolatilityMixin documentation for details on calculation logic.
        """
        return self.calculate_volatility()

    @property
    def variance(self) -> Tensor:
        """Returns the variance of the instrument.

        Returns a tensor filled with sigma squared.
        """
        return torch.full_like(self.get_buffer("spot"), self.sigma**2)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        """Simulate Bitcoin perpetual paths using geometric Brownian motion.

        This generates multiple synthetic price paths for Monte Carlo
        training of deep hedging models.

        Args:
            n_paths: Number of paths to simulate (can be large for training)
            time_horizon: Time period to simulate
            init_state: Initial price state, default (50000.0,)

        Examples:
            >>> btc = BitcoinPerpetualBrownian(sigma=0.8)
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

        # Generate funding rates
        # Funding tends to be positive in bull markets (longs pay shorts)
        # and negative in bear markets (shorts pay longs)

        # Calculate price momentum
        returns = torch.log(spot[:, 1:] / spot[:, :-1])
        momentum = torch.cat(
            [
                torch.zeros(n_paths, 1, dtype=self.dtype, device=self.device),
                returns.cumsum(dim=1)
                / (torch.arange(1, n_steps, dtype=self.dtype, device=self.device) + 1),
            ],
            dim=1,
        )

        # Funding correlates with momentum (positive momentum -> positive funding)
        funding_base = torch.randn(
            n_paths, n_steps, dtype=self.dtype, device=self.device
        )
        funding_rates = (
            self.funding_mean
            + momentum * self.funding_mean * 2  # Momentum effect
            + funding_base * self.funding_std  # Random component
        )

        # Clip extreme funding rates
        funding_rates = torch.clamp(funding_rates, -0.01, 0.01)  # ±1% max

        self.register_buffer("_funding_rate", funding_rates)

        # Index price (slightly different from perpetual due to basis)
        basis = torch.randn_like(spot) * 0.0005  # Small random basis
        index_price = spot * (1 + basis)
        self.register_buffer("index_price", index_price)

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
                - 'funding_mean': Historical mean funding
                - 'funding_std': Historical funding volatility
                - 'init_price': Starting price

        Examples:
            >>> # Calibrate from historical data
            >>> hist_params = {
            ...     'volatility': 0.75,
            ...     'drift': 0.15,
            ...     'funding_mean': 0.0002,
            ...     'init_price': 55000
            ... }
            >>> btc = BitcoinPerpetualBrownian()
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
            if "funding_mean" in historical_data:
                self.funding_mean = historical_data["funding_mean"]
            if "funding_std" in historical_data:
                self.funding_std = historical_data["funding_std"]

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
            f"leverage={self.leverage}",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinPerpetualBrownian({', '.join(params)})"
