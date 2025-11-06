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
        return self.calculate_volatility()

    @property
    def variance(self) -> Tensor:
        return torch.full_like(self.get_buffer("spot"), self.sigma**2)

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
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
