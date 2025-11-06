from math import ceil
from typing import Optional, Tuple, cast
import torch
from torch import Tensor

from pfhedge.stochastic import generate_geometric_brownian
from pfhedge._utils.typing import TensorOrScalar
from pfhedge.instruments import BasePrimary
from .volatility_mixin import VolatilityMixin


class BitcoinSpotBrownian(VolatilityMixin, BasePrimary):

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
        super().__init__()

        self.cost = cost
        self.dt = dt
        self.sigma = sigma
        self.mu = mu
        self.volatility_window = volatility_window

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def is_listed(self) -> bool:
        return True

    @property
    def volatility(self) -> Tensor:
        return self.calculate_volatility()

    @property
    def variance(self) -> Tensor:
        return torch.full_like(self.get_buffer("spot"), self.sigma**2)

    @property
    def has_funding(self) -> bool:
        return False

    @property
    def max_leverage(self) -> float:
        return 1.0

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
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"BitcoinSpotBrownian({', '.join(params)})"
