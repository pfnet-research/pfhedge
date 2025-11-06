from typing import Optional, Tuple, cast
import torch
from torch import Tensor

from pfhedge._utils.typing import TensorOrScalar


class HistoricalDataMixin:

    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        if init_state is None:
            init_state = cast(Tuple[float], self.default_init_state)

        # Delegate to data loader for bootstrap path generation
        # The data loader knows how to resample historical data
        paths = self.data_loader.generate_bootstrap_paths(
            n_paths=n_paths, time_horizon=time_horizon, init_state=init_state
        )

        # Register as spot buffer
        self.register_buffer("spot", paths)

        # Generate bid/ask spreads (same logic as Brownian simulations)
        base_spread = 0.0001  # 1 basis point
        volatility_adjustment = torch.abs(torch.randn_like(paths) * base_spread)
        spread = base_spread + volatility_adjustment

        bid = paths * (1 - spread / 2)
        ask = paths * (1 + spread / 2)

        self.register_buffer("bid", bid)
        self.register_buffer("ask", ask)
        self.register_buffer("mid", (bid + ask) / 2)
