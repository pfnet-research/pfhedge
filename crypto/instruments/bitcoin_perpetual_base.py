from abc import abstractmethod
from typing import Optional, Tuple
import torch
from torch import Tensor

from pfhedge.instruments import BasePrimary
from pfhedge._utils.typing import TensorOrScalar


class BitcoinPerpetualBase(BasePrimary):

    def __init__(
        self,
        cost: float = 0.0006,  # Lower cost for perpetual (taker fee)
        dt: float = 1 / 24 / 12,  # 5-minute bars
        leverage: float = 20.0,  # Maximum leverage
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        self.cost = cost
        self.dt = dt
        self.leverage = leverage
        # Funding every 8 hours (expressed in years to match dt units)
        self.funding_interval = (8 / 24) / 365

        self.to(dtype=dtype, device=device)

    @property
    def default_init_state(self) -> Tuple[float, ...]:
        return (1.0,)

    @property
    def is_listed(self) -> bool:
        return True

    @property
    def has_funding(self) -> bool:
        return True

    @property
    def max_leverage(self) -> float:
        return self.leverage

    @abstractmethod
    def simulate(
        self,
        n_paths: int = 1,
        time_horizon: float = 20 / 250,
        init_state: Optional[Tuple[TensorOrScalar, ...]] = None,
    ) -> None:
        pass

    @property
    def funding_rate(self) -> Tensor:
        if not hasattr(self, "_buffers") or "_funding_rate" not in self._buffers:
            raise ValueError("No funding rate data. Call simulate() first.")
        return self.get_buffer("_funding_rate")

    def cumulative_funding_cost(self, position_size: Optional[Tensor] = None) -> Tensor:
        if "_funding_rate" not in self._buffers:
            raise ValueError("No funding rate data loaded")

        funding = self.get_buffer("_funding_rate")
        spot = self.get_buffer("spot")

        if position_size is None:
            position_size = torch.ones(
                funding.shape[0], 1, dtype=self.dtype, device=self.device
            )
        elif not isinstance(position_size, Tensor):
            position_size = torch.tensor(
                position_size, dtype=self.dtype, device=self.device
            )

        # Ensure position_size has correct shape
        if position_size.dim() == 0:
            position_size = position_size.unsqueeze(0).unsqueeze(0)
        elif position_size.dim() == 1:
            position_size = position_size.unsqueeze(1)

        # Funding cost = position_size * funding_rate * notional_value
        funding_cost = position_size * funding * spot

        # Cumulative sum over time
        return funding_cost.cumsum(dim=1)

    def margin_requirement(self, position_size: float) -> float:
        current_price = self.spot[:, -1].mean().item() if self.buffers() else 50000.0
        notional_value = abs(position_size * current_price)
        return notional_value / self.leverage

    def funding_payment_times(self) -> Tensor:
        if not self.buffers():
            raise ValueError("No data loaded. Call simulate() first.")

        n_steps = self.get_buffer("spot").shape[1]
        steps_per_funding = int(self.funding_interval / self.dt)

        # Create boolean mask for funding times
        funding_times = torch.zeros(n_steps, dtype=torch.bool, device=self.device)
        funding_times[::steps_per_funding] = True

        return funding_times

    def __repr__(self) -> str:
        params = [
            f"cost={self.cost}",
            f"dt={self.dt}",
            f"leverage={self.leverage}",
        ]
        if hasattr(self, "dtype") and self.dtype is not None:
            params.append(f"dtype={self.dtype}")
        if hasattr(self, "device") and self.device is not None:
            params.append(f"device='{self.device}'")
        return f"{self.__class__.__name__}({', '.join(params)})"
