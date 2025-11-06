import torch
from typing import Optional, TYPE_CHECKING, List

from .constants import FUNDING_TIME_TOLERANCE_FACTOR, MONEYNESS_TOLERANCE

if TYPE_CHECKING:
    from .config import BacktestConfig
    from crypto.instruments import BitcoinEuropeanOption


class BacktestValidators:

    @staticmethod
    def check_funding_alignment(
        option: "BitcoinEuropeanOption", config: "BacktestConfig"
    ) -> Optional[List[float]]:
        if option is None:
            return None

        if (
            hasattr(option.underlier, "has_funding")
            and not option.underlier.has_funding
        ):
            return None

        if not hasattr(option.underlier, "funding_payment_times"):
            return None

        funding_times = option.underlier.funding_payment_times()
        if funding_times is None or len(funding_times) == 0:
            return None

        dt = config.dt
        n_steps = option.underlier.spot.shape[1]
        time_grid = torch.arange(0, n_steps) * dt

        tolerance = dt * FUNDING_TIME_TOLERANCE_FACTOR

        misaligned_times = []
        for t in funding_times:
            diff = torch.abs(time_grid - float(t))
            closest_idx = torch.argmin(diff)
            if diff[closest_idx] > tolerance:
                misaligned_times.append(float(t))

        return misaligned_times if misaligned_times else None

    @staticmethod
    def validate_moneyness_consistency(
        initial_spots: torch.Tensor,
        strike: float,
        target_moneyness: float,
        tolerance: float = MONEYNESS_TOLERANCE,
    ) -> tuple[float, float]:
        initial_moneyness = initial_spots / strike
        log_moneyness = torch.log(initial_moneyness)

        std_log_moneyness = log_moneyness.std().item()
        mean_log_moneyness = log_moneyness.mean().item()

        return mean_log_moneyness, std_log_moneyness
