"""Validation utilities for backtesting."""

import torch
from typing import Optional, TYPE_CHECKING, List

from .constants import FUNDING_TIME_TOLERANCE_FACTOR, MONEYNESS_TOLERANCE

if TYPE_CHECKING:
    from .config import BacktestConfig
    from crypto.instruments import BitcoinEuropeanOption


class BacktestValidators:
    """Validation utilities for backtesting."""

    @staticmethod
    def check_funding_alignment(
        option: "BitcoinEuropeanOption", config: "BacktestConfig"
    ) -> Optional[List[float]]:
        """Check if funding payment times align with resampled grid.

        Returns:
            List of misaligned funding times, or None if no issues

        Raises:
            Exception: Only if check itself fails (not for misalignment)
        """
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
        """Validate moneyness matches target within tolerance.

        Args:
            initial_spots: Initial spot prices across paths
            strike: Strike price
            target_moneyness: Expected moneyness (spot/strike)
            tolerance: Acceptable deviation

        Returns:
            Tuple of (mean_log_moneyness, std_log_moneyness)
        """
        initial_moneyness = initial_spots / strike
        log_moneyness = torch.log(initial_moneyness)

        std_log_moneyness = log_moneyness.std().item()
        mean_log_moneyness = log_moneyness.mean().item()

        return mean_log_moneyness, std_log_moneyness
