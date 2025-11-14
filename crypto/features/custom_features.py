from typing import Optional
import torch
from torch import Tensor

from pfhedge.features._base import StateIndependentFeature
from pfhedge.features._getter import FeatureFactory


class VolatilityChange(StateIndependentFeature):
    """Change in volatility between consecutive time steps.

    Captures volatility dynamics crucial for spot hedging.
    Computed as: current_volatility - previous_volatility
    First time step has zero change (no previous volatility).

    Name:
        'volatility_change'
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "volatility_change"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        # Call the derivative's volatility_change method
        # This expects BitcoinEuropeanOption to have volatility_change() method
        if hasattr(self.derivative, "volatility_change"):
            vol_change = self.derivative.volatility_change()
            if time_step is not None:
                vol_change = vol_change[:, time_step : time_step + 1]
            return vol_change.unsqueeze(-1)
        else:
            # Fallback: compute from volatility feature
            if hasattr(self.derivative, "volatility"):
                current_vol = self.derivative.volatility()
                # Compute change
                vol_change = torch.cat(
                    [
                        torch.zeros_like(current_vol[:, [0]]),
                        current_vol[:, 1:] - current_vol[:, :-1],
                    ],
                    dim=1,
                )
                if time_step is not None:
                    vol_change = vol_change[:, time_step : time_step + 1]
                return vol_change.unsqueeze(-1)
            else:
                raise AttributeError(
                    "Derivative must have volatility_change() or volatility() method"
                )


class MoneynessSquared(StateIndependentFeature):
    """Squared log-moneyness for capturing non-linear gamma effects.

    Computed as: (log(S / K))^2
    Helps capture non-linear effects near at-the-money strikes.

    Name:
        'moneyness_squared'
    """

    def __init__(self) -> None:
        super().__init__()

    def __str__(self) -> str:
        return "moneyness_squared"

    def get(self, time_step: Optional[int] = None) -> Tensor:
        # Use the derivative's moneyness_squared method if available
        if hasattr(self.derivative, "moneyness_squared"):
            m_squared = self.derivative.moneyness_squared()
            if time_step is not None:
                m_squared = m_squared[:, time_step : time_step + 1]
            return m_squared.unsqueeze(-1)
        else:
            # Fallback: compute from log_moneyness
            log_m = self.derivative.moneyness(time_step, log=True)
            if log_m.dim() == 2:
                log_m = log_m.unsqueeze(-1)
            return log_m**2


# Register the custom features
FeatureFactory().register_feature("volatility_change", VolatilityChange)
FeatureFactory().register_feature("moneyness_squared", MoneynessSquared)
