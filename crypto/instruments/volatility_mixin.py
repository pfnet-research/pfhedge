import torch
from torch import Tensor
from typing import Optional


class VolatilityMixin:

    def calculate_volatility(
        self,
        constant_volatility: Optional[float] = None,
        volatility_window: Optional[int] = None,
        sigma: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> Tensor:
        # Get parameters from instance if not provided
        if constant_volatility is None:
            constant_volatility = getattr(self, "constant_volatility", None)
        if volatility_window is None:
            volatility_window = getattr(self, "volatility_window", 0)
        if sigma is None:
            sigma = getattr(self, "sigma", 0.8)
        if dt is None:
            dt = getattr(self, "dt", 8 / 24 / 365)

        # Get spot prices
        if not hasattr(self, "spot"):
            raise ValueError("No spot data. Call simulate() first.")
        spot = self.get_buffer("spot")

        # Priority 1: Override with constant volatility
        if constant_volatility is not None:
            return torch.full_like(spot, constant_volatility)

        # Priority 2: Calculate rolling window realized volatility
        if volatility_window > 0:
            from crypto.features.volatility import calculate_realized_volatility

            # Annualization factor: sqrt(periods per year)
            # Example: 8-hour bars -> 1095 periods/year -> sqrt(1095) ≈ 33
            annualization_factor = torch.sqrt(torch.tensor(1.0 / dt))

            realized_vol = calculate_realized_volatility(
                spot,
                window=volatility_window,
                annualization_factor=annualization_factor.item(),
            )

            # Fill NaN values at beginning with constant volatility
            # (First N periods don't have enough history for window)
            realized_vol = torch.where(
                torch.isnan(realized_vol),
                torch.full_like(realized_vol, sigma),
                realized_vol,
            )

            return realized_vol

        # Priority 3: Fallback to constant sigma
        return torch.full_like(spot, sigma)

    @property
    def volatility(self) -> Tensor:
        return self.calculate_volatility()
