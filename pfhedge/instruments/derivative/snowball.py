from dataclasses import dataclass
from typing import Any
from typing import Optional
from typing import override

import torch
from torch import Tensor

from pfhedge.instruments.derivative.base import BaseDerivative
from pfhedge.instruments.derivative.base import OptionMixin
from pfhedge.nn.functional import european_payoff


@dataclass
class Snowball(BaseDerivative, OptionMixin):
    """Snowball option.

    A Snowball option is a structured product with potential early redemption
    and knockout features.
    """

    notional: float
    init_spot: float
    strike: float
    maturity: float
    knockin_barrier: float
    no_touch_coupon: float
    observations: list[float]
    knockout_barriers: list[float]
    knockout_coupons: list[float]
    is_knocked_in: bool

    def __init__(
        self,
        underlier: Any,
        notional: float,
        init_spot: float,
        strike: float,
        maturity: float,
        knockin_barrier: float,
        observations: list[float],
        knockout_barriers: list[float],
        knockout_coupons: list[float],
        no_touch_coupon: float = 0.0,
        is_knocked_in: bool = False,
    ) -> None:
        """Initialize the Snowball option.

        Args:
            underlier: The underlying asset
            notional: Notional amount of the contract
            init_spot: Initial spot price of the underlier
            strike: Strike price
            maturity: Time to maturity in years
            knockin_barrier: Knock-in barrier level
            observations: List of observation times
            knockout_barriers: List of knockout barrier levels for each observation
            knockout_coupons: List of coupon rates for each knockout observation
            no_touch_coupon: Fixed coupon paid if not knocked out/in
        """
        super().__init__()
        self.register_underlier("underlier", underlier)
        self.notional = notional
        self.init_spot = init_spot
        self.strike = strike
        self.maturity = maturity
        self.knockin_barrier = knockin_barrier
        self.observations = torch.tensor(observations)
        self.knockout_barriers = torch.tensor(knockout_barriers)
        self.knockout_coupons = torch.tensor(knockout_coupons)
        self.no_touch_coupon = no_touch_coupon
        self.is_knocked_in = is_knocked_in

    def payoff_fn(self) -> Tensor:
        """Defines the payoff function of the Snowball option.

        Returns:
            torch.Tensor: The payoff of the option.
        """
        spot = self.ul().spot
        payoff = (
            -european_payoff(spot, call=False, strike=self.strike) * self.notional / self.init_spot
        )
        if not self.is_knocked_in:
            payoff = payoff.where(
                self.knocked_in(spot.size(-1) - 1)[..., -1],
                torch.full_like(spot[..., -1], self.no_touch_coupon),
            )
        for o, b, c in zip(
            reversed(self.observations),
            reversed(self.knockout_barriers),
            reversed(self.knockout_coupons),
        ):
            idx = torch.round(o / self.underlier.dt).long().item()
            payoff = payoff.where(spot[..., idx] < b, torch.full_like(spot[..., idx], c))
        return payoff

    def knocked_in(self, time_step: Optional[int] = None) -> Tensor:
        spot = self.ul().spot
        if time_step is None:
            return spot.cummin(-1).values <= self.knockin_barrier
        return spot[..., : time_step + 1].min(-1, keepdim=True).values <= self.knockin_barrier

    @override
    def extra_repr(self) -> str:
        params = []
        params.append("notional=" + _format_float(self.notional))
        params.append("init_spot=" + _format_float(self.init_spot))
        params.append("strike=" + _format_float(self.strike))
        params.append("maturity=" + _format_float(self.maturity))
        params.append("knockin_barrier=" + _format_float(self.knockin_barrier))
        params.append("observations=" + str(self.observations))
        params.append("knockout_barriers=" + str(self.knockout_barriers))
        params.append("knockout_coupons=" + str(self.knockout_coupons))
        params.append("no_touch_coupon=" + _format_float(self.no_touch_coupon))
        params.append("is_knocked_in=" + str(self.is_knocked_in))
        return "\n".join(params)
