import logging
import torch
from torch import Tensor
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import BacktestConfig
    from crypto.instruments import BitcoinEuropeanOption
    from pfhedge.nn import Hedger


class StrategyExecutor:

    def __init__(self, config: "BacktestConfig"):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.diagnostics = None
        self.positions = None

    def run_deep_hedge(
        self, option: "BitcoinEuropeanOption", model: "Hedger"
    ) -> Tensor:
        from crypto.strategies.deep_hedge_utils import (
            compute_funding_cum_cost,
            apply_no_trade_band,
        )

        if option is None:
            raise ValueError(
                "No option provided. Call create_bootstrap_option() first or provide an option."
            )
        if model is None:
            raise ValueError(
                "No model loaded. Call load_model() first or provide a model."
            )

        self.logger.info("Running deep hedging strategy...")

        model.eval()

        self._move_to_device(option, model)

        if self.config.enable_diagnostics:
            self._attach_diagnostics(model)

        with torch.no_grad():
            from pfhedge.nn.functional import cum_pl

            hedge_positions = model.compute_hedge(option).squeeze(1)
            hedge_positions = apply_no_trade_band(
                hedge_positions, self.config.band_width
            )

            spot = option.underlier.spot
            cost = option.underlier.cost
            payoff = option.payoff()

            cum_pnl = cum_pl(
                spot=spot.unsqueeze(1),
                unit=hedge_positions.unsqueeze(1),
                cost=[cost],
                payoff=payoff,
                deduct_first_cost=True,
            )

            cum_pnl = self._apply_funding_costs(cum_pnl, hedge_positions, option)

        self.logger.info(f"✅ Deep hedge strategy computed")
        self.logger.info(f"   Positions shape: {hedge_positions.shape}")
        self.logger.info(f"   PnL shape: {cum_pnl.shape}")
        self.logger.info(
            f"   Final PnL: ${cum_pnl[:, -1].mean().item():.2f} ± ${cum_pnl[:, -1].std().item():.2f}"
        )

        if self.config.enable_diagnostics and self.diagnostics is not None:
            self._print_diagnostics()

        self.positions = hedge_positions

        return cum_pnl

    def run_bs_baseline(self, option: "BitcoinEuropeanOption") -> Tensor:
        from crypto.strategies.deep_hedge_utils import (
            calculate_bs_hedge_pnl,
            apply_no_trade_band,
        )

        if option is None:
            raise ValueError(
                "No option provided. Call create_bootstrap_option() first or provide an option."
            )

        self.logger.info("Running Black-Scholes delta hedge baseline...")

        bs_delta = option.black_scholes_delta()
        spots = option.underlier.spot
        payoffs = option.payoff()
        cost = option.underlier.cost

        funding_rate = None
        funding_times = None

        if (
            hasattr(option.underlier, "has_funding")
            and option.underlier.has_funding
            and hasattr(option.underlier, "funding_rate")
            and hasattr(option.underlier, "funding_payment_times")
        ):
            funding_rate = option.underlier.funding_rate
            funding_times = option.underlier.funding_payment_times()

        cum_pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=bs_delta,
            payoffs=payoffs,
            cost=cost,
            funding_rate=funding_rate,
            funding_times=funding_times,
            band_width=self.config.band_width,
        )

        filtered_delta = apply_no_trade_band(bs_delta, self.config.band_width)

        self.logger.info(f"✅ Black-Scholes baseline computed")
        self.logger.info(f"   Delta shape: {bs_delta.shape}")
        self.logger.info(f"   PnL shape: {cum_pnl.shape}")
        self.logger.info(
            f"   Final PnL: ${cum_pnl[:, -1].mean().item():.2f} ± ${cum_pnl[:, -1].std().item():.2f}"
        )

        self.positions = filtered_delta

        return cum_pnl

    def _move_to_device(self, option: "BitcoinEuropeanOption", model: "Hedger") -> None:
        model_device = next(model.parameters()).device
        if hasattr(option.underlier, "spot"):
            if option.underlier.spot.device != model_device:
                self.logger.info(
                    f"   Moving option tensors from {option.underlier.spot.device} to {model_device}"
                )
                option.underlier.to(model_device)

    def _attach_diagnostics(self, model: "Hedger") -> None:
        from crypto.training.diagnostics import MLPDiagnostics

        self.diagnostics = MLPDiagnostics(model, sample_frequency=1)
        self.diagnostics.attach()
        self.logger.info(
            "\n🔍 Diagnostics enabled - will track MLP inputs/outputs during hedging\n"
        )

    def _apply_funding_costs(
        self,
        cum_pnl: Tensor,
        positions: Tensor,
        option: "BitcoinEuropeanOption",
    ) -> Tensor:
        from crypto.strategies.deep_hedge_utils import compute_funding_cum_cost

        if (
            hasattr(option.underlier, "has_funding")
            and option.underlier.has_funding
            and hasattr(option.underlier, "funding_rate")
            and hasattr(option.underlier, "funding_payment_times")
        ):
            spots = option.underlier.spot
            funding_rate = option.underlier.funding_rate
            funding_times = option.underlier.funding_payment_times()

            funding_costs = compute_funding_cum_cost(
                spots=spots,
                positions=positions,
                funding_rate=funding_rate,
                funding_times=funding_times,
            )

            cum_pnl = cum_pnl - funding_costs

            self.logger.info(
                f"   Applied funding costs: ${funding_costs[:, -1].mean().item():.2f} avg per path"
            )

        return cum_pnl

    def _print_diagnostics(self) -> None:
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST DIAGNOSTICS")
        self.logger.info("=" * 70)
        self.logger.info("\nMLP behavior during hedging on bootstrap paths:\n")
        self.diagnostics.print_summary(verbose=True)
        self.diagnostics.detach()
        self.logger.info("=" * 70)
