import logging
import numpy as np
import torch
from typing import Union, TYPE_CHECKING

from .constants import DAYS_PER_YEAR, MONEYNESS_TOLERANCE

if TYPE_CHECKING:
    from .config import BacktestConfig
    from crypto.data.loader import CryptoDataLoader
    from crypto.instruments import (
        BitcoinEuropeanOption,
        BitcoinPerpetualHistorical,
        BitcoinSpotHistorical,
    )


class BootstrapOptionGenerator:

    def __init__(self, config: "BacktestConfig", data_loader: "CryptoDataLoader"):
        self.config = config
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__)

    def create_option(self) -> "BitcoinEuropeanOption":
        from crypto.instruments import BitcoinEuropeanOption

        if self.data_loader is None:
            raise ValueError(
                "No data loaded. Call load_data() first or provide a data_loader."
            )

        if (
            self.data_loader.perpetual_data is None
            or self.data_loader.perpetual_data.empty
        ):
            raise ValueError("Data loader has no perpetual data")

        self.logger.info("Creating bootstrap option from historical data...")
        self.logger.info(
            f"Using time-varying realized volatility ({self.config.volatility_window}-period rolling window)"
        )
        self.logger.info("This is more realistic than constant vol from training")

        underlier = self._create_underlier()

        time_horizon = self.config.maturity_days / DAYS_PER_YEAR

        self._simulate_bootstrap_paths(underlier, time_horizon)

        option = BitcoinEuropeanOption(
            underlier=underlier,
            strike=self.config.strike,
            maturity=time_horizon,
            call=self.config.call,
            cost=0.0,
        )

        return option

    def _create_underlier(
        self,
    ) -> Union["BitcoinSpotHistorical", "BitcoinPerpetualHistorical"]:
        from crypto.instruments import (
            BitcoinPerpetualHistorical,
            BitcoinSpotHistorical,
        )

        underlying_type = self.config.underlying_type
        self.logger.info(f"Creating {underlying_type} underlier for backtest")

        constant_vol = None

        if underlying_type == "spot":
            underlier = BitcoinSpotHistorical(
                data_loader=self.data_loader,
                cost=self.config.transaction_cost,
                dt=self.config.dt,
                dtype=torch.float32,
                device="cpu",
                constant_volatility=constant_vol,
                volatility_window=self.config.volatility_window,
            )
        elif underlying_type == "perpetual":
            underlier = BitcoinPerpetualHistorical(
                data_loader=self.data_loader,
                cost=self.config.transaction_cost,
                dt=self.config.dt,
                dtype=torch.float32,
                device="cpu",
                constant_volatility=constant_vol,
                volatility_window=self.config.volatility_window,
            )
        else:
            raise ValueError(
                f"Invalid underlying_type: {underlying_type}. Must be 'spot' or 'perpetual'"
            )

        return underlier

    def _simulate_bootstrap_paths(
        self,
        underlier: Union["BitcoinSpotHistorical", "BitcoinPerpetualHistorical"],
        time_horizon: float,
    ) -> None:
        bootstrap_mode = self.config.bootstrap_mode

        if bootstrap_mode == "normalize_spot":
            self._simulate_normalize_spot(underlier, time_horizon)
        elif bootstrap_mode == "absolute_strike":
            self._simulate_absolute_strike(underlier, time_horizon)
        else:
            raise ValueError(
                f"Invalid bootstrap_mode: {bootstrap_mode}. Must be 'normalize_spot' or 'absolute_strike'"
            )

    def _simulate_normalize_spot(
        self,
        underlier: Union["BitcoinSpotHistorical", "BitcoinPerpetualHistorical"],
        time_horizon: float,
    ) -> None:
        target_moneyness = self.config.effective_target_moneyness
        target_initial_spot = self.config.strike * target_moneyness

        self.logger.info(f"Bootstrap mode: normalize_spot")
        self.logger.info(f"  Target moneyness: {target_moneyness:.4f}")
        self.logger.info(f"  Target initial spot: ${target_initial_spot:,.2f}")
        self.logger.info(f"  Strike: ${self.config.strike:,.2f}")
        self.logger.info(f"  Target log_moneyness: {np.log(target_moneyness):.4f}")

        try:
            underlier.simulate_bootstrap(
                n_paths=self.config.n_bootstrap_paths,
                time_horizon=time_horizon,
                window_size=None,
                target_initial_spot=target_initial_spot,
                max_date=None,
                store_scale_factors=True,
            )
        except Exception as e:
            raise ValueError(f"Failed to generate bootstrap paths: {e}")

        self.logger.info(f"✅ Generated {underlier.spot.shape[0]:,} bootstrap paths")
        self.logger.info(f"   Each path has {underlier.spot.shape[1]} time steps")

        self._validate_moneyness_consistency(underlier, target_moneyness)

    def _simulate_absolute_strike(
        self,
        underlier: Union["BitcoinSpotHistorical", "BitcoinPerpetualHistorical"],
        time_horizon: float,
    ) -> None:
        self.logger.info(f"Bootstrap mode: absolute_strike (no rescaling)")
        self.logger.warning(
            "Using absolute_strike mode - paths will have varying moneyness. "
            "Consider bootstrap_mode='normalize_spot' for consistent results."
        )

        try:
            underlier.simulate_bootstrap(
                n_paths=self.config.n_bootstrap_paths,
                time_horizon=time_horizon,
                window_size=None,
                target_initial_spot=None,
                max_date=None,
                store_scale_factors=False,
            )
        except Exception as e:
            raise ValueError(f"Failed to generate bootstrap paths: {e}")

        self.logger.info(f"✅ Generated {underlier.spot.shape[0]:,} bootstrap paths")
        self.logger.info(f"   Each path has {underlier.spot.shape[1]} time steps")

        self._log_moneyness_distribution(underlier)

    def _validate_moneyness_consistency(
        self,
        underlier: Union["BitcoinSpotHistorical", "BitcoinPerpetualHistorical"],
        target_moneyness: float,
    ) -> None:
        initial_spots = underlier.spot[:, 0]
        initial_moneyness = initial_spots / self.config.strike
        log_moneyness = torch.log(initial_moneyness)

        std_log_moneyness = log_moneyness.std().item()
        mean_log_moneyness = log_moneyness.mean().item()
        target_log_moneyness = np.log(target_moneyness)

        if std_log_moneyness > MONEYNESS_TOLERANCE:
            self.logger.warning(
                f"Initial log_moneyness varies across paths (std={std_log_moneyness:.6f})"
            )

        if abs(mean_log_moneyness - target_log_moneyness) > MONEYNESS_TOLERANCE:
            self.logger.warning(
                f"Mean log_moneyness ({mean_log_moneyness:.4f}) differs from target ({target_log_moneyness:.4f})"
            )

        self.logger.info(
            f"✅ Moneyness verified: mean={mean_log_moneyness:.4f}, std={std_log_moneyness:.6f}, "
            f"initial_spot: ${initial_spots.mean().item():,.2f} "
            f"(${initial_spots.min().item():,.2f}-${initial_spots.max().item():,.2f})"
        )

    def _log_moneyness_distribution(
        self,
        underlier: Union["BitcoinSpotHistorical", "BitcoinPerpetualHistorical"],
    ) -> None:
        initial_spots = underlier.spot[:, 0]
        initial_moneyness = initial_spots / self.config.strike
        log_moneyness = torch.log(initial_moneyness)

        p5 = torch.quantile(log_moneyness, 0.05).item()
        p50 = torch.quantile(log_moneyness, 0.50).item()
        p95 = torch.quantile(log_moneyness, 0.95).item()

        self.logger.info(
            f"ℹ️  Initial log_moneyness distribution: "
            f"5th={p5:.4f}, median={p50:.4f}, 95th={p95:.4f} "
            f"(range: {np.exp(p5):.3f}x to {np.exp(p95):.3f}x)"
        )
