import logging
from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .config import BacktestConfig
    from crypto.data.loader import CryptoDataLoader
    from crypto.instruments import BitcoinEuropeanOption
    from .results import BacktestResults


class BacktestLogger:

    def __init__(self, verbosity: str = "INFO"):
        self.logger = logging.getLogger(__name__)
        self._configure_logging(verbosity)

    def _configure_logging(self, verbosity: str) -> None:
        level = getattr(logging, verbosity.upper(), logging.INFO)
        self.logger.setLevel(level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(levelname)s - %(message)s"))
            self.logger.addHandler(handler)

    def log_model_loaded(
        self, model_config: dict, device: str, model_path: str
    ) -> None:
        self.logger.info("✅ Model loaded successfully")
        self.logger.info(f"   Device: {device}")
        self.logger.info(
            f"   Architecture: {model_config['n_layers']} layers × {model_config['n_units']} units"
        )
        self.logger.info(f"   Features: {model_config.get('features', 'N/A')}")
        criterion = model_config.get("criterion") or model_config.get("risk_measure")
        self.logger.info(
            f"   Criterion: {criterion} (param={model_config['risk_param']})"
        )

    def log_data_loading_start(
        self, data_dir: str, data_file: Optional[str] = None
    ) -> None:
        self.logger.info(f"Loading historical data from {data_dir}...")
        if data_file:
            self.logger.info(f"Loading data from specified file: {data_file}")

    def log_data_loaded(self, n_records: int, data_type: str = "data") -> None:
        self.logger.info(f"✅ Loaded {n_records:,} raw {data_type} records")

    def log_resampling(self, frequency: str) -> None:
        self.logger.info(f"Resampling to {frequency} frequency...")

    def log_resampled(self, n_records: int, frequency: str) -> None:
        self.logger.info(
            f"✅ Resampled to {n_records:,} records at {frequency} intervals"
        )

    def log_date_filtering(self, start_date: str, end_date: str) -> None:
        self.logger.info(f"Filtering data from {start_date} to {end_date}...")

    def log_filtered(self, n_records: int) -> None:
        self.logger.info(f"✅ Filtered to {n_records:,} records in date range")

    def log_funding_loaded(self, n_records: int) -> None:
        self.logger.info(f"✅ Loaded {n_records:,} funding rate records in date range")

    def log_no_funding_data(self) -> None:
        self.logger.warning(
            "⚠️  No funding data found (this is okay, but funding costs won't be applied)"
        )

    def log_no_options_data(self) -> None:
        self.logger.warning(
            "⚠️  No options data found (this is okay for basic backtesting)"
        )

    def log_data_summary(self, loader: "CryptoDataLoader", frequency: str) -> None:
        summary = loader.summary()
        self.logger.info("=" * 60)
        self.logger.info("DATA SUMMARY (After Filtering & Resampling)")
        self.logger.info("=" * 60)

        if "perpetual" in summary:
            perp = summary["perpetual"]
            self.logger.info(f"\nPerpetual data:")
            self.logger.info(f"  Records: {perp['records']:,}")
            self.logger.info(f"  Frequency: {frequency}")
            self.logger.info(
                f"  Date range: {perp['date_range'][0]} to {perp['date_range'][1]}"
            )
            self.logger.info(
                f"  Price range: ${perp['price_range'][0]:.2f} - ${perp['price_range'][1]:.2f}"
            )
            if perp.get("avg_spread_pct"):
                self.logger.info(f"  Avg spread: {perp['avg_spread_pct'] * 100:.4f}%")

        if "options" in summary:
            opts = summary["options"]
            self.logger.info(f"\nOptions data:")
            self.logger.info(f"  Records: {opts['records']:,}")
            self.logger.info(f"  Unique strikes: {opts['unique_strikes']}")
            self.logger.info(f"  Calls: {opts['call_count']:,}")
            self.logger.info(f"  Puts: {opts['put_count']:,}")
            if opts.get("avg_iv"):
                self.logger.info(f"  Avg IV: {opts['avg_iv']:.2%}")

        self.logger.info("=" * 60)
        self.logger.info("✅ Data loading complete\n")

    def log_bootstrap_creation_start(
        self, underlying_type: str, vol_window: int
    ) -> None:
        self.logger.info("Creating bootstrap option from historical data...")
        self.logger.info(
            f"Using time-varying realized volatility ({vol_window}-period rolling window)"
        )
        self.logger.info("This is more realistic than constant vol from training")
        self.logger.info(f"Creating {underlying_type} underlier for backtest")

    def log_bootstrap_mode(self, mode: str, details: dict) -> None:
        self.logger.info(f"Bootstrap mode: {mode}")
        if mode == "normalize_spot":
            self.logger.info(f"  Target moneyness: {details['target_moneyness']:.4f}")
            self.logger.info(
                f"  Target initial spot: ${details['target_initial_spot']:,.2f}"
            )
            self.logger.info(f"  Strike: ${details['strike']:,.2f}")
            self.logger.info(
                f"  Target log_moneyness: {details['target_log_moneyness']:.4f}"
            )
        elif mode == "absolute_strike":
            self.logger.warning(
                "Using absolute_strike mode - paths will have varying moneyness. "
                "Consider bootstrap_mode='normalize_spot' for consistent results."
            )

    def log_bootstrap_paths_generated(self, n_paths: int, n_steps: int) -> None:
        self.logger.info(f"✅ Generated {n_paths:,} bootstrap paths")
        self.logger.info(f"   Each path has {n_steps} time steps")

    def log_moneyness_verification(
        self, mean: float, std: float, min_spot: float, max_spot: float, avg_spot: float
    ) -> None:
        self.logger.info(
            f"✅ Moneyness verified: mean={mean:.4f}, std={std:.6f}, "
            f"initial_spot: ${avg_spot:,.2f} (${min_spot:,.2f}-${max_spot:,.2f})"
        )

    def log_moneyness_warning(self, std: float) -> None:
        self.logger.warning(
            f"Initial log_moneyness varies across paths (std={std:.6f})"
        )

    def log_moneyness_mismatch(self, mean: float, target: float) -> None:
        self.logger.warning(
            f"Mean log_moneyness ({mean:.4f}) differs from target ({target:.4f})"
        )

    def log_moneyness_distribution(self, p5: float, p50: float, p95: float) -> None:
        import numpy as np

        self.logger.info(
            f"ℹ️  Initial log_moneyness distribution: "
            f"5th={p5:.4f}, median={p50:.4f}, 95th={p95:.4f} "
            f"(range: {np.exp(p5):.3f}x to {np.exp(p95):.3f}x)"
        )

    def log_bootstrap_summary(
        self, option: "BitcoinEuropeanOption", config: "BacktestConfig"
    ) -> None:
        summary = option.summary()
        self.logger.info("=" * 60)
        self.logger.info("BOOTSTRAP OPTION SUMMARY")
        self.logger.info("=" * 60)
        self.logger.info(f"\nOption:")
        self.logger.info(f"  Type: {'Call' if config.call else 'Put'}")
        self.logger.info(f"  Strike: ${config.strike:,.2f}")
        self.logger.info(f"  Maturity: {config.maturity_days} days")
        self.logger.info(f"  Paths: {summary['n_paths']:,}")
        self.logger.info(f"  Steps per path: {summary['n_steps']}")

        if summary.get("simulated", False):
            self.logger.info(f"\nMarket Data:")
            self.logger.info(f"  Initial spot (avg): ${summary['spot_initial']:,.2f}")
            self.logger.info(
                f"  Final spot (avg): ${summary['spot_final_mean']:,.2f} ± ${summary['spot_final_std']:,.2f}"
            )
            self.logger.info(f"\nOption Statistics:")
            self.logger.info(
                f"  Payoff (avg): ${summary['payoff_mean']:,.2f} ± ${summary['payoff_std']:,.2f}"
            )
            self.logger.info(f"  ITM ratio: {summary['itm_ratio']:.1%}")

        self.logger.info("=" * 60)
        self.logger.info("✅ Bootstrap option created successfully\n")

    def log_strategy_execution_start(self, strategy_name: str) -> None:
        self.logger.info(f"Running {strategy_name} strategy...")

    def log_device_move(self, from_device: str, to_device: str) -> None:
        self.logger.info(f"   Moving option tensors from {from_device} to {to_device}")

    def log_diagnostics_enabled(self) -> None:
        self.logger.info(
            "\n🔍 Diagnostics enabled - will track MLP inputs/outputs during hedging\n"
        )

    def log_funding_applied(self, avg_cost: float) -> None:
        self.logger.info(f"   Applied funding costs: ${avg_cost:.2f} avg per path")

    def log_strategy_results(
        self,
        strategy_name: str,
        positions_shape: tuple,
        pnl_shape: tuple,
        final_pnl_mean: float,
        final_pnl_std: float,
    ) -> None:
        self.logger.info(f"✅ {strategy_name} strategy computed")
        self.logger.info(f"   Positions shape: {positions_shape}")
        self.logger.info(f"   PnL shape: {pnl_shape}")
        self.logger.info(f"   Final PnL: ${final_pnl_mean:.2f} ± ${final_pnl_std:.2f}")

    def log_diagnostics_summary(self, diagnostics) -> None:
        self.logger.info("=" * 70)
        self.logger.info("BACKTEST DIAGNOSTICS")
        self.logger.info("=" * 70)
        self.logger.info("\nMLP behavior during hedging on bootstrap paths:\n")
        diagnostics.print_summary(verbose=True)
        self.logger.info("=" * 70)

    def log_backtest_config(self, config: "BacktestConfig") -> None:
        self.logger.info("=" * 60)
        self.logger.info("STARTING BACKTEST")
        self.logger.info("=" * 60)
        self.logger.info(f"\nConfiguration:")
        self.logger.info(f"  Date range: {config.start_date} to {config.end_date}")
        self.logger.info(
            f"  Option: {'Call' if config.call else 'Put'} @ ${config.strike:,.2f}"
        )
        self.logger.info(f"  Maturity: {config.maturity_days} days")
        self.logger.info(f"  Bootstrap paths: {config.n_bootstrap_paths}")
        self.logger.info(f"  Time step: {config.dt_hours} hours")
        self.logger.info(f"  Transaction cost: {config.transaction_cost * 100:.3f}%")
        self.logger.info(f"  Model: {config.model_path}")
        self.logger.info(f"  Data directory: {config.data_dir}")
        self.logger.info("=" * 60)

    def log_backtest_step(self, step: int, total: int, description: str) -> None:
        self.logger.info(f"[Step {step}/{total}] {description}...")

    def log_results_creation(self) -> None:
        self.logger.info("\nCreating results object...")

    def log_seed_set(self, seed: int, cuda_available: bool) -> None:
        if cuda_available:
            self.logger.info(f"🔒 Seed={seed} (random, numpy, torch, CUDA)")
        else:
            self.logger.info(f"🔒 Seed={seed} (random, numpy, torch)")

    def log_no_seed_warning(self) -> None:
        self.logger.warning("⚠️  No seed set - results may vary between runs")

    def log_funding_alignment_warning(
        self, n_misaligned: int, misaligned_times: list, dt: float
    ) -> None:
        self.logger.warning("⚠️  " * 20)
        self.logger.warning("⚠️  WARNING: Funding Payment Time Alignment Issue")
        self.logger.warning("⚠️  " * 20)
        self.logger.warning(
            f"\nFound {n_misaligned} funding payment times that don't align"
        )
        self.logger.warning(f"with the resampled time grid (dt={dt:.6f} years).")
        self.logger.warning(f"\nMisaligned times (first 5): {misaligned_times[:5]}")
        self.logger.warning(
            f"\nThis may cause funding costs to be applied at slightly different"
        )
        self.logger.warning(
            f"times than intended. Consider adjusting dt_hours to align with funding"
        )
        self.logger.warning(
            f"payment frequency (typically 8 hours for perpetual futures)."
        )
        self.logger.warning("⚠️  " * 20)

    def log_funding_alignment_check_failed(self, error: Exception) -> None:
        self.logger.warning(f"⚠️  Note: Could not check funding alignment: {error}")

    def log_final_summary(self, results: "BacktestResults") -> None:
        summary = results.summary()

        self.logger.info("=" * 60)
        self.logger.info("BACKTEST COMPLETE")
        self.logger.info("=" * 60)
        self.logger.info("\nPERFORMANCE SUMMARY")
        self.logger.info("-" * 60)

        deep = summary["deep_hedge"]
        self.logger.info(f"\nDeep Hedge:")
        self.logger.info(f"  Mean PnL: ${deep['mean']:,.2f}")
        self.logger.info(f"  Std PnL: ${deep['std']:,.2f}")
        self.logger.info(f"  Sharpe Ratio: {deep['sharpe_ratio']:.3f}")
        self.logger.info(f"  Sortino Ratio: {deep['sortino_ratio']:.3f}")
        self.logger.info(f"  Max Drawdown: ${deep['max_drawdown']:.2f}")
        self.logger.info(f"  CVaR (95%): ${deep['cvar_95']:.2f}")
        self.logger.info(f"  Win Rate: {deep['win_rate']:.1%}")

        bs = summary["bs_baseline"]
        self.logger.info(f"\nBlack-Scholes Baseline:")
        self.logger.info(f"  Mean PnL: ${bs['mean']:,.2f}")
        self.logger.info(f"  Std PnL: ${bs['std']:,.2f}")
        self.logger.info(f"  Sharpe Ratio: {bs['sharpe_ratio']:.3f}")
        self.logger.info(f"  Sortino Ratio: {bs['sortino_ratio']:.3f}")
        self.logger.info(f"  Max Drawdown: ${bs['max_drawdown']:.2f}")
        self.logger.info(f"  CVaR (95%): ${bs['cvar_95']:.2f}")
        self.logger.info(f"  Win Rate: {bs['win_rate']:.1%}")

        mean_improvement = deep["mean"] - bs["mean"]
        sharpe_improvement = deep["sharpe_ratio"] - bs["sharpe_ratio"]
        self.logger.info(f"\nComparison (Deep Hedge vs BS):")
        self.logger.info(f"  Mean PnL improvement: ${mean_improvement:+,.2f}")
        self.logger.info(f"  Sharpe improvement: {sharpe_improvement:+.3f}")
        self.logger.info("=" * 60)

    def log_error(self, error: Exception, context: str) -> None:
        self.logger.info("=" * 60)
        self.logger.error(f"❌ BACKTEST FAILED: {context}")
        self.logger.info("=" * 60)
        self.logger.error(f"\nError type: {type(error).__name__}")
        self.logger.error(f"Error message: {error}")

        if isinstance(error, FileNotFoundError):
            self.logger.info("\nCommon causes:")
            self.logger.info("  - Model checkpoint path is incorrect")
            self.logger.info("  - Data directory doesn't exist or is empty")
            self.logger.info("  - Missing perpetual data files (*perpetual*.parquet)")
            self.logger.info("\nPlease check your configuration and file paths.")
        elif isinstance(error, ValueError):
            self.logger.info("\nCommon causes:")
            self.logger.info("  - Data directory is empty or has no matching files")
            self.logger.info("  - Date range doesn't overlap with available data")
            self.logger.info("  - Invalid configuration parameters")
            self.logger.info("  - Bootstrap path generation failed")
            self.logger.info("\nPlease check your data and configuration.")
        elif isinstance(error, (RuntimeError, KeyError)):
            self.logger.info("\nCommon causes:")
            self.logger.info("  - Model checkpoint is corrupted or incompatible")
            self.logger.info("  - Model architecture doesn't match checkpoint")
            self.logger.info("  - CUDA/device mismatch")
            self.logger.info("  - Tensor shape mismatch during computation")
            self.logger.info(
                "\nPlease check your model checkpoint and device settings."
            )
        else:
            self.logger.info("\nPlease check the full traceback above for details.")

        self.logger.info("=" * 60)
