from typing import Optional, TYPE_CHECKING

import torch
from torch import Tensor

from .bootstrap_generator import BootstrapOptionGenerator
from .config import BacktestConfig
from .data_loader import BacktestDataLoader
from .logger import BacktestLogger
from .model_loader import BacktestModelLoader
from .strategy_executor import StrategyExecutor
from .validators import BacktestValidators

if TYPE_CHECKING:
    from pfhedge.nn import Hedger
    from crypto.data.loader import CryptoDataLoader
    from crypto.instruments import BitcoinEuropeanOption


class Backtester:
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.logger = BacktestLogger(verbosity="INFO")

        self.model = None
        self.data_loader = None
        self.option = None

        self.deep_positions = None
        self.bs_positions = None

        self.diagnostics = None

    def load_model(self, device: Optional[str] = None) -> "Hedger":
        if device is None:
            device = "cpu"

        # Load checkpoint once and cache it
        checkpoint = BacktestModelLoader._load_checkpoint_safely(
            self.config.model_path, device
        )
        model_config = BacktestModelLoader._extract_config(checkpoint)

        # Create model using cached checkpoint
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        self.model = create_deep_hedger(
            model_type=model_config.get("model_type", "mlp"),
            n_layers=model_config["n_layers"],
            n_units=model_config["n_units"],
            risk_measure=model_config["criterion"],
            risk_param=model_config["risk_param"],
            features=model_config["features"],
        )
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()

        self.logger.log_model_loaded(
            model_config=model_config, device=device, model_path=self.config.model_path
        )

        return self.model

    def load_data(self) -> "CryptoDataLoader":
        data_loader_helper = BacktestDataLoader(self.config)
        self.data_loader = data_loader_helper.load_and_prepare_data()

        frequency = (
            f"{int(self.config.dt_hours)}H"
            if self.config.dt_hours == int(self.config.dt_hours)
            else f"{int(round(self.config.dt_hours * 60))}T"
        )

        self.logger.log_data_summary(self.data_loader, frequency)

        return self.data_loader

    def create_bootstrap_option(
        self, data_loader: Optional["CryptoDataLoader"] = None
    ) -> "BitcoinEuropeanOption":
        if data_loader is None:
            data_loader = self.data_loader

        if data_loader is None:
            raise ValueError(
                "No data loaded. Call load_data() first or provide a data_loader."
            )

        generator = BootstrapOptionGenerator(self.config, data_loader)
        self.option = generator.create_option()

        self.logger.log_bootstrap_summary(self.option, self.config)

        return self.option

    def run_deep_hedge(self, option=None, model=None) -> Tensor:
        if option is None:
            option = self.option
        if model is None:
            model = self.model

        executor = StrategyExecutor(self.config)
        deep_pnl = executor.run_deep_hedge(option, model)

        self.deep_positions = executor.positions
        self.diagnostics = executor.diagnostics

        return deep_pnl

    def run_bs_baseline(self, option=None) -> Tensor:
        if option is None:
            option = self.option

        executor = StrategyExecutor(self.config)
        bs_pnl = executor.run_bs_baseline(option)

        self.bs_positions = executor.positions

        return bs_pnl

    def run(self, seed: Optional[int] = None):
        # Import BacktestResults
        import logging
        import random
        import numpy as np
        from .results import BacktestResults

        logger = logging.getLogger(__name__)

        # Set random seeds for reproducibility if requested
        if seed is not None:
            # Seed ALL random generators
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Set CUDA seeds for GPU reproducibility
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
                # Enable deterministic mode for cuDNN
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                logger.info(f"🔒 Seed={seed} (random, numpy, torch, CUDA)")
            else:
                logger.info(f"🔒 Seed={seed} (random, numpy, torch)")
        else:
            logger.warning("⚠️  No seed set - results may vary between runs")

        print("\n" + "=" * 60)
        print("STARTING BACKTEST")
        print("=" * 60)
        print(f"\nConfiguration:")
        print(f"  Date range: {self.config.start_date} to {self.config.end_date}")
        print(
            f"  Option: {'Call' if self.config.call else 'Put'} @ ${self.config.strike:,.2f}"
        )
        print(f"  Maturity: {self.config.maturity_days} days")
        print(f"  Bootstrap paths: {self.config.n_bootstrap_paths}")
        print(f"  Time step: {self.config.dt_hours} hours")
        print(f"  Transaction cost: {self.config.transaction_cost * 100:.3f}%")
        print(f"  Model: {self.config.model_path}")
        print(f"  Data directory: {self.config.data_dir}")
        print("=" * 60 + "\n")

        # Wrap execution in try-except to provide helpful error messages
        try:
            # Step 1: Load model
            print("[Step 1/5] Loading model...")
            self.load_model()

            # Step 2: Load data
            print("[Step 2/5] Loading data...")
            self.load_data()

            # Step 3: Create bootstrap option
            print("[Step 3/5] Creating bootstrap option...")
            self.create_bootstrap_option()

            # Check funding alignment (warning only, doesn't stop execution)
            misaligned = BacktestValidators.check_funding_alignment(
                self.option, self.config
            )
            if misaligned:
                self.logger.log_funding_alignment_warning(
                    len(misaligned), misaligned, self.config.dt
                )

            # Step 4: Run deep hedge strategy
            print("[Step 4/5] Running deep hedge strategy...")
            deep_pnl = self.run_deep_hedge()

            # Step 5: Run BS baseline strategy
            print("[Step 5/5] Running BS baseline strategy...")
            bs_pnl = self.run_bs_baseline()

            # Create results object
            print("\nCreating results object...")
            results = BacktestResults(
                deep_pnl=deep_pnl,
                bs_pnl=bs_pnl,
                deep_positions=self.deep_positions,
                bs_positions=self.bs_positions,
                spots=self.option.underlier.spot,
                config=self.config,
            )

        except FileNotFoundError as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: File Not Found")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nCommon causes:")
            print("  - Model checkpoint path is incorrect")
            print("  - Data directory doesn't exist or is empty")
            print("  - Missing perpetual data files (*perpetual*.parquet)")
            print("\nPlease check your configuration and file paths.")
            print("=" * 60 + "\n")
            raise

        except ValueError as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: Invalid Data or Configuration")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nCommon causes:")
            print("  - Data directory is empty or has no matching files")
            print("  - Date range doesn't overlap with available data")
            print("  - Invalid configuration parameters")
            print("  - Bootstrap path generation failed")
            print("\nPlease check your data and configuration.")
            print("=" * 60 + "\n")
            raise

        except (RuntimeError, KeyError) as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: Model or Execution Error")
            print("=" * 60)
            print(f"\nError: {e}")
            print("\nCommon causes:")
            print("  - Model checkpoint is corrupted or incompatible")
            print("  - Model architecture doesn't match checkpoint")
            print("  - CUDA/device mismatch")
            print("  - Tensor shape mismatch during computation")
            print("\nPlease check your model checkpoint and device settings.")
            print("=" * 60 + "\n")
            raise

        except Exception as e:
            print("\n" + "=" * 60)
            print("❌ BACKTEST FAILED: Unexpected Error")
            print("=" * 60)
            print(f"\nError type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nPlease check the full traceback above for details.")
            print("=" * 60 + "\n")
            raise

        # Print final summary
        print("\n" + "=" * 60)
        print("BACKTEST COMPLETE")
        print("=" * 60)

        summary = results.summary()

        print("\nPERFORMANCE SUMMARY")
        print("-" * 60)

        # Deep Hedge metrics
        deep = summary["deep_hedge"]
        print(f"\nDeep Hedge:")
        print(f"  Mean PnL: ${deep['mean']:,.2f}")
        print(f"  Std PnL: ${deep['std']:,.2f}")
        print(f"  Sharpe Ratio: {deep['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {deep['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: ${deep['max_drawdown']:.2f}")
        print(f"  CVaR (95%): ${deep['cvar_95']:.2f}")
        print(f"  Win Rate: {deep['win_rate']:.1%}")

        # BS Baseline metrics
        bs = summary["bs_baseline"]
        print(f"\nBlack-Scholes Baseline:")
        print(f"  Mean PnL: ${bs['mean']:,.2f}")
        print(f"  Std PnL: ${bs['std']:,.2f}")
        print(f"  Sharpe Ratio: {bs['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {bs['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: ${bs['max_drawdown']:.2f}")
        print(f"  CVaR (95%): ${bs['cvar_95']:.2f}")
        print(f"  Win Rate: {bs['win_rate']:.1%}")

        # Comparison
        print(f"\nComparison (Deep Hedge vs BS):")
        mean_improvement = deep["mean"] - bs["mean"]
        sharpe_improvement = deep["sharpe_ratio"] - bs["sharpe_ratio"]
        print(f"  Mean PnL improvement: ${mean_improvement:+,.2f}")
        print(f"  Sharpe improvement: {sharpe_improvement:+.3f}")

        print("=" * 60 + "\n")

        return results

    def __repr__(self) -> str:
        return f"Backtester(config={self.config})"
