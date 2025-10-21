"""Training framework for deep hedging strategies."""

from typing import Optional, TYPE_CHECKING, List, Dict, Any
import os
import torch
from torch import Tensor

from .config import TrainingConfig

if TYPE_CHECKING:
    from pfhedge.nn import Hedger
    from crypto.instruments import BitcoinEuropeanOption


class Trainer:
    """Train deep hedging models with comprehensive experiment tracking.

    This class orchestrates the training process:
    1. Create training option with simulated paths
    2. Create test option for evaluation
    3. Create and configure the model
    4. Train the model
    5. Evaluate on test set
    6. Save model checkpoint

    Args:
        config: Training configuration

    Examples:
        >>> from crypto.training import TrainingConfig, Trainer
        >>> config = TrainingConfig(
        ...     strike=50000,
        ...     maturity_days=14,
        ...     volatility=0.8,
        ...     n_epochs=80
        ... )
        >>> trainer = Trainer(config)
        >>> results = trainer.train(seed=42)
        >>> print(results.summary())
    """

    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration.

        Args:
            config: Training configuration
        """
        self.config = config

        # Placeholders for created components
        self.train_option = None
        self.test_option = None
        self.model = None

    def create_training_option(self) -> "BitcoinEuropeanOption":
        """Create option for training with simulated Brownian paths.

        Returns:
            BitcoinEuropeanOption with synthetic paths for training

        Examples:
            >>> trainer = Trainer(config)
            >>> option = trainer.create_training_option()
            >>> print(option.summary())
        """
        from crypto.instruments import create_bitcoin_option_from_config

        print("Creating training option with simulated paths...")

        # Create config dict for option creation
        option_config = {
            "strike": self.config.strike,
            "maturity_days": self.config.maturity_days,
            "call": self.config.call,
            "cost": 0.0,  # No option transaction cost, only underlier cost
            "sigma": self.config.volatility,
            "mu": self.config.drift,
            "underlier_cost": self.config.transaction_cost,
            "dt": self.config.dt,
            "n_paths": self.config.n_paths,
            "seed": self.config.train_seed,
        }

        option, _ = create_bitcoin_option_from_config(option_config)

        print(f"✅ Training option created")
        print(f"   Type: {'Call' if self.config.call else 'Put'}")
        print(f"   Strike: ${self.config.strike:,.2f}")
        print(f"   Maturity: {self.config.maturity_days} days")
        print(f"   Paths: {self.config.n_paths:,}")
        print(f"   Volatility: {self.config.volatility:.1%}")

        # Store for later use
        self.train_option = option

        return option

    def create_test_option(self) -> "BitcoinEuropeanOption":
        """Create option for testing with different seed.

        Returns:
            BitcoinEuropeanOption with synthetic paths for testing

        Examples:
            >>> trainer = Trainer(config)
            >>> test_option = trainer.create_test_option()
            >>> print(test_option.summary())
        """
        from crypto.instruments import create_bitcoin_option_from_config

        print("\nCreating test option with different seed...")

        # Create config dict for test option (same params, different seed and paths)
        option_config = {
            "strike": self.config.strike,
            "maturity_days": self.config.maturity_days,
            "call": self.config.call,
            "cost": 0.0,
            "sigma": self.config.volatility,
            "mu": self.config.drift,
            "underlier_cost": self.config.transaction_cost,
            "dt": self.config.dt,
            "n_paths": self.config.test_n_paths,
            "seed": self.config.test_seed,
        }

        option, _ = create_bitcoin_option_from_config(option_config)

        print(f"✅ Test option created")
        print(f"   Paths: {self.config.test_n_paths:,}")
        print(f"   Seed: {self.config.test_seed}")

        # Store for later use
        self.test_option = option

        return option

    def create_model(self) -> "Hedger":
        """Create deep hedger model with configured architecture.

        Returns:
            Hedger model ready for training

        Examples:
            >>> trainer = Trainer(config)
            >>> model = trainer.create_model()
            >>> print(model)
        """
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        print("\nCreating deep hedger model...")

        # Create model with configured architecture
        model = create_deep_hedger(
            n_layers=self.config.n_layers,
            n_units=self.config.n_units,
            risk_measure=self.config.risk_measure,
            risk_param=self.config.risk_param,
        )

        # Move model to configured device
        device = self.config.device
        model = model.to(device)

        print(f"✅ Model created")
        print(
            f"   Architecture: {self.config.n_layers} layers × {self.config.n_units} units"
        )
        print(
            f"   Risk measure: {self.config.risk_measure} (param={self.config.risk_param})"
        )
        print(f"   Device: {device}")

        # Store for later use
        self.model = model

        return model

    def train_model(
        self,
        model: Optional["Hedger"] = None,
        option: Optional["BitcoinEuropeanOption"] = None,
    ) -> List[float]:
        """Train the model on the training option.

        Args:
            model: Hedger model to train. If None, uses self.model.
            option: Option to train on. If None, uses self.train_option.

        Returns:
            Training history (list of loss values per epoch)

        Raises:
            ValueError: If model or option is None and not previously created

        Examples:
            >>> trainer = Trainer(config)
            >>> trainer.create_model()
            >>> trainer.create_training_option()
            >>> history = trainer.train_model()
            >>> print(f"Final loss: {history[-1]:.6f}")
        """
        # Use provided model/option or fall back to stored ones
        if model is None:
            model = self.model
        if option is None:
            option = self.train_option

        if model is None:
            raise ValueError(
                "No model provided. Call create_model() first or provide a model."
            )
        if option is None:
            raise ValueError(
                "No training option provided. Call create_training_option() first or provide an option."
            )

        print(f"\nTraining model for {self.config.n_epochs} epochs...")
        print(f"  Training paths: {self.config.n_paths:,}")
        print(f"  This may take a few minutes...\n")

        # Ensure option is on same device as model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        if hasattr(option.underlier, "spot"):
            option_device = option.underlier.spot.device
            option_dtype = option.underlier.spot.dtype

            # Move to correct device if needed
            if option_device != model_device:
                print(f"   Moving option from {option_device} to {model_device}...")
                option.underlier.to(model_device)

            # Verify dtype consistency (warn if mismatch)
            if option_dtype != model_dtype:
                print(
                    f"   ⚠️  Warning: Option dtype ({option_dtype}) differs from model dtype ({model_dtype})"
                )
                print(
                    f"   This may cause mixed precision issues. Consider converting option to {model_dtype}."
                )

        # Train the model
        history = model.fit(
            option,
            n_paths=self.config.n_paths,
            n_epochs=self.config.n_epochs,
            verbose=True,
        )

        print(f"\n✅ Training complete!")
        if len(history) > 0:
            print(f"  Initial loss: {history[0]:.6f}")
            print(f"  Final loss: {history[-1]:.6f}")
            improvement = (history[0] - history[-1]) / history[0] * 100
            print(f"  Improvement: {improvement:.1f}%")

        return history

    def evaluate(
        self,
        model: Optional["Hedger"] = None,
        option: Optional["BitcoinEuropeanOption"] = None,
    ) -> Dict[str, Any]:
        """Evaluate model on test set.

        Args:
            model: Trained model to evaluate. If None, uses self.model.
            option: Test option to evaluate on. If None, uses self.test_option.

        Returns:
            Dictionary with test metrics (mean_pnl, std_pnl, sharpe_ratio, etc.)

        Raises:
            ValueError: If model or option is None and not previously created

        Examples:
            >>> trainer = Trainer(config)
            >>> # ... train model ...
            >>> test_metrics = trainer.evaluate()
            >>> print(f"Test Sharpe: {test_metrics['sharpe_ratio']:.3f}")
        """
        from crypto.strategies.deep_hedge_utils import (
            calculate_bs_hedge_pnl,
            compute_funding_cum_cost,
            compare_hedge_performance,
        )

        # Use provided model/option or fall back to stored ones
        if model is None:
            model = self.model
        if option is None:
            option = self.test_option

        if model is None:
            raise ValueError("No model provided. Train model first or provide a model.")
        if option is None:
            raise ValueError(
                "No test option provided. Call create_test_option() first or provide an option."
            )

        print("\nEvaluating model on test set...")

        # Set model to eval mode
        model.eval()

        # Ensure option is on same device and dtype as model
        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        if hasattr(option.underlier, "spot"):
            option_device = option.underlier.spot.device
            option_dtype = option.underlier.spot.dtype

            # Move to correct device if needed
            if option_device != model_device:
                print(
                    f"   Moving test option from {option_device} to {model_device}..."
                )
                option.underlier.to(model_device)

            # Verify dtype consistency (warn if mismatch)
            if option_dtype != model_dtype:
                print(
                    f"   ⚠️  Warning: Test option dtype ({option_dtype}) differs from model dtype ({model_dtype})"
                )
                print(f"   This may cause mixed precision issues during evaluation.")

        with torch.no_grad():
            # Compute deep hedging strategy
            deep_positions = model.compute_hedge(option).squeeze(1)
            deep_pnl = model.compute_cum_pl(option)

            # Get necessary tensors
            spots = option.underlier.spot

            # Add funding costs if applicable
            if hasattr(option.underlier, "funding_rate") and hasattr(
                option.underlier, "funding_payment_times"
            ):
                funding_rate = option.underlier.funding_rate
                funding_times = option.underlier.funding_payment_times()

                funding_costs = compute_funding_cum_cost(
                    spots=spots,
                    positions=deep_positions,
                    funding_rate=funding_rate,
                    funding_times=funding_times,
                )
                deep_pnl = deep_pnl - funding_costs

            # Compute Black-Scholes baseline
            bs_delta = option.black_scholes_delta()
            payoffs = option.payoff()
            cost = option.underlier.cost

            funding_rate = None
            funding_times = None
            if hasattr(option.underlier, "funding_rate"):
                funding_rate = option.underlier.funding_rate
                funding_times = option.underlier.funding_payment_times()

            bs_pnl = calculate_bs_hedge_pnl(
                spots=spots,
                bs_delta=bs_delta,
                payoffs=payoffs,
                cost=cost,
                funding_rate=funding_rate,
                funding_times=funding_times,
            )

        # Compare performance
        results = compare_hedge_performance(deep_pnl, bs_pnl)

        print(f"✅ Evaluation complete")
        print(f"   Test paths: {deep_pnl.shape[0]:,}")
        print(
            f"   Deep hedge PnL: ${deep_pnl[:, -1].mean().item():.2f} ± ${deep_pnl[:, -1].std().item():.2f}"
        )
        print(
            f"   BS baseline PnL: ${bs_pnl[:, -1].mean().item():.2f} ± ${bs_pnl[:, -1].std().item():.2f}"
        )

        # Extract metrics from comparison results
        # compare_hedge_performance returns: {"Deep Hedge": {...}, "Black-Scholes": {...}}
        deep_results = results["Deep Hedge"]
        bs_results = results["Black-Scholes"]

        # Store evaluation results
        test_metrics = {
            "deep_hedge": {
                "mean_pnl": deep_results["mean"],
                "std_pnl": deep_results["std"],
                "sharpe_ratio": deep_results["sharpe"],
                "sortino_ratio": None,  # Not computed by compare_hedge_performance
            },
            "bs_baseline": {
                "mean_pnl": bs_results["mean"],
                "std_pnl": bs_results["std"],
                "sharpe_ratio": bs_results["sharpe"],
                "sortino_ratio": None,  # Not computed by compare_hedge_performance
            },
            "comparison": {
                "sharpe_improvement": deep_results["sharpe"] - bs_results["sharpe"],
                "mean_pnl_improvement": deep_results["mean"] - bs_results["mean"],
            },
        }

        return test_metrics

    def save_model(
        self,
        model: Optional["Hedger"] = None,
        history: Optional[List[float]] = None,
    ) -> str:
        """Save trained model checkpoint.

        Args:
            model: Trained model to save. If None, uses self.model.
            history: Training history to save. If None, uses empty list.

        Returns:
            Path where model was saved

        Raises:
            ValueError: If model is None and not previously created

        Examples:
            >>> trainer = Trainer(config)
            >>> # ... train model ...
            >>> model_path = trainer.save_model()
            >>> print(f"Model saved to: {model_path}")
        """
        # Use provided model or fall back to stored one
        if model is None:
            model = self.model

        if model is None:
            raise ValueError("No model provided. Train model first or provide a model.")

        if history is None:
            history = []

        print("\nSaving model checkpoint...")

        # Create model directory if it doesn't exist
        model_dir = os.path.dirname(self.config.model_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)

        # Get features from model (use DEFAULT_FEATURES if not available)
        from crypto.strategies.deep_hedge_utils import DEFAULT_FEATURES

        if hasattr(model, "features"):
            features = model.features
        else:
            features = DEFAULT_FEATURES

        # Create checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "n_layers": self.config.n_layers,
                "n_units": self.config.n_units,
                "in_features": 4,  # log_moneyness, expiry_time, volatility, prev_hedge
                "out_features": 1,  # hedge ratio
                "risk_param": self.config.risk_param,
                "risk_measure": self.config.risk_measure,
                "features": features,  # Include features for reproducibility
            },
            "training_config": {
                "strike": self.config.strike,
                "maturity_days": self.config.maturity_days,
                "volatility": self.config.volatility,
                "cost": self.config.transaction_cost,
                "dt": self.config.dt,
                "n_epochs": self.config.n_epochs,
                "train_seed": self.config.train_seed,
                "device": self.config.device,
            },
            "training_history": history,
        }

        # Save checkpoint
        torch.save(checkpoint, self.config.model_path)

        print(f"✅ Model checkpoint saved to: {self.config.model_path}")
        print(
            f"   Architecture: {self.config.n_layers} layers × {self.config.n_units} units"
        )
        print(f"   Training epochs: {self.config.n_epochs}")
        if len(history) > 0:
            print(f"   Final loss: {history[-1]:.6f}")

        return self.config.model_path

    def train(self, seed: Optional[int] = None):
        """Run full training pipeline.

        This orchestrates the entire training process:
        1. Set random seeds for reproducibility
        2. Create training option
        3. Create test option
        4. Create model
        5. Train model
        6. Evaluate on test set
        7. Save model checkpoint
        8. Return training results

        Args:
            seed: Random seed for reproducibility. If provided, overrides
                  config.train_seed. If None, uses config.train_seed.

        Returns:
            TrainingResults object with all results and metrics

        Raises:
            ValueError: If configuration is invalid

        Examples:
            >>> from crypto.training import TrainingConfig, Trainer
            >>> config = TrainingConfig(
            ...     strike=50000,
            ...     maturity_days=14,
            ...     n_epochs=80
            ... )
            >>> trainer = Trainer(config)
            >>> results = trainer.train(seed=42)
            >>> print(results.summary())
        """
        from .results import TrainingResults
        import numpy as np

        # Use provided seed or config seed
        if seed is None:
            seed = self.config.train_seed

        # Set random seeds for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Set CUDA seeds if using CUDA
        if torch.cuda.is_available() and "cuda" in self.config.device:
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU
            # Make cuDNN deterministic (may impact performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print("\n" + "=" * 70)
        print("DEEP HEDGING TRAINING")
        print("=" * 70)
        print(f"\nConfiguration:")
        print(
            f"  Option: {'Call' if self.config.call else 'Put'} @ ${self.config.strike:,.2f}"
        )
        print(f"  Maturity: {self.config.maturity_days} days")
        print(f"  Volatility: {self.config.volatility:.1%}")
        print(f"  Transaction cost: {self.config.transaction_cost:.2%}")
        print(f"  Training paths: {self.config.n_paths:,}")
        print(f"  Training epochs: {self.config.n_epochs}")
        print(f"  Test paths: {self.config.test_n_paths:,}")
        print(f"  Random seed: {seed}")
        print(f"  Model: {self.config.n_layers}×{self.config.n_units} units")
        print(f"  Device: {self.config.device}")
        print("=" * 70 + "\n")

        try:
            # Step 1: Create training option
            print("[Step 1/6] Creating training option...")
            self.create_training_option()

            # Step 2: Create test option
            print("\n[Step 2/6] Creating test option...")
            self.create_test_option()

            # Step 3: Create model
            print("\n[Step 3/6] Creating model...")
            self.create_model()

            # Step 4: Train model
            print("\n[Step 4/6] Training model...")
            history = self.train_model()

            # Step 5: Evaluate on test set
            print("\n[Step 5/6] Evaluating on test set...")
            test_metrics = self.evaluate()

            # Step 6: Save model
            print("\n[Step 6/6] Saving model...")
            model_path = self.save_model(history=history)

            # Create results object
            results = TrainingResults(
                training_history=history,
                test_metrics=test_metrics,
                model_config=self.config.to_dict(),
                model_path=model_path,
            )

            # Print summary
            print("\n" + "=" * 70)
            print("TRAINING COMPLETE")
            print("=" * 70)
            print(f"\nTraining Summary:")
            print(f"  Initial loss: {history[0]:.6f}")
            print(f"  Final loss: {history[-1]:.6f}")
            improvement = (history[0] - history[-1]) / history[0] * 100
            print(f"  Improvement: {improvement:.1f}%")

            print(f"\nTest Performance:")
            deep = test_metrics["deep_hedge"]
            bs = test_metrics["bs_baseline"]
            print(
                f"  Deep hedge: ${deep['mean_pnl']:.2f} ± ${deep['std_pnl']:.2f} (Sharpe: {deep['sharpe_ratio']:.3f})"
            )
            print(
                f"  BS baseline: ${bs['mean_pnl']:.2f} ± ${bs['std_pnl']:.2f} (Sharpe: {bs['sharpe_ratio']:.3f})"
            )
            print(
                f"  Improvement: ${test_metrics['comparison']['mean_pnl_improvement']:+.2f} (Sharpe: {test_metrics['comparison']['sharpe_improvement']:+.3f})"
            )

            print(f"\nModel saved to: {model_path}")
            print("=" * 70 + "\n")

            return results

        except Exception as e:
            print("\n" + "=" * 70)
            print("❌ TRAINING FAILED")
            print("=" * 70)
            print(f"\nError type: {type(e).__name__}")
            print(f"Error message: {e}")
            print("\nPlease check the configuration and error details.")
            print("=" * 70 + "\n")
            raise

    def __repr__(self) -> str:
        """String representation."""
        return f"Trainer(config={self.config})"
