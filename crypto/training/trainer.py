from typing import Optional, TYPE_CHECKING, List, Dict, Any
import os
import torch
from torch import Tensor

from .config import TrainingConfig
from torch.optim import Adam, AdamW, SGD

if TYPE_CHECKING:
    from pfhedge.nn import Hedger
    from crypto.instruments import BitcoinEuropeanOption


def _validate_cuda_available(device: str):
    if "cuda" in device.lower() and not torch.cuda.is_available():
        raise ValueError(
            f"CUDA device requested (device='{device}'), but CUDA is not available.\n"
            f"Suggestions: Use device='cpu' or install CUDA support"
        )


def _create_optimizer(config: TrainingConfig, model_parameters):
    optimizer_class = {"adam": Adam, "adamw": AdamW, "sgd": SGD}[config.optimizer]

    kwargs = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }

    if config.optimizer == "sgd":
        kwargs["momentum"] = 0.9

    return optimizer_class(model_parameters, **kwargs)


def _move_option_to_device(option, target_device, verbose=False):
    # CRITICAL: Move underlier to target device BEFORE any simulation
    # This ensures data generation happens directly on GPU, avoiding CPU bottleneck
    current_device = getattr(option.underlier, "device", torch.device("cpu"))

    if current_device != target_device:
        if verbose:
            print(f"   Moving underlier from {current_device} to {target_device}...")
        option.underlier.to(target_device)

        # Also ensure the option itself is on the correct device
        if hasattr(option, "to"):
            option.to(target_device)


def _check_dtype_consistency(option, model_dtype, verbose=False):
    if not hasattr(option.underlier, "spot"):
        return

    option_dtype = option.underlier.spot.dtype
    if option_dtype != model_dtype and verbose:
        print(
            f"   ⚠️  Warning: Option dtype ({option_dtype}) differs from model dtype ({model_dtype})"
        )


def _print_optimizer_info(config: TrainingConfig, verbose=False):
    if not verbose:
        return

    print(f"\n🔧 Optimizer: {config.optimizer.upper()}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Weight decay: {config.weight_decay}")
    if config.early_stopping:
        print(
            f"   Early stopping: enabled (patience={config.patience}, min_delta={config.min_delta})"
        )


def _print_training_summary(history: List[float], verbose=False):
    if not verbose or len(history) == 0:
        return

    print(f"\n✅ Training complete!")
    print(f"  Initial loss: {history[0]:.6f}")
    print(f"  Final loss: {history[-1]:.6f}")
    improvement = (history[0] - history[-1]) / history[0] * 100
    print(f"  Improvement: {improvement:.1f}%")


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        verbose: bool = False,
        enable_diagnostics: bool = False,
    ):
        _validate_cuda_available(config.device)

        self.config = config
        self.verbose = verbose
        self.enable_diagnostics = enable_diagnostics
        self.diagnostics = None

        self.train_option = None
        self.test_option = None
        self.model = None

    def create_option(self, n_paths: int, seed: int) -> "BitcoinEuropeanOption":
        from crypto.instruments import BitcoinEuropeanOption
        import torch
        import numpy as np

        if self.config.underlying_type == "spot":
            from crypto.instruments import BitcoinSpotBrownian

            underlier_class = BitcoinSpotBrownian
        else:
            from crypto.instruments import BitcoinPerpetualBrownian

            underlier_class = BitcoinPerpetualBrownian

        # Set random seed if provided
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Determine the actual device to use
        device_str = self.config.device
        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(device_str)

        # Create underlier directly on the target device to avoid CPU bottleneck
        underlier = underlier_class(
            sigma=self.config.volatility,
            mu=self.config.drift,
            cost=self.config.transaction_cost,
            dt=self.config.dt,
            volatility_window=self.config.volatility_window,
            device=device,  # CRITICAL: Set device here to generate data directly on GPU
        )

        # Create option
        maturity_time = self.config.maturity_days / 365
        option = BitcoinEuropeanOption(
            underlier,
            strike=self.config.strike,
            maturity=maturity_time,
            call=self.config.call,
        )

        # Simulate underlying paths on the correct device
        underlier.simulate(n_paths=n_paths, time_horizon=maturity_time)

        return option

    def create_model(self) -> "Hedger":
        from crypto.strategies import create_deep_hedger
        import torch.nn as nn

        hedger = create_deep_hedger(
            model_type=self.config.model_type,
            n_layers=self.config.n_layers,
            n_units=self.config.n_units,
            risk_measure=self.config.risk_measure,
            risk_param=self.config.risk_param,
            features=self.config.features,
        )

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        hedger.to(device)

        # Apply better initialization for MLP output head
        if self.config.model_type == "mlp":
            self._initialize_model_head(hedger.model)

        if self.verbose:
            try:
                n_params = sum(
                    p.numel() for p in hedger.parameters() if p.is_meta is False
                )
                print(f"✅ Model created: {self.config.model_type.upper()}")
                print(f"   Device: {device}")
                print(f"   Parameters: {n_params:,}")
                print(f"   Risk measure: {self.config.risk_measure}")
                if self.config.grad_clip_norm:
                    print(f"   Gradient clipping: {self.config.grad_clip_norm}")
                if self.config.bs_warmup_epochs > 0:
                    print(f"   BS warmup: {self.config.bs_warmup_epochs} epochs")
            except (ValueError, RuntimeError):
                print(f"✅ Model created: {self.config.model_type.upper()}")
                print(f"   Device: {device}")
                print(f"   Parameters: (lazy, will be initialized on first forward)")
                print(f"   Risk measure: {self.config.risk_measure}")

        return hedger

    def _initialize_model_head(self, model):
        """Initialize MLP output head with Xavier initialization for better gradient flow."""
        import torch.nn as nn

        last_linear = None
        for module in model.modules():
            if isinstance(module, nn.Linear):
                last_linear = module

        if last_linear is not None:
            with torch.no_grad():
                nn.init.xavier_uniform_(last_linear.weight, gain=0.5)
                last_linear.bias.fill_(0.0)

    def train_model(
        self,
        model: Optional["Hedger"] = None,
        option: Optional["BitcoinEuropeanOption"] = None,
    ) -> List[float]:
        model = model or self.model
        option = option or self.train_option

        if model is None:
            raise ValueError("No model provided. Call create_model() first.")
        if option is None:
            raise ValueError("No training option provided. Call create_option() first.")

        if self.verbose:
            print(f"\nTraining model for {self.config.n_epochs} epochs...")
            print(f"  Training paths: {self.config.n_paths:,}")

        model_device = next(model.parameters()).device
        model_dtype = next(model.parameters()).dtype

        _move_option_to_device(option, model_device, self.verbose)
        _check_dtype_consistency(option, model_dtype, self.verbose)

        if self.enable_diagnostics:
            self._attach_diagnostics(model)

        optimizer = _create_optimizer(self.config, model.parameters())
        _print_optimizer_info(self.config, self.verbose)

        # Use curriculum learning if warmup epochs specified
        if self.config.bs_warmup_epochs > 0:
            history = self._train_with_curriculum(
                model, option, optimizer, model_device
            )
        elif self.config.early_stopping:
            history = self._train_with_early_stopping(
                model, option, optimizer, model_device
            )
        else:
            # Standard training with optional gradient clipping
            history = self._train_standard(model, option, optimizer, model_device)

        if self.enable_diagnostics and self.diagnostics is not None:
            self._print_diagnostics()

        _print_training_summary(history, self.verbose)

        return history

    def _train_standard(
        self, model: "Hedger", option: "BitcoinEuropeanOption", optimizer, device
    ) -> List[float]:
        """Standard training with optional gradient clipping."""
        if self.config.grad_clip_norm:
            # Custom training loop with gradient clipping
            return self._train_with_grad_clipping(model, option, optimizer, device)
        else:
            # Use PFHedge's built-in fit method
            return model.fit(
                option,
                n_paths=self.config.n_paths,
                n_epochs=self.config.n_epochs,
                optimizer=optimizer,
                verbose=self.verbose,
                use_amp=self.config.use_amp and device.type == "cuda",
                validation_freq=self.config.validation_freq,
            )

    def _compute_hedge(
        self, model: "Hedger", option: "BitcoinEuropeanOption"
    ) -> Tensor:
        """Compute hedge positions and reduce dimensions if needed."""
        hedge_positions = model.compute_hedge(option)
        if hedge_positions.dim() == 3:
            hedge_positions = hedge_positions.squeeze(1)
        return hedge_positions

    def _compute_loss(
        self, model: "Hedger", option: "BitcoinEuropeanOption", hedge_positions: Tensor
    ) -> Tensor:
        """Compute loss including CVaR and optional penalties."""
        from crypto.strategies import calculate_bs_hedge_pnl

        spots = option.underlier.spot
        payoffs = option.payoff()
        pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=hedge_positions,
            payoffs=payoffs,
            cost=self.config.transaction_cost,
        )

        # Normalize PnL by initial spot for stable loss scale
        initial_spot = spots[:, 0].mean()
        normalized_pnl = pnl[:, -1] / initial_spot

        loss = model.criterion(normalized_pnl)

        # Add penalty for constant positions
        if self.config.const_position_penalty > 0:
            time_changes = (
                (hedge_positions[:, 1:] - hedge_positions[:, :-1]).abs().mean()
            )
            const_penalty = self.config.const_position_penalty / (time_changes + 1e-6)
            loss = loss + const_penalty

        return loss

    def _backward_and_step(
        self, loss: Tensor, model: "Hedger", optimizer, scaler=None
    ) -> float:
        """Unified backward pass with optional AMP and gradient clipping.

        Returns:
            grad_norm: Gradient norm value (float)
        """
        import torch.nn as nn

        if scaler:
            # AMP backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if self.config.grad_clip_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.config.grad_clip_norm
                )
            else:
                grad_norm = sum(
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard backward pass
            loss.backward()
            if self.config.grad_clip_norm:
                grad_norm = nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.config.grad_clip_norm
                )
            else:
                grad_norm = sum(
                    p.grad.norm().item()
                    for p in model.parameters()
                    if p.grad is not None
                )
            optimizer.step()

        # Convert to float if tensor
        return grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

    def _log_epoch(
        self,
        epoch: int,
        loss: Tensor,
        grad_norm: float,
        hedge_positions: Tensor,
        extra_info: str = "",
    ):
        """Log training progress for an epoch."""
        if not self.verbose:
            return
        if epoch % 10 != 0 and epoch != self.config.n_epochs - 1:
            return

        with torch.no_grad():
            pos_mean = hedge_positions.mean().item()
            pos_std = hedge_positions.std().item()
            pos_min = hedge_positions.min().item()
            pos_max = hedge_positions.max().item()

        base_msg = (
            f"Epoch {epoch+1}/{self.config.n_epochs}{extra_info}: "
            f"Loss={loss.item():.6f}, GradNorm={grad_norm:.2f}, "
            f"Hedge: μ={pos_mean:.3f} σ={pos_std:.3f} range=[{pos_min:.3f}, {pos_max:.3f}]"
        )
        print(base_msg)

    def _train_with_grad_clipping(
        self, model: "Hedger", option: "BitcoinEuropeanOption", optimizer, device
    ) -> List[float]:
        """Training loop with gradient clipping."""
        history = []
        use_amp = self.config.use_amp and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        model.train()

        for epoch in range(self.config.n_epochs):
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    hedge_positions = self._compute_hedge(model, option)
                    loss = self._compute_loss(model, option, hedge_positions)
            else:
                hedge_positions = self._compute_hedge(model, option)
                loss = self._compute_loss(model, option, hedge_positions)

            # Backward pass (unified)
            grad_norm = self._backward_and_step(loss, model, optimizer, scaler)

            history.append(loss.item())
            self._log_epoch(epoch, loss, grad_norm, hedge_positions)

        return history

    def _compute_curriculum_loss(
        self,
        model: "Hedger",
        option: "BitcoinEuropeanOption",
        hedge_positions: Tensor,
        alpha: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute curriculum loss with MSE and CVaR components.

        Returns:
            tuple: (total_loss, mse_loss, cvar_loss)
        """
        from crypto.strategies import calculate_bs_hedge_pnl

        # Compute Black-Scholes delta for curriculum
        bs_delta = option.black_scholes_delta(option.underlier.spot)
        mse_loss = torch.mean((hedge_positions - bs_delta) ** 2)

        # Compute CVaR loss
        spots = option.underlier.spot
        payoffs = option.payoff()
        pnl = calculate_bs_hedge_pnl(
            spots=spots,
            bs_delta=hedge_positions,
            payoffs=payoffs,
            cost=self.config.transaction_cost,
        )

        initial_spot = spots[:, 0].mean()
        normalized_pnl = pnl[:, -1] / initial_spot
        cvar_loss = model.criterion(normalized_pnl)

        # Combined loss based on curriculum stage
        if alpha < 1.0:
            loss = (1.0 - alpha) * mse_loss + alpha * cvar_loss
        else:
            # After warmup, optionally keep BS anchor
            loss = cvar_loss + self.config.bs_anchor_weight * mse_loss

        # Add penalty for constant positions
        if self.config.const_position_penalty > 0:
            time_changes = (
                (hedge_positions[:, 1:] - hedge_positions[:, :-1]).abs().mean()
            )
            const_penalty = self.config.const_position_penalty / (time_changes + 1e-6)
            loss = loss + const_penalty

        return loss, mse_loss, cvar_loss

    def _train_with_curriculum(
        self, model: "Hedger", option: "BitcoinEuropeanOption", optimizer, device
    ) -> List[float]:
        """Training with BS-delta warmup curriculum."""
        history = []
        warmup = self.config.bs_warmup_epochs
        ramp = self.config.curriculum_ramp_epochs
        total_transition = warmup + ramp
        use_amp = self.config.use_amp and device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        model.train()

        if self.verbose:
            print(f"\n📚 Curriculum Learning:")
            print(f"  Warmup (BS imitation): epochs 1-{warmup}")
            if ramp > 0:
                print(f"  Transition: epochs {warmup+1}-{total_transition}")
            print(f"  Full training: epochs {total_transition+1}+")

        for epoch in range(self.config.n_epochs):
            optimizer.zero_grad(set_to_none=True)

            # Curriculum weight: 0 (pure BS) -> 1 (pure CVaR)
            if epoch < warmup:
                alpha = 0.0
            elif epoch < total_transition:
                alpha = (epoch - warmup) / ramp if ramp > 0 else 1.0
            else:
                alpha = 1.0

            # Forward pass with optional AMP
            if use_amp:
                with torch.cuda.amp.autocast():
                    hedge_positions = self._compute_hedge(model, option)
                    loss, mse_loss, cvar_loss = self._compute_curriculum_loss(
                        model, option, hedge_positions, alpha
                    )
            else:
                hedge_positions = self._compute_hedge(model, option)
                loss, mse_loss, cvar_loss = self._compute_curriculum_loss(
                    model, option, hedge_positions, alpha
                )

            # Backward pass (unified)
            grad_norm = self._backward_and_step(loss, model, optimizer, scaler)

            history.append(loss.item())

            # Custom logging for curriculum with phase info
            if self.verbose and (epoch % 10 == 0 or epoch == self.config.n_epochs - 1):
                phase = (
                    "Warmup"
                    if epoch < warmup
                    else ("Transition" if epoch < total_transition else "Full")
                )
                extra_info = f" [{phase}, α={alpha:.2f}]: Loss={loss.item():.6f} (MSE={mse_loss.item():.6f}, CVaR={cvar_loss.item():.6f})"

                with torch.no_grad():
                    pos_mean = hedge_positions.mean().item()
                    pos_std = hedge_positions.std().item()

                print(
                    f"Epoch {epoch+1}/{self.config.n_epochs}{extra_info}, "
                    f"GradNorm={grad_norm:.2f}, Hedge: μ={pos_mean:.3f} σ={pos_std:.3f}"
                )

        return history

    def _attach_diagnostics(self, model):
        from crypto.training.diagnostics import MLPDiagnostics

        self.diagnostics = MLPDiagnostics(model, sample_frequency=5)
        self.diagnostics.attach()
        if self.verbose:
            print("\n🔍 Diagnostics enabled - will track MLP inputs/outputs/gradients")

    def _print_diagnostics(self):
        print("\n" + "=" * 70)
        print("TRAINING DIAGNOSTICS")
        print("=" * 70)
        self.diagnostics.print_summary(verbose=self.verbose)
        self.diagnostics.detach()

    def _train_with_early_stopping(
        self, model: "Hedger", option: "BitcoinEuropeanOption", optimizer, device
    ) -> List[float]:
        history = []
        best_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.config.n_epochs):
            epoch_history = model.fit(
                option,
                n_paths=self.config.n_paths,
                n_epochs=1,
                optimizer=optimizer,
                verbose=False,
                use_amp=self.config.use_amp and device.type == "cuda",
                validation_freq=1,
            )

            loss = epoch_history[0] if epoch_history else float("inf")
            history.append(loss)

            if loss < best_loss - self.config.min_delta:
                best_loss = loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.config.n_epochs}: loss={loss:.6f} ⭐ (new best)"
                    )
            else:
                patience_counter += 1
                if self.verbose:
                    print(
                        f"Epoch {epoch+1}/{self.config.n_epochs}: loss={loss:.6f} (patience: {patience_counter}/{self.config.patience})"
                    )

            if patience_counter >= self.config.patience:
                if self.verbose:
                    print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
                    print(
                        f"   Best loss: {best_loss:.6f} (epoch {epoch+1-patience_counter})"
                    )

                if best_state is not None:
                    model.load_state_dict(
                        {k: v.to(device) for k, v in best_state.items()}
                    )
                    if self.verbose:
                        print(f"   Restored best model weights")
                break

        return history

    def evaluate(
        self,
        model: Optional["Hedger"] = None,
        option: Optional["BitcoinEuropeanOption"] = None,
    ) -> Dict[str, Any]:
        model = model or self.model
        option = option or self.test_option

        if model is None:
            raise ValueError("No model provided.")
        if option is None:
            raise ValueError("No test option provided.")

        if self.verbose:
            print(f"\nEvaluating on test set ({self.config.test_n_paths} paths)...")

        model_device = next(model.parameters()).device
        _move_option_to_device(option, model_device, self.verbose)

        from crypto.strategies import calculate_bs_hedge_pnl

        model.eval()
        with torch.no_grad():
            # Compute model hedge
            hedge_positions = model.compute_hedge(option)
            if hedge_positions.dim() == 3:
                hedge_positions = hedge_positions.squeeze(1)

            spots = option.underlier.spot
            payoffs = option.payoff()

            # Compute model PnL
            model_pnl = calculate_bs_hedge_pnl(
                spots=spots,
                bs_delta=hedge_positions,
                payoffs=payoffs,
                cost=self.config.transaction_cost,
                band_width=self.config.band_width,
            )

            # Compute BS baseline hedge
            bs_delta = option.black_scholes_delta(spots)

            # Compute BS baseline PnL
            bs_pnl = calculate_bs_hedge_pnl(
                spots=spots,
                bs_delta=bs_delta,
                payoffs=payoffs,
                cost=self.config.transaction_cost,
                band_width=self.config.band_width,
            )

        model_final_pnl = model_pnl[:, -1]
        bs_final_pnl = bs_pnl[:, -1]

        # Compute hedging effectiveness metrics
        model_std = model_final_pnl.std().item()
        bs_std = bs_final_pnl.std().item()
        variability_ratio = model_std / (bs_std + 1e-8)

        # Compute correlation between model hedge and BS delta
        # Flatten to (n_paths * n_steps) for correlation
        hedge_flat = hedge_positions.reshape(-1)
        bs_delta_flat = bs_delta.reshape(-1)

        hedge_mean = hedge_flat.mean()
        bs_mean = bs_delta_flat.mean()

        cov = ((hedge_flat - hedge_mean) * (bs_delta_flat - bs_mean)).mean()
        hedge_std_corr = hedge_flat.std()
        bs_std_corr = bs_delta_flat.std()
        correlation = cov / (hedge_std_corr * bs_std_corr + 1e-8)

        metrics = {
            "pnl_mean": model_final_pnl.mean().item(),
            "pnl_std": model_std,
            "pnl_min": model_final_pnl.min().item(),
            "pnl_max": model_final_pnl.max().item(),
            "sharpe": model_final_pnl.mean().item() / (model_std + 1e-8),
            "variability_ratio": variability_ratio,
            "bs_correlation": correlation.item(),
            "bs_pnl_std": bs_std,
        }

        if self.verbose:
            print(f"  Mean PnL: {metrics['pnl_mean']:.6f}")
            print(f"  Std PnL: {metrics['pnl_std']:.6f}")
            print(f"  Sharpe: {metrics['sharpe']:.6f}")
            print(
                f"  Variability Ratio: {metrics['variability_ratio']:.3f} ({'better' if variability_ratio < 1.0 else 'worse'} than BS)"
            )
            print(f"  BS Correlation: {metrics['bs_correlation']:.3f}")

        return metrics

    def save_checkpoint(
        self, model: Optional["Hedger"] = None, path: Optional[str] = None
    ):
        model = model or self.model
        path = path or self.config.model_path

        if model is None:
            raise ValueError("No model to save.")

        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

        model_config = {
            "model_type": self.config.model_type,
            "n_layers": self.config.n_layers,
            "n_units": self.config.n_units,
            "risk_param": self.config.risk_param,
            "criterion": self.config.risk_measure,
            "features": self.config.features
            or [
                "log_moneyness",
                "expiry_time",
                "volatility",
                "prev_hedge",
            ],
        }

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": self.config.to_dict(),
            "training_config": self.config.to_dict(),
            "model_config": model_config,
        }

        torch.save(checkpoint, path)

        if self.verbose:
            print(f"\n✅ Model saved to: {path}")

    def train(self, seed: Optional[int] = None):
        from .results import TrainingResults

        seed = seed if seed is not None else self.config.train_seed

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING DEEP HEDGING MODEL")
            print(f"{'='*60}")
            print(f"  Seed: {seed}")
            print(f"  Underlying: {self.config.underlying_type}")
            print(f"  Architecture: {self.config.model_type.upper()}")

        self.train_option = self.create_option(n_paths=self.config.n_paths, seed=seed)

        self.test_option = self.create_option(
            n_paths=self.config.test_n_paths, seed=self.config.test_seed
        )

        self.model = self.create_model()

        train_history = self.train_model()

        self.save_checkpoint()

        test_metrics = self.evaluate()

        results = TrainingResults(
            config=self.config,
            train_history=train_history,
            test_metrics=test_metrics,
            model=self.model,
        )

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING COMPLETE")
            print(f"{'='*60}\n")

        return results
