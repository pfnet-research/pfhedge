"""Model checkpoint loading utilities for backtesting."""

import os
import pickle
import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from pfhedge.nn import Hedger


class BacktestModelLoader:
    """Handles model checkpoint loading with backward compatibility."""

    @staticmethod
    def load_from_checkpoint(model_path: str, device: str = "cpu") -> "Hedger":
        """Load model from checkpoint with safety fallbacks.

        Args:
            model_path: Path to model checkpoint file
            device: Target device ('cpu', 'cuda', etc.)

        Returns:
            Loaded Hedger model in evaluation mode

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            KeyError: If checkpoint is missing required keys
            RuntimeError: If state dict doesn't match model architecture
        """
        from crypto.strategies.deep_hedge_utils import create_deep_hedger

        logger = logging.getLogger(__name__)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        logger.info(f"Loading model from {model_path}...")

        checkpoint = BacktestModelLoader._load_checkpoint_safely(model_path, device)

        BacktestModelLoader._validate_checkpoint(checkpoint)

        model_config = BacktestModelLoader._extract_config(checkpoint)

        model = create_deep_hedger(
            model_type=model_config.get("model_type", "mlp"),
            n_layers=model_config["n_layers"],
            n_units=model_config["n_units"],
            risk_measure=model_config["criterion"],
            risk_param=model_config["risk_param"],
            features=model_config["features"],
        )

        try:
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        except RuntimeError as e:
            raise RuntimeError(
                f"Failed to load model weights. State dict mismatch: {e}\n"
                f"This usually means the checkpoint was saved with a different model architecture."
            )

        model.eval()

        return model

    @staticmethod
    def _load_checkpoint_safely(model_path: str, device: str) -> dict:
        """Try weights_only first, fallback to unsafe loading.

        Args:
            model_path: Path to checkpoint
            device: Target device

        Returns:
            Loaded checkpoint dictionary
        """
        try:
            return torch.load(model_path, map_location=device, weights_only=True)
        except (TypeError, RuntimeError, pickle.UnpicklingError):
            return torch.load(model_path, map_location=device)

    @staticmethod
    def _validate_checkpoint(checkpoint: dict) -> None:
        """Validate checkpoint has required structure.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Raises:
            KeyError: If required keys are missing
        """
        if "model_state_dict" not in checkpoint:
            raise KeyError("Checkpoint missing 'model_state_dict'")
        if "model_config" not in checkpoint:
            raise KeyError("Checkpoint missing 'model_config'")

        model_config = checkpoint["model_config"]

        required_keys = ["n_layers", "n_units", "risk_param"]
        for key in required_keys:
            if key not in model_config:
                raise KeyError(f"Model config missing required key: '{key}'")

        if "criterion" not in model_config and "risk_measure" not in model_config:
            raise KeyError("Model config missing 'criterion' or 'risk_measure'")

        if "features" not in model_config:
            raise KeyError(
                "Checkpoint missing 'features' in model_config. "
                "This model was likely trained before the feature bug fix. "
                "Please retrain so features are saved in the checkpoint."
            )

    @staticmethod
    def _extract_config(checkpoint: dict) -> dict:
        """Extract and normalize model config from checkpoint.

        Args:
            checkpoint: Loaded checkpoint dictionary

        Returns:
            Normalized model config dictionary with keys:
                - n_layers, n_units, risk_param, criterion, features
        """
        model_config = checkpoint["model_config"]

        if "criterion" in model_config:
            criterion = model_config["criterion"]
        else:
            criterion = model_config["risk_measure"]

        raw_features = model_config["features"]
        if isinstance(raw_features, (list, tuple)):
            features = [str(f) for f in raw_features]
        else:
            features = [str(raw_features)]

        return {
            "model_type": model_config.get("model_type", "mlp"),
            "n_layers": model_config["n_layers"],
            "n_units": model_config["n_units"],
            "risk_param": model_config["risk_param"],
            "criterion": criterion,
            "features": features,
        }
