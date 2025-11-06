from typing import Dict, Optional, Any, List
import json
from datetime import datetime
from pathlib import Path


class TrainingResults:

    def __init__(
        self,
        config=None,
        train_history=None,
        test_metrics=None,
        model=None,
        training_history=None,
        model_config=None,
        model_path=None,
    ):
        if config is not None:
            self.training_history = train_history or []
            self.test_metrics = test_metrics or {}
            self.model_config = (
                config.to_dict() if hasattr(config, "to_dict") else config
            )
            self.model_path = config.model_path if hasattr(config, "model_path") else ""
            self.model = model
        else:
            self.training_history = training_history or []
            self.test_metrics = test_metrics or {}
            self.model_config = model_config or {}
            self.model_path = model_path or ""
            self.model = None

        self.created_at = datetime.now()
        self.n_epochs = len(self.training_history)

    def summary(self) -> Dict[str, Any]:
        if len(self.training_history) == 0:
            initial_loss = None
            final_loss = None
            improvement_pct = None
        else:
            initial_loss = self.training_history[0]
            final_loss = self.training_history[-1]
            if initial_loss != 0:
                improvement_pct = (initial_loss - final_loss) / initial_loss * 100
            else:
                improvement_pct = 0.0

        return {
            "n_epochs": self.n_epochs,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "improvement_pct": improvement_pct,
            "test_metrics": self.test_metrics,
            "model_path": self.model_path,
            "created_at": self.created_at.isoformat(),
        }

    def to_dict(self, include_raw: bool = True) -> Dict[str, Any]:
        data = {
            "summary": self.summary(),
            "model_config": self.model_config,
            "test_metrics": self.test_metrics,
            "model_path": self.model_path,
            "created_at": self.created_at.isoformat(),
        }

        # Add full training history if requested
        if include_raw:
            data["training_history"] = self.training_history

        return data

    @staticmethod
    def _safe_json_normalize(obj: Any) -> Any:
        if isinstance(obj, (datetime,)):
            return obj.isoformat()
        elif isinstance(obj, (Path,)):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: TrainingResults._safe_json_normalize(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [TrainingResults._safe_json_normalize(item) for item in obj]
        else:
            return obj

    def to_json(
        self, filepath: Optional[str] = None, include_raw: bool = False, **kwargs
    ) -> Optional[str]:
        # Get dictionary and normalize for JSON
        data = self.to_dict(include_raw=include_raw)
        normalized = self._safe_json_normalize(data)

        # Set default indent if not specified
        if "indent" not in kwargs:
            kwargs["indent"] = 2

        # Write to file or return string
        if filepath is not None:
            with open(filepath, "w") as f:
                json.dump(normalized, f, **kwargs)
            return None
        else:
            return json.dumps(normalized, **kwargs)

    def __repr__(self) -> str:
        summary = self.summary()
        final_loss = summary.get("final_loss")
        improvement = summary.get("improvement_pct")

        loss_str = (
            f"final_loss={final_loss:.6f}" if final_loss is not None else "no_training"
        )
        improvement_str = (
            f", improvement={improvement:.1f}%" if improvement is not None else ""
        )

        # Get test Sharpe if available
        test_sharpe_str = ""
        if self.test_metrics and "deep_hedge" in self.test_metrics:
            test_sharpe = self.test_metrics["deep_hedge"].get("sharpe_ratio")
            if test_sharpe is not None:
                test_sharpe_str = f", test_sharpe={test_sharpe:.3f}"

        return (
            f"TrainingResults(n_epochs={self.n_epochs}, "
            f"{loss_str}{improvement_str}{test_sharpe_str})"
        )
