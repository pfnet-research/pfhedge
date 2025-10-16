"""Results container for training framework."""

from typing import Dict, Optional, Any, List
import json
from datetime import datetime
from pathlib import Path


class TrainingResults:
    """Container for training results with summary statistics.

    Stores training history, test metrics, and model configuration.
    Provides methods to export data for analysis and reproducibility.

    Args:
        training_history: List of loss values per epoch
        test_metrics: Dictionary with test performance metrics
        model_config: Model and training configuration dictionary
        model_path: Path where model checkpoint was saved

    Examples:
        >>> results = TrainingResults(
        ...     training_history=history,
        ...     test_metrics=test_metrics,
        ...     model_config=config.to_dict(),
        ...     model_path="models/model.pth"
        ... )
        >>> summary = results.summary()
        >>> print(f"Final loss: {summary['final_loss']:.6f}")
    """

    def __init__(
        self,
        training_history: List[float],
        test_metrics: Dict[str, Any],
        model_config: Dict[str, Any],
        model_path: str,
    ):
        """Initialize results container.

        Args:
            training_history: List of loss values per epoch
            test_metrics: Test performance metrics
            model_config: Model and training configuration
            model_path: Path where model was saved
        """
        self.training_history = training_history
        self.test_metrics = test_metrics
        self.model_config = model_config
        self.model_path = model_path

        # Store creation timestamp
        self.created_at = datetime.now()

        # Derived properties
        self.n_epochs = len(training_history)

    def summary(self) -> Dict[str, Any]:
        """Get training and test summary.

        Returns:
            Dictionary with training and test metrics

        Examples:
            >>> summary = results.summary()
            >>> print(f"Final loss: {summary['final_loss']:.6f}")
            >>> print(f"Test Sharpe: {summary['test_metrics']['deep_hedge']['sharpe_ratio']:.3f}")
        """
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
        """Export all data to dictionary.

        Args:
            include_raw: If True (default), includes full training history.
                         If False, includes only summary statistics.
                         Set to False for large training runs to reduce size.

        Returns:
            Dictionary containing:
            - 'summary': Summary statistics
            - 'model_config': Model and training configuration
            - 'training_history': Full loss history (if include_raw=True)
            - 'test_metrics': Test performance metrics
            - 'model_path': Path to saved model
            - 'created_at': Timestamp

        Examples:
            >>> # Full export with training history
            >>> data = results.to_dict()
            >>>
            >>> # Lightweight export without full history
            >>> summary_only = results.to_dict(include_raw=False)
            >>>
            >>> # Save to JSON
            >>> import json
            >>> with open('results.json', 'w') as f:
            ...     json.dump(data, f)
        """
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
        """Recursively normalize data for JSON serialization.

        Handles common non-JSON types:
        - datetime objects -> ISO format strings
        - Path objects -> strings
        - Recursively processes dicts and lists

        Args:
            obj: Object to normalize

        Returns:
            JSON-serializable version of obj
        """
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
        """Export results to JSON format.

        Convenience method that handles non-JSON types and optionally
        writes to file.

        Args:
            filepath: Optional path to write JSON file. If None, returns JSON string.
            include_raw: If True, includes full training history. If False, only
                         summary and test metrics (default False for smaller size).
            **kwargs: Additional arguments passed to json.dumps() (e.g., indent=2)

        Returns:
            JSON string if filepath is None, otherwise None (writes to file)

        Examples:
            >>> # Get JSON string without full history
            >>> json_str = results.to_json(include_raw=False)
            >>>
            >>> # Write to file with pretty formatting
            >>> results.to_json('results.json', include_raw=True, indent=2)
            >>>
            >>> # Lightweight export (summary only)
            >>> results.to_json('summary.json', include_raw=False, indent=2)
        """
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
        """String representation."""
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
