import torch
import torch.nn as nn
from typing import List


class EnhancedMLP(nn.Module):
    """MLP with layer normalization and dropout for better generalization.

    This is an enhanced version of pfhedge's MultiLayerPerceptron that adds:
    - Layer normalization (memory-efficient for time series)
    - Dropout for regularization
    - Better suited for deep networks with many layers

    Note: Uses LayerNorm instead of BatchNorm to avoid memory explosion
    with large batch sizes (n_paths * time_steps).
    """

    def __init__(
        self,
        n_layers: int,
        n_units: List[int],
        dropout: float = 0.1,
        use_layer_norm: bool = True,
    ):
        super().__init__()

        if isinstance(n_units, int):
            n_units = [n_units] * n_layers

        if len(n_units) != n_layers:
            raise ValueError(
                f"n_units must have {n_layers} elements, got {len(n_units)}"
            )

        self.n_layers = n_layers
        self.n_units = n_units
        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # Build layers
        layers = []

        # First layer is lazy (input size determined at runtime)
        layers.append(nn.LazyLinear(n_units[0]))
        if use_layer_norm:
            layers.append(nn.LayerNorm(n_units[0]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Hidden layers
        for i in range(1, n_layers):
            layers.append(nn.Linear(n_units[i - 1], n_units[i]))
            if use_layer_norm:
                layers.append(nn.LayerNorm(n_units[i]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Output layer (no activation, layer norm, or dropout)
        layers.append(nn.Linear(n_units[-1], 1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle time series input: (batch, time, features) -> (batch, features)
        if x.dim() == 3:
            batch_size, time_steps, n_features = x.shape
            # Flatten batch and time dimensions
            x = x.reshape(batch_size * time_steps, n_features)
            output = self.layers(x)
            # Reshape back
            output = output.reshape(batch_size, time_steps, 1)
        else:
            output = self.layers(x)

        return output

    def extra_repr(self) -> str:
        return (
            f"n_layers={self.n_layers}, "
            f"n_units={self.n_units}, "
            f"dropout={self.dropout}, "
            f"layer_norm={self.use_layer_norm}"
        )
