import torch
import torch.nn as nn


class GatedRecurrentUnit(nn.Module):

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = False,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        # Create GRU layer
        self.gru = nn.GRU(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # Output layer
        gru_out_size = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(gru_out_size, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Handle 2D input (N, H_in) by adding time dimension
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (N, H_in) -> (N, 1, H_in)

        # GRU forward pass
        # gru_out shape: (N, T, hidden_size * num_directions)
        gru_out, _ = self.gru(x)

        # Apply fully connected layer to each time step
        # output shape: (N, T, out_features)
        output = self.fc(gru_out)

        return output

    def __repr__(self) -> str:
        return (
            f"GatedRecurrentUnit(\n"
            f"  in_features={self.in_features}, "
            f"  out_features={self.out_features}, "
            f"  hidden_size={self.hidden_size}, "
            f"  num_layers={self.num_layers}, "
            f"  dropout={self.dropout}, "
            f"  bidirectional={self.bidirectional}\n"
            f")"
        )
