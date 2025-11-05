"""GRU-based model for deep hedging with transaction costs."""

import torch
import torch.nn as nn


class GatedRecurrentUnit(nn.Module):
    """GRU model for modeling temporal dependencies in hedging.

    This model uses GRU (Gated Recurrent Unit) layers to capture the sequential
    nature of hedging decisions. GRU is similar to LSTM but simpler and often
    faster to train, making it a good choice for deep hedging applications.

    The GRU processes features at each time step and outputs hedge ratios,
    maintaining hidden state across time steps to model the path-dependent
    nature of optimal hedging under transaction costs.

    Args:
        in_features (int): Size of each input sample (required).
        out_features (int, default=1): Size of each output sample (hedge ratio).
        hidden_size (int, default=64): Number of features in hidden state.
        num_layers (int, default=2): Number of stacked GRU layers.
        dropout (float, default=0.2): Dropout probability for regularization.
            Applied between GRU layers (not used if num_layers=1).
        bidirectional (bool, default=False): If True, becomes bidirectional GRU.
            Note: For hedging, typically should be False since we can't see future.

    Shape:
        - Input: :math:`(N, T, H_{in})` where
          :math:`N` is batch size (number of paths),
          :math:`T` is sequence length (time steps), and
          :math:`H_{in}` is number of input features.
        - Output: :math:`(N, T, H_{out})` where
          :math:`H_{out}` is number of output features (typically 1 for hedge ratio).

    Examples:
        >>> import torch
        >>> from crypto.strategies import GatedRecurrentUnit
        >>>
        >>> # Create GRU model with 4 input features
        >>> model = GatedRecurrentUnit(in_features=4, hidden_size=64, num_layers=2)
        >>> print(model)
        GatedRecurrentUnit(
          (gru): GRU(4, 64, num_layers=2, dropout=0.2, batch_first=True)
          (fc): Linear(in_features=64, out_features=1, bias=True)
        )
        >>>
        >>> # Forward pass with batch of 100 paths, 42 time steps, 4 features
        >>> x = torch.randn(100, 42, 4)
        >>> output = model(x)
        >>> output.shape
        torch.Size([100, 42, 1])

    Notes:
        - GRU is often preferred over LSTM when:
          * Training speed is important
          * Memory efficiency is important (fewer parameters)
          * Simpler architecture is desired
        - LSTM may be preferred when:
          * Longer-term dependencies need to be captured
          * More complex gating mechanisms are beneficial

    References:
        - Buehler, H., Gonon, L., Teichmann, J. and Wood, B., 2019.
          Deep hedging. Quantitative Finance, 19(8), pp.1271-1291.
        - Cho, K., Van Merriënboer, B., Gulcehre, C., Bahdanau, D., Bougares, F.,
          Schwenk, H. and Bengio, Y., 2014.
          Learning phrase representations using RNN encoder-decoder for statistical
          machine translation. arXiv preprint arXiv:1406.1078.
    """

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
        """Forward pass through GRU model.

        Args:
            x: Input tensor of shape (N, T, H_in) where
                N = batch size (number of paths)
                T = sequence length (time steps)
                H_in = number of input features

        Returns:
            Output tensor of shape (N, T, H_out) where
                H_out = number of output features (hedge ratios)
        """
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
        """String representation."""
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
