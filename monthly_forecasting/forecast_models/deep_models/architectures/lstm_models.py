import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from ..utils.lightning_base import LightningForecastBase

logger = logging.getLogger(__name__)


class LSTMForecaster(nn.Module):
    """
    LSTM-based forecasting model for monthly discharge prediction.

    Supports various LSTM configurations including bidirectional, stacked,
    and multi-input architectures.
    """

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        now_dim: int = 0,
        lookback: int = 365,
        future_known_steps: int = 30,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        bidirectional: bool = False,
        use_attention: bool = False,
        use_nan_mask: bool = True,
        **kwargs,
    ):
        """
        Initialize LSTM forecaster.

        Args:
            past_dim: Dimension of past features
            future_dim: Dimension of future features
            static_dim: Dimension of static features
            now_dim: Dimension of current time step features
            lookback: Number of past time steps
            future_known_steps: Number of future time steps
            hidden_dim: Hidden layer dimension
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_dim: Output dimension
            bidirectional: Whether to use bidirectional LSTM
            use_attention: Whether to use attention mechanism
            use_nan_mask: Whether to use NaN masking
            **kwargs: Additional parameters
        """
        super().__init__()

        self.past_dim = past_dim
        self.future_dim = future_dim
        self.static_dim = static_dim
        self.now_dim = now_dim
        self.lookback = lookback
        self.future_known_steps = future_known_steps
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.use_nan_mask = use_nan_mask

        # LSTM for processing past sequence
        if past_dim > 0:
            lstm_input_dim = past_dim
            if use_nan_mask:
                lstm_input_dim += past_dim  # Add NaN mask dimension

            self.past_lstm = nn.LSTM(
                input_size=lstm_input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional,
            )

            lstm_output_dim = hidden_dim * (2 if bidirectional else 1)

            # Attention mechanism for LSTM outputs
            if use_attention:
                self.attention = AttentionLayer(lstm_output_dim)

        else:
            self.past_lstm = None
            lstm_output_dim = 0

        # LSTM for processing future sequence (if available)
        if future_dim > 0:
            self.future_lstm = nn.LSTM(
                input_size=future_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True,
            )
            future_output_dim = hidden_dim // 2
        else:
            self.future_lstm = None
            future_output_dim = 0

        # Process current features
        if now_dim > 0:
            self.now_fc = nn.Sequential(
                nn.Linear(now_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            now_output_dim = hidden_dim
        else:
            self.now_fc = None
            now_output_dim = 0

        # Process static features
        if static_dim > 0:
            self.static_fc = nn.Sequential(
                nn.Linear(static_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            static_output_dim = hidden_dim
        else:
            self.static_fc = None
            static_output_dim = 0

        # Compute final input dimension
        final_input_dim = (
            lstm_output_dim + future_output_dim + now_output_dim + static_output_dim
        )

        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        logger.info(
            f"LSTMForecaster initialized with final_input_dim={final_input_dim}"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of LSTM forecaster.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            Dictionary with predictions
        """
        batch_size = (
            batch["x_past"].size(0) if "x_past" in batch else batch["x_static"].size(0)
        )
        features = []

        # Process past sequence with LSTM
        if self.past_lstm is not None and "x_past" in batch:
            x_past = batch["x_past"]  # (batch, lookback, past_dim)

            if self.use_nan_mask and "x_nan_mask" in batch:
                x_nan_mask = batch["x_nan_mask"]  # (batch, lookback, past_dim)
                # Concatenate features with NaN mask
                lstm_input = torch.cat([x_past, x_nan_mask], dim=-1)
            else:
                lstm_input = x_past

            # LSTM forward pass
            lstm_out, (hidden, cell) = self.past_lstm(lstm_input)

            if self.use_attention:
                # Use attention to aggregate LSTM outputs
                past_features = self.attention(lstm_out)
            else:
                # Use last hidden state
                if self.bidirectional:
                    # Concatenate forward and backward hidden states
                    past_features = torch.cat([hidden[-2], hidden[-1]], dim=-1)
                else:
                    past_features = hidden[-1]

            features.append(past_features)

        # Process future sequence with LSTM
        if self.future_lstm is not None and "x_future" in batch:
            x_future = batch["x_future"]  # (batch, lookback + future_steps, future_dim)
            future_out, (future_hidden, _) = self.future_lstm(x_future)
            future_features = future_hidden[-1]  # Use last hidden state
            features.append(future_features)

        # Process current features
        if self.now_fc is not None and "x_now" in batch:
            x_now = batch["x_now"]  # (batch, 1, now_dim)
            x_now_flat = x_now.view(batch_size, -1)
            now_features = self.now_fc(x_now_flat)
            features.append(now_features)

        # Process static features
        if self.static_fc is not None and "x_static" in batch:
            x_static = batch["x_static"]  # (batch, static_dim)
            static_features = self.static_fc(x_static)
            features.append(static_features)

        # Concatenate all features
        if features:
            combined_features = torch.cat(features, dim=-1)
        else:
            raise ValueError("No input features provided")

        # Final prediction
        predictions = self.final_layers(combined_features)

        return {
            "predictions": predictions,
        }


class BidirectionalLSTMForecaster(LSTMForecaster):
    """Bidirectional LSTM forecaster."""

    def __init__(self, **kwargs):
        kwargs["bidirectional"] = True
        super().__init__(**kwargs)


class AttentionLSTMForecaster(LSTMForecaster):
    """LSTM forecaster with attention mechanism."""

    def __init__(self, **kwargs):
        kwargs["use_attention"] = True
        super().__init__(**kwargs)


class AttentionLayer(nn.Module):
    """
    Attention mechanism for aggregating LSTM outputs.
    """

    def __init__(self, hidden_dim: int, attention_dim: Optional[int] = None):
        """
        Initialize attention layer.

        Args:
            hidden_dim: Dimension of LSTM hidden states
            attention_dim: Dimension of attention mechanism
        """
        super().__init__()

        if attention_dim is None:
            attention_dim = hidden_dim

        self.attention_dim = attention_dim
        self.hidden_dim = hidden_dim

        # Attention mechanism
        self.attention_fc = nn.Linear(hidden_dim, attention_dim)
        self.context_vector = nn.Parameter(torch.randn(attention_dim))

    def forward(self, lstm_outputs: torch.Tensor) -> torch.Tensor:
        """
        Apply attention to LSTM outputs.

        Args:
            lstm_outputs: LSTM outputs (batch, seq_len, hidden_dim)

        Returns:
            Attended features (batch, hidden_dim)
        """
        batch_size, seq_len, hidden_dim = lstm_outputs.size()

        # Compute attention scores
        attention_weights = torch.tanh(
            self.attention_fc(lstm_outputs)
        )  # (batch, seq_len, attention_dim)
        attention_scores = torch.matmul(
            attention_weights, self.context_vector
        )  # (batch, seq_len)
        attention_scores = F.softmax(attention_scores, dim=1)  # (batch, seq_len)

        # Apply attention weights
        attended_features = torch.sum(
            lstm_outputs * attention_scores.unsqueeze(-1), dim=1
        )  # (batch, hidden_dim)

        return attended_features


class MultiLayerLSTMForecaster(nn.Module):
    """
    Multi-layer LSTM with skip connections for deep learning.
    """

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        now_dim: int = 0,
        hidden_dim: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1,
        use_skip_connections: bool = True,
        use_nan_mask: bool = True,
        **kwargs,
    ):
        """
        Initialize multi-layer LSTM.

        Args:
            past_dim: Dimension of past features
            future_dim: Dimension of future features
            static_dim: Dimension of static features
            now_dim: Dimension of current features
            hidden_dim: Hidden dimension for each layer
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_dim: Output dimension
            use_skip_connections: Whether to use skip connections
            use_nan_mask: Whether to use NaN masking
            **kwargs: Additional parameters
        """
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_skip_connections = use_skip_connections
        self.use_nan_mask = use_nan_mask

        # Input dimension
        lstm_input_dim = past_dim
        if use_nan_mask:
            lstm_input_dim += past_dim

        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        input_dim = lstm_input_dim

        for i in range(num_layers):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=input_dim,
                    hidden_size=hidden_dim,
                    num_layers=1,
                    batch_first=True,
                    dropout=0,  # Apply dropout manually for better control
                )
            )
            input_dim = hidden_dim

        # Dropout layers
        self.dropout = nn.Dropout(dropout)

        # Process other inputs (same as LSTMForecaster)
        if future_dim > 0:
            self.future_fc = nn.Sequential(
                nn.Linear(future_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.future_fc = None

        if now_dim > 0:
            self.now_fc = nn.Sequential(
                nn.Linear(now_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.now_fc = None

        if static_dim > 0:
            self.static_fc = nn.Sequential(
                nn.Linear(static_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.static_fc = None

        # Calculate final input dimension
        final_input_dim = hidden_dim  # From LSTM
        if self.future_fc is not None:
            final_input_dim += hidden_dim
        if self.now_fc is not None:
            final_input_dim += hidden_dim
        if self.static_fc is not None:
            final_input_dim += hidden_dim

        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through multi-layer LSTM."""
        features = []

        # Process past sequence through multiple LSTM layers
        if "x_past" in batch:
            x_past = batch["x_past"]

            if self.use_nan_mask and "x_nan_mask" in batch:
                x_nan_mask = batch["x_nan_mask"]
                lstm_input = torch.cat([x_past, x_nan_mask], dim=-1)
            else:
                lstm_input = x_past

            # Pass through LSTM layers with optional skip connections
            layer_input = lstm_input
            layer_outputs = []

            for i, lstm_layer in enumerate(self.lstm_layers):
                layer_output, (hidden, cell) = lstm_layer(layer_input)
                layer_output = self.dropout(layer_output)
                layer_outputs.append(layer_output)

                # Skip connection (add input to output)
                if (
                    self.use_skip_connections
                    and i > 0
                    and layer_input.size(-1) == layer_output.size(-1)
                ):
                    layer_output = layer_output + layer_input

                layer_input = layer_output

            # Use final layer's last hidden state
            final_lstm_features = hidden[-1]
            features.append(final_lstm_features)

        # Process other inputs (same as LSTMForecaster)
        batch_size = (
            batch["x_past"].size(0) if "x_past" in batch else batch["x_static"].size(0)
        )

        if self.future_fc is not None and "x_future" in batch:
            x_future = batch["x_future"].view(batch_size, -1)
            future_features = self.future_fc(x_future)
            features.append(future_features)

        if self.now_fc is not None and "x_now" in batch:
            x_now = batch["x_now"].view(batch_size, -1)
            now_features = self.now_fc(x_now)
            features.append(now_features)

        if self.static_fc is not None and "x_static" in batch:
            x_static = batch["x_static"]
            static_features = self.static_fc(x_static)
            features.append(static_features)

        # Combine and predict
        combined_features = torch.cat(features, dim=-1)
        predictions = self.final_layers(combined_features)

        return {"predictions": predictions}


class LSTMLightningModel(LightningForecastBase):
    """
    PyTorch Lightning wrapper for LSTM forecasting models.
    """

    def __init__(
        self,
        architecture: str = "basic",
        past_dim: int = 0,
        future_dim: int = 0,
        static_dim: int = 0,
        now_dim: int = 0,
        **kwargs,
    ):
        """
        Initialize Lightning LSTM model.

        Args:
            architecture: Type of LSTM ('basic', 'bidirectional', 'attention', 'multilayer')
            past_dim: Past features dimension
            future_dim: Future features dimension
            static_dim: Static features dimension
            now_dim: Current features dimension
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)

        # Create the appropriate LSTM model
        model_kwargs = {
            "past_dim": past_dim,
            "future_dim": future_dim,
            "static_dim": static_dim,
            "now_dim": now_dim,
            **kwargs,
        }

        if architecture == "basic":
            self.model = LSTMForecaster(**model_kwargs)
        elif architecture == "bidirectional":
            self.model = BidirectionalLSTMForecaster(**model_kwargs)
        elif architecture == "attention":
            self.model = AttentionLSTMForecaster(**model_kwargs)
        elif architecture == "multilayer":
            self.model = MultiLayerLSTMForecaster(**model_kwargs)
        else:
            raise ValueError(f"Unknown LSTM architecture: {architecture}")

        logger.info(f"LSTMLightningModel initialized with architecture: {architecture}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through LSTM model."""
        return self.model(batch)
