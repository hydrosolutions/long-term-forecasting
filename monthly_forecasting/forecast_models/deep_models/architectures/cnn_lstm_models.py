import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import logging

from ..utils.lightning_base import LightningForecastBase

logger = logging.getLogger(__name__)


class CNNLSTMForecaster(nn.Module):
    """
    CNN-LSTM hybrid model for monthly discharge forecasting.
    
    Uses CNN layers to extract local temporal patterns followed by LSTM
    to capture long-term dependencies.
    """

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        now_dim: int = 0,
        lookback: int = 365,
        future_known_steps: int = 30,
        cnn_filters: List[int] = [32, 64],
        kernel_sizes: List[int] = [3, 3],
        pool_sizes: List[int] = [2, 2],
        lstm_hidden_dim: int = 64,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
        use_nan_mask: bool = True,
        **kwargs
    ):
        """
        Initialize CNN-LSTM forecaster.
        
        Args:
            past_dim: Dimension of past features
            future_dim: Dimension of future features
            static_dim: Dimension of static features
            now_dim: Dimension of current time step features
            lookback: Number of past time steps
            future_known_steps: Number of future time steps
            cnn_filters: List of CNN filter sizes
            kernel_sizes: List of CNN kernel sizes
            pool_sizes: List of pooling sizes
            lstm_hidden_dim: LSTM hidden dimension
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout rate
            output_dim: Output dimension
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
        self.dropout = dropout
        self.output_dim = output_dim
        self.use_nan_mask = use_nan_mask
        
        # CNN layers for past sequence processing
        if past_dim > 0:
            cnn_input_dim = past_dim
            if use_nan_mask:
                cnn_input_dim += past_dim  # Add NaN mask dimension
                
            self.cnn_layers = self._build_cnn_layers(
                input_channels=cnn_input_dim,
                filters=cnn_filters,
                kernel_sizes=kernel_sizes,
                pool_sizes=pool_sizes,
                dropout=dropout
            )
            
            # Calculate CNN output dimension
            cnn_output_length = self._calculate_cnn_output_length(
                lookback, kernel_sizes, pool_sizes
            )
            
            # LSTM after CNN - use the final filter size as input
            self.past_lstm = nn.LSTM(
                input_size=cnn_filters[-1],
                hidden_size=lstm_hidden_dim,
                num_layers=lstm_num_layers,
                dropout=dropout if lstm_num_layers > 1 else 0,
                batch_first=True
            )
            
            past_output_dim = lstm_hidden_dim
        else:
            self.cnn_layers = None
            self.past_lstm = None
            past_output_dim = 0
            
        # Process future features with separate CNN-LSTM
        if future_dim > 0:
            self.future_cnn = self._build_cnn_layers(
                input_channels=future_dim,
                filters=[16, 32],  # Smaller filters for future
                kernel_sizes=[3, 3],
                pool_sizes=[2, 2],
                dropout=dropout
            )
            
            future_cnn_output_length = self._calculate_cnn_output_length(
                lookback + future_known_steps, [3, 3], [2, 2]
            )
            
            self.future_lstm = nn.LSTM(
                input_size=32,  # Use the CNN filter size
                hidden_size=lstm_hidden_dim // 2,
                num_layers=1,
                batch_first=True
            )
            
            future_output_dim = lstm_hidden_dim // 2
        else:
            self.future_cnn = None
            self.future_lstm = None
            future_output_dim = 0
            
        # Process current features
        if now_dim > 0:
            self.now_fc = nn.Sequential(
                nn.Linear(now_dim, lstm_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            now_output_dim = lstm_hidden_dim
        else:
            self.now_fc = None
            now_output_dim = 0
            
        # Process static features
        if static_dim > 0:
            self.static_fc = nn.Sequential(
                nn.Linear(static_dim, lstm_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            static_output_dim = lstm_hidden_dim
        else:
            self.static_fc = None
            static_output_dim = 0
            
        # Compute final input dimension
        final_input_dim = past_output_dim + future_output_dim + now_output_dim + static_output_dim
        
        # Final prediction layers
        self.final_layers = nn.Sequential(
            nn.Linear(final_input_dim, lstm_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim, lstm_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim // 2, output_dim)
        )
        
        logger.info(f"CNNLSTMForecaster initialized with final_input_dim={final_input_dim}")

    def _build_cnn_layers(
        self,
        input_channels: int,
        filters: List[int],
        kernel_sizes: List[int],
        pool_sizes: List[int],
        dropout: float
    ) -> nn.Sequential:
        """Build CNN layers."""
        layers = []
        in_channels = 1  # We'll treat features as channels, time as width
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(filters, kernel_sizes, pool_sizes)):
            # For first layer, input channels is the feature dimension
            if i == 0:
                # Conv1d expects (batch, channels, length)
                # We need to reshape from (batch, length, features) to (batch, features, length)
                layers.append(
                    nn.Conv1d(
                        in_channels=input_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
            else:
                layers.append(
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2
                    )
                )
                
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(kernel_size=pool_size))
            layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
            
        return nn.Sequential(*layers)

    def _calculate_cnn_output_length(
        self,
        input_length: int,
        kernel_sizes: List[int],
        pool_sizes: List[int]
    ) -> int:
        """Calculate output length after CNN layers."""
        length = input_length
        
        for kernel_size, pool_size in zip(kernel_sizes, pool_sizes):
            # After conv with padding: same length
            # After pooling: length // pool_size
            length = length // pool_size
            
        return max(1, length)  # Ensure at least 1

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of CNN-LSTM forecaster.
        
        Args:
            batch: Dictionary containing input tensors
            
        Returns:
            Dictionary with predictions
        """
        batch_size = batch['x_past'].size(0) if 'x_past' in batch else batch['x_static'].size(0)
        features = []
        
        # Process past sequence with CNN-LSTM
        if self.cnn_layers is not None and self.past_lstm is not None and 'x_past' in batch:
            x_past = batch['x_past']  # (batch, lookback, past_dim)
            
            if self.use_nan_mask and 'x_nan_mask' in batch:
                x_nan_mask = batch['x_nan_mask']  # (batch, lookback, past_dim)
                # Concatenate features with NaN mask
                cnn_input = torch.cat([x_past, x_nan_mask], dim=-1)
            else:
                cnn_input = x_past
                
            # Reshape for CNN: (batch, features, time)
            cnn_input = cnn_input.transpose(1, 2)  # (batch, features, lookback)
            
            # CNN forward pass
            cnn_output = self.cnn_layers(cnn_input)  # (batch, cnn_filters[-1], reduced_length)
            
            # Reshape for LSTM: (batch, reduced_length, cnn_filters[-1])
            cnn_output = cnn_output.transpose(1, 2)  # (batch, reduced_length, cnn_filters[-1])
            
            # LSTM forward pass
            lstm_out, (hidden, cell) = self.past_lstm(cnn_output)
            past_features = hidden[-1]  # Use last hidden state
            features.append(past_features)
        
        # Process future sequence with CNN-LSTM
        if self.future_cnn is not None and self.future_lstm is not None and 'x_future' in batch:
            x_future = batch['x_future']  # (batch, lookback + future_steps, future_dim)
            
            # Reshape for CNN: (batch, features, time)
            x_future = x_future.transpose(1, 2)
            
            # CNN forward pass
            future_cnn_output = self.future_cnn(x_future)
            
            # Reshape for LSTM
            future_cnn_output = future_cnn_output.transpose(1, 2)  # (batch, reduced_length, filters)
            
            # LSTM forward pass
            future_lstm_out, (future_hidden, _) = self.future_lstm(future_cnn_output)
            future_features = future_hidden[-1]
            features.append(future_features)
            
        # Process current features
        if self.now_fc is not None and 'x_now' in batch:
            x_now = batch['x_now']  # (batch, 1, now_dim)
            x_now_flat = x_now.view(batch_size, -1)
            now_features = self.now_fc(x_now_flat)
            features.append(now_features)
            
        # Process static features
        if self.static_fc is not None and 'x_static' in batch:
            x_static = batch['x_static']  # (batch, static_dim)
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
            'predictions': predictions,
        }


class ResidualCNNLSTMForecaster(nn.Module):
    """
    CNN-LSTM with residual connections for deeper networks.
    """
    
    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        now_dim: int = 0,
        num_residual_blocks: int = 3,
        **kwargs
    ):
        """
        Initialize residual CNN-LSTM.
        
        Args:
            past_dim: Past features dimension
            future_dim: Future features dimension
            static_dim: Static features dimension
            now_dim: Current features dimension
            num_residual_blocks: Number of residual blocks
            **kwargs: Additional parameters
        """
        super().__init__()
        
        self.past_dim = past_dim
        self.num_residual_blocks = num_residual_blocks
        self.use_nan_mask = kwargs.get('use_nan_mask', True)
        
        if past_dim > 0:
            input_dim = past_dim
            if self.use_nan_mask:
                input_dim += past_dim
                
            # Initial CNN layer
            self.initial_conv = nn.Conv1d(
                in_channels=input_dim,
                out_channels=64,
                kernel_size=3,
                padding=1
            )
            
            # Residual blocks
            self.residual_blocks = nn.ModuleList([
                ResidualBlock(64, 64) for _ in range(num_residual_blocks)
            ])
            
            # LSTM after residual CNN
            self.lstm = nn.LSTM(
                input_size=64,
                hidden_size=kwargs.get('lstm_hidden_dim', 64),
                num_layers=kwargs.get('lstm_num_layers', 2),
                batch_first=True
            )
            
        # Other processing layers (same as CNNLSTMForecaster)
        # ... (implementation similar to CNNLSTMForecaster)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through residual CNN-LSTM."""
        # Implementation similar to CNNLSTMForecaster but with residual blocks
        pass


class ResidualBlock(nn.Module):
    """Residual block for CNN."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.skip_connection = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
    def forward(self, x):
        identity = self.skip_connection(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = F.relu(out)
        
        return out


class TemporalCNN(nn.Module):
    """
    Temporal Convolutional Network (TCN) for sequence modeling.
    """
    
    def __init__(
        self,
        input_size: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize TCN.
        
        Args:
            input_size: Input feature dimension
            num_channels: List of channel sizes for each layer
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )
            )
            
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)


class TemporalBlock(nn.Module):
    """Temporal block for TCN."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Chomp1d(nn.Module):
    """Remove padding from the end of tensor."""
    
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class CNNLSTMLightningModel(LightningForecastBase):
    """
    PyTorch Lightning wrapper for CNN-LSTM forecasting models.
    """

    def __init__(
        self,
        architecture: str = 'basic',
        past_dim: int = 0,
        future_dim: int = 0,
        static_dim: int = 0,
        now_dim: int = 0,
        **kwargs
    ):
        """
        Initialize Lightning CNN-LSTM model.
        
        Args:
            architecture: Type of CNN-LSTM ('basic', 'residual', 'tcn')
            past_dim: Past features dimension
            future_dim: Future features dimension
            static_dim: Static features dimension
            now_dim: Current features dimension
            **kwargs: Additional parameters
        """
        super().__init__(**kwargs)
        
        # Create the appropriate CNN-LSTM model
        model_kwargs = {
            'past_dim': past_dim,
            'future_dim': future_dim,
            'static_dim': static_dim,
            'now_dim': now_dim,
            **kwargs
        }
        
        if architecture == 'basic':
            self.model = CNNLSTMForecaster(**model_kwargs)
        elif architecture == 'residual':
            self.model = ResidualCNNLSTMForecaster(**model_kwargs)
        else:
            raise ValueError(f"Unknown CNN-LSTM architecture: {architecture}")
        
        logger.info(f"CNNLSTMLightningModel initialized with architecture: {architecture}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through CNN-LSTM model."""
        return self.model(batch)