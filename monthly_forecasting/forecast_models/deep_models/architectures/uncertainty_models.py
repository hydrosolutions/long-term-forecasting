import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from ..utils.lightning_base import LightningMetaForecastBase

logger = logging.getLogger(__name__)


def predict_quantile(mu: np.ndarray, b: np.ndarray, tau: float, q: float) -> np.ndarray:
    """
    Predict quantile for Asymmetric Laplace distribution.

    Args:
        mu: Location parameter (mean)
        b: Scale parameter
        tau: Asymmetry parameter
        q: Quantile to predict

    Returns:
        Quantile prediction
    """
    scale = b / (tau * (1 - tau))
    output = np.where(
        q < tau,
        mu + scale * tau * np.log(2 * q),
        mu - scale * (1 - tau) * np.log(2 * (1 - q)),
    )
    return output


class UncertaintyNet(nn.Module):
    """
    Neural network for uncertainty quantification in discharge forecasting.

    Based on the original AL_UncertaintyNet but restructured for the new architecture.
    Predicts uncertainty parameters for Asymmetric Laplace distribution.
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
        use_nan_mask: bool = True,
        **kwargs,
    ):
        """
        Initialize UncertaintyNet.

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
            output_dim: Output dimension (typically 1 for discharge)
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
            )
        else:
            self.past_lstm = None

        # Process future features
        if future_dim > 0:
            self.future_fc = nn.Sequential(
                nn.Linear(future_dim * (lookback + future_known_steps), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.future_fc = None

        # Process current features
        if now_dim > 0:
            self.now_fc = nn.Sequential(
                nn.Linear(now_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.now_fc = None

        # Process static features
        if static_dim > 0:
            self.static_fc = nn.Sequential(
                nn.Linear(static_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )
        else:
            self.static_fc = None

        # Compute final input dimension
        final_input_dim = 0
        if self.past_lstm is not None:
            final_input_dim += hidden_dim
        if self.future_fc is not None:
            final_input_dim += hidden_dim
        if self.now_fc is not None:
            final_input_dim += hidden_dim
        if self.static_fc is not None:
            final_input_dim += hidden_dim

        # Final layers for uncertainty prediction
        # Predict both location (mu) and scale (b) parameters for Asymmetric Laplace
        self.final_layers = nn.Sequential(
            nn.Linear(final_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Separate heads for mu and b parameters
        self.mu_head = nn.Linear(hidden_dim // 2, output_dim)
        self.b_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus(),  # Ensure positive scale parameter
        )

        logger.info(
            f"UncertaintyNet initialized with final_input_dim={final_input_dim}"
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of UncertaintyNet.

        Args:
            batch: Dictionary containing input tensors

        Returns:
            Dictionary with mu and b parameters for Asymmetric Laplace distribution
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
            # Use last hidden state
            past_features = hidden[-1]  # (batch, hidden_dim)
            features.append(past_features)

        # Process future features
        if self.future_fc is not None and "x_future" in batch:
            x_future = batch["x_future"]  # (batch, lookback + future_steps, future_dim)
            # Flatten temporal dimension
            x_future_flat = x_future.view(batch_size, -1)
            future_features = self.future_fc(x_future_flat)
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

        # Final prediction layers
        hidden_output = self.final_layers(combined_features)

        # Predict uncertainty parameters
        mu = self.mu_head(hidden_output)  # Location parameter
        b = self.b_head(hidden_output)  # Scale parameter (positive)

        # Add small epsilon to avoid numerical issues
        b = b + 1e-6

        return {
            "mu": mu,
            "b": b,
            "predictions": mu,  # For compatibility with base class
        }


class UncertaintyLightningModel(LightningMetaForecastBase):
    """
    PyTorch Lightning wrapper for UncertaintyNet.

    Handles training with Asymmetric Laplace loss and uncertainty quantification.
    """

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        now_dim: int = 0,
        tau: float = 0.5,  # Asymmetry parameter for Asymmetric Laplace
        **kwargs,
    ):
        """
        Initialize the Lightning model.

        Args:
            past_dim: Dimension of past features
            future_dim: Dimension of future features
            static_dim: Dimension of static features
            now_dim: Dimension of current features
            tau: Asymmetry parameter for Asymmetric Laplace distribution
            **kwargs: Additional parameters for parent classes
        """
        # Set loss function to asymmetric laplace
        kwargs.setdefault("loss_function", "asymmetric_laplace")
        kwargs.setdefault("loss_params", {"tau": tau})

        super().__init__(uncertainty_output=True, **kwargs)

        self.tau = tau

        # Create the uncertainty network
        self.model = UncertaintyNet(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            now_dim=now_dim,
            **kwargs,
        )

        logger.info(f"UncertaintyLightningModel initialized with tau={tau}")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the uncertainty model."""
        return self.model(batch)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Training step with Asymmetric Laplace loss."""
        outputs = self.forward(batch)
        mu = outputs["mu"]
        b = outputs["b"]
        targets = batch["y"]

        # Asymmetric Laplace loss
        loss = self._asymmetric_laplace_loss(
            mu.squeeze(-1), b.squeeze(-1), targets, self.tau
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-end metrics
        self.training_outputs.append(
            {
                "loss": loss.detach(),
                "predictions": mu.detach().squeeze(-1),
                "targets": targets.detach(),
                "b": b.detach().squeeze(-1),
            }
        )

        return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Validation step with uncertainty metrics."""
        outputs = self.forward(batch)
        mu = outputs["mu"]
        b = outputs["b"]
        targets = batch["y"]

        # Asymmetric Laplace loss
        loss = self._asymmetric_laplace_loss(
            mu.squeeze(-1), b.squeeze(-1), targets, self.tau
        )

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # Store outputs for epoch-end metrics
        self.validation_outputs.append(
            {
                "loss": loss.detach(),
                "predictions": mu.detach().squeeze(-1),
                "targets": targets.detach(),
                "b": b.detach().squeeze(-1),
            }
        )

        return loss

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Prediction step with uncertainty quantiles."""
        outputs = self.forward(batch)
        mu = outputs["mu"].squeeze(-1)
        b = outputs["b"].squeeze(-1)

        # Generate quantile predictions
        quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        quantile_predictions = {}

        for q in quantiles:
            q_pred = self._predict_quantile_torch(mu, b, self.tau, q)
            quantile_predictions[f"Q_{int(q * 100):02d}"] = q_pred

        outputs.update(quantile_predictions)
        outputs["metadata"] = batch["metadata"]

        return outputs

    def _asymmetric_laplace_loss(
        self, mu: torch.Tensor, b: torch.Tensor, y: torch.Tensor, tau: float
    ) -> torch.Tensor:
        """Compute Asymmetric Laplace loss."""
        # Compute the loss for Asymmetric Laplace distribution
        diff = y - mu
        loss = torch.where(diff >= 0, tau * diff / b, (tau - 1) * diff / b) + torch.log(
            b / (tau * (1 - tau))
        )

        return loss.mean()

    def _predict_quantile_torch(
        self, mu: torch.Tensor, b: torch.Tensor, tau: float, q: float
    ) -> torch.Tensor:
        """Predict quantile using PyTorch tensors."""
        scale = b / (tau * (1 - tau))

        quantile_pred = torch.where(
            q < tau,
            mu + scale * tau * torch.log(torch.tensor(2 * q)),
            mu - scale * (1 - tau) * torch.log(torch.tensor(2 * (1 - q))),
        )

        return quantile_pred
