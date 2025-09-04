import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
import logging
import pytorch_lightning as pl
from torch.optim.lr_scheduler import StepLR
import pandas as pd

from ..losses import AsymmetricLaplaceLoss


logger = logging.getLogger(__name__)


class ResidualBlock(torch.nn.Module):
    def __init__(
        self,
        in_dim: int,
        h_dim: int,
        out_dim: int,
        dropout: float = 0,
    ) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.hidden_layer = torch.nn.Linear(in_dim, h_dim)
        self.output_layer = torch.nn.Linear(h_dim, out_dim)
        self.residual_layer = torch.nn.Linear(in_dim, out_dim)
        self.act = torch.nn.ReLU()

    def forward(self, x: torch.Tensor):
        hid = self.act(self.hidden_layer(x))
        out = self.output_layer(hid)
        res = self.residual_layer(x)
        out = out + res
        return out


class MLPUncertaintyModel(pl.LightningModule):
    """
    PyTorch Lightning model for predicting asymmetric Laplace distribution parameters
    using residual blocks. The model takes (batch, features) input and outputs
    parameters for the asymmetric Laplace distribution: location (μ), scale (σ), and asymmetry (p).

    The first feature can optionally be used directly as the location parameter (μ).
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_residual_blocks: int,
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        gradient_clip_val: float = 1.0,
        lr_scheduler: Optional[Dict] = None,
        use_ensemble_mean: bool = False,
        quantiles: Optional[List] = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
    ):
        """
        Initialize the MLP Uncertainty Model.

        Args:
            num_features: Number of input features.
            hidden_size: Size of hidden layers in residual blocks.
            num_residual_blocks: Number of residual blocks.
            dropout: Dropout probability.
            learning_rate: Learning rate for optimizer.
            weight_decay: Weight decay for optimizer.
            gradient_clip_val: Value for gradient clipping.
            lr_scheduler: Configuration for learning rate scheduler (optional).
            use_ensemble_mean: If True, use the first feature as μ directly.
        """
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles

        # Input layer
        self.input_layer = nn.Linear(num_features, hidden_size)

        # Residual blocks
        self.residual_blocks = nn.ModuleList(
            [
                ResidualBlock(hidden_size, hidden_size, hidden_size, dropout)
                for _ in range(num_residual_blocks)
            ]
        )

        # Output layer: predicts μ, σ, p (or σ, p if use_ensemble_mean)
        output_size = 2 if use_ensemble_mean else 3
        self.output_layer = nn.Linear(hidden_size, output_size)

        self.loss_fn = AsymmetricLaplaceLoss()

        # Activation for scale parameter to ensure positivity
        self.softplus = nn.Softplus()

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, num_features)

        Returns:
            Tuple of (μ, β, τ) for asymmetric Laplace distribution
            μ is the location
            β is the scale
            τ is the asymmetry
        """
        # Input projection
        out = F.relu(self.input_layer(x))

        # Residual blocks
        for block in self.residual_blocks:
            out = block(out)

        # Output parameters
        params = self.output_layer(out)

        if self.hparams.use_ensemble_mean:
            # Use first feature as μ
            mu = x[:, 0]
            beta = self.softplus(params[:, 0])
            tau = torch.sigmoid(params[:, 1])
        else:
            mu = params[:, 0]
            beta = self.softplus(params[:, 1])
            tau = torch.sigmoid(params[:, 2])

        return mu, beta, tau

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Training step.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x = batch["X"]
        y = batch["y"]
        mu, beta, tau = self(x)
        loss = self.loss_fn(mu, beta, tau, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Validation step.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Batch index

        Returns:
            Loss value
        """
        x = batch["X"]
        y = batch["y"]
        mu, beta, tau = self(x)
        loss = self.loss_fn(mu, beta, tau, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def predict(self, batch: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """
        Make predictions for a batch of input data, computing parameters of the
        asymmetric Laplace distribution.

        Args:
            batch: Dictionary containing input tensors with keys:
                - 'X': Input features (batch, num_features)
                - 'day', 'month', 'year': Date components (optional)
                - 'code': Basin or location code (optional)

        Returns:
            DataFrame with columns: date, code, loc (μ), sigma (σ), p (asymmetry)
        """
        x = batch["X"]
        mu, beta, tau = self(x)

        # Convert tensors to numpy for quantile computation
        mu_np = mu.detach().cpu().numpy()
        beta_np = beta.detach().cpu().numpy()
        tau_np = tau.detach().cpu().numpy()

        # Extract date components
        day = batch.get("day", None)
        month = batch.get("month", None)
        year = batch.get("year", None)

        if day is not None and month is not None and year is not None:
            day_np = day.detach().cpu().numpy().astype(int)
            month_np = month.detach().cpu().numpy().astype(int)
            year_np = year.detach().cpu().numpy().astype(int)
            dates = pd.to_datetime({"year": year_np, "month": month_np, "day": day_np})
        else:
            dates = [pd.NaT] * len(mu_np)

        # Extract code
        code = batch.get("code", None)
        if code is not None:
            code_np = code.detach().cpu().numpy()
        else:
            code_np = [None] * len(mu_np)

        # Create DataFrame
        data = {
            "date": dates,
            "code": code_np,
            "loc": mu_np,
            "scale": beta_np,
            "asymmetry": tau_np,
        }

        prediction_df = pd.DataFrame(data)

        return prediction_df

    def predict_step(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def configure_optimizers(self) -> Dict:
        """
        Configure optimizer and learning rate scheduler.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler:
            scheduler_config = self.hparams.lr_scheduler
            scheduler_type = scheduler_config.get("type", "step")

            if scheduler_type == "step":
                scheduler = StepLR(
                    optimizer,
                    step_size=scheduler_config.get("step_size", 10),
                    gamma=scheduler_config.get("gamma", 0.1),
                )
            else:
                # Default to no scheduler
                return optimizer

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return optimizer
