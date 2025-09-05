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

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
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
        use_pred_mean: bool = False,
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
        output_size = 2 if use_pred_mean else 3
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

        if self.hparams.use_pred_mean:
            # Use first feature as μ
            mu = x[:, 0]
            beta = self.softplus(params[:, 0]) + 1e-2
            tau = torch.sigmoid(params[:, 1]) 

            #clip tau to be in range
            tau = torch.clamp(tau, min=1e-2, max=0.99)

            #detach mu to avoid gradients flowing into input feature
            #mu = mu.detach()
        else:
            mu = params[:, 0]
            beta = self.softplus(params[:, 1]) + 1e-2
            tau = torch.sigmoid(params[:, 2]) 
            #clip tau to be in range
            tau = torch.clamp(tau, min=1e-3, max=0.999)

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

    def predict_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int, dataloader_idx: int = 0
    ) -> pd.DataFrame:
        """
        Perform a prediction step on a single batch of data.

        This method is called by the PyTorch Lightning Trainer during the
        prediction loop. It processes a batch, computes the distribution
        parameters, and returns them along with identifiers in a DataFrame.

        Args:
            batch: A dictionary containing the input tensors for the batch.
                   Expected keys include 'X' for features, and optionally
                   'date', 'code', 'year', 'month', 'day' for metadata.
            batch_idx: The index of the current batch.
            dataloader_idx: The index of the dataloader (if multiple are used).

        Returns:
            A pandas DataFrame containing the predictions (loc, scale, asymmetry)
            and identifiers for the batch.
        """
        if "X" not in batch:
            raise KeyError("Batch dictionary must contain an 'X' key for features.")

        x = batch["X"]
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            mu, beta, tau = self(x)

        # Prepare data for DataFrame creation
        data: dict[str, np.ndarray | list[pd.Timestamp] | list[None]] = {
            "loc": mu.cpu().numpy(),
            "scale": beta.cpu().numpy(),
            "asymmetry": tau.cpu().numpy(),
        }

        # Safely extract optional metadata
        batch_size = x.shape[0]
        if all(k in batch for k in ["year", "month", "day"]):
            data["date"] = pd.to_datetime({
                'year': batch["year"].cpu().numpy(),
                'month': batch["month"].cpu().numpy(),
                'day': batch["day"].cpu().numpy()
            }, errors="coerce")
        else:
            data["date"] = [pd.NaT] * batch_size

        if "code" in batch:
            data["code"] = batch["code"].cpu().numpy()
        else:
            data["code"] = [None] * batch_size

        return pd.DataFrame(data)


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


def test_mlp_uncertainty_model():
    """
    Simple test function to verify the MLP Uncertainty Model works correctly.
    Creates a model with dummy data and performs forward pass and loss computation.
    """
    print("Testing MLP Uncertainty Model...")
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Model parameters
    num_features = 10
    batch_size = 32
    hidden_size = 64
    num_residual_blocks = 3
    
    # Create model
    model = MLPUncertaintyModel(
        num_features=num_features,
        hidden_size=hidden_size,
        num_residual_blocks=num_residual_blocks,
        dropout=0.1,
        learning_rate=1e-3,
        use_ensemble_mean=False
    )
    
    print(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy data
    x = torch.randn(batch_size, num_features)
    y = torch.randn(batch_size)  # Target values
    
    # Test forward pass
    print("\nTesting forward pass...")
    model.eval()
    with torch.no_grad():
        mu, beta, tau = model(x)
        
    print(f"Output shapes - μ: {mu.shape}, β: {beta.shape}, τ: {tau.shape}")
    print(f"μ range: [{mu.min().item():.3f}, {mu.max().item():.3f}]")
    print(f"β range: [{beta.min().item():.3f}, {beta.max().item():.3f}] (should be positive)")
    print(f"τ range: [{tau.min().item():.3f}, {tau.max().item():.3f}] (should be in [0,1])")
    
    # Test training step
    print("\nTesting training step...")
    model.train()
    batch = {"X": x, "y": y}
    loss = model.training_step(batch, 0)
    print(f"Training loss: {loss.item():.4f}")
    
    # Test prediction
    print("\nTesting prediction...")
    model.eval()
    batch_with_metadata = {
        "X": x,
        "day": torch.ones(batch_size) * 15,
        "month": torch.ones(batch_size) * 6,
        "year": torch.ones(batch_size) * 2023,
        "code": torch.arange(batch_size)
    }
    
    prediction_df = model.predict(batch_with_metadata)
    print(f"Prediction DataFrame shape: {prediction_df.shape}")
    print("First few predictions:")
    print(prediction_df.head())
    
    # Test with ensemble mean mode
    print("\nTesting with use_ensemble_mean=True...")
    model_ensemble = MLPUncertaintyModel(
        num_features=num_features,
        hidden_size=hidden_size,
        num_residual_blocks=num_residual_blocks,
        use_ensemble_mean=True
    )
    
    model_ensemble.eval()
    with torch.no_grad():
        mu_ens, beta_ens, tau_ens = model_ensemble(x)
    
    print(f"Ensemble mode - μ uses first feature: {torch.allclose(mu_ens, x[:, 0])}")
    
    print("\n✅ All tests passed! MLP Uncertainty Model is working correctly.")


if __name__ == "__main__":
    test_mlp_uncertainty_model()
