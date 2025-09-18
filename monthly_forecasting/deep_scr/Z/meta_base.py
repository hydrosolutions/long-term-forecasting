import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from abc import ABC, abstractmethod
import pandas as pd
from .loss_function import QuantileLoss, AsymmetricLaplaceLoss

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


class LitMetaForecastBase(pl.LightningModule, ABC):
    """
    Abstract LightningModule that handles:
      - Quantile vs MSE loss setup
      - training/validation/test steps
      - predict_step → pandas DataFrame of (date, code, Q… / prediction)
      - optimizer configuration

    Subclasses must implement:
      - build_network(self, in_dim: int, out_dim: int) -> torch.nn.Module
    """

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        static_dim: int,
        lookback: int,
        future_known_steps: int,
        base_learner_dim: int,
        base_learner_error_dim: int,
        hidden_dim: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        loss_fn: str = "QuantileLoss",
        output_dim: int = 1,
        dropout: float = 0.0,
        lr_scheduler_factor: float = 0.1,
        lr_scheduler_patience: int = 3,
        quantiles=None,
        center_weight: float = 0.0,
        adaptive_weighting: bool = False,
        correction_term: bool = False,
        weight_by_metrics: bool = False,
        **kwargs,
    ):
        super().__init__()
        # save everything except the net itself
        self.save_hyperparameters()

        self.quantiles = quantiles

        # pick loss and determine output_dim
        if loss_fn == "MSE":
            self.criterion = F.mse_loss
            output_dim = 1
        elif loss_fn == "QuantileLoss":
            if quantiles is None:
                raise ValueError("Quantiles must be provided for QuantileLoss")
            self.criterion = QuantileLoss(quantiles=quantiles)
            output_dim = len(quantiles)
        elif loss_fn == "AsymmetricLaplaceLoss":
            self.criterion = AsymmetricLaplaceLoss()
            self.output_dim = 2
        else:
            raise ValueError(f"Unknown loss_fn={loss_fn}")

        self.net = self.build_network(output_dim)

    @abstractmethod
    def build_network(self, out_dim: int) -> torch.nn.Module:
        """
        Return a torch.nn.Module mapping a tensor of shape (B, in_dim)
        → (B, out_dim).  Subclasses must override this.
        """
        pass

    def forward(self, batch):
        # net should output shape (B, out_dim)
        return self.net(batch).squeeze(1)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.criterion(y_hat, batch["y"])
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.criterion(y_hat, batch["y"])
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        loss = self.criterion(y_hat, batch["y"])
        self.log("test_loss", loss, on_epoch=True, prog_bar=True)

    def predict_step(self, batch, batch_idx):
        # 1) forward
        y_hat = self(batch)
        y_np = y_hat.detach().cpu().numpy()

        # 2) extract date components & code
        day_np = batch["day"].detach().cpu().numpy().ravel().astype(int)
        month_np = batch["month"].detach().cpu().numpy().ravel().astype(int)
        year_np = batch["year"].detach().cpu().numpy().ravel().astype(int)

        if isinstance(batch["code"], torch.Tensor):
            code_np = batch["code"].detach().cpu().numpy().ravel()
        else:
            code_np = list(batch["code"])

        # 3) build pandas dates
        dates = pd.to_datetime({"year": year_np, "month": month_np, "day": day_np})

        # 4) assemble DataFrame
        data = {"date": dates, "code": code_np}

        # handle quantile vs single‐output
        if y_np.ndim == 2 and self.quantiles is not None:
            B, Q = y_np.shape
            if Q != len(self.quantiles):
                raise ValueError(
                    f"Output shape {y_np.shape} ≠ {len(self.quantiles)} quantiles"
                )
            for i, q in enumerate(self.quantiles):
                data[f"Q{int(q * 100)}"] = y_np[:, i]
        elif y_np.ndim == 1:
            # name median/Q50 by default if quantiles given, else 'prediction'
            name = "Q50" if self.quantiles is not None else "Q_pred"
            data[name] = y_np.ravel()
        else:
            raise ValueError(f"Unexpected output shape {y_np.shape}")

        return pd.DataFrame(data)

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sched,
                "monitor": "val_loss",
                "interval": "epoch",
            },
        }
