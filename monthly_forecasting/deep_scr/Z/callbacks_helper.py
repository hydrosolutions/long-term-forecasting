from pytorch_lightning.callbacks import Callback


# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


class MetricsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.train_losses = []
        self.val_losses = []
        self.test_loss = None

    def on_train_epoch_end(self, trainer, pl_module):
        # Look for epoch-level train loss as you're logging with on_epoch=True
        if "train_loss" in trainer.callback_metrics:
            train_loss = trainer.callback_metrics["train_loss"]
            self.train_losses.append(train_loss.item())
            logger.info(f"Epoch {trainer.current_epoch}: Train Loss = {train_loss:.4f}")
        else:
            logger.warning(
                "No train loss found in callback metrics. Ensure you're logging it correctly."
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation loss
        val_loss = trainer.callback_metrics.get("val_loss")
        if val_loss is not None:
            self.val_losses.append(val_loss.item())
            logger.info(
                f"Epoch {trainer.current_epoch}: Validation Loss = {val_loss:.4f}"
            )

    def on_test_end(self, trainer, pl_module):
        test_loss = trainer.callback_metrics.get("test_loss")
        if test_loss is not None:
            self.test_loss = test_loss.item()
            logger.info(f"Test Loss = {self.test_loss:.4f}")
