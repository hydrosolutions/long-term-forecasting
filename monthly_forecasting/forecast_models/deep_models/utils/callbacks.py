import os
import logging
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping, 
    LearningRateMonitor,
    RichProgressBar
)
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)


class DeepForecastingCallbacks:
    """
    Factory class for creating PyTorch Lightning callbacks for deep learning forecasting models.
    
    Provides commonly used callbacks configured for forecasting workflows.
    """

    @staticmethod
    def get_standard_callbacks(
        save_dir: str,
        model_name: str,
        monitor: str = 'val_loss',
        patience: int = 10,
        min_delta: float = 0.001,
        save_top_k: int = 1,
        save_last: bool = True,
        enable_progress_bar: bool = True,
        log_lr: bool = True,
        **kwargs
    ) -> List[pl.Callback]:
        """
        Get standard callbacks for deep learning training.
        
        Args:
            save_dir: Directory to save model checkpoints
            model_name: Name of the model for checkpoint naming
            monitor: Metric to monitor for checkpointing and early stopping
            patience: Number of epochs to wait before early stopping
            min_delta: Minimum change to qualify as improvement
            save_top_k: Number of best models to save
            save_last: Whether to save last model
            enable_progress_bar: Whether to show progress bar
            log_lr: Whether to log learning rate
            **kwargs: Additional callback parameters
            
        Returns:
            List of configured callbacks
        """
        callbacks = []
        
        # Model checkpointing
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(save_dir, model_name, 'checkpoints'),
            filename=f"{model_name}_{{epoch:02d}}_{{val_loss:.4f}}",
            monitor=monitor,
            mode='min' if 'loss' in monitor.lower() else 'max',
            save_top_k=save_top_k,
            save_last=save_last,
            verbose=True,
            **kwargs.get('checkpoint_params', {})
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor=monitor,
            mode='min' if 'loss' in monitor.lower() else 'max',
            patience=patience,
            min_delta=min_delta,
            verbose=True,
            **kwargs.get('early_stopping_params', {})
        )
        callbacks.append(early_stopping)
        
        # Learning rate monitoring
        if log_lr:
            lr_monitor = LearningRateMonitor(
                logging_interval='epoch',
                **kwargs.get('lr_monitor_params', {})
            )
            callbacks.append(lr_monitor)
        
        # Progress bar
        if enable_progress_bar:
            try:
                progress_bar = RichProgressBar()
                callbacks.append(progress_bar)
            except ImportError:
                # Fallback to default progress bar if rich not available
                logger.warning("Rich progress bar not available, using default")
        
        return callbacks

    @staticmethod
    def get_meta_learning_callbacks(
        save_dir: str,
        model_name: str,
        monitor: str = 'val_loss',
        patience: int = 15,  # Longer patience for meta-learning
        min_delta: float = 0.0001,  # Smaller delta for meta-learning
        save_top_k: int = 3,  # Save more models for ensemble
        **kwargs
    ) -> List[pl.Callback]:
        """
        Get callbacks specifically configured for meta-learning models.
        
        Args:
            save_dir: Directory to save model checkpoints
            model_name: Name of the model for checkpoint naming
            monitor: Metric to monitor
            patience: Patience for early stopping (longer for meta-learning)
            min_delta: Minimum delta for improvement
            save_top_k: Number of best models to save
            **kwargs: Additional parameters
            
        Returns:
            List of configured callbacks
        """
        # Use standard callbacks with meta-learning specific defaults
        return DeepForecastingCallbacks.get_standard_callbacks(
            save_dir=save_dir,
            model_name=model_name,
            monitor=monitor,
            patience=patience,
            min_delta=min_delta,
            save_top_k=save_top_k,
            **kwargs
        )

    @staticmethod
    def get_uncertainty_callbacks(
        save_dir: str,
        model_name: str,
        monitor: str = 'val_loss',
        **kwargs
    ) -> List[pl.Callback]:
        """
        Get callbacks for uncertainty quantification models.
        
        Args:
            save_dir: Directory to save model checkpoints
            model_name: Name of the model
            monitor: Metric to monitor
            **kwargs: Additional parameters
            
        Returns:
            List of configured callbacks
        """
        # Uncertainty models might need different monitoring
        callbacks = DeepForecastingCallbacks.get_standard_callbacks(
            save_dir=save_dir,
            model_name=model_name,
            monitor=monitor,
            patience=20,  # Longer patience for uncertainty models
            min_delta=0.0001,
            save_top_k=2,
            **kwargs
        )
        
        return callbacks


class ForecastingMetricsCallback(pl.Callback):
    """
    Custom callback for logging forecasting-specific metrics during training.
    
    Computes and logs additional metrics relevant to discharge forecasting
    such as Nash-Sutcliffe Efficiency, relative errors, etc.
    """

    def __init__(self, log_every_n_epochs: int = 1):
        """
        Initialize the metrics callback.
        
        Args:
            log_every_n_epochs: Frequency of metric logging
        """
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute and log forecasting metrics at the end of validation epoch."""
        if (trainer.current_epoch + 1) % self.log_every_n_epochs != 0:
            return
            
        # Get validation outputs if available
        if hasattr(pl_module, 'validation_outputs') and pl_module.validation_outputs:
            self._compute_forecasting_metrics(trainer, pl_module)

    def _compute_forecasting_metrics(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Compute forecasting-specific metrics."""
        import torch
        import numpy as np
        
        # Get predictions and targets
        all_predictions = torch.cat([x['predictions'] for x in pl_module.validation_outputs])
        all_targets = torch.cat([x['targets'] for x in pl_module.validation_outputs])
        
        # Handle multi-output predictions
        if all_predictions.dim() > 1:
            all_predictions = all_predictions.mean(dim=-1)
        
        # Convert to numpy for metric computation
        predictions = all_predictions.cpu().numpy()
        targets = all_targets.cpu().numpy()
        
        # Compute Nash-Sutcliffe Efficiency
        nse = self._compute_nse(predictions, targets)
        pl_module.log('val_nse', nse, on_epoch=True)
        
        # Compute relative error metrics
        relative_error = np.abs((predictions - targets) / (targets + 1e-8))
        mean_relative_error = np.mean(relative_error)
        pl_module.log('val_mean_relative_error', mean_relative_error, on_epoch=True)
        
        # Compute correlation coefficient
        correlation = np.corrcoef(predictions, targets)[0, 1]
        pl_module.log('val_correlation', correlation, on_epoch=True)

    @staticmethod
    def _compute_nse(predictions: np.ndarray, targets: np.ndarray) -> float:
        """Compute Nash-Sutcliffe Efficiency."""
        mean_target = np.mean(targets)
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - mean_target) ** 2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else -np.inf
        
        nse = 1 - (ss_res / ss_tot)
        return float(nse)


class GradientClippingCallback(pl.Callback):
    """
    Custom callback for gradient clipping with logging.
    
    Monitors gradient norms and applies clipping when necessary.
    """

    def __init__(self, max_norm: float = 1.0, log_grad_norm: bool = True):
        """
        Initialize gradient clipping callback.
        
        Args:
            max_norm: Maximum gradient norm
            log_grad_norm: Whether to log gradient norms
        """
        super().__init__()
        self.max_norm = max_norm
        self.log_grad_norm = log_grad_norm

    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer, optimizer_idx: int):
        """Apply gradient clipping before optimizer step."""
        import torch.nn.utils as utils
        
        if self.log_grad_norm:
            # Compute gradient norm before clipping
            total_norm = 0
            for p in pl_module.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            
            pl_module.log('grad_norm', total_norm, on_step=True)
        
        # Apply gradient clipping
        utils.clip_grad_norm_(pl_module.parameters(), self.max_norm)


def get_default_callbacks(
    save_dir: str,
    model_name: str,
    model_type: str = 'standard',
    **kwargs
) -> List[pl.Callback]:
    """
    Get default callbacks based on model type.
    
    Args:
        save_dir: Directory to save models
        model_name: Name of the model
        model_type: Type of model ('standard', 'meta_learning', 'uncertainty')
        **kwargs: Additional parameters
        
    Returns:
        List of appropriate callbacks
    """
    if model_type == 'meta_learning':
        callbacks = DeepForecastingCallbacks.get_meta_learning_callbacks(
            save_dir=save_dir,
            model_name=model_name,
            **kwargs
        )
    elif model_type == 'uncertainty':
        callbacks = DeepForecastingCallbacks.get_uncertainty_callbacks(
            save_dir=save_dir,
            model_name=model_name,
            **kwargs
        )
    else:
        callbacks = DeepForecastingCallbacks.get_standard_callbacks(
            save_dir=save_dir,
            model_name=model_name,
            **kwargs
        )
    
    # Add forecasting-specific metrics
    callbacks.append(ForecastingMetricsCallback())
    
    # Add gradient clipping if requested
    if kwargs.get('gradient_clipping', False):
        max_norm = kwargs.get('max_grad_norm', 1.0)
        callbacks.append(GradientClippingCallback(max_norm=max_norm))
    
    return callbacks