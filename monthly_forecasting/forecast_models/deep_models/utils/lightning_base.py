import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, List
import logging
from abc import abstractmethod

logger = logging.getLogger(__name__)


class LightningForecastBase(pl.LightningModule):
    """
    Base PyTorch Lightning module for deep learning forecasting models.
    
    Provides common functionality for training, validation, and testing
    across different neural network architectures.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer: str = 'adam',
        scheduler: Optional[str] = None,
        scheduler_params: Optional[Dict[str, Any]] = None,
        loss_function: str = 'mse',
        loss_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the Lightning module.
        
        Args:
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            optimizer: Optimizer type ('adam', 'adamw', 'sgd')
            scheduler: Learning rate scheduler ('step', 'cosine', 'plateau')
            scheduler_params: Parameters for scheduler
            loss_function: Loss function type ('mse', 'mae', 'huber', 'quantile', 'asymmetric_laplace')
            loss_params: Parameters for loss function
            **kwargs: Additional parameters
        """
        super().__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Training configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer
        self.scheduler_name = scheduler
        self.scheduler_params = scheduler_params or {}
        
        # Loss configuration
        self.loss_function_name = loss_function
        self.loss_params = loss_params or {}
        self.loss_function = self._create_loss_function()
        
        # Metrics storage
        self.training_outputs = []
        self.validation_outputs = []

    def _create_loss_function(self) -> nn.Module:
        """Create the loss function based on configuration."""
        if self.loss_function_name == 'mse':
            return nn.MSELoss()
        elif self.loss_function_name == 'mae':
            return nn.L1Loss()
        elif self.loss_function_name == 'huber':
            delta = self.loss_params.get('delta', 1.0)
            return nn.HuberLoss(delta=delta)
        elif self.loss_function_name == 'quantile':
            # Import from losses module when available
            from ..losses.quantile_loss import QuantileLoss
            quantiles = self.loss_params.get('quantiles', [0.5])
            return QuantileLoss(quantiles=quantiles)
        elif self.loss_function_name == 'asymmetric_laplace':
            # Import from losses module when available
            from ..losses.asymmetric_laplace_loss import AsymmetricLaplaceLoss
            return AsymmetricLaplaceLoss(**self.loss_params)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_function_name}")

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            batch: Dictionary containing:
                - x_past: (batch, past_time_steps, past_features)
                - x_nan_mask: (batch, past_time_steps, past_features)
                - x_future: (batch, future_time_steps, future_features)
                - x_now: (batch, 1, now_features)  
                - x_static: (batch, static_features)
                
        Returns:
            Dictionary containing model outputs:
                - predictions: (batch,) or (batch, n_quantiles)
                - uncertainty: (batch,) optional uncertainty estimates
        """
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        # Forward pass
        outputs = self.forward(batch)
        predictions = outputs['predictions']
        targets = batch['y']
        
        # Compute loss
        if self.loss_function_name in ['quantile', 'asymmetric_laplace']:
            # These loss functions handle multiple outputs
            loss = self.loss_function(predictions, targets.unsqueeze(-1))
        else:
            # Standard loss functions expect single output
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
            loss = self.loss_function(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end processing
        self.training_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        # Forward pass
        outputs = self.forward(batch)
        predictions = outputs['predictions']
        targets = batch['y']
        
        # Compute loss
        if self.loss_function_name in ['quantile', 'asymmetric_laplace']:
            loss = self.loss_function(predictions, targets.unsqueeze(-1))
        else:
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
            loss = self.loss_function(predictions, targets)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Store for epoch-end processing
        self.validation_outputs.append({
            'loss': loss.detach(),
            'predictions': predictions.detach(),
            'targets': targets.detach()
        })
        
        return loss

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step."""
        outputs = self.forward(batch)
        predictions = outputs['predictions']
        targets = batch['y']
        
        # Compute loss
        if self.loss_function_name in ['quantile', 'asymmetric_laplace']:
            loss = self.loss_function(predictions, targets.unsqueeze(-1))
        else:
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
            loss = self.loss_function(predictions, targets)
        
        # Log metrics
        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        return {
            'test_loss': loss,
            'predictions': predictions,
            'targets': targets
        }

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Prediction step."""
        outputs = self.forward(batch)
        
        # Include metadata in outputs
        outputs['metadata'] = batch['metadata']
        
        return outputs

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""
        if self.training_outputs:
            # Compute epoch-level metrics
            all_predictions = torch.cat([x['predictions'] for x in self.training_outputs])
            all_targets = torch.cat([x['targets'] for x in self.training_outputs])
            
            # Compute R² score
            if all_predictions.dim() > 1:
                # For multi-output models, use mean prediction
                all_predictions = all_predictions.mean(dim=-1)
            
            ss_res = torch.sum((all_targets - all_predictions) ** 2)
            ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            self.log('train_r2', r2, on_epoch=True)
            
            # Clear outputs
            self.training_outputs.clear()

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if self.validation_outputs:
            # Compute epoch-level metrics
            all_predictions = torch.cat([x['predictions'] for x in self.validation_outputs])
            all_targets = torch.cat([x['targets'] for x in self.validation_outputs])
            
            # Compute R² score
            if all_predictions.dim() > 1:
                all_predictions = all_predictions.mean(dim=-1)
            
            ss_res = torch.sum((all_targets - all_predictions) ** 2)
            ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
            r2 = 1 - ss_res / ss_tot
            
            self.log('val_r2', r2, on_epoch=True, prog_bar=True)
            
            # Clear outputs
            self.validation_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizers and learning rate schedulers."""
        # Create optimizer
        if self.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.scheduler_params.get('momentum', 0.9)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Add scheduler if specified
        if self.scheduler_name is None:
            return optimizer
        
        if self.scheduler_name.lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_params.get('step_size', 10),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.scheduler_name.lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_params.get('T_max', 100),
                eta_min=self.scheduler_params.get('eta_min', 0)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'epoch'
                }
            }
        elif self.scheduler_name.lower() == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_params.get('factor', 0.5),
                patience=self.scheduler_params.get('patience', 5),
                verbose=True
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val_loss',
                    'interval': 'epoch'
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler_name}")


class LightningMetaForecastBase(LightningForecastBase):
    """
    Base class for meta-learning forecasting models.
    
    Extends LightningForecastBase with meta-learning specific functionality
    such as handling base model predictions and uncertainty quantification.
    """

    def __init__(
        self,
        base_model_dim: int = 0,
        base_model_error_dim: int = 0,
        uncertainty_output: bool = False,
        **kwargs
    ):
        """
        Initialize the meta-learning Lightning module.
        
        Args:
            base_model_dim: Dimension of base model predictions
            base_model_error_dim: Dimension of base model error features
            uncertainty_output: Whether to output uncertainty estimates
            **kwargs: Additional parameters for parent class
        """
        super().__init__(**kwargs)
        
        self.base_model_dim = base_model_dim
        self.base_model_error_dim = base_model_error_dim
        self.uncertainty_output = uncertainty_output

    @abstractmethod
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass for meta-learning model.
        
        Args:
            batch: Dictionary containing all inputs including base model predictions
                
        Returns:
            Dictionary containing:
                - predictions: Point predictions or uncertainty parameters
                - uncertainty: Uncertainty estimates (if applicable)
                - quantiles: Quantile predictions (if applicable)
        """
        pass

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with meta-learning specific handling."""
        # Forward pass
        outputs = self.forward(batch)
        targets = batch['y']
        
        # Handle different output types
        if 'quantiles' in outputs:
            # Quantile regression
            quantiles = outputs['quantiles']
            loss = self.loss_function(quantiles, targets.unsqueeze(-1))
        elif 'mu' in outputs and 'sigma' in outputs:
            # Uncertainty prediction (e.g., for Asymmetric Laplace)
            mu = outputs['mu']
            sigma = outputs['sigma']
            loss = self.loss_function(mu, sigma, targets)
        else:
            # Standard prediction
            predictions = outputs['predictions']
            if predictions.dim() > 1:
                predictions = predictions.squeeze(-1)
            loss = self.loss_function(predictions, targets)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss