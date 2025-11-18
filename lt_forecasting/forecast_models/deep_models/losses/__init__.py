"""
Loss functions for deep learning forecasting models.
"""

from .quantile_loss import QuantileLoss
from .asymmetric_laplace_loss import AsymmetricLaplaceLoss

__all__ = ["QuantileLoss", "AsymmetricLaplaceLoss"]
