"""
Deep learning models for monthly discharge forecasting.

This package provides PyTorch-based deep learning models that inherit from 
BaseForecastModel and integrate seamlessly with the existing forecasting workflow.
"""

from .deep_regressor import DeepRegressor
from .deep_meta_learner import DeepMetaLearner

__all__ = ['DeepRegressor', 'DeepMetaLearner']