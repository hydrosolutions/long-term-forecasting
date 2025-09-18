"""
Deep learning models for monthly discharge forecasting.

This package provides PyTorch-based deep learning models that inherit from
BaseForecastModel and integrate seamlessly with the existing forecasting workflow.
"""

from .uncertainty_mixture import UncertaintyMixtureModel

__all__ = ["UncertaintyMixtureModel"]
