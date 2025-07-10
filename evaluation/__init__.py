"""
Evaluation module for monthly discharge forecasting.

This module provides comprehensive evaluation capabilities for forecasting models,
including individual model evaluation, ensemble creation, and dashboard integration.
"""

__version__ = "1.0.0"
__author__ = "Sandro Hunziker"

# Import main components
from .evaluate_models import calculate_metrics, evaluate_per_code, evaluate_per_month
from .prediction_loader import scan_prediction_files, load_predictions
from .ensemble_builder import create_family_ensemble, create_global_ensemble
from .evaluate_pipeline import run_evaluation_pipeline

__all__ = [
    "calculate_metrics",
    "evaluate_per_code",
    "evaluate_per_month",
    "scan_prediction_files",
    "load_predictions",
    "create_family_ensemble",
    "create_global_ensemble",
    "run_evaluation_pipeline",
]
