"""
Meta-Learning Framework for Monthly Discharge Forecasting

This module provides meta-learning capabilities for combining multiple base model
predictions through intelligent ensemble weighting and advanced meta-modeling techniques.

Classes:
    BaseMetaLearner: Abstract base class for meta-learning models
    HistoricalMetaLearner: Performance-based weighting using historical metrics
    AdvancedMetaLearner: Sophisticated meta-modeling with rich feature engineering
    DistributionalMetaLearner: Neural network for distributional predictions

The meta-learning framework is designed to be production-ready and operates
independently of development tools, ensuring operational deployment readiness.
"""

from .base_meta_learner import BaseMetaLearner
from .historical_meta_learner import HistoricalMetaLearner

__all__ = [
    "BaseMetaLearner",
    "HistoricalMetaLearner",
]

__version__ = "1.0.0"
