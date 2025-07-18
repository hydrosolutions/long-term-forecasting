"""
Meta-learning models for hydrological forecasting.

This module contains meta-learning approaches that combine predictions from multiple
base models to improve forecast accuracy. The meta-learners use historical performance
data to weight the contributions of different base models.
"""

from .base_meta_learner import BaseMetaLearner
from .historical_meta_learner import HistoricalMetaLearner

__all__ = ["BaseMetaLearner", "HistoricalMetaLearner"]