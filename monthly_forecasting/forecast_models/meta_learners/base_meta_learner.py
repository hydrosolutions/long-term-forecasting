"""
Base Meta-Learner class for monthly discharge forecasting.

This module provides the abstract base class for all meta-learning models,
defining the interface and common functionality for meta-learning approaches.
"""

import os
import pandas as pd
import numpy as np
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging

from monthly_forecasting.forecast_models.base_class import BaseForecastModel
from monthly_forecasting.scr.evaluation_utils import calculate_all_metrics

logger = logging.getLogger(__name__)


class BaseMetaLearner(BaseForecastModel):
    """
    Abstract base class for meta-learning models.
    
    Meta-learners combine predictions from multiple base models using
    intelligent weighting strategies or advanced meta-modeling techniques.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
        base_model_predictions: Dict[str, pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the meta-learner.
        
        Args:
            data: Historical data for training
            static_data: Static basin characteristics
            general_config: General configuration
            model_config: Meta-learning specific configuration
            feature_config: Feature engineering configuration
            path_config: Path configuration
            base_model_predictions: Dictionary of base model predictions
        """
        super().__init__(data, static_data, general_config, model_config, feature_config, path_config)
        
        # Meta-learning specific attributes
        self.base_model_predictions = base_model_predictions or {}
        self.meta_model = None
        self.performance_weights = {}
        self.feature_importance = {}
        
        # Configuration
        self.meta_learning_config = model_config.get('meta_learning', {})
        self.ensemble_method = self.meta_learning_config.get('ensemble_method', 'weighted_mean')
        self.weighting_strategy = self.meta_learning_config.get('weighting_strategy', 'performance_based')
        self.performance_metric = self.meta_learning_config.get('performance_metric', 'rmse')
        
        # Validation
        self._validate_configuration()
        
        logger.info(f"Initialized {self.__class__.__name__} with {len(self.base_model_predictions)} base models")
    
    def _validate_configuration(self) -> None:
        """Validate meta-learning configuration."""
        valid_methods = ['mean', 'weighted_mean', 'meta_model']
        if self.ensemble_method not in valid_methods:
            raise ValueError(f"Invalid ensemble_method: {self.ensemble_method}. Must be one of {valid_methods}")
        
        valid_strategies = ['performance_based', 'uniform', 'basin_specific', 'temporal']
        if self.weighting_strategy not in valid_strategies:
            raise ValueError(f"Invalid weighting_strategy: {self.weighting_strategy}. Must be one of {valid_strategies}")
        
        valid_metrics = ['rmse', 'r2', 'nse', 'mae', 'kge']
        if self.performance_metric not in valid_metrics:
            raise ValueError(f"Invalid performance_metric: {self.performance_metric}. Must be one of {valid_metrics}")
    
    def add_base_model_predictions(self, model_id: str, predictions: pd.DataFrame) -> None:
        """
        Add predictions from a base model.
        
        Args:
            model_id: Unique identifier for the base model
            predictions: DataFrame with predictions (columns: date, code, Q_obs, Q_pred)
        """
        required_columns = ['date', 'code', 'Q_obs', 'Q_pred']
        missing_columns = [col for col in required_columns if col not in predictions.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns in predictions: {missing_columns}")
        
        self.base_model_predictions[model_id] = predictions.copy()
        logger.info(f"Added predictions for base model: {model_id}")
    
    def get_base_model_ids(self) -> List[str]:
        """Get list of base model IDs."""
        return list(self.base_model_predictions.keys())
    
    def get_common_prediction_index(self) -> pd.MultiIndex:
        """
        Get common prediction index across all base models.
        
        Returns:
            MultiIndex with (date, code) combinations present in all base models
        """
        if not self.base_model_predictions:
            return pd.MultiIndex.from_tuples([], names=['date', 'code'])
        
        # Get common index across all base models
        common_index = None
        for model_id, predictions in self.base_model_predictions.items():
            model_index = pd.MultiIndex.from_frame(predictions[['date', 'code']])
            
            if common_index is None:
                common_index = model_index
            else:
                common_index = common_index.intersection(model_index)
        
        return common_index
    
    def calculate_base_model_performance(
        self,
        model_id: str,
        metric: str = None,
        group_by: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a base model.
        
        Args:
            model_id: Base model identifier
            metric: Specific metric to calculate (if None, calculate all)
            group_by: List of columns to group by (e.g., ['code', 'month'])
            
        Returns:
            Dictionary of performance metrics
        """
        if model_id not in self.base_model_predictions:
            raise ValueError(f"Base model {model_id} not found")
        
        predictions = self.base_model_predictions[model_id]
        
        if group_by:
            # Calculate metrics for each group
            def calculate_group_metrics(group_df):
                return pd.Series(calculate_all_metrics(
                    group_df['Q_obs'], 
                    group_df['Q_pred']
                ))
            
            grouped_metrics = predictions.groupby(group_by).apply(calculate_group_metrics)
            return grouped_metrics.to_dict()
        else:
            # Calculate overall metrics
            metrics = calculate_all_metrics(predictions['Q_obs'], predictions['Q_pred'])
            return metrics if metric is None else {metric: metrics[metric]}
    
    def calculate_all_base_model_performance(
        self,
        metric: str = None,
        group_by: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for all base models.
        
        Args:
            metric: Specific metric to calculate (if None, calculate all)
            group_by: List of columns to group by
            
        Returns:
            Nested dictionary {model_id: {metric: value}}
        """
        all_performance = {}
        
        for model_id in self.get_base_model_ids():
            all_performance[model_id] = self.calculate_base_model_performance(
                model_id, metric, group_by
            )
        
        return all_performance
    
    @abstractmethod
    def compute_weights(self, **kwargs) -> Dict[str, float]:
        """
        Compute weights for ensemble combination.
        
        Returns:
            Dictionary mapping model IDs to weights
        """
        pass
    
    @abstractmethod
    def train_meta_model(self, **kwargs) -> None:
        """
        Train the meta-model (if using meta-modeling approach).
        """
        pass
    
    def create_ensemble_predictions(
        self,
        weights: Dict[str, float] = None,
        common_index_only: bool = True
    ) -> pd.DataFrame:
        """
        Create ensemble predictions from base models.
        
        Args:
            weights: Model weights (if None, compute using configured method)
            common_index_only: Whether to use only common prediction index
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not self.base_model_predictions:
            raise ValueError("No base model predictions available")
        
        if weights is None:
            weights = self.compute_weights()
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total weight is zero")
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Get prediction index
        if common_index_only:
            pred_index = self.get_common_prediction_index()
        else:
            # Use union of all indices
            all_indices = []
            for predictions in self.base_model_predictions.values():
                all_indices.append(pd.MultiIndex.from_frame(predictions[['date', 'code']]))
            pred_index = all_indices[0].union_many(all_indices[1:]) if len(all_indices) > 1 else all_indices[0]
        
        if pred_index.empty:
            raise ValueError("No common prediction index found")
        
        # Create ensemble predictions
        ensemble_predictions = []
        observations = []
        
        for date, code in pred_index:
            weighted_pred = 0.0
            obs_value = None
            weight_sum = 0.0
            
            for model_id, weight in normalized_weights.items():
                if model_id in self.base_model_predictions:
                    model_preds = self.base_model_predictions[model_id]
                    
                    # Find matching prediction
                    mask = (model_preds['date'] == date) & (model_preds['code'] == code)
                    matching_preds = model_preds[mask]
                    
                    if not matching_preds.empty:
                        pred_value = matching_preds['Q_pred'].iloc[0]
                        if not np.isnan(pred_value):
                            weighted_pred += weight * pred_value
                            weight_sum += weight
                        
                        # Get observation (should be same across models)
                        if obs_value is None:
                            obs_value = matching_preds['Q_obs'].iloc[0]
            
            # Renormalize if some models didn't have predictions
            if weight_sum > 0:
                ensemble_predictions.append(weighted_pred / weight_sum)
            else:
                ensemble_predictions.append(np.nan)
            
            observations.append(obs_value)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'date': [idx[0] for idx in pred_index],
            'code': [idx[1] for idx in pred_index],
            'Q_obs': observations,
            'Q_pred': ensemble_predictions,
            'model': [self.name] * len(pred_index)
        })
        
        return result_df
    
    def evaluate_ensemble_performance(
        self,
        ensemble_predictions: pd.DataFrame = None,
        metric: str = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.
        
        Args:
            ensemble_predictions: Ensemble predictions DataFrame
            metric: Specific metric to calculate
            
        Returns:
            Dictionary of performance metrics
        """
        if ensemble_predictions is None:
            ensemble_predictions = self.create_ensemble_predictions()
        
        metrics = calculate_all_metrics(
            ensemble_predictions['Q_obs'], 
            ensemble_predictions['Q_pred']
        )
        
        return metrics if metric is None else {metric: metrics[metric]}
    
    def save_model(self) -> None:
        """Save meta-learning model to file."""
        # Implementation depends on specific meta-learning approach
        logger.info(f"Saving {self.__class__.__name__} model")
        # TODO: Implement model saving
        pass
    
    def load_model(self) -> None:
        """Load meta-learning model from file."""
        # Implementation depends on specific meta-learning approach
        logger.info(f"Loading {self.__class__.__name__} model")
        # TODO: Implement model loading
        pass