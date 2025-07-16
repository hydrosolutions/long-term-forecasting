"""
Production ensemble utilities for monthly discharge forecasting.

This module provides utilities for creating and managing ensemble predictions
in a production environment, independent of development tools.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
import logging
from datetime import datetime
import warnings

from .evaluation_utils import calculate_all_metrics
from .meta_utils import validate_prediction_format, align_predictions

logger = logging.getLogger(__name__)


class EnsembleBuilder:
    """
    Build ensemble predictions from multiple base models.
    
    This class provides various methods for combining predictions from
    multiple models into ensemble predictions.
    """
    
    def __init__(self, ensemble_method: str = 'mean', handle_missing: str = 'skip'):
        """
        Initialize ensemble builder.
        
        Args:
            ensemble_method: Method for combining predictions ('mean', 'weighted_mean', 'median')
            handle_missing: How to handle missing predictions ('skip', 'zero', 'uniform')
        """
        self.ensemble_method = ensemble_method
        self.handle_missing = handle_missing
        
        # Validate ensemble method
        valid_methods = ['mean', 'weighted_mean', 'median', 'max', 'min']
        if ensemble_method not in valid_methods:
            raise ValueError(f"Invalid ensemble method: {ensemble_method}. Must be one of {valid_methods}")
        
        # Validate missing handling
        valid_missing = ['skip', 'zero', 'uniform']
        if handle_missing not in valid_missing:
            raise ValueError(f"Invalid missing handling: {handle_missing}. Must be one of {valid_missing}")
        
        logger.info(f"Initialized EnsembleBuilder with method={ensemble_method}, missing={handle_missing}")
    
    def create_simple_ensemble(
        self,
        predictions_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float] = None,
        alignment: str = 'inner'
    ) -> pd.DataFrame:
        """
        Create simple ensemble from multiple predictions.
        
        Args:
            predictions_dict: Dictionary of {model_id: predictions_df}
            weights: Model weights (if None, use uniform weights)
            alignment: How to align predictions ('inner', 'outer')
            
        Returns:
            DataFrame with ensemble predictions
        """
        if not predictions_dict:
            return pd.DataFrame()
        
        # Validate all predictions
        for model_id, predictions in predictions_dict.items():
            validate_prediction_format(predictions, model_id)
        
        # Set default weights
        if weights is None:
            weights = {model_id: 1.0 for model_id in predictions_dict.keys()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total weight is zero")
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Align predictions
        aligned_predictions = align_predictions(
            predictions_dict, 
            alignment_method=alignment,
            fill_missing=(self.handle_missing != 'skip')
        )
        
        if not aligned_predictions:
            return pd.DataFrame()
        
        # Get common structure
        first_model = list(aligned_predictions.keys())[0]
        ensemble_df = aligned_predictions[first_model][['date', 'code', 'Q_obs']].copy()
        
        ensemble_preds = []
        ensemble_weights = []
        
        for idx, row in ensemble_df.iterrows():
            date = row['date']
            code = row['code']
            
            # Collect predictions from all models
            model_preds = []
            model_weights = []
            
            for model_id, predictions in aligned_predictions.items():
                mask = (predictions['date'] == date) & (predictions['code'] == code)
                matching_rows = predictions[mask]
                
                if not matching_rows.empty:
                    pred_value = matching_rows['Q_pred'].iloc[0]
                    
                    if not np.isnan(pred_value):
                        model_preds.append(pred_value)
                        model_weights.append(normalized_weights.get(model_id, 0.0))
                    elif self.handle_missing == 'zero':
                        model_preds.append(0.0)
                        model_weights.append(normalized_weights.get(model_id, 0.0))
            
            # Create ensemble prediction
            if model_preds:
                if self.ensemble_method == 'mean':
                    ensemble_pred = np.mean(model_preds)
                elif self.ensemble_method == 'weighted_mean':
                    if sum(model_weights) > 0:
                        ensemble_pred = np.average(model_preds, weights=model_weights)
                    else:
                        ensemble_pred = np.mean(model_preds)
                elif self.ensemble_method == 'median':
                    ensemble_pred = np.median(model_preds)
                elif self.ensemble_method == 'max':
                    ensemble_pred = np.max(model_preds)
                elif self.ensemble_method == 'min':
                    ensemble_pred = np.min(model_preds)
                else:
                    ensemble_pred = np.mean(model_preds)
                
                ensemble_preds.append(ensemble_pred)
                ensemble_weights.append(len(model_preds))
            else:
                ensemble_preds.append(np.nan)
                ensemble_weights.append(0)
        
        ensemble_df['Q_pred'] = ensemble_preds
        ensemble_df['n_models'] = ensemble_weights
        ensemble_df['model'] = 'ensemble'
        
        logger.info(f"Created simple ensemble with {len(ensemble_df)} predictions using {self.ensemble_method}")
        
        return ensemble_df
    
    def create_weighted_ensemble(
        self,
        predictions_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float],
        normalize_weights: bool = True
    ) -> pd.DataFrame:
        """
        Create weighted ensemble from multiple predictions.
        
        Args:
            predictions_dict: Dictionary of {model_id: predictions_df}
            weights: Model weights
            normalize_weights: Whether to normalize weights to sum to 1
            
        Returns:
            DataFrame with weighted ensemble predictions
        """
        if normalize_weights:
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in weights.items()}
        
        return self.create_simple_ensemble(predictions_dict, weights)
    
    def create_performance_weighted_ensemble(
        self,
        predictions_dict: Dict[str, pd.DataFrame],
        performance_metrics: Dict[str, float],
        metric_type: str = 'error',
        epsilon: float = 1e-10
    ) -> pd.DataFrame:
        """
        Create ensemble weighted by performance metrics.
        
        Args:
            predictions_dict: Dictionary of {model_id: predictions_df}
            performance_metrics: Dictionary of {model_id: performance_value}
            metric_type: Type of metric ('error' or 'skill')
            epsilon: Small value to avoid division by zero
            
        Returns:
            DataFrame with performance-weighted ensemble predictions
        """
        if not performance_metrics:
            return self.create_simple_ensemble(predictions_dict)
        
        # Calculate weights based on performance
        weights = {}
        
        if metric_type == 'error':
            # For error metrics, use inverse weighting (lower error = higher weight)
            for model_id in predictions_dict.keys():
                if model_id in performance_metrics:
                    perf_value = performance_metrics[model_id]
                    if not np.isnan(perf_value) and np.isfinite(perf_value):
                        weights[model_id] = 1.0 / (perf_value + epsilon)
                    else:
                        weights[model_id] = 0.0
                else:
                    weights[model_id] = 0.0
        else:  # skill
            # For skill metrics, use direct weighting (higher skill = higher weight)
            for model_id in predictions_dict.keys():
                if model_id in performance_metrics:
                    perf_value = performance_metrics[model_id]
                    if not np.isnan(perf_value) and np.isfinite(perf_value) and perf_value > 0:
                        weights[model_id] = perf_value
                    else:
                        weights[model_id] = 0.0
                else:
                    weights[model_id] = 0.0
        
        # Fallback to uniform weights if all weights are zero
        if sum(weights.values()) == 0:
            weights = {model_id: 1.0 for model_id in predictions_dict.keys()}
        
        return self.create_weighted_ensemble(predictions_dict, weights)
    
    def create_adaptive_ensemble(
        self,
        predictions_dict: Dict[str, pd.DataFrame],
        basin_performance: Dict[str, Dict[str, float]] = None,
        temporal_performance: Dict[str, Dict[int, float]] = None,
        fallback_uniform: bool = True
    ) -> pd.DataFrame:
        """
        Create adaptive ensemble with basin-specific and temporal weighting.
        
        Args:
            predictions_dict: Dictionary of {model_id: predictions_df}
            basin_performance: Performance per basin {model_id: {basin_code: performance}}
            temporal_performance: Performance per month {model_id: {month: performance}}
            fallback_uniform: Whether to use uniform weights as fallback
            
        Returns:
            DataFrame with adaptive ensemble predictions
        """
        if not predictions_dict:
            return pd.DataFrame()
        
        # Align predictions
        aligned_predictions = align_predictions(predictions_dict, alignment_method='inner')
        
        if not aligned_predictions:
            return pd.DataFrame()
        
        # Get common structure
        first_model = list(aligned_predictions.keys())[0]
        ensemble_df = aligned_predictions[first_model][['date', 'code', 'Q_obs']].copy()
        
        ensemble_preds = []
        
        for idx, row in ensemble_df.iterrows():
            date = row['date']
            code = row['code']
            month = pd.to_datetime(date).month
            
            # Calculate adaptive weights for this specific context
            context_weights = {}
            
            for model_id in aligned_predictions.keys():
                weight = 1.0  # Base weight
                
                # Basin-specific weight
                if basin_performance and model_id in basin_performance:
                    basin_perf = basin_performance[model_id].get(code, None)
                    if basin_perf is not None and not np.isnan(basin_perf):
                        weight *= basin_perf
                
                # Temporal weight
                if temporal_performance and model_id in temporal_performance:
                    temporal_perf = temporal_performance[model_id].get(month, None)
                    if temporal_perf is not None and not np.isnan(temporal_perf):
                        weight *= temporal_perf
                
                context_weights[model_id] = weight
            
            # Normalize weights
            total_weight = sum(context_weights.values())
            if total_weight > 0:
                context_weights = {k: v / total_weight for k, v in context_weights.items()}
            elif fallback_uniform:
                context_weights = {k: 1.0 / len(context_weights) for k in context_weights.keys()}
            
            # Create weighted prediction for this context
            weighted_pred = 0.0
            total_weight = 0.0
            
            for model_id, predictions in aligned_predictions.items():
                mask = (predictions['date'] == date) & (predictions['code'] == code)
                matching_rows = predictions[mask]
                
                if not matching_rows.empty:
                    pred_value = matching_rows['Q_pred'].iloc[0]
                    
                    if not np.isnan(pred_value) and model_id in context_weights:
                        weighted_pred += context_weights[model_id] * pred_value
                        total_weight += context_weights[model_id]
            
            if total_weight > 0:
                ensemble_preds.append(weighted_pred / total_weight)
            else:
                ensemble_preds.append(np.nan)
        
        ensemble_df['Q_pred'] = ensemble_preds
        ensemble_df['model'] = 'adaptive_ensemble'
        
        logger.info(f"Created adaptive ensemble with {len(ensemble_df)} predictions")
        
        return ensemble_df


def create_family_ensemble(
    predictions_dict: Dict[str, pd.DataFrame],
    family_mapping: Dict[str, str],
    family_name: str,
    ensemble_method: str = 'mean'
) -> Optional[pd.DataFrame]:
    """
    Create ensemble for a specific model family.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        family_mapping: Dictionary mapping {model_id: family_name}
        family_name: Name of the family to create ensemble for
        ensemble_method: Ensemble method to use
        
    Returns:
        DataFrame with family ensemble predictions or None if no models found
    """
    # Filter predictions for this family
    family_predictions = {}
    
    for model_id, predictions in predictions_dict.items():
        if family_mapping.get(model_id) == family_name:
            family_predictions[model_id] = predictions
    
    if not family_predictions:
        logger.warning(f"No models found for family: {family_name}")
        return None
    
    # Create ensemble
    ensemble_builder = EnsembleBuilder(ensemble_method=ensemble_method)
    ensemble_df = ensemble_builder.create_simple_ensemble(family_predictions)
    
    if not ensemble_df.empty:
        ensemble_df['model'] = f"{family_name}_ensemble"
        ensemble_df['family'] = family_name
        ensemble_df['ensemble_type'] = 'family'
        
        logger.info(f"Created {family_name} family ensemble with {len(ensemble_df)} predictions")
    
    return ensemble_df


def create_global_ensemble(
    predictions_dict: Dict[str, pd.DataFrame],
    ensemble_method: str = 'mean',
    use_all_models: bool = True
) -> Optional[pd.DataFrame]:
    """
    Create global ensemble from all available models.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        ensemble_method: Ensemble method to use
        use_all_models: Whether to use all models (if False, could implement selection)
        
    Returns:
        DataFrame with global ensemble predictions or None if no models found
    """
    if not predictions_dict:
        logger.warning("No models available for global ensemble")
        return None
    
    # Create ensemble
    ensemble_builder = EnsembleBuilder(ensemble_method=ensemble_method)
    ensemble_df = ensemble_builder.create_simple_ensemble(predictions_dict)
    
    if not ensemble_df.empty:
        ensemble_df['model'] = 'global_ensemble'
        ensemble_df['ensemble_type'] = 'global'
        
        logger.info(f"Created global ensemble with {len(ensemble_df)} predictions from {len(predictions_dict)} models")
    
    return ensemble_df


def evaluate_ensemble_performance(
    ensemble_predictions: pd.DataFrame,
    base_predictions: Dict[str, pd.DataFrame],
    metrics: List[str] = None
) -> Dict[str, Any]:
    """
    Evaluate ensemble performance against base models.
    
    Args:
        ensemble_predictions: DataFrame with ensemble predictions
        base_predictions: Dictionary of base model predictions
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary with performance comparison
    """
    if metrics is None:
        metrics = ['r2', 'rmse', 'nse', 'mae', 'kge']
    
    # Calculate ensemble metrics
    ensemble_metrics = calculate_all_metrics(
        ensemble_predictions['Q_obs'],
        ensemble_predictions['Q_pred']
    )
    
    # Calculate base model metrics
    base_metrics = {}
    for model_id, predictions in base_predictions.items():
        base_metrics[model_id] = calculate_all_metrics(
            predictions['Q_obs'],
            predictions['Q_pred']
        )
    
    # Calculate improvements
    improvements = {}
    for metric in metrics:
        if metric in ensemble_metrics:
            ensemble_value = ensemble_metrics[metric]
            base_values = [base_metrics[model_id].get(metric, np.nan) for model_id in base_metrics.keys()]
            base_values = [v for v in base_values if not np.isnan(v)]
            
            if base_values:
                best_base = max(base_values) if metric in ['r2', 'nse', 'kge'] else min(base_values)
                avg_base = np.mean(base_values)
                
                if metric in ['r2', 'nse', 'kge']:
                    # Higher is better
                    improvements[f'{metric}_vs_best'] = (ensemble_value - best_base) / best_base * 100 if best_base != 0 else 0
                    improvements[f'{metric}_vs_avg'] = (ensemble_value - avg_base) / avg_base * 100 if avg_base != 0 else 0
                else:
                    # Lower is better
                    improvements[f'{metric}_vs_best'] = (best_base - ensemble_value) / best_base * 100 if best_base != 0 else 0
                    improvements[f'{metric}_vs_avg'] = (avg_base - ensemble_value) / avg_base * 100 if avg_base != 0 else 0
    
    return {
        'ensemble_metrics': ensemble_metrics,
        'base_metrics': base_metrics,
        'improvements': improvements,
        'n_base_models': len(base_predictions),
        'n_ensemble_predictions': len(ensemble_predictions)
    }


def optimize_ensemble_weights(
    predictions_dict: Dict[str, pd.DataFrame],
    optimization_metric: str = 'nse',
    method: str = 'grid_search',
    cv_folds: int = 5
) -> Dict[str, float]:
    """
    Optimize ensemble weights using cross-validation.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        optimization_metric: Metric to optimize for
        method: Optimization method ('grid_search', 'random_search')
        cv_folds: Number of cross-validation folds
        
    Returns:
        Dictionary with optimal weights
    """
    if len(predictions_dict) < 2:
        return {model_id: 1.0 for model_id in predictions_dict.keys()}
    
    model_ids = list(predictions_dict.keys())
    
    # Simple grid search for now (can be extended)
    best_weights = None
    best_score = -np.inf if optimization_metric in ['r2', 'nse', 'kge'] else np.inf
    
    # Generate weight combinations
    from itertools import product
    
    weight_options = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    for weights_tuple in product(weight_options, repeat=len(model_ids)):
        if sum(weights_tuple) == 0:
            continue
        
        # Normalize weights
        total_weight = sum(weights_tuple)
        normalized_weights = {model_ids[i]: weights_tuple[i] / total_weight for i in range(len(model_ids))}
        
        # Create ensemble with these weights
        ensemble_builder = EnsembleBuilder(ensemble_method='weighted_mean')
        ensemble_df = ensemble_builder.create_weighted_ensemble(predictions_dict, normalized_weights)
        
        if not ensemble_df.empty:
            # Calculate performance
            metrics = calculate_all_metrics(ensemble_df['Q_obs'], ensemble_df['Q_pred'])
            score = metrics.get(optimization_metric, np.nan)
            
            if not np.isnan(score):
                if optimization_metric in ['r2', 'nse', 'kge']:
                    # Higher is better
                    if score > best_score:
                        best_score = score
                        best_weights = normalized_weights
                else:
                    # Lower is better
                    if score < best_score:
                        best_score = score
                        best_weights = normalized_weights
    
    if best_weights is None:
        # Fallback to uniform weights
        best_weights = {model_id: 1.0 / len(model_ids) for model_id in model_ids}
    
    logger.info(f"Optimized ensemble weights: {best_weights}, best {optimization_metric}: {best_score:.4f}")
    
    return best_weights


def create_ensemble_summary(
    ensemble_predictions: pd.DataFrame,
    base_predictions: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create summary of ensemble performance.
    
    Args:
        ensemble_predictions: DataFrame with ensemble predictions
        base_predictions: Dictionary of base model predictions
        
    Returns:
        DataFrame with ensemble summary
    """
    summary_data = []
    
    # Ensemble summary
    ensemble_metrics = calculate_all_metrics(
        ensemble_predictions['Q_obs'],
        ensemble_predictions['Q_pred']
    )
    
    summary_data.append({
        'model': 'ensemble',
        'type': 'ensemble',
        'n_predictions': len(ensemble_predictions),
        'n_codes': ensemble_predictions['code'].nunique(),
        **ensemble_metrics
    })
    
    # Base model summaries
    for model_id, predictions in base_predictions.items():
        base_metrics = calculate_all_metrics(
            predictions['Q_obs'],
            predictions['Q_pred']
        )
        
        summary_data.append({
            'model': model_id,
            'type': 'base',
            'n_predictions': len(predictions),
            'n_codes': predictions['code'].nunique(),
            **base_metrics
        })
    
    return pd.DataFrame(summary_data)