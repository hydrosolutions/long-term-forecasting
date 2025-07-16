"""
Meta-learning utilities for monthly discharge forecasting.

This module provides utility functions for meta-learning operations,
including feature engineering, model combination, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime
import warnings

from .evaluation_utils import calculate_all_metrics

logger = logging.getLogger(__name__)


def validate_prediction_format(predictions: pd.DataFrame, model_id: str = None) -> bool:
    """
    Validate that predictions DataFrame has the required format.
    
    Args:
        predictions: DataFrame with model predictions
        model_id: Model identifier for logging
        
    Returns:
        True if format is valid
        
    Raises:
        ValueError: If format is invalid
    """
    required_columns = ['date', 'code', 'Q_obs', 'Q_pred']
    missing_columns = [col for col in required_columns if col not in predictions.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns in predictions{' for ' + model_id if model_id else ''}: {missing_columns}")
    
    # Check for empty DataFrame
    if predictions.empty:
        raise ValueError(f"Empty predictions DataFrame{' for ' + model_id if model_id else ''}")
    
    # Check for required data types
    try:
        pd.to_datetime(predictions['date'])
    except Exception as e:
        raise ValueError(f"Invalid date format in predictions{' for ' + model_id if model_id else ''}: {e}")
    
    # Check for numeric columns
    numeric_columns = ['Q_obs', 'Q_pred']
    for col in numeric_columns:
        if not pd.api.types.is_numeric_dtype(predictions[col]):
            raise ValueError(f"Column {col} must be numeric in predictions{' for ' + model_id if model_id else ''}")
    
    return True


def align_predictions(
    predictions_dict: Dict[str, pd.DataFrame],
    alignment_method: str = 'inner',
    fill_missing: bool = False,
    fill_value: float = np.nan
) -> Dict[str, pd.DataFrame]:
    """
    Align predictions from multiple models to common index.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        alignment_method: 'inner' (intersection) or 'outer' (union)
        fill_missing: Whether to fill missing values
        fill_value: Value to use for filling missing predictions
        
    Returns:
        Dictionary of aligned predictions
    """
    if not predictions_dict:
        return {}
    
    # Validate all predictions
    for model_id, predictions in predictions_dict.items():
        validate_prediction_format(predictions, model_id)
    
    # Create common index
    all_indices = []
    for model_id, predictions in predictions_dict.items():
        model_index = pd.MultiIndex.from_frame(predictions[['date', 'code']])
        all_indices.append(model_index)
    
    if alignment_method == 'inner':
        # Find intersection of all indices
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.intersection(idx)
    else:  # outer
        # Find union of all indices
        common_index = all_indices[0]
        for idx in all_indices[1:]:
            common_index = common_index.union(idx)
    
    logger.info(f"Common index has {len(common_index)} entries after {alignment_method} alignment")
    
    # Align predictions to common index
    aligned_predictions = {}
    
    for model_id, predictions in predictions_dict.items():
        # Create aligned DataFrame
        aligned_df = pd.DataFrame(
            index=common_index,
            columns=['Q_obs', 'Q_pred']
        )
        
        # Fill with actual predictions
        for date, code in common_index:
            mask = (predictions['date'] == date) & (predictions['code'] == code)
            matching_rows = predictions[mask]
            
            if not matching_rows.empty:
                aligned_df.loc[(date, code), 'Q_obs'] = matching_rows['Q_obs'].iloc[0]
                aligned_df.loc[(date, code), 'Q_pred'] = matching_rows['Q_pred'].iloc[0]
            elif fill_missing:
                aligned_df.loc[(date, code), 'Q_pred'] = fill_value
        
        # Convert back to standard format
        aligned_df = aligned_df.reset_index()
        aligned_df = aligned_df.rename(columns={'level_0': 'date', 'level_1': 'code'})
        aligned_df['model'] = model_id
        
        aligned_predictions[model_id] = aligned_df
    
    return aligned_predictions


def create_meta_features(
    predictions_dict: Dict[str, pd.DataFrame],
    performance_metrics: Dict[str, Dict[str, float]] = None,
    basin_features: pd.DataFrame = None,
    include_temporal: bool = True,
    include_performance: bool = True
) -> pd.DataFrame:
    """
    Create meta-features for advanced meta-learning.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        performance_metrics: Historical performance metrics per model
        basin_features: Static basin characteristics
        include_temporal: Whether to include temporal features
        include_performance: Whether to include performance features
        
    Returns:
        DataFrame with meta-features
    """
    # Align predictions
    aligned_predictions = align_predictions(predictions_dict, alignment_method='inner')
    
    if not aligned_predictions:
        return pd.DataFrame()
    
    # Get common index
    first_model = list(aligned_predictions.keys())[0]
    common_df = aligned_predictions[first_model][['date', 'code', 'Q_obs']].copy()
    
    meta_features = []
    
    for idx, row in common_df.iterrows():
        date = row['date']
        code = row['code']
        obs = row['Q_obs']
        
        feature_dict = {
            'date': date,
            'code': code,
            'Q_obs': obs
        }
        
        # Base model predictions
        model_predictions = {}
        for model_id, predictions in aligned_predictions.items():
            mask = (predictions['date'] == date) & (predictions['code'] == code)
            matching_rows = predictions[mask]
            
            if not matching_rows.empty:
                pred_value = matching_rows['Q_pred'].iloc[0]
                model_predictions[model_id] = pred_value
                feature_dict[f'pred_{model_id}'] = pred_value
        
        # Statistical features from base predictions
        if len(model_predictions) > 1:
            pred_values = list(model_predictions.values())
            pred_array = np.array([v for v in pred_values if not np.isnan(v)])
            
            if len(pred_array) > 0:
                feature_dict['pred_mean'] = np.mean(pred_array)
                feature_dict['pred_std'] = np.std(pred_array)
                feature_dict['pred_median'] = np.median(pred_array)
                feature_dict['pred_min'] = np.min(pred_array)
                feature_dict['pred_max'] = np.max(pred_array)
                feature_dict['pred_range'] = np.max(pred_array) - np.min(pred_array)
                
                if len(pred_array) > 1:
                    feature_dict['pred_cv'] = np.std(pred_array) / np.mean(pred_array) if np.mean(pred_array) != 0 else 0
        
        # Temporal features
        if include_temporal:
            date_obj = pd.to_datetime(date)
            feature_dict['year'] = date_obj.year
            feature_dict['month'] = date_obj.month
            feature_dict['day_of_year'] = date_obj.dayofyear
            feature_dict['quarter'] = date_obj.quarter
            
            # Seasonal indicators
            feature_dict['is_winter'] = int(date_obj.month in [12, 1, 2])
            feature_dict['is_spring'] = int(date_obj.month in [3, 4, 5])
            feature_dict['is_summer'] = int(date_obj.month in [6, 7, 8])
            feature_dict['is_autumn'] = int(date_obj.month in [9, 10, 11])
        
        # Performance features
        if include_performance and performance_metrics:
            for model_id, perf_dict in performance_metrics.items():
                if isinstance(perf_dict, dict):
                    for metric_name, metric_value in perf_dict.items():
                        if not np.isnan(metric_value):
                            feature_dict[f'perf_{model_id}_{metric_name}'] = metric_value
        
        # Basin features
        if basin_features is not None and code in basin_features.index:
            basin_row = basin_features.loc[code]
            for col in basin_features.columns:
                feature_dict[f'basin_{col}'] = basin_row[col]
        
        meta_features.append(feature_dict)
    
    meta_features_df = pd.DataFrame(meta_features)
    
    logger.info(f"Created meta-features with {len(meta_features_df)} rows and {len(meta_features_df.columns)} columns")
    
    return meta_features_df


def ensemble_predictions(
    predictions_dict: Dict[str, pd.DataFrame],
    weights: Dict[str, float] = None,
    method: str = 'weighted_mean',
    handle_missing: str = 'skip'
) -> pd.DataFrame:
    """
    Create ensemble predictions from multiple models.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        weights: Model weights (if None, use uniform weights)
        method: Ensemble method ('mean', 'weighted_mean', 'median')
        handle_missing: How to handle missing predictions ('skip', 'zero', 'uniform')
        
    Returns:
        DataFrame with ensemble predictions
    """
    if not predictions_dict:
        return pd.DataFrame()
    
    # Validate predictions
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
        
        # Collect predictions from all models
        model_preds = {}
        for model_id, predictions in aligned_predictions.items():
            mask = (predictions['date'] == date) & (predictions['code'] == code)
            matching_rows = predictions[mask]
            
            if not matching_rows.empty:
                pred_value = matching_rows['Q_pred'].iloc[0]
                if not np.isnan(pred_value):
                    model_preds[model_id] = pred_value
        
        # Create ensemble prediction
        if model_preds:
            if method == 'mean':
                ensemble_pred = np.mean(list(model_preds.values()))
            elif method == 'median':
                ensemble_pred = np.median(list(model_preds.values()))
            elif method == 'weighted_mean':
                weighted_sum = 0.0
                weight_sum = 0.0
                
                for model_id, pred_value in model_preds.items():
                    if model_id in normalized_weights:
                        weighted_sum += normalized_weights[model_id] * pred_value
                        weight_sum += normalized_weights[model_id]
                
                if weight_sum > 0:
                    ensemble_pred = weighted_sum / weight_sum
                else:
                    ensemble_pred = np.nan
            else:
                raise ValueError(f"Unknown ensemble method: {method}")
            
            ensemble_preds.append(ensemble_pred)
        else:
            ensemble_preds.append(np.nan)
    
    ensemble_df['Q_pred'] = ensemble_preds
    ensemble_df['model'] = 'ensemble'
    
    logger.info(f"Created ensemble with {len(ensemble_df)} predictions using {method} method")
    
    return ensemble_df


def calculate_model_agreement(
    predictions_dict: Dict[str, pd.DataFrame],
    metric: str = 'correlation'
) -> pd.DataFrame:
    """
    Calculate agreement between model predictions.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        metric: Agreement metric ('correlation', 'rmse', 'mae')
        
    Returns:
        DataFrame with pairwise agreement metrics
    """
    # Align predictions
    aligned_predictions = align_predictions(predictions_dict, alignment_method='inner')
    
    if len(aligned_predictions) < 2:
        return pd.DataFrame()
    
    model_ids = list(aligned_predictions.keys())
    agreement_matrix = pd.DataFrame(index=model_ids, columns=model_ids)
    
    for i, model_i in enumerate(model_ids):
        for j, model_j in enumerate(model_ids):
            if i == j:
                agreement_matrix.loc[model_i, model_j] = 1.0 if metric == 'correlation' else 0.0
            else:
                pred_i = aligned_predictions[model_i]['Q_pred'].values
                pred_j = aligned_predictions[model_j]['Q_pred'].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(pred_i) | np.isnan(pred_j))
                pred_i_clean = pred_i[valid_mask]
                pred_j_clean = pred_j[valid_mask]
                
                if len(pred_i_clean) > 1:
                    if metric == 'correlation':
                        agreement_matrix.loc[model_i, model_j] = np.corrcoef(pred_i_clean, pred_j_clean)[0, 1]
                    elif metric == 'rmse':
                        agreement_matrix.loc[model_i, model_j] = np.sqrt(np.mean((pred_i_clean - pred_j_clean) ** 2))
                    elif metric == 'mae':
                        agreement_matrix.loc[model_i, model_j] = np.mean(np.abs(pred_i_clean - pred_j_clean))
                else:
                    agreement_matrix.loc[model_i, model_j] = np.nan
    
    return agreement_matrix


def analyze_prediction_diversity(
    predictions_dict: Dict[str, pd.DataFrame],
    diversity_metrics: List[str] = None
) -> Dict[str, float]:
    """
    Analyze diversity of predictions across models.
    
    Args:
        predictions_dict: Dictionary of {model_id: predictions_df}
        diversity_metrics: List of diversity metrics to calculate
        
    Returns:
        Dictionary with diversity metrics
    """
    if diversity_metrics is None:
        diversity_metrics = ['std', 'range', 'iqr', 'entropy']
    
    # Align predictions
    aligned_predictions = align_predictions(predictions_dict, alignment_method='inner')
    
    if len(aligned_predictions) < 2:
        return {}
    
    # Collect all predictions
    all_predictions = []
    for model_id, predictions in aligned_predictions.items():
        all_predictions.append(predictions['Q_pred'].values)
    
    pred_matrix = np.array(all_predictions).T  # Shape: (n_samples, n_models)
    
    diversity_results = {}
    
    for metric in diversity_metrics:
        if metric == 'std':
            # Average standard deviation across samples
            diversity_results[metric] = np.nanmean(np.nanstd(pred_matrix, axis=1))
        elif metric == 'range':
            # Average range across samples
            diversity_results[metric] = np.nanmean(np.nanmax(pred_matrix, axis=1) - np.nanmin(pred_matrix, axis=1))
        elif metric == 'iqr':
            # Average interquartile range across samples
            q75 = np.nanpercentile(pred_matrix, 75, axis=1)
            q25 = np.nanpercentile(pred_matrix, 25, axis=1)
            diversity_results[metric] = np.nanmean(q75 - q25)
        elif metric == 'entropy':
            # Approximate entropy based on prediction distributions
            # This is a simplified version - could be improved
            diversity_results[metric] = np.nanmean(np.nanstd(pred_matrix, axis=1) / np.nanmean(pred_matrix, axis=1))
    
    return diversity_results


def evaluate_meta_learning_performance(
    predictions_dict: Dict[str, pd.DataFrame],
    ensemble_predictions: pd.DataFrame,
    baseline_method: str = 'mean'
) -> Dict[str, Any]:
    """
    Evaluate meta-learning performance against baseline.
    
    Args:
        predictions_dict: Dictionary of base model predictions
        ensemble_predictions: Meta-learning ensemble predictions
        baseline_method: Baseline ensemble method for comparison
        
    Returns:
        Dictionary with performance comparison results
    """
    # Create baseline ensemble
    baseline_ensemble = ensemble_predictions(
        predictions_dict, 
        method=baseline_method
    )
    
    # Calculate metrics for individual models
    individual_metrics = {}
    for model_id, predictions in predictions_dict.items():
        individual_metrics[model_id] = calculate_all_metrics(
            predictions['Q_obs'], 
            predictions['Q_pred']
        )
    
    # Calculate metrics for baseline ensemble
    baseline_metrics = calculate_all_metrics(
        baseline_ensemble['Q_obs'], 
        baseline_ensemble['Q_pred']
    )
    
    # Calculate metrics for meta-learning ensemble
    meta_metrics = calculate_all_metrics(
        ensemble_predictions['Q_obs'], 
        ensemble_predictions['Q_pred']
    )
    
    # Calculate improvements
    improvements = {}
    for metric in meta_metrics:
        if metric in baseline_metrics:
            baseline_value = baseline_metrics[metric]
            meta_value = meta_metrics[metric]
            
            if baseline_value != 0:
                improvement = (meta_value - baseline_value) / baseline_value * 100
                improvements[metric] = improvement
    
    return {
        'individual_metrics': individual_metrics,
        'baseline_metrics': baseline_metrics,
        'meta_metrics': meta_metrics,
        'improvements': improvements,
        'best_individual': max(individual_metrics.keys(), 
                              key=lambda x: individual_metrics[x].get('nse', -np.inf))
    }


def cross_validate_meta_learning(
    predictions_dict: Dict[str, pd.DataFrame],
    meta_learner_class,
    cv_folds: int = 5,
    cv_method: str = 'time_series',
    **meta_learner_kwargs
) -> Dict[str, Any]:
    """
    Perform cross-validation for meta-learning approach.
    
    Args:
        predictions_dict: Dictionary of base model predictions
        meta_learner_class: Meta-learner class to evaluate
        cv_folds: Number of cross-validation folds
        cv_method: Cross-validation method ('time_series', 'random')
        **meta_learner_kwargs: Additional arguments for meta-learner
        
    Returns:
        Dictionary with cross-validation results
    """
    # This is a placeholder for cross-validation implementation
    # Full implementation would require more complex time series CV
    
    logger.info(f"Cross-validation for meta-learning with {cv_folds} folds")
    
    # For now, return basic structure
    return {
        'cv_method': cv_method,
        'cv_folds': cv_folds,
        'cv_scores': {},
        'mean_score': np.nan,
        'std_score': np.nan
    }