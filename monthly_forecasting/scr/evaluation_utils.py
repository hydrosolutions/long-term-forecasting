"""
Production-ready evaluation utilities for monthly discharge forecasting.

This module provides metric calculation functions for model evaluation,
designed to be production-ready and independent of development tools.
"""

import logging
import numpy as np
import pandas as pd
from typing import Union, Tuple, Dict, Any

logger = logging.getLogger(__name__)


def _validate_inputs(observed: np.ndarray, predicted: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and clean input arrays for metric calculation.
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        Tuple of cleaned observed and predicted arrays
        
    Raises:
        ValueError: If inputs are invalid
    """
    # Convert to numpy arrays
    observed = np.asarray(observed, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    
    # Check array shapes
    if observed.shape != predicted.shape:
        raise ValueError(f"Shape mismatch: observed {observed.shape} vs predicted {predicted.shape}")
    
    # Check for empty arrays
    if observed.size == 0 or predicted.size == 0:
        raise ValueError("Empty arrays provided")
    
    # Create mask for valid (non-NaN, non-inf) values
    mask = ~(np.isnan(observed) | np.isnan(predicted) | 
             np.isinf(observed) | np.isinf(predicted))
    
    if not np.any(mask):
        raise ValueError("No valid (non-NaN, non-inf) data points found")
    
    # Filter arrays
    observed_clean = observed[mask]
    predicted_clean = predicted[mask]
    
    if len(observed_clean) < 2:
        raise ValueError("At least 2 valid data points required for metric calculation")
    
    return observed_clean, predicted_clean


def r2_score(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate R-squared (coefficient of determination).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        R-squared score
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate R-squared
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        
        if ss_tot < 1e-10:
            logger.warning("Zero variance in observed data, R2 undefined")
            return np.nan
        
        r2 = 1 - (ss_res / ss_tot)
        return float(r2)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in r2_score calculation: {str(e)}")
        return np.nan


def rmse(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Root Mean Square Error (RMSE).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        RMSE value
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate RMSE
        mse = np.mean((obs - pred) ** 2)
        rmse_value = np.sqrt(mse)
        
        return float(rmse_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in rmse calculation: {str(e)}")
        return np.nan


def mae(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Error (MAE).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        MAE value
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate MAE
        mae_value = np.mean(np.abs(obs - pred))
        
        return float(mae_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in mae calculation: {str(e)}")
        return np.nan


def nse(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        NSE value
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate NSE
        obs_mean = np.mean(obs)
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - obs_mean) ** 2)
        
        if ss_tot < 1e-10:
            logger.warning("Zero variance in observed data, NSE undefined")
            return np.nan
        
        nse_value = 1 - (ss_res / ss_tot)
        
        return float(nse_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in nse calculation: {str(e)}")
        return np.nan


def kge(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Kling-Gupta Efficiency (KGE).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        KGE value
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate components
        obs_mean = np.mean(obs)
        pred_mean = np.mean(pred)
        obs_std = np.std(obs, ddof=1)
        pred_std = np.std(pred, ddof=1)
        
        # Correlation coefficient
        correlation = np.corrcoef(obs, pred)[0, 1]
        
        # Bias ratio
        bias_ratio = pred_mean / obs_mean if obs_mean != 0 else np.nan
        
        # Variability ratio
        var_ratio = pred_std / obs_std if obs_std != 0 else np.nan
        
        # Check for invalid components
        if np.isnan(correlation) or np.isnan(bias_ratio) or np.isnan(var_ratio):
            logger.warning("Invalid components in KGE calculation")
            return np.nan
        
        # Calculate KGE
        kge_value = 1 - np.sqrt((correlation - 1)**2 + (bias_ratio - 1)**2 + (var_ratio - 1)**2)
        
        return float(kge_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in kge calculation: {str(e)}")
        return np.nan


def bias(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate bias (mean error).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        Bias value
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate bias
        bias_value = np.mean(pred - obs)
        
        return float(bias_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in bias calculation: {str(e)}")
        return np.nan


def nrmse(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Normalized Root Mean Square Error (NRMSE).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        NRMSE value
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate RMSE
        rmse_value = rmse(obs, pred)
        
        # Calculate mean of observed values
        obs_mean = np.mean(obs)
        
        if obs_mean == 0:
            logger.warning("Zero mean in observed data, NRMSE undefined")
            return np.nan
        
        # Calculate NRMSE
        nrmse_value = rmse_value / obs_mean
        
        return float(nrmse_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in nrmse calculation: {str(e)}")
        return np.nan


def mape(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        MAPE value (in percentage)
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Avoid division by zero
        mask = np.abs(obs) > 1e-10
        if not np.any(mask):
            logger.warning("All observed values near zero, MAPE undefined")
            return np.nan
        
        # Calculate MAPE for non-zero observations
        obs_nonzero = obs[mask]
        pred_nonzero = pred[mask]
        
        mape_value = np.mean(np.abs((obs_nonzero - pred_nonzero) / obs_nonzero)) * 100
        
        return float(mape_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in mape calculation: {str(e)}")
        return np.nan


def pbias(observed: Union[np.ndarray, pd.Series], predicted: Union[np.ndarray, pd.Series]) -> float:
    """
    Calculate Percent Bias (PBIAS).
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        PBIAS value (in percentage)
    """
    try:
        obs, pred = _validate_inputs(observed, predicted)
        
        # Calculate bias
        bias_value = bias(obs, pred)
        
        # Calculate mean of observed values
        obs_mean = np.mean(obs)
        
        if obs_mean == 0:
            logger.warning("Zero mean in observed data, PBIAS undefined")
            return np.nan
        
        # Calculate PBIAS
        pbias_value = (bias_value / obs_mean) * 100
        
        return float(pbias_value)
        
    except (ValueError, RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Error in pbias calculation: {str(e)}")
        return np.nan


def calculate_all_metrics(
    observed: Union[np.ndarray, pd.Series], 
    predicted: Union[np.ndarray, pd.Series]
) -> Dict[str, float]:
    """
    Calculate all available metrics for model evaluation.
    
    Args:
        observed: Array of observed values
        predicted: Array of predicted values
        
    Returns:
        Dictionary containing all calculated metrics
    """
    metrics = {
        'r2': r2_score(observed, predicted),
        'rmse': rmse(observed, predicted),
        'nrmse': nrmse(observed, predicted),
        'mae': mae(observed, predicted),
        'mape': mape(observed, predicted),
        'nse': nse(observed, predicted),
        'kge': kge(observed, predicted),
        'bias': bias(observed, predicted),
        'pbias': pbias(observed, predicted)
    }
    
    return metrics


def evaluate_predictions(
    predictions_df: pd.DataFrame,
    observed_col: str = 'Q_obs',
    predicted_col: str = 'Q_pred',
    group_cols: list = None
) -> pd.DataFrame:
    """
    Evaluate predictions using all available metrics.
    
    Args:
        predictions_df: DataFrame with predictions and observations
        observed_col: Column name for observed values
        predicted_col: Column name for predicted values
        group_cols: List of columns to group by (e.g., ['code', 'month'])
        
    Returns:
        DataFrame with evaluation metrics
    """
    if group_cols is None:
        # Calculate metrics for entire dataset
        metrics = calculate_all_metrics(
            predictions_df[observed_col], 
            predictions_df[predicted_col]
        )
        return pd.DataFrame([metrics])
    else:
        # Calculate metrics for each group
        def calculate_group_metrics(group_df):
            return pd.Series(calculate_all_metrics(
                group_df[observed_col], 
                group_df[predicted_col]
            ))
        
        return predictions_df.groupby(group_cols).apply(calculate_group_metrics).reset_index()


# Legacy compatibility functions for existing code
def calculate_metrics_dict(observed: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Legacy function for backward compatibility."""
    return calculate_all_metrics(observed, predicted)