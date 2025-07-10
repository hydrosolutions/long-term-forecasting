"""
Model evaluation module for monthly discharge forecasting.

This module provides comprehensive evaluation capabilities using the eval_scr.metric_functions
module for consistent metric calculations across all models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Import metric functions from eval_scr
from eval_scr.metric_functions import (
    r2_score, rmse, mae, nse, kge, bias,
    calculate_R2, calculate_RMSE, calculate_MAE, calculate_NSE
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_metrics(observed: Union[pd.Series, np.ndarray], 
                     predicted: Union[pd.Series, np.ndarray],
                     model_id: str = "unknown") -> Dict[str, float]:
    """
    Calculate comprehensive metrics for observed vs predicted values.
    
    Parameters:
    -----------
    observed : Union[pd.Series, np.ndarray]
        Observed discharge values
    predicted : Union[pd.Series, np.ndarray]
        Predicted discharge values
    model_id : str
        Model identifier for logging
        
    Returns:
    --------
    Dict[str, float]
        Dictionary containing all calculated metrics
    """
    # Convert to numpy arrays
    obs = np.asarray(observed)
    pred = np.asarray(predicted)
    
    # Remove NaN values
    valid_mask = ~(np.isnan(obs) | np.isnan(pred))
    obs_valid = obs[valid_mask]
    pred_valid = pred[valid_mask]
    
    # Check if we have enough valid data
    if len(obs_valid) < 2:
        logger.warning(f"Insufficient valid data for {model_id}: {len(obs_valid)} points")
        return {
            'r2': np.nan,
            'rmse': np.nan,
            'nrmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'nse': np.nan,
            'kge': np.nan,
            'bias': np.nan,
            'pbias': np.nan,
            'n_samples': len(obs_valid)
        }
    
    try:
        # Calculate basic metrics using eval_scr functions
        r2 = r2_score(obs_valid, pred_valid)
        rmse_val = rmse(obs_valid, pred_valid)
        mae_val = mae(obs_valid, pred_valid)
        nse_val = nse(obs_valid, pred_valid)
        kge_val = kge(obs_valid, pred_valid)
        bias_val = bias(obs_valid, pred_valid)
        
        # Calculate derived metrics
        obs_mean = np.mean(obs_valid)
        
        # Normalized RMSE
        nrmse_val = rmse_val / obs_mean if obs_mean != 0 else np.nan
        
        # Mean Absolute Percentage Error
        mape_val = (mae_val / obs_mean) * 100 if obs_mean != 0 else np.nan
        
        # Percent Bias
        pbias_val = (bias_val / obs_mean) * 100 if obs_mean != 0 else np.nan
        
        metrics = {
            'r2': r2,
            'rmse': rmse_val,
            'nrmse': nrmse_val,
            'mae': mae_val,
            'mape': mape_val,
            'nse': nse_val,
            'kge': kge_val,
            'bias': bias_val,
            'pbias': pbias_val,
            'n_samples': len(obs_valid)
        }
        
        # Log if any metrics are invalid
        invalid_metrics = [k for k, v in metrics.items() if np.isnan(v) or np.isinf(v)]
        if invalid_metrics:
            logger.warning(f"Invalid metrics for {model_id}: {invalid_metrics}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics for {model_id}: {str(e)}")
        return {
            'r2': np.nan,
            'rmse': np.nan,
            'nrmse': np.nan,
            'mae': np.nan,
            'mape': np.nan,
            'nse': np.nan,
            'kge': np.nan,
            'bias': np.nan,
            'pbias': np.nan,
            'n_samples': len(obs_valid)
        }

def evaluate_per_code(df: pd.DataFrame, 
                     model_id: str = "unknown",
                     min_samples: int = 5) -> pd.DataFrame:
    """
    Evaluate model performance for each basin code.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing observations and predictions
    model_id : str
        Model identifier
    min_samples : int
        Minimum number of samples required for evaluation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for each basin code
    """
    results = []
    
    for code in df['code'].unique():
        df_code = df[df['code'] == code].copy()
        
        # Skip if insufficient data
        if len(df_code) < min_samples:
            logger.warning(f"Insufficient data for code {code} in {model_id}: {len(df_code)} samples")
            continue
        
        # Calculate metrics for this code
        metrics = calculate_metrics(df_code['Q_obs'], df_code['Q_pred'], f"{model_id}_code_{code}")
        
        # Add metadata
        metrics['code'] = code
        metrics['model_id'] = model_id
        metrics['level'] = 'per_code'
        
        results.append(metrics)
    
    if not results:
        logger.warning(f"No valid codes found for {model_id}")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def evaluate_per_month(df: pd.DataFrame, 
                      model_id: str = "unknown",
                      min_samples: int = 3) -> pd.DataFrame:
    """
    Evaluate model performance for each month.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing observations and predictions
    model_id : str
        Model identifier
    min_samples : int
        Minimum number of samples required for evaluation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for each month
    """
    results = []
    
    # Add month column if not present
    if 'month' not in df.columns:
        df = df.copy()
        df['month'] = pd.to_datetime(df['date']).dt.month
    
    for month in sorted(df['month'].unique()):
        df_month = df[df['month'] == month].copy()
        
        # Skip if insufficient data
        if len(df_month) < min_samples:
            logger.warning(f"Insufficient data for month {month} in {model_id}: {len(df_month)} samples")
            continue
        
        # Calculate metrics for this month
        metrics = calculate_metrics(df_month['Q_obs'], df_month['Q_pred'], f"{model_id}_month_{month}")
        
        # Add metadata
        metrics['month'] = month
        metrics['model_id'] = model_id
        metrics['level'] = 'per_month'
        
        results.append(metrics)
    
    if not results:
        logger.warning(f"No valid months found for {model_id}")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def evaluate_per_code_month(df: pd.DataFrame, 
                           model_id: str = "unknown",
                           min_samples: int = 2) -> pd.DataFrame:
    """
    Evaluate model performance for each basin code and month combination.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing observations and predictions
    model_id : str
        Model identifier
    min_samples : int
        Minimum number of samples required for evaluation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with metrics for each code-month combination
    """
    results = []
    
    # Add month column if not present
    if 'month' not in df.columns:
        df = df.copy()
        df['month'] = pd.to_datetime(df['date']).dt.month
    
    for code in df['code'].unique():
        for month in sorted(df['month'].unique()):
            df_subset = df[(df['code'] == code) & (df['month'] == month)].copy()
            
            # Skip if insufficient data
            if len(df_subset) < min_samples:
                continue
            
            # Calculate metrics for this code-month combination
            metrics = calculate_metrics(df_subset['Q_obs'], df_subset['Q_pred'], 
                                      f"{model_id}_code_{code}_month_{month}")
            
            # Add metadata
            metrics['code'] = code
            metrics['month'] = month
            metrics['model_id'] = model_id
            metrics['level'] = 'per_code_month'
            
            results.append(metrics)
    
    if not results:
        logger.warning(f"No valid code-month combinations found for {model_id}")
        return pd.DataFrame()
    
    return pd.DataFrame(results)

def evaluate_overall(df: pd.DataFrame, 
                    model_id: str = "unknown",
                    min_samples: int = 10) -> pd.DataFrame:
    """
    Evaluate overall model performance across all data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing observations and predictions
    model_id : str
        Model identifier
    min_samples : int
        Minimum number of samples required for evaluation
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with overall metrics
    """
    # Skip if insufficient data
    if len(df) < min_samples:
        logger.warning(f"Insufficient data for overall evaluation of {model_id}: {len(df)} samples")
        return pd.DataFrame()
    
    # Calculate overall metrics
    metrics = calculate_metrics(df['Q_obs'], df['Q_pred'], f"{model_id}_overall")
    
    # Add metadata
    metrics['model_id'] = model_id
    metrics['level'] = 'overall'
    
    return pd.DataFrame([metrics])

def evaluate_model_comprehensive(df: pd.DataFrame, 
                               model_id: str = "unknown",
                               min_samples_overall: int = 10,
                               min_samples_code: int = 5,
                               min_samples_month: int = 3,
                               min_samples_code_month: int = 2,
                               include_code_month: bool = False) -> pd.DataFrame:
    """
    Perform comprehensive evaluation of a model at all levels.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing observations and predictions
    model_id : str
        Model identifier
    min_samples_overall : int
        Minimum samples for overall evaluation
    min_samples_code : int
        Minimum samples for per-code evaluation
    min_samples_month : int
        Minimum samples for per-month evaluation
    min_samples_code_month : int
        Minimum samples for per-code-month evaluation
    include_code_month : bool
        Whether to include per-code-month evaluation
        
    Returns:
    --------
    pd.DataFrame
        Combined DataFrame with all evaluation results
    """
    all_results = []
    
    # Overall evaluation
    logger.info(f"Evaluating {model_id} - Overall")
    overall_results = evaluate_overall(df, model_id, min_samples_overall)
    if not overall_results.empty:
        all_results.append(overall_results)
    
    # Per-code evaluation
    logger.info(f"Evaluating {model_id} - Per Code")
    code_results = evaluate_per_code(df, model_id, min_samples_code)
    if not code_results.empty:
        all_results.append(code_results)
    
    # Per-month evaluation
    logger.info(f"Evaluating {model_id} - Per Month")
    month_results = evaluate_per_month(df, model_id, min_samples_month)
    if not month_results.empty:
        all_results.append(month_results)
    
    # Per-code-month evaluation (optional)
    if include_code_month:
        logger.info(f"Evaluating {model_id} - Per Code-Month")
        code_month_results = evaluate_per_code_month(df, model_id, min_samples_code_month)
        if not code_month_results.empty:
            all_results.append(code_month_results)
    
    # Combine all results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)
        logger.info(f"Completed evaluation for {model_id}: {len(combined_results)} result rows")
        return combined_results
    else:
        logger.warning(f"No valid evaluation results for {model_id}")
        return pd.DataFrame()

def evaluate_multiple_models(loaded_predictions: Dict[str, pd.DataFrame],
                           include_code_month: bool = False,
                           min_samples_overall: int = 10,
                           min_samples_code: int = 5,
                           min_samples_month: int = 3,
                           min_samples_code_month: int = 2) -> pd.DataFrame:
    """
    Evaluate multiple models comprehensively.
    
    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of loaded prediction DataFrames
    include_code_month : bool
        Whether to include per-code-month evaluation
    min_samples_overall : int
        Minimum samples for overall evaluation
    min_samples_code : int
        Minimum samples for per-code evaluation
    min_samples_month : int
        Minimum samples for per-month evaluation
    min_samples_code_month : int
        Minimum samples for per-code-month evaluation
        
    Returns:
    --------
    pd.DataFrame
        Combined evaluation results for all models
    """
    all_model_results = []
    
    for model_id, df in loaded_predictions.items():
        logger.info(f"Evaluating model: {model_id}")
        
        # Get model metadata
        family = df['family'].iloc[0] if 'family' in df.columns else 'unknown'
        model_name = df['model_name'].iloc[0] if 'model_name' in df.columns else model_id
        
        # Evaluate model
        model_results = evaluate_model_comprehensive(
            df, model_id, 
            min_samples_overall=min_samples_overall,
            min_samples_code=min_samples_code,
            min_samples_month=min_samples_month,
            min_samples_code_month=min_samples_code_month,
            include_code_month=include_code_month
        )
        
        if not model_results.empty:
            # Add family and model name metadata
            model_results['family'] = family
            model_results['model_name'] = model_name
            all_model_results.append(model_results)
        
    # Combine all results
    if all_model_results:
        combined_results = pd.concat(all_model_results, ignore_index=True)
        logger.info(f"Completed evaluation for {len(loaded_predictions)} models: {len(combined_results)} total result rows")
        return combined_results
    else:
        logger.warning("No valid evaluation results for any model")
        return pd.DataFrame()

def calculate_model_rankings(evaluation_results: pd.DataFrame,
                           metric: str = 'r2',
                           level: str = 'overall',
                           ascending: bool = False) -> pd.DataFrame:
    """
    Calculate model rankings based on specified metric and level.
    
    Parameters:
    -----------
    evaluation_results : pd.DataFrame
        DataFrame with evaluation results
    metric : str
        Metric to rank by
    level : str
        Evaluation level to consider
    ascending : bool
        Whether to sort in ascending order
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with model rankings
    """
    # Filter to specified level
    level_results = evaluation_results[evaluation_results['level'] == level].copy()
    
    if level_results.empty:
        logger.warning(f"No results found for level: {level}")
        return pd.DataFrame()
    
    # Sort by metric
    ranked_results = level_results.sort_values(metric, ascending=ascending)
    
    # Add ranking
    ranked_results['rank'] = range(1, len(ranked_results) + 1)
    
    return ranked_results[['rank', 'model_id', 'family', 'model_name', metric, 'n_samples']].copy()

if __name__ == "__main__":
    # Example usage
    try:
        from .prediction_loader import load_all_predictions
    except ImportError:
        from prediction_loader import load_all_predictions
    
    # Load predictions
    predictions, validation = load_all_predictions()
    
    # Evaluate all models
    results = evaluate_multiple_models(predictions, include_code_month=False)
    
    print("\n=== MODEL EVALUATION SUMMARY ===")
    print(f"Evaluated {len(predictions)} models")
    print(f"Generated {len(results)} evaluation records")
    
    # Show overall rankings
    overall_rankings = calculate_model_rankings(results, metric='r2', level='overall')
    print("\n=== TOP 10 MODELS BY R2 ===")
    print(overall_rankings.head(10).to_string(index=False))