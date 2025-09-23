"""
Utility functions for meta-learning models.

This module contains helper functions specifically designed for meta-learning
workflows, including weight calculation, ensemble creation, and performance
analysis utilities.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from scipy.special import softmax

logger = logging.getLogger(__name__)


def calculate_weights_softmax(
    performance_values: np.ndarray, temperature: float = 1.0, invert: bool = False
) -> np.ndarray:
    """
    Calculate weights using softmax transformation.

    Parameters:
    -----------
    performance_values : np.ndarray
        Array of performance values
    temperature : float
        Temperature parameter for softmax (default: 1.0)
    invert : bool
        Whether to invert the performance values (for error metrics)

    Returns:
    --------
    np.ndarray
        Array of weights summing to 1.0
    """
    # Handle NaN values
    if np.all(np.isnan(performance_values)):
        return np.full(len(performance_values), 1.0 / len(performance_values))

    # Replace NaN with worst performance
    perf_values = performance_values.copy()
    if invert:
        # For error metrics, higher is worse
        perf_values = np.where(
            np.isnan(perf_values), np.nanmax(perf_values), perf_values
        )
        # Invert so that better models get higher weights
        perf_values = -perf_values
    else:
        # For accuracy metrics, lower is worse
        perf_values = np.where(
            np.isnan(perf_values), np.nanmin(perf_values), perf_values
        )

    max_value = np.nanmax(perf_values)
    perf_values = perf_values - max_value  # Shift for numerical stability
    # Apply softmax with temperature
    weights = softmax(perf_values / temperature)
    # Ensure weights are non-negative
    weights = np.clip(weights, 0, 1)

    return weights


def top_n_uniform_weights(
    performance_values: np.ndarray, top_n: int = 3, invert: bool = False
) -> np.ndarray:
    """
    Calculate uniform weights for the top N models based on performance values.

    Parameters:
    -----------
    performance_values : np.ndarray
        Array of performance values
    top_n : int
        Number of top models to consider (default: 3)
    invert : bool
        Whether to invert the performance values (for error metrics)

    Returns:
    --------
    np.ndarray
        Array of weights for the top N models, uniform distribution
    """
    # Handle NaN values
    if np.all(np.isnan(performance_values)):
        return np.full(len(performance_values), 1.0 / len(performance_values))

    perf_values = performance_values.copy()

    # Replace NaN with worst performance
    if invert:
        perf_values = np.where(
            np.isnan(perf_values), np.nanmax(perf_values), perf_values
        )
    else:
        perf_values = np.where(
            np.isnan(perf_values), np.nanmin(perf_values), perf_values
        )
        # invert for accuracy metrics (lower is better)
        perf_values = -perf_values

    # Get indices of top N models
    top_indices = np.argsort(perf_values)[:top_n]

    # Create uniform weights for top N models
    weights = np.zeros_like(perf_values, dtype=float)
    weights[top_indices] = 1.0 / top_n

    # Ensure weights are non-negative
    weights = np.clip(weights, 0, 1)
    return weights


def calculate_weights_normalized(
    performance_values: np.ndarray, invert: bool = False
) -> np.ndarray:
    """
    Calculate weights using normalized transformation.

    Parameters:
    -----------
    performance_values : np.ndarray
        Array of performance values
    invert : bool
        Whether to invert the performance values (for error metrics)

    Returns:
    --------
    np.ndarray
        Array of weights summing to 1.0
    """
    # Handle NaN values
    if np.all(np.isnan(performance_values)):
        return np.full(len(performance_values), 1.0 / len(performance_values))

    perf_values = performance_values.copy()

    # Replace NaN with worst performance
    if invert:
        # For error metrics, higher is worse
        perf_values = np.where(
            np.isnan(perf_values), np.nanmax(perf_values), perf_values
        )
        # Invert so that better models get higher weights
        perf_values = 1.0 / (1.0 + perf_values)
    else:
        # For accuracy metrics, lower is worse
        perf_values = np.where(
            np.isnan(perf_values), np.nanmin(perf_values), perf_values
        )

    # Handle edge cases
    if np.all(perf_values == 0):
        return np.full(len(perf_values), 1.0 / len(perf_values))

    # Normalize to sum to 1
    weights = perf_values / np.sum(perf_values)
    # Ensure weights are non-negative
    weights = np.clip(weights, 0, 1)

    return weights


def weights_hybrid(
    performance_values: np.ndarray,
    delta: float = 0.1,  # threshold for "small difference" (10%)
    top_n: int = 3,  # average top-n when differences are tiny
    temperature: float = 0.5,  # softmax temperature on *relative* gaps
    is_error_metric: bool = True,  # whether higher is worse (error metric)
) -> np.ndarray:
    e = np.asarray(performance_values, dtype=float)
    # All NaN -> uniform
    if np.all(np.isnan(e)):
        return np.full(e.size, 1.0 / e.size)

    if not is_error_metric:
        e = -e  # Invert for accuracy metrics (higher is better)

    # Replace NaNs with "worst" so they get tiny weight
    worst = np.nanmax(e)
    e = np.where(np.isnan(e), worst, e)

    # Relative gaps to the best
    e_min = np.min(e)
    rel_gap = (e - e_min) / max(e_min, 1e-12)  # dimensionless

    # If all within delta -> average top-n
    if np.max(rel_gap) <= delta:
        idx = np.argsort(e)[:top_n]
        w = np.zeros_like(e, dtype=float)
        w[idx] = 1.0 / top_n
        return w

    # Otherwise softmax on "higher is better" = negative relative error
    x = -rel_gap
    w = softmax(x / max(temperature, 1e-8))

    w = np.clip(w, 0, 1)  # Ensure non-negative weights

    return w


def create_weighted_ensemble(
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    model_columns: List[str],
    group_columns: List[str] = None,
    debug: bool = False,
    naive_approach: bool = False,
) -> pd.DataFrame:
    """
    Create weighted ensemble predictions with proper NaN handling.

    Parameters:
    -----------
    predictions : pd.DataFrame
        DataFrame with model predictions
    weights : pd.DataFrame
        DataFrame with model weights
    model_columns : List[str]
        List of model column names
    group_columns : List[str], optional
        Columns to group by for weight application
    debug : bool, optional
        If True, print debug information

    Returns:
    --------
    pd.DataFrame
        DataFrame with ensemble predictions
    """
    result = predictions.copy()

    if group_columns is None:
        group_columns = ["code", "period"]

    # Validate that all model columns exist in predictions
    missing_cols = set(model_columns) - set(predictions.columns)
    if missing_cols:
        raise ValueError(
            f"Model columns {missing_cols} not found in predictions DataFrame"
        )

    if naive_approach:
        # Naive approach: simple average of model predictions
        logger.info("Using naive approach: simple average of model predictions")
        result["ensemble"] = result[model_columns].mean(axis=1, skipna=True)
        return result

    # Initialize ensemble column
    result["ensemble"] = np.nan

    # Iterate over each group defined by group_columns
    for group_idx, group in result.groupby(group_columns):
        # Get the weights for this group
        weight_mask = True
        for col in group_columns:
            weight_mask = weight_mask & (weights[col] == group[col].iloc[0])

        group_weights = weights.loc[weight_mask]

        if group_weights.empty:
            logger.warning(
                f"No weights found for group {dict(zip(group_columns, group_idx))}"
            )
            continue

        # Get predictions for this group (can be multiple rows)
        group_predictions = group[model_columns].values  # Shape: (n_rows, n_models)

        # Get weights as 1D array
        base_weights = group_weights[
            model_columns
        ].values.flatten()  # Shape: (n_models,)

        if debug:
            print(f"\nGroup {dict(zip(group_columns, group_idx))}:")
            print(f"Base weights: {base_weights}")
            print(f"Base weights sum: {base_weights.sum()}")

        # Check if all predictions are NaN
        if np.all(np.isnan(group_predictions)):
            logger.warning(
                f"All predictions are NaN for group {dict(zip(group_columns, group_idx))}"
            )
            continue

        # Process each row in the group
        ensemble_values = []
        for row_predictions in group_predictions:
            # Create mask for non-NaN predictions
            valid_mask = ~np.isnan(row_predictions)

            if not valid_mask.any():
                # All predictions in this row are NaN
                ensemble_values.append(np.nan)
                continue

            # Apply weights only to valid predictions
            row_weights = base_weights.copy()
            row_weights[~valid_mask] = 0  # Set weights to 0 for NaN predictions

            # Normalize weights to sum to 1
            weight_sum = row_weights.sum()
            if weight_sum > 0:
                row_weights = row_weights / weight_sum

                # Calculate weighted average
                # Use only valid predictions and their normalized weights
                valid_predictions = row_predictions[valid_mask]
                valid_weights = row_weights[valid_mask]

                if debug:
                    print(f"  Row predictions: {row_predictions}")
                    print(f"  Valid mask: {valid_mask}")
                    print(f"  Valid predictions: {valid_predictions}")
                    print(f"  Valid weights: {valid_weights}")
                    print(f"  Valid weights sum: {valid_weights.sum()}")

                # Double-check weights sum to 1
                assert np.abs(valid_weights.sum() - 1.0) < 1e-10, (
                    f"Weights don't sum to 1: {valid_weights.sum()}"
                )

                weighted_sum = np.sum(valid_predictions * valid_weights)

                if debug:
                    print(f"  Weighted sum: {weighted_sum}")
                    print(f"  Simple mean of valid: {np.mean(valid_predictions)}")

                ensemble_values.append(weighted_sum)
            else:
                # All weights are 0 (shouldn't happen if we have valid predictions)
                ensemble_values.append(np.nanmean(row_predictions[valid_mask]))

        # Assign ensemble values back to result DataFrame
        result.loc[group.index, "ensemble"] = ensemble_values

    return result


def validate_performance_data(
    performance_df: pd.DataFrame, required_columns: List[str], min_samples: int = 5
) -> Tuple[bool, str]:
    """
    Validate performance data for meta-learning.

    Parameters:
    -----------
    performance_df : pd.DataFrame
        DataFrame containing performance metrics
    required_columns : List[str]
        List of required column names
    min_samples : int
        Minimum number of samples required

    Returns:
    --------
    Tuple[bool, str]
        (is_valid, error_message)
    """
    # Check if DataFrame is empty
    if performance_df.empty:
        return False, "Performance DataFrame is empty"

    # Check required columns
    missing_columns = [
        col for col in required_columns if col not in performance_df.columns
    ]
    if missing_columns:
        return False, f"Missing required columns: {missing_columns}"

    # Check minimum samples
    if len(performance_df) < min_samples:
        return False, f"Insufficient samples: {len(performance_df)} < {min_samples}"

    # Check for all NaN values in metric columns
    metric_columns = [
        col for col in performance_df.columns if col not in ["code", "period"]
    ]
    if performance_df[metric_columns].isna().all().all():
        return False, "All metric values are NaN"

    return True, "Valid"


def aggregate_performance_across_basins(
    performance_df: pd.DataFrame,
    metric_columns: List[str],
    group_by: str = "period",
    aggregation_method: str = "mean",
) -> pd.DataFrame:
    """
    Aggregate performance metrics across basins.

    Parameters:
    -----------
    performance_df : pd.DataFrame
        DataFrame with performance metrics
    metric_columns : List[str]
        List of metric column names
    group_by : str
        Column to group by for aggregation
    aggregation_method : str
        Method for aggregation ('mean', 'median', 'std')

    Returns:
    --------
    pd.DataFrame
        Aggregated performance DataFrame
    """
    if aggregation_method == "mean":
        agg_func = np.nanmean
    elif aggregation_method == "median":
        agg_func = np.nanmedian
    elif aggregation_method == "std":
        agg_func = np.nanstd
    else:
        raise ValueError(f"Unsupported aggregation method: {aggregation_method}")

    # Group by specified column and aggregate
    result = (
        performance_df.groupby(group_by)[metric_columns].agg(agg_func).reset_index()
    )

    return result


def get_fallback_weights(
    model_columns: List[str], fallback_strategy: str = "equal"
) -> Dict[str, float]:
    """
    Get fallback weights when insufficient historical data is available.

    Parameters:
    -----------
    model_columns : List[str]
        List of model names
    fallback_strategy : str
        Strategy for fallback weights ('equal', 'random')

    Returns:
    --------
    Dict[str, float]
        Dictionary of model weights
    """
    if fallback_strategy == "equal":
        weight_value = 1.0 / len(model_columns)
        return {model: weight_value for model in model_columns}
    else:
        raise ValueError(f"Unsupported fallback strategy: {fallback_strategy}")


def check_data_sufficiency(
    data: pd.DataFrame, min_samples_per_basin: int = 5, min_samples_per_period: int = 5
) -> Dict[str, Any]:
    """
    Check if data is sufficient for meta-learning.

    Parameters:
    -----------
    data : pd.DataFrame
        Input data with 'code' and 'period' columns
    min_samples_per_basin : int
        Minimum samples required per basin
    min_samples_per_period : int
        Minimum samples required per period

    Returns:
    --------
    Dict[str, Any]
        Dictionary with sufficiency information
    """
    result = {
        "sufficient": True,
        "total_samples": len(data),
        "unique_basins": data["code"].nunique(),
        "unique_periods": data["period"].nunique(),
        "issues": [],
    }

    # Check samples per basin
    basin_counts = data["code"].value_counts()
    insufficient_basins = basin_counts[basin_counts < min_samples_per_basin]
    if len(insufficient_basins) > 0:
        result["sufficient"] = False
        result["issues"].append(
            f"Insufficient samples for basins: {insufficient_basins.index.tolist()}"
        )

    # Check samples per period
    period_counts = data["period"].value_counts()
    insufficient_periods = period_counts[period_counts < min_samples_per_period]
    if len(insufficient_periods) > 0:
        result["sufficient"] = False
        result["issues"].append(
            f"Insufficient samples for periods: {insufficient_periods.index.tolist()}"
        )

    return result
