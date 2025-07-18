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


def get_periods(data: pd.DataFrame, period_type: str = "monthly") -> pd.DataFrame:
    """
    Create period columns for temporal grouping.

    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame with 'date' column
    period_type : str
        Type of period grouping ('monthly', 'weekly', 'daily')

    Returns:
    --------
    pd.DataFrame
        DataFrame with additional period columns
    """
    df = data.copy()

    if "date" not in df.columns:
        raise ValueError("DataFrame must contain a 'date' column")

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"])

    if period_type == "monthly":
        df["period"] = df["date"].dt.month
        df["period_name"] = df["date"].dt.month_name()
    elif period_type == "weekly":
        df["period"] = df["date"].dt.isocalendar().week
        df["period_name"] = "Week_" + df["period"].astype(str)
    elif period_type == "daily":
        df["period"] = df["date"].dt.dayofyear
        df["period_name"] = "Day_" + df["period"].astype(str)
    else:
        raise ValueError(f"Unsupported period_type: {period_type}")

    return df


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

    # Apply softmax with temperature
    weights = softmax(perf_values / temperature)

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

    return weights


def create_weighted_ensemble(
    predictions: pd.DataFrame,
    weights: pd.DataFrame,
    model_columns: List[str],
    group_columns: List[str] = None,
) -> pd.DataFrame:
    """
    Create weighted ensemble predictions.

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

    Returns:
    --------
    pd.DataFrame
        DataFrame with ensemble predictions
    """
    result = predictions.copy()

    if group_columns is None:
        group_columns = ["code", "period"]

    # Initialize ensemble column
    result["ensemble"] = np.nan

    # Group by specified columns
    for group_keys, group_data in result.groupby(group_columns):
        # Get weights for this group
        weight_mask = weights[group_columns[0]] == group_keys[0]
        if len(group_columns) > 1:
            for i, col in enumerate(group_columns[1:], 1):
                weight_mask = weight_mask & (weights[col] == group_keys[i])

        group_weights = weights[weight_mask]

        if len(group_weights) == 0:
            # No weights found, use equal weights
            group_weights = pd.Series(
                [1.0 / len(model_columns)] * len(model_columns), index=model_columns
            )
        else:
            # Get weights for this group
            group_weights = group_weights.iloc[0]

        # Calculate weighted average
        group_indices = group_data.index
        ensemble_values = np.zeros(len(group_indices))

        for i, idx in enumerate(group_indices):
            model_preds = []
            model_weights = []

            for model in model_columns:
                if model in group_data.columns and not pd.isna(
                    group_data.loc[idx, model]
                ):
                    model_preds.append(group_data.loc[idx, model])
                    model_weights.append(group_weights.get(model, 0.0))

            if len(model_preds) > 0:
                model_preds = np.array(model_preds)
                model_weights = np.array(model_weights)

                # Normalize weights
                if np.sum(model_weights) > 0:
                    model_weights = model_weights / np.sum(model_weights)
                    ensemble_values[i] = np.sum(model_preds * model_weights)
                else:
                    ensemble_values[i] = np.mean(model_preds)
            else:
                ensemble_values[i] = np.nan

        result.loc[group_indices, "ensemble"] = ensemble_values

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
    elif fallback_strategy == "random":
        # Generate random weights and normalize
        weights = np.random.random(len(model_columns))
        weights = weights / np.sum(weights)
        return {model: weight for model, weight in zip(model_columns, weights)}
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
