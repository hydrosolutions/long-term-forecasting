"""
Metric functions for evaluating model performance.

This module contains production-ready metric functions that can be used
for model evaluation in the main forecasting library.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union, List

logger = logging.getLogger(__name__)


def calculate_NSE(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) with robust error handling.

    NSE = 1 - Σ(O_i - S_i)² / Σ(O_i - O_mean)²
    where O_i are observed values and S_i are simulated values

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values

    Returns:
    --------
    float
        NSE value (higher is better, 1 is perfect)
    """
    # Convert inputs to numpy arrays if they aren't already
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Ensure same length
    if observed.shape != simulated.shape:
        raise ValueError(
            f"Observed and simulated arrays must have same shape. "
            f"Got shapes {observed.shape} and {simulated.shape}"
        )

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have enough valid data points
    n_valid = len(valid_observed)
    if n_valid < 2:  # Need at least 2 points to calculate variance
        return np.nan

    # Calculate observed mean
    obs_mean = np.mean(valid_observed)

    # Calculate denominator (variance of observations)
    denominator = np.sum((valid_observed - obs_mean) ** 2)

    # Check if denominator is too close to zero
    if denominator < 1e-10:
        return np.nan

    # Calculate numerator (sum of squared residuals)
    numerator = np.sum((valid_observed - valid_simulated) ** 2)

    # Calculate NSE
    nse = 1 - (numerator / denominator)

    # Handle edge cases
    if np.isinf(nse) or np.isnan(nse):
        return np.nan

    return nse


def calculate_R2(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination) with robust error handling.

    R² = 1 - (SS_res / SS_tot)
    where SS_res is the sum of squares of residuals and SS_tot is the total sum of squares

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values

    Returns:
    --------
    float
        R-squared value (between 0 and 1, higher is better)
    """
    from sklearn.metrics import r2_score

    # Convert inputs to numpy arrays if they aren't already
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Ensure same length
    if observed.shape != simulated.shape:
        raise ValueError(
            f"Observed and simulated arrays must have same shape. "
            f"Got shapes {observed.shape} and {simulated.shape}"
        )

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have enough valid data points
    n_valid = len(valid_observed)
    if n_valid < 2:  # Need at least 2 points to calculate variance
        return np.nan

    # Calculate R-squared using sklearn's r2_score
    try:
        r2 = r2_score(valid_observed, valid_simulated)
    except ValueError as e:
        logger.debug(f"Error calculating R-squared: {str(e)}")
        return np.nan

    # Handle edge cases
    if np.isinf(r2) or np.isnan(r2):
        logger.debug(f"Invalid R-squared value: {r2}")
        return np.nan

    return r2


def calculate_RMSE(
    observed: np.ndarray, simulated: np.ndarray, normalize: bool = True
) -> float:
    """
    Calculate Root Mean Square Error with proper handling of NaN values.

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values
    normalize : bool, optional
        If True, normalize RMSE by mean of observed values (default: True)

    Returns:
    --------
    float
        RMSE value (lower is better)
    """
    # Ensure inputs are numpy arrays
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have any valid data points
    n_valid = len(valid_observed)
    if n_valid == 0:
        return np.nan

    # Calculate squared differences
    squared_diff = (valid_observed - valid_simulated) ** 2

    # Calculate RMSE
    rmse = np.sqrt(np.mean(squared_diff))

    if normalize:
        mean_observed = np.mean(valid_observed)
        if mean_observed == 0:
            return np.nan
        return rmse / mean_observed

    return rmse


def calculate_MAE(
    observed: np.ndarray, simulated: np.ndarray, normalize: bool = True
) -> float:
    """
    Calculate Mean Absolute Error with proper handling of NaN values.

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values
    normalize : bool, optional
        If True, normalize MAE by mean of observed values (default: True)

    Returns:
    --------
    float
        MAE value (lower is better)
    """
    # Ensure inputs are numpy arrays
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have any valid data points
    n_valid = len(valid_observed)
    if n_valid == 0:
        return np.nan

    # Calculate absolute differences
    abs_diff = np.abs(valid_observed - valid_simulated)

    # Calculate MAE
    mae = np.mean(abs_diff)

    if normalize:
        mean_observed = np.mean(valid_observed)
        if mean_observed == 0:
            return np.nan
        return mae / mean_observed

    return mae


def calculate_NMSE(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate Normalized Mean Squared Error.

    NMSE = MSE / variance(observed)

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values

    Returns:
    --------
    float
        NMSE value (lower is better, 0 is perfect)
    """
    # Ensure inputs are numpy arrays
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have any valid data points
    n_valid = len(valid_observed)
    if n_valid < 2:
        return np.nan

    # Calculate MSE
    mse = np.mean((valid_observed - valid_simulated) ** 2)

    # Calculate variance of observed values
    obs_var = np.var(valid_observed, ddof=1)

    # Check if variance is too close to zero
    if obs_var < 1e-10:
        return np.nan

    # Calculate NMSE
    nmse = mse / obs_var

    return nmse


def calculate_NRMSE(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate Normalized Root Mean Squared Error.

    NRMSE = RMSE / std(observed)

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values

    Returns:
    --------
    float
        NRMSE value (lower is better, 0 is perfect)
    """
    # Ensure inputs are numpy arrays
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have any valid data points
    n_valid = len(valid_observed)
    if n_valid < 2:
        return np.nan

    # Calculate RMSE
    rmse = np.sqrt(np.mean((valid_observed - valid_simulated) ** 2))

    # Calculate standard deviation of observed values
    obs_std = np.std(valid_observed, ddof=1)

    # Check if standard deviation is too close to zero
    if obs_std < 1e-10:
        return np.nan

    # Calculate NRMSE
    nrmse = rmse / obs_std

    return nrmse


def calculate_NMAE(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate Normalized Mean Absolute Error.

    NMAE = MAE / mean(observed)

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values

    Returns:
    --------
    float
        NMAE value (lower is better, 0 is perfect)
    """
    # Ensure inputs are numpy arrays
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have any valid data points
    n_valid = len(valid_observed)
    if n_valid == 0:
        return np.nan

    # Calculate MAE
    mae = np.mean(np.abs(valid_observed - valid_simulated))

    # Calculate mean of observed values
    obs_mean = np.mean(valid_observed)

    # Check if mean is too close to zero
    if obs_mean < 1e-10:
        return np.nan

    # Calculate NMAE
    nmae = mae / obs_mean

    return nmae


# Metric registry for easy access
METRIC_FUNCTIONS = {
    "nse": calculate_NSE,
    "r2": calculate_R2,
    "rmse": calculate_RMSE,
    "mae": calculate_MAE,
    "nmse": calculate_NMSE,
    "nrmse": calculate_NRMSE,
    "nmae": calculate_NMAE,
}


def get_metric_function(metric_name: str):
    """
    Get a metric function by name.

    Parameters:
    -----------
    metric_name : str
        Name of the metric function

    Returns:
    --------
    callable
        The metric function

    Raises:
    -------
    ValueError
        If metric_name is not supported
    """
    if metric_name not in METRIC_FUNCTIONS:
        available_metrics = list(METRIC_FUNCTIONS.keys())
        raise ValueError(
            f"Metric '{metric_name}' is not supported. Available metrics: {available_metrics}"
        )

    return METRIC_FUNCTIONS[metric_name]
