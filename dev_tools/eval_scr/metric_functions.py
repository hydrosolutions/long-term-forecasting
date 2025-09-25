import logging

logger = logging.getLogger(__name__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import datetime

from typing import Union, List, Tuple


def sdivsigma_nse(data: pd.DataFrame, observed_col: str, simulated_col: str):
    """
    Calculate the forecast efficacy and the Nash-Sutcliffe Efficiency (NSE) for the observed and simulated data.

    NSE = 1 - s/sigma

    Args:
        data (pandas.DataFrame): The input data containing the observed and simulated data.
        observed_col (str): The name of the column containing the observed data.
        simulated_col (str): The name of the column containing the simulated data.

    Returns:
        pandas.Series: A pandas Series containing the forecast efficacy and the NSE value.

    Raises:
        ValueError: If the input data is missing one or more required columns.

    """
    # Test the input. Make sure that the DataFrame contains the required columns
    if not all(column in data.columns for column in [observed_col, simulated_col]):
        raise ValueError(
            f"DataFrame is missing one or more required columns: {observed_col, simulated_col}"
        )

    # print("DEBUG: forecasting:sdivsigma_nse: data: \n", data)

    # Convert to numpy arrays for faster computation
    # Use float64 for better numerical stability
    observed = data[observed_col].to_numpy(dtype=np.float64)
    simulated = data[simulated_col].to_numpy(dtype=np.float64)

    # Check for empty data after dropping NaNs
    mask = ~(np.isnan(observed) | np.isnan(simulated))
    if not np.any(mask):
        return pd.Series([np.nan, np.nan], index=["sdivsigma", "nse"])

    # Filter arrays using mask
    observed = observed[mask]
    simulated = simulated[mask]

    # Early return if not enough data points
    if len(observed) < 2:  # Need at least 2 points for std calculation
        logger.info(f"Not enough data points for sdivsigma_nse calculation.")
        print(f"Not enough data points for sdivsigma_nse calculation.")
        return pd.Series([np.nan, np.nan], index=["sdivsigma", "nse"])

    # Calculate mean once for reuse
    observed_mean = np.mean(observed)

    # Count the number of data points
    n = len(observed)

    # Calculate denominators
    denominator_nse = np.sum((observed - observed_mean) ** 2)
    # sigma: Standard deviation of the observed data
    denominator_sdivsigma = np.std(observed, ddof=1)  # ddof=1 for sample std

    # Check for numerical stability
    if denominator_nse < 1e-10 or denominator_sdivsigma < 1e-10:
        logger.debug(f"Numerical stability issue in sdivsigma_nse:")
        logger.debug(f"denominator_nse: {denominator_nse}")
        logger.debug(f"denominator_sdivsigma: {denominator_sdivsigma}")
        return pd.Series([np.nan, np.nan], index=["sdivsigma", "nse"])

    try:
        # Calculate differences once for reuse
        differences = observed - simulated

        # Calculate NSE
        numerator_nse = np.sum(differences**2)
        nse_value = 1 - (numerator_nse / denominator_nse)

        # Calculate sdivsigma
        # s: Average of squared differences between observed and simulated data
        numerator_sdivsigma = np.sqrt(np.sum(differences**2) / (n - 1))
        # s/sigma: Efficacy of the model
        sdivsigma = numerator_sdivsigma / denominator_sdivsigma

        # Sanity checks
        if not (-np.inf < nse_value < np.inf) or not (0 <= sdivsigma < np.inf):
            return pd.Series([np.nan, np.nan], index=["sdivsigma", "nse"])

        return pd.Series([sdivsigma, nse_value], index=["sdivsigma", "nse"])

    except (RuntimeWarning, FloatingPointError) as e:
        logger.debug(f"Numerical computation error in sdivsigma_nse: {str(e)}")
        return pd.Series([np.nan, np.nan], index=["sdivsigma", "nse"])


def calc_accuracy(
    data: pd.DataFrame, observed_col: str, simulated_col: str, threshold_col: str
):
    # Test the input. Make sure that the DataFrame contains the required columns
    if not all(column in data.columns for column in [observed_col, simulated_col]):
        raise ValueError(
            f"DataFrame is missing one or more required columns: {observed_col, simulated_col}"
        )
    # Drop NaN values in columns with observed and simulated data
    dataN = data.dropna(subset=[observed_col, simulated_col]).copy()
    # Test if we still have data after dropping NaN values
    if dataN.empty:
        logger.debug("accuracy: data is empty after dropping NaN values")
        return pd.Series([np.nan], index=["accuracy"])

    # Calculate the accuracy of the model
    residual = abs(dataN[observed_col] - dataN[simulated_col])
    accuracy = (residual < dataN[threshold_col]).astype(int).mean()

    return pd.Series([accuracy], index=["accuracy"])


def forecast_accuracy_hydromet(
    data: pd.DataFrame, observed_col: str, simulated_col: str, delta_col: str
):
    """
    Calculate the forecast accuracy for the observed and simulated data.

    Args:
        data (pandas.DataFrame): The input data containing the observed and simulated data.
        observed_col (str): The name of the column containing the observed data.
        simulated_col (str): The name of the column containing the simulated data.

    Returns:
        pandas.Series: A pandas Series containing the forecast accuracy.

    Raises:
        ValueError: If the input data is missing one or more required columns.

    """
    # Test the input. Make sure that the DataFrame contains the required columns
    if not all(
        column in data.columns for column in [observed_col, simulated_col, delta_col]
    ):
        raise ValueError(
            f"DataFrame is missing one or more required columns: {observed_col, simulated_col, delta_col}"
        )

    # Convert to numpy arrays for faster computation
    observed = data[observed_col].to_numpy(dtype=np.float64)
    simulated = data[simulated_col].to_numpy(dtype=np.float64)
    delta_values = data[delta_col].to_numpy(dtype=np.float64)

    # Check for empty data after dropping NaNs
    mask = ~(np.isnan(observed) | np.isnan(simulated) | np.isnan(delta_values))
    if not np.any(mask):
        return pd.Series([np.nan, np.nan], index=["delta", "accuracy"])

    # Also drop rows where observed, simulated or delta_valus is inf
    mask = mask & ~(np.isinf(observed) | np.isinf(simulated) | np.isinf(delta_values))
    if not np.any(mask):
        return pd.Series([np.nan, np.nan], index=["delta", "accuracy"])

    # Filter arrays using mask
    observed = observed[mask]
    simulated = simulated[mask]
    delta_values = delta_values[mask]

    # Early return if not enough data points
    if len(observed) < 1:
        return pd.Series([np.nan, np.nan], index=["delta", "accuracy"])

    try:
        # Calculate absolute differences once
        abs_diff = np.abs(observed - simulated)

        # Calculate accuracy using vectorized operations
        accuracy = np.mean(abs_diff <= delta_values)

        # Get the last delta value (they are all the same)
        delta = delta_values[-1]

        # Sanity checks
        if not (0 <= accuracy <= 1) or not (0 <= delta < np.inf):
            return pd.Series([np.nan, np.nan], index=["delta", "accuracy"])

        return pd.Series([delta, accuracy], index=["delta", "accuracy"])

    except (RuntimeWarning, FloatingPointError) as e:
        logger.debug(
            f"Numerical computation error in forecast_accuracy_hydromet: {str(e)}"
        )
        return pd.Series([np.nan, np.nan], index=["delta", "accuracy"])


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
    return_components : bool, optional
        If True, returns additional information about the calculation

    Returns:
    --------
    Union[float, Tuple[float, int, dict]]
        If return_components=False: returns NSE value
        If return_components=True: returns (NSE value, number of valid samples, components dict)
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


def calculate_RMSE(
    observed: np.ndarray, simulated: np.ndarray, normalize: bool = True
) -> Union[float, Tuple[float, int]]:
    """
    Calculate Root Mean Square Error with proper handling of NaN values and empty slices.

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
    float or Tuple[float, int]
        If normalize=True: returns normalized RMSE
        If normalize=False: returns (RMSE, number of valid samples)
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
) -> Union[float, Tuple[float, int]]:
    """
    Calculate Mean Absolute Error with proper handling of NaN values and empty slices.

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
    float or Tuple[float, int]
        If normalize=True: returns normalized MAE
        If normalize=False: returns (MAE, number of valid samples)
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


def calculate_QuantileLoss(
    observed: np.ndarray,
    simulated: np.ndarray,
    quantile_levels: np.ndarray,
    normalize: bool = False,
) -> float:
    """
    Calculate Pinball Loss for quantile forecasts.
    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray (2D array)
        Array of simulated values (quantile forecasts)
    quantile_levels : np.ndarray
        Array of quantile levels (e.g., [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    Formula:
    QL = max(q * (y-y_pred), (1-q) * (y_pred-y))
    Returns:
    --------
    float
        Pinball Loss value (lower is better)
    """
    # Ensure inputs are numpy arrays
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)

    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated).any(axis=1)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]

    # Check if we have any valid data points
    n_valid = len(valid_observed)
    if n_valid == 0:
        return np.nan

    # Calculate Pinball Loss
    pinball_loss = np.zeros_like(quantile_levels)
    for i, q in enumerate(quantile_levels):
        pinball_loss[i] = np.mean(
            np.maximum(
                q * (valid_observed - valid_simulated[:, i]),
                (1 - q) * (valid_simulated[:, i] - valid_observed),
            )
        )

    # Calculate mean Pinball Loss
    mean_pinball_loss = np.mean(pinball_loss)

    if normalize:
        mean_observed = np.mean(valid_observed)
        if mean_observed == 0:
            return np.nan, 0
        return mean_pinball_loss / mean_observed

    return mean_pinball_loss


def calculate_R2(observed: np.ndarray, simulated: np.ndarray) -> float:
    """
    Calculate R-squared (coefficient of determination) with robust error handling.

    R² = 1 - (SS_res / SS_tot)
    where SS_res is the sum of squares of residuals and SS_tot is the total sum of squares

    import r2_score from sklearn.metrics

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


def calculate_prob_exceedance(
        observed: np.ndarray, 
        simulated: np.ndarray,
) -> float:
    """
    Calculate the Probability of Exceedance (POE) for observed and simulated data.

    POE = P(Simulated > Observed)

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values

    Returns:
    --------
    float
        Probability of Exceedance value (between 0 and 1)
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
    if n_valid == 0:
        return np.nan

    # Calculate Probability of Exceedance
    poe = np.mean(valid_simulated > valid_observed)

    # Handle edge cases
    if np.isinf(poe) or np.isnan(poe):
        logger.debug(f"Invalid Probability of Exceedance value: {poe}")
        return np.nan

    return poe

def calculate_coverage(
    observed: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Calculate the Coverage of prediction intervals.

    Coverage = P(Lower Bound <= Observed <= Upper Bound)

    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    lower_bound : np.ndarray
        Array of lower bounds of prediction intervals
    upper_bound : np.ndarray
        Array of upper bounds of prediction intervals

    Returns:
    --------
    float
        Coverage value (between 0 and 1)
    """
    # Convert inputs to numpy arrays if they aren't already
    observed = np.asarray(observed)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    # Ensure same length
    if not (observed.shape == lower_bound.shape == upper_bound.shape):
        raise ValueError(
            f"Observed, lower bound, and upper bound arrays must have same shape. "
            f"Got shapes {observed.shape}, {lower_bound.shape}, and {upper_bound.shape}"
        )

    # Find valid indices (non-NaN in all arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(lower_bound) & ~np.isnan(upper_bound)
    valid_observed = observed[valid_mask]
    valid_lower = lower_bound[valid_mask]
    valid_upper = upper_bound[valid_mask]

    # Check if we have enough valid data points
    n_valid = len(valid_observed)
    if n_valid == 0:
        return np.nan

    # Calculate Coverage
    coverage = np.mean((valid_observed >= valid_lower) & (valid_observed <= valid_upper))

    # Handle edge cases
    if np.isinf(coverage) or np.isnan(coverage):
        logger.debug(f"Invalid Coverage value: {coverage}")
        return np.nan

    return coverage



from scipy import integrate


def calculate_crps_from_quantiles(quantile_levels, quantile_forecasts, observation):
    """
    Calculate CRPS from quantile forecasts using numerical integration.

    Parameters:
    -----------
    quantile_levels : list or array
        The probability levels of the quantiles (e.g., [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    quantile_forecasts : list or array
        The forecasted values at each quantile level
    observation : float
        The observed value

    Returns:
    --------
    float
        The CRPS value (lower is better)
    """
    # Convert inputs to numpy arrays
    quantile_levels = np.array(quantile_levels)
    quantile_forecasts = np.array(quantile_forecasts)

    # Ensure quantiles are sorted (should be true for proper quantile forecasts)
    sorted_indices = np.argsort(quantile_forecasts)
    quantile_forecasts = quantile_forecasts[sorted_indices]
    quantile_levels = quantile_levels[sorted_indices]

    # Interpolate the CDF
    def cdf(x):
        return np.interp(x, quantile_forecasts, quantile_levels, left=0.0, right=1.0)

    # Heaviside function for the observation (CDF of the actual observation)
    def heaviside(x):
        return 1.0 if x >= observation else 0.0

    # Function to integrate: (CDF_forecast - CDF_observation)^2
    def integrand(x):
        return (cdf(x) - heaviside(x)) ** 2

    # Define integration range
    min_x = min(quantile_forecasts[0], observation) - 0.1 * (
        quantile_forecasts[-1] - quantile_forecasts[0]
    )
    max_x = max(quantile_forecasts[-1], observation) + 0.1 * (
        quantile_forecasts[-1] - quantile_forecasts[0]
    )

    # Compute the CRPS by numerical integration
    crps, _ = integrate.quad(integrand, min_x, max_x)

    return crps


def calculate_mean_CRPS(
    observed: Union[List, np.ndarray],
    forecasts: Union[pd.DataFrame, np.ndarray],
    quantile_levels: np.ndarray,
) -> float:
    """
    Calculate the Mean Continuous Ranked Probability Score (CRPS) for a set of observations
    and corresponding ensemble forecasts.

    Parameters:
    -----------
    observed : array-like
        Array of observed values
    forecasts : DataFrame or array-like
        Ensemble forecasts, where each column represents one ensemble member
        If DataFrame, columns should be the ensemble members (Q1, Q2, etc.)
        If array, shape should be (n_timesteps, n_members)

    Returns:
    --------
    float
        Normalized mean CRPS score
    """
    # Convert inputs to numpy arrays
    observed = np.array(observed).flatten()
    if isinstance(forecasts, pd.DataFrame):
        forecasts = forecasts.values

    # Ensure forecasts is 2D array
    if forecasts.ndim == 1:
        forecasts = forecasts.reshape(-1, 1)

    n_timesteps = len(observed)
    crps_array = []

    for t in range(n_timesteps):
        # Skip if observation is NaN
        if np.isnan(observed[t]):
            continue

        # Get forecast ensemble for this timestep
        ensemble = forecasts[t]
        ensemble = ensemble[~np.isnan(ensemble)]  # Remove NaN values

        if len(ensemble) == 0:
            continue

        # Calculate CRPS for this timestep
        crps = calculate_crps_from_quantiles(quantile_levels, ensemble, observed[t])
        crps_array.append(crps)

    # Normalize by number of valid timesteps and mean observation
    valid_timesteps = np.sum(~np.isnan(observed))
    mean_obs = np.nanmean(observed)

    if valid_timesteps == 0 or mean_obs == 0:
        return np.nan

    crps_array = np.array(crps_array)
    mean_crps = np.nanmean(crps_array) / mean_obs
    return mean_crps


def calculate_metrics_pentad(
    df,
    forecast_col,
    observed_col,
    delta_col,
):
    df = df.copy()
    metrics = pd.DataFrame()
    for pentad in df.pentad_in_year.unique():
        df_pentad = df[df["pentad_in_year"] == pentad].copy()
        accuracy = forecast_accuracy_hydromet(
            df_pentad, observed_col, forecast_col, delta_col
        ).loc["accuracy"]
        efficiency = sdivsigma_nse(df_pentad, observed_col, forecast_col).loc[
            "sdivsigma"
        ]
        nse = sdivsigma_nse(df_pentad, observed_col, forecast_col).loc["nse"]

        new_row = {
            "pentad_in_year": pentad,
            "accuracy": accuracy,
            "efficiency": efficiency,
            "nse": nse,
        }

        metrics = pd.concat([metrics, pd.DataFrame(new_row, index=[0])])

    return metrics


def calculate_metrics_decad(
    df,
    forecast_col,
    observed_col,
    delta_col,
):
    df = df.copy()
    metrics = pd.DataFrame()
    for decad in df.decad_in_year.unique():
        df_decad = df[df["decad_in_year"] == decad].copy()
        accuracy = forecast_accuracy_hydromet(
            df_decad, observed_col, forecast_col, delta_col
        ).loc["accuracy"]
        efficiency = sdivsigma_nse(df_decad, observed_col, forecast_col).loc[
            "sdivsigma"
        ]
        nse = sdivsigma_nse(df_decad, observed_col, forecast_col).loc["nse"]

        new_row = {
            "decad_in_year": decad,
            "accuracy": accuracy,
            "efficiency": efficiency,
            "nse": nse,
        }

        metrics = pd.concat([metrics, pd.DataFrame(new_row, index=[0])])

    return metrics


# Convenience functions for calibrate_hindcast.py
def r2_score(observed, predicted):
    """Convenience wrapper for R2 calculation."""
    return calculate_R2(observed, predicted)


def rmse(observed, predicted):
    """Convenience wrapper for RMSE calculation."""
    return calculate_RMSE(observed, predicted, normalize=False)


def mae(observed, predicted):
    """Convenience wrapper for MAE calculation."""
    return calculate_MAE(observed, predicted, normalize=False)


def bias(observed, predicted):
    """Calculate bias (mean error)."""
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    # Find valid indices
    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)
    if not np.any(valid_mask):
        return np.nan

    valid_obs = observed[valid_mask]
    valid_pred = predicted[valid_mask]

    return np.mean(valid_pred - valid_obs)


def nse(observed, predicted):
    """Convenience wrapper for NSE calculation."""
    return calculate_NSE(observed, predicted)

def prob_exceedance(observed, predicted):
    """Convenience wrapper for Probability of Exceedance calculation."""
    return calculate_prob_exceedance(observed, predicted)

def coverage(observed, lower_bound, upper_bound):
    """Convenience wrapper for Coverage calculation."""
    return calculate_coverage(observed, lower_bound, upper_bound)

def kge(observed, predicted):
    """Calculate Kling-Gupta Efficiency."""
    observed = np.asarray(observed)
    predicted = np.asarray(predicted)

    # Find valid indices
    valid_mask = ~np.isnan(observed) & ~np.isnan(predicted)
    if not np.any(valid_mask):
        return np.nan

    valid_obs = observed[valid_mask]
    valid_pred = predicted[valid_mask]

    if len(valid_obs) < 2:
        return np.nan

    # Calculate correlation coefficient
    try:
        r = np.corrcoef(valid_obs, valid_pred)[0, 1]
        if np.isnan(r):
            return np.nan
    except:
        return np.nan

    # Calculate bias ratio (mean predicted / mean observed)
    mean_obs = np.mean(valid_obs)
    mean_pred = np.mean(valid_pred)

    if mean_obs == 0:
        return np.nan

    bias_ratio = mean_pred / mean_obs

    # Calculate variability ratio (std predicted / std observed)
    std_obs = np.std(valid_obs, ddof=1)
    std_pred = np.std(valid_pred, ddof=1)

    if std_obs == 0:
        return np.nan

    var_ratio = std_pred / std_obs

    # Calculate KGE
    kge_value = 1 - np.sqrt((r - 1) ** 2 + (bias_ratio - 1) ** 2 + (var_ratio - 1) ** 2)

    return kge_value
