import sys
import glob
import os
import re

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import stats

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

import tqdm
from collections import defaultdict

# Shared logging
import logging
from log_config import setup_logging

setup_logging()


def get_periods(df) -> pd.DataFrame:
    """
    Get unique periods from the data.
    Creates 36 periods per year: days 10, 20, and end of each month.
    The period name is <month>-<day> where day is "10", "20", or "end".
    if it is the last day of the month, the period name is <month>-end. (accounts for february)

    Returns:
        pd.DataFrame: DataFrame containing unique periods.
    """
    df = df.copy()
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month

    # Determine period suffix based on day of month
    # Days 1-10 -> "10"
    # Days 11-20 -> "20"
    # Days 21-end -> "end"
    conditions = [df["day"] <= 10, df["day"] <= 20, df["day"] > 20]
    choices = ["10", "20", "end"]

    df["period_suffix"] = np.select(conditions, choices, default="end")
    df["period"] = df["month"].astype(str) + "-" + df["period_suffix"]

    df.drop(columns=["period_suffix"], inplace=True)

    return df


def discharge_m3_to_mm(df):
    pass


def discharge_mm_to_m3(df):
    pass


def create_target(df, column="discharge", prediction_horizon=30, offset=None):
    """
    Create target variable: average discharge for next N days, by basin.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input DataFrame with 'code' column
    column : str, default='discharge'
        Column to create target from

    Returns:
    --------
    pandas.DataFrame
        DataFrame with target column added
    """
    if offset is None:
        offset = prediction_horizon

    min_periods = min(int(prediction_horizon * 0.75), 15)
    target = pd.Series(index=df.index, dtype=float)

    # Calculate target for each basin separately
    for code in df["code"].unique():
        basin_data = df[df["code"] == code][column]

        # Calculate future average
        future_avg = (
            basin_data.rolling(window=prediction_horizon, min_periods=min_periods)
            .mean()
            .shift(-offset)
        )

        # Assign values back to the correct rows
        target.loc[basin_data.index] = future_avg

    #
    # Add target to the original DataFrame
    df["target"] = target
    return df


def glacier_mapper_features(
    df: pd.DataFrame,
    static: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    static = static.copy()

    for code in df["code"].unique():
        area = static.loc[static["code"] == code, "area_km2"].values[0]
        gl_fr = static.loc[static["code"] == code, "gl_fr"].values[0]
        glacier_area = area * gl_fr
        h_min = static.loc[static["code"] == code, "h_min"].values[0]
        h_max = static.loc[static["code"] == code, "h_max"].values[0]

        df.loc[df["code"] == code, "gla_area_below_sl50"] /= glacier_area
        df.loc[df["code"] == code, "gla_area_below_sl50"] *= 100
        df.loc[df["code"] == code, "gla_fsc_total"] *= 100
        df.loc[df["code"] == code, "gla_fsc_below_sl50"] *= 100
        df.loc[df["code"] == code, "fsc_basin"] *= 100

        # glacier_melt_potential =(100 - gla_fsc_total) * gl_fr
        df.loc[df["code"] == code, "glacier_melt_potential"] = (
            100 - df.loc[df["code"] == code, "gla_fsc_total"]
        ) * gl_fr

        # Normalize the SLA values   with SLA_norm = (SLA - h_min) / (h_max - h_min)
        df.loc[df["code"] == code, "SLA_East"] = (
            df.loc[df["code"] == code, "SLA_East"] - h_min
        ) / (h_max - h_min)
        df.loc[df["code"] == code, "SLA_West"] = (
            df.loc[df["code"] == code, "SLA_West"] - h_min
        ) / (h_max - h_min)
        df.loc[df["code"] == code, "SLA_North"] = (
            df.loc[df["code"] == code, "SLA_North"] - h_min
        ) / (h_max - h_min)
        df.loc[df["code"] == code, "SLA_South"] = (
            df.loc[df["code"] == code, "SLA_South"] - h_min
        ) / (h_max - h_min)
        df.loc[df["code"] == code, "SLA_Avr"] = (
            df.loc[df["code"] == code, "SLA_East"]
            + df.loc[df["code"] == code, "SLA_West"]
            + df.loc[df["code"] == code, "SLA_North"]
            + df.loc[df["code"] == code, "SLA_South"]
        ) / 4
    return df


def get_elevation_bands_per_percentile(
    elevation_band_code, num_bands=3, rel_area_col="relative_a"
):
    """
    Assigns elevation bands to percentiles based on the elevation.
    returns: a dictionary with {P1 : {bands : [1,2,3...], relative_area : [0.1, 0.2, 0.3...]}
    """
    # Calculate percentiles based on elevation
    elevation_band_code = elevation_band_code.copy()
    elevation_band_code["elevation_band"] = elevation_band_code["name"].apply(
        lambda x: int(str(x).split("_")[-1]) if "_" in str(x) else int(x)
    )
    elevation_band_code = elevation_band_code.sort_values(by="elevation_band")

    # Calculate cumulative area
    total_area = elevation_band_code[rel_area_col].sum()
    elevation_band_code["cum_area"] = (
        elevation_band_code[rel_area_col].cumsum() / total_area
    )

    # Define percentile thresholds based on num_bands
    thresholds = [i / num_bands for i in range(1, num_bands)]

    # Initialize result dictionary
    result = {}
    for i in range(num_bands):
        band_name = f"Perc_Elev_{i + 1}"
        result[band_name] = {"bands": [], "relative_area": []}

    # Assign bands to percentiles
    current_percentile = 0

    for i, row in elevation_band_code.iterrows():
        # Find which percentile this band belongs to
        cum_area = row["cum_area"]
        band_idx = 0

        # Determine which percentile band this elevation belongs to
        for j, threshold in enumerate(thresholds):
            if cum_area <= threshold:
                band_idx = j
                break
            band_idx = j + 1

        band_name = f"Perc_Elev_{band_idx + 1}"
        result[band_name]["bands"].append(int(row["elevation_band"]))
        result[band_name]["relative_area"].append(row[rel_area_col])

    return result


def calculate_percentile_snow_bands(
    hydro_df, elevation_band_df, num_bands=3, col_name="SWE"
):
    """
    Calculates SWE for n elevation bands based on area percentiles: example col name = SWE
        - SWE_P1: 0-p1 percentile
        - SWE_P2: p1-p2 percentile
        -....
        - SWE_Pn: pn-100 percentile
    """
    hydro_df = hydro_df.copy()

    # Initialize the new SWE percentile columns
    for i in range(1, num_bands + 1):
        hydro_df[f"{col_name}_Perc_Elev_{i}"] = 0.0

    # Process each basin code separately
    for code in hydro_df.code.unique():
        # Filter data for current basin
        basin_mask = elevation_band_df["code"] == code
        basin_elev_data = elevation_band_df[basin_mask].copy()

        if basin_elev_data.empty:
            continue

        # Get elevation bands per percentile for this basin
        percentile_bands = get_elevation_bands_per_percentile(
            basin_elev_data, num_bands=num_bands, rel_area_col="relative_a"
        )

        # Calculate SWE for each percentile band
        hydro_mask = hydro_df["code"] == code

        for percentile, data in percentile_bands.items():
            percentile_col_name = f"{col_name}_{percentile}"
            bands = data["bands"]
            rel_areas = data["relative_area"]

            # Total weight for normalization
            total_weight = sum(rel_areas)

            if total_weight == 0:
                continue

            # Calculate weighted SWE for this percentile
            for band, rel_area in zip(bands, rel_areas):
                swe_col = f"{col_name}_{band}"

                if swe_col not in hydro_df.columns:
                    continue

                # Add weighted contribution
                hydro_df.loc[hydro_mask, percentile_col_name] += (
                    hydro_df.loc[hydro_mask, swe_col] * rel_area
                )

            # Normalize by total weight
            hydro_df.loc[hydro_mask, percentile_col_name] /= total_weight

    return hydro_df


def get_normalization_params(df_train, features, target=None):
    """
    Calculate normalization parameters (mean and std) from training data.

    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe
    features : list
        List of feature columns to normalize
    target : str, optional
        Target column to normalize. If None, only features are normalized.

    Returns:
    --------
    dict
        Dictionary containing mean and std for each column
    """
    scaler = {}
    cols_to_normalize = features.copy()
    if target is not None:
        cols_to_normalize.append(target)

    for col in cols_to_normalize:
        mean_ = df_train[col].astype(float).mean()
        std_ = df_train[col].astype(float).std()
        std_ = 1 if std_ == 0 else std_  # Avoid division by zero
        scaler[col] = (mean_, std_)

    return scaler


def apply_normalization(df, scaler, features):
    """
    Apply normalization using pre-computed parameters.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to normalize
    scaler : dict
        Dictionary containing (mean, std) for each column
    features : list
        List of feature columns to normalize

    Returns:
    --------
    pd.DataFrame
        Normalized dataframe
    """
    df = df.copy()

    for col in features:
        if col in scaler:
            df[col] = df[col].astype(float)
            mean_, std_ = scaler[col]
            df[col] = (df[col] - mean_) / std_

    return df


def normalize_features(df_train, df_test, features, target):
    """
    Normalize features and target using mean and standard deviation from training data.
    """
    # Get normalization parameters from training data
    scaler = get_normalization_params(df_train, features, target)

    # Apply normalization to both datasets
    cols_to_normalize = features + [target]
    df_train_normalized = apply_normalization(df_train, scaler, cols_to_normalize)
    df_test_normalized = apply_normalization(df_test, scaler, cols_to_normalize)

    return df_train_normalized, df_test_normalized, scaler


def inverse_normalization(
    df: pd.DataFrame, scaler: dict, var_to_scale: str, var_used_for_scaling: str
):
    df = df.copy()
    if var_to_scale in df.columns and var_used_for_scaling in df.columns:
        df[var_to_scale] = df[var_to_scale].astype(float)
        mean_val, std_val = scaler[var_used_for_scaling]
        df[var_to_scale] = df[var_to_scale] * std_val + mean_val

    return df


def apply_inverse_normalization(
    df: pd.DataFrame, scaler: dict, var_to_scale: str, var_used_for_scaling: str
) -> pd.DataFrame:
    """
    Apply inverse normalization (denormalization) using global scaling parameters.
    Transforms scaled values back to original scale: x_original = x_scaled * std + mean

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing the variable to inverse scale
    scaler : dict
        Dictionary containing (mean, std) tuples for each variable
    var_to_scale : str
        Column name to inverse transform
    var_used_for_scaling : str
        Variable name whose scaling parameters should be used

    Returns:
    --------
    pd.DataFrame
        DataFrame with inverse scaled variable
    """
    df = df.copy()

    if var_to_scale in df.columns and var_used_for_scaling in scaler:
        df[var_to_scale] = df[var_to_scale].astype(float)
        mean_val, std_val = scaler[var_used_for_scaling]
        # Apply inverse transformation: multiply by std and add mean
        df[var_to_scale] = (df[var_to_scale] * std_val) + mean_val

    return df


def get_normalization_params_per_basin(df_train, features, target=None):
    """
    Calculate normalization parameters per basin from training data.

    Parameters:
    -----------
    df_train : pd.DataFrame
        Training dataframe with 'code' column for basin identification
    features : list
        List of feature columns to normalize
    target : str, optional
        Target column to normalize. If None, only features are normalized.

    Returns:
    --------
    dict
        Nested dictionary: {basin_code: {column: (mean, std)}}
    """
    cols_to_normalize = features.copy()
    if target is not None:
        cols_to_normalize.append(target)

    # Pre-compute statistics for all basins at once
    basin_stats = df_train.groupby("code")[cols_to_normalize].agg(["mean", "std"])

    # Replace zero std with 1
    basin_stats.loc[:, (slice(None), "std")] = basin_stats.loc[
        :, (slice(None), "std")
    ].replace(0, 1)

    # Initialize scaler dictionary
    scaler = {}

    for code in df_train.code.unique():
        scaler[code] = {}
        for col in cols_to_normalize:
            mean_val = basin_stats.loc[code, (col, "mean")]
            std_val = basin_stats.loc[code, (col, "std")]
            scaler[code][col] = (mean_val, std_val)

    return scaler


def apply_normalization_per_basin(df, scaler, features):
    """
    Apply per-basin normalization using pre-computed parameters.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe to normalize with 'code' column for basin identification
    scaler : dict
        Nested dictionary: {basin_code: {column: (mean, std)}}
    features : list
        List of feature columns to normalize

    Returns:
    --------
    pd.DataFrame
        Normalized dataframe
    """
    df = df.copy()

    for code in df.code.unique():
        if code in scaler:
            basin_mask = df["code"] == code

            for col in features:
                if col in scaler[code]:
                    # Ensure column is float type before normalization to avoid dtype warnings
                    df[col] = df[col].astype(float)
                    mean_val, std_val = scaler[code][col]
                    mean_val = float(mean_val)
                    std_val = float(std_val)
                    df.loc[basin_mask, col] = (
                        df.loc[basin_mask, col] - mean_val
                    ) / std_val

    return df


def apply_inverse_normalization_per_basin(
    df: pd.DataFrame, scaler: dict, var_to_scale: str, var_used_for_scaling: str
):
    """
    Apply inverse normalization (denormalization) per basin.
    Transforms scaled values back to original scale: x_original = x_scaled * std + mean
    """
    df = df.copy()

    for code in df.code.unique():
        if code in scaler:
            basin_mask = df["code"] == code

            if var_to_scale in df.columns and var_used_for_scaling in scaler[code]:
                df.loc[basin_mask, var_to_scale] = df.loc[
                    basin_mask, var_to_scale
                ].astype(float)
                mean_val, std_val = scaler[code][var_used_for_scaling]
                # Apply inverse transformation: multiply by std and add mean
                df.loc[basin_mask, var_to_scale] = (
                    df.loc[basin_mask, var_to_scale] * std_val
                ) + mean_val

    return df


def normalize_features_per_basin(df_train, df_test, features, target):
    """
    Fast normalization of features and target per basin using vectorized operations.
    """
    # Get normalization parameters from training data
    scaler = get_normalization_params_per_basin(df_train, features, target)

    # Apply normalization to both datasets
    cols_to_normalize = features + [target]
    df_train_normalized = apply_normalization_per_basin(
        df_train, scaler, cols_to_normalize
    )
    df_test_normalized = apply_normalization_per_basin(
        df_test, scaler, cols_to_normalize
    )

    return df_train_normalized, df_test_normalized, scaler


def get_relative_scaling_features(features, relative_scaling_vars):
    """
    Identify features that should use relative scaling based on pattern matching.

    Parameters:
    -----------
    features : list
        List of all feature column names
    relative_scaling_vars : list
        List of variable patterns to match (e.g., ["SWE", "T", "discharge"])

    Returns:
    --------
    list
        List of features that match any of the patterns

    Example:
    --------
    >>> features = ["SWE_1", "SWE_2", "SWE_Perc_Elev_1", "P_1", "discharge"]
    >>> relative_scaling_vars = ["SWE", "discharge"]
    >>> get_relative_scaling_features(features, relative_scaling_vars)
    ["SWE_1", "SWE_2", "SWE_Perc_Elev_1", "discharge"]
    """
    if not relative_scaling_vars:
        return []

    relative_features = []
    for feature in features:
        for var_pattern in relative_scaling_vars:
            # Check if the feature starts with the pattern followed by underscore or is exact match
            if feature == var_pattern or feature.startswith(f"{var_pattern}_"):
                relative_features.append(feature)
                break  # No need to check other patterns for this feature

    return relative_features


def get_long_term_mean_per_basin(df, features):
    """
    Calculate long-term mean and standard deviation for each feature per basin and period.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with 'date' and 'code' columns plus feature columns
    features : list
        List of feature columns to calculate statistics for

    Returns:
    --------
    pd.DataFrame
        DataFrame with long-term mean and std for each feature per basin and period
    """
    df = df.copy()

    # Add period column using existing get_periods function
    df = get_periods(df)

    # Group by code and period (36 groups per basin per year)
    groupby_cols = ["code", "period"]

    grouped = df.groupby(groupby_cols)

    # Calculate both mean and std
    long_term_stats = grouped[features].agg(["mean", "std"]).reset_index()

    # Handle zero std by replacing with 1 (to avoid division by zero)
    for feature in features:
        std_col = (feature, "std")
        if std_col in long_term_stats.columns:
            long_term_stats.loc[long_term_stats[std_col] == 0, std_col] = 1.0
            # Also replace NaN std with 1.0 (happens when group has only one value)
            long_term_stats[std_col] = long_term_stats[std_col].fillna(1.0)

    return long_term_stats


def apply_long_term_mean(df, long_term_mean, features):
    """
    Apply long-term mean to the DataFrame in a vectorized way:
      - merge on basin code and month
      - fill missing feature values with the corresponding long-term mean
    """
    df = df.copy()
    df["month"] = df["date"].dt.month  # ensure month col exists

    # --- 1) flatten multi-index columns if needed ---
    ltm = long_term_mean.copy()
    if isinstance(ltm.columns, pd.MultiIndex):
        # after agg(['mean']), cols look like (feature, 'mean')
        ltm.columns = ["code", "month"] + [f"{feat}_mean" for feat in features]
    else:
        # if you already ran grouped.mean(), you just need to rename
        rename_map = {feat: f"{feat}_mean" for feat in features}
        ltm = ltm.rename(columns=rename_map)

    # --- 2) merge the long-term means back onto the original ---
    df = df.merge(ltm, on=["code", "month"], how="left")

    # --- 3) fill in missing values from the long-term mean ---
    for feat in features:
        mean_col = f"{feat}_mean"
        # Check if we have duplicate columns that cause df[mean_col] to return a DataFrame
        if isinstance(df[mean_col], pd.DataFrame):
            # Use the first occurrence of the column
            df[feat] = df[feat].fillna(df[mean_col].iloc[:, 0])
        else:
            # Normal case - mean_col is a Series
            df[feat] = df[feat].fillna(df[mean_col])

    # --- 4) drop helper columns and return ---
    drop_cols = [f"{feat}_mean" for feat in features]

    return df.drop(columns=drop_cols)


def apply_long_term_mean_scaling(df, long_term_stats, features, features_to_scale=None):
    """
    Apply standardization scaling using long-term statistics:
      - merge on basin code and period
      - scale features using formula: (x - mean) / std

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features to scale
    long_term_stats : pd.DataFrame
        DataFrame with long-term mean and std for each feature per basin and period
    features : list
        List of all feature columns in the data
    features_to_scale : list, optional
        List of features to apply scaling to. If None, scales all features.

    Returns:
    --------
    pd.DataFrame
        DataFrame with selected features scaled
    """
    df = df.copy()

    # Add period column using existing get_periods function
    df = get_periods(df)

    # Default to scaling all features if not specified
    if features_to_scale is None:
        features_to_scale = features

    # --- 1) flatten multi-index columns if needed ---
    lts = long_term_stats.copy()
    if isinstance(lts.columns, pd.MultiIndex):
        # after agg(['mean', 'std']), cols look like (feature, 'mean'), (feature, 'std')
        new_columns = []
        for col in lts.columns:
            if col[1] == "":  # This is for 'code' and 'period' columns
                new_columns.append(col[0])
            elif col[1] in ["mean", "std"]:  # This is for feature columns
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                # Handle any other column structure
                new_columns.append("_".join(str(x) for x in col if x))
        lts.columns = new_columns

    # --- 2) merge the long-term stats back onto the original ---
    df = df.merge(lts, on=["code", "period"], how="left")

    # --- 3) apply standardization formula to selected features ---
    for feat in features_to_scale:
        if feat in features:  # Only scale if feature exists
            mean_col = f"{feat}_mean"
            std_col = f"{feat}_std"

            if mean_col in df.columns and std_col in df.columns:
                # Apply standardization formula: (x - mean) / std
                df[feat] = (df[feat] - df[mean_col]) / df[std_col]

    # --- 4) drop helper columns and return ---
    drop_cols = []
    for feat in features:  # Drop stats for all features, not just scaled ones
        drop_cols.extend([f"{feat}_mean", f"{feat}_std"])
    drop_cols = [col for col in drop_cols if col in df.columns]

    # Also drop the period column we added
    if "period" in df.columns:
        drop_cols.append("period")

    return df.drop(columns=drop_cols)


def apply_inverse_long_term_mean_scaling(
    df: pd.DataFrame,
    long_term_stats: pd.DataFrame,
    features_to_inverse_scale: list,
):
    """
    Inverse standardization scaling using long-term statistics.
    Transforms scaled values back to original scale using: x_original = x_scaled * std + mean

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with features to inverse scale
    long_term_stats : pd.DataFrame
        DataFrame with long-term mean and std for each feature per basin and period
    features_to_inverse_scale : list
        List of feature columns to inverse scale

    Returns:
    --------
    pd.DataFrame
        DataFrame with features inverse scaled to original values
    """
    df = df.copy()

    # Add period column using existing get_periods function
    df = get_periods(df)

    # --- 1) flatten multi-index columns if needed ---
    lts = long_term_stats.copy()
    if isinstance(lts.columns, pd.MultiIndex):
        new_columns = []
        for col in lts.columns:
            if col[1] == "":  # This is for 'code' and 'period' columns
                new_columns.append(col[0])
            elif col[1] in ["mean", "std"]:  # This is for feature columns
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append("_".join(str(x) for x in col if x))
        lts.columns = new_columns

    # --- 2) merge the long-term stats back onto the original ---
    df = df.merge(lts, on=["code", "period"], how="left")

    # --- 3) apply inverse standardization formula ---
    for feat in features_to_inverse_scale:
        mean_col = f"{feat}_mean"
        std_col = f"{feat}_std"

        if mean_col in df.columns and std_col in df.columns:
            # Apply inverse standardization: x_original = x_scaled * std + mean
            df[feat] = df[feat] * df[std_col] + df[mean_col]

    # --- 4) drop helper columns and return ---
    drop_cols = []
    for feat in features_to_inverse_scale:
        drop_cols.extend([f"{feat}_mean", f"{feat}_std"])
    drop_cols = [col for col in drop_cols if col in df.columns]

    # Also drop the period column we added
    if "period" in df.columns:
        drop_cols.append("period")

    return df.drop(columns=drop_cols)


def apply_inverse_long_term_mean_scaling_predictions(
    df: pd.DataFrame,
    long_term_stats: pd.DataFrame,
    prediction_col: str,
    target_col: str,
):
    """
    Inverse standardization scaling specifically for predictions.
    Uses the target variable's statistics to transform predictions back to original scale.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with predictions to inverse scale (must have 'date' and 'code' columns)
    long_term_stats : pd.DataFrame
        DataFrame with long-term mean and std for each feature per basin and period
    prediction_col : str
        Name of the column containing predictions to inverse scale
    target_col : str
        Name of the target variable whose statistics should be used

    Returns:
    --------
    pd.DataFrame
        DataFrame with predictions inverse scaled to original values

    Note:
    -----
    This function is critical for fixing the R2 degradation issue. It ensures that
    predictions are inverse-transformed using the correct (target) statistics.
    """
    df = df.copy()

    # Add period column using existing get_periods function
    df = get_periods(df)

    # --- 1) flatten multi-index columns if needed ---
    lts = long_term_stats.copy()
    if isinstance(lts.columns, pd.MultiIndex):
        new_columns = []
        for col in lts.columns:
            if col[1] == "":  # This is for 'code' and 'period' columns
                new_columns.append(col[0])
            elif col[1] in ["mean", "std"]:  # This is for feature columns
                new_columns.append(f"{col[0]}_{col[1]}")
            else:
                new_columns.append("_".join(str(x) for x in col if x))
        lts.columns = new_columns

    # --- 2) merge the long-term stats back onto the original ---
    df = df.merge(lts, on=["code", "period"], how="left")

    # --- 3) apply inverse standardization using target statistics ---
    target_mean_col = f"{target_col}_mean"
    target_std_col = f"{target_col}_std"

    if target_mean_col in df.columns and target_std_col in df.columns:
        # Apply inverse standardization: x_original = x_scaled * std + mean
        df[prediction_col] = (
            df[prediction_col] * df[target_std_col] + df[target_mean_col]
        )
    else:
        raise ValueError(f"Target statistics not found for {target_col}")

    # --- 4) drop all helper columns and return ---
    # Get all stat columns to drop
    stat_cols = [
        col for col in df.columns if col.endswith("_mean") or col.endswith("_std")
    ]
    drop_cols = stat_cols.copy()

    # Also drop the period column we added
    if "period" in df.columns:
        drop_cols.append("period")

    return df.drop(columns=drop_cols)


def calculate_elevation_band_areas(gdf_path):
    """
    Calculate relative areas for each elevation band by code.

    Parameters:
    -----------
    gdf : GeoDataFrame
        Geodataframe containing elevation band polygons with names in format 'code_band'

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns for code and relative areas of each elevation band
    """
    # Read GeoDataFrame
    gdf = gpd.read_file(gdf_path)
    # Split name into code and band
    gdf["code"] = gdf["name"].str.split("_").str[0].astype(int)
    gdf["band"] = gdf["name"].str.split("_").str[1].astype(int)

    # Calculate areas
    gdf["area"] = gdf.geometry.area

    # Create dictionary to store results
    results = defaultdict(dict)

    # Process each code
    for code in gdf["code"].unique():
        # Get data for this code
        code_data = gdf[gdf["code"] == code]

        # Calculate total area for this code
        total_area = code_data["area"].sum()

        # Calculate relative areas for each band
        for _, row in code_data.iterrows():
            relative_area = row["area"] / total_area
            results[code][f"relative_area_{row['band']}"] = relative_area

        # Add code identifier
        results[code]["code"] = code

    # Convert results to DataFrame
    df_results = pd.DataFrame.from_dict(results, orient="index")

    # Ensure all elevation bands are present (fill missing with 0)
    all_bands = [f"relative_area_{i}" for i in range(1, gdf["band"].max() + 1)]
    for band in all_bands:
        if band not in df_results.columns:
            df_results[band] = 0.0

    # Sort columns
    cols = ["code"] + sorted([col for col in df_results.columns if col != "code"])
    df_results = df_results[cols]

    # nan to 0
    df_results = df_results.fillna(0)

    # Reset index
    df_results = df_results.reset_index(drop=True)

    return df_results


def average_by_elevation_band(df, elevation_band_areas):
    df = df.copy()

    for code in df.code.unique():
        mask_df = df.code == code
        mask_elev = elevation_band_areas.code == code

        for col in df.columns:
            if "_" in col:
                try:
                    band = int(col.split("_")[1])
                    relative_area = elevation_band_areas.loc[
                        mask_elev, f"relative_area_{band}"
                    ].values[0]
                    df.loc[mask_df, col] = df.loc[mask_df, col] * relative_area
                except Exception as e:
                    print("-" * 50)
                    print(f"Error processing column {col}")
                    print(e)

    return df


def create_lag_features(df, features, lags):
    df = df.copy()

    for code in df.code.unique():
        mask = df.code == code

        for feature in features:
            for lag in lags:
                df.loc[mask, f"{feature}_lag_{lag}"] = df.loc[mask, feature].shift(lag)

    return df


def agg_with_min_obs(x, func="mean", min_obs=15):
    """
    Aggregate data only if there are enough valid observations

    Parameters:
    -----------
    x : Series
        Data to aggregate
    func : str
        'mean' or 'sum'
    min_obs : int
        Minimum number of non-NaN observations required

    Returns:
    --------
    float or NaN
        Aggregated value if enough observations, NaN otherwise
    """
    valid_count = x.notna().sum()
    if valid_count >= min_obs:
        if func == "last":
            return np.nanmean(x.iloc[-5:])  # Get last value this way instead
        return x.agg(func)
    return np.nan


def create_monthly_df(df, feature_cols):
    df = df.copy()

    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    groupby_cols = ["year", "month"]

    # Base aggregations
    agg_dict = {
        "discharge": lambda y: agg_with_min_obs(y, "mean"),
        "T": "mean",
        "P": "sum",
    }

    # Add feature column aggregations
    for col in feature_cols:
        df[f"{col}_mean"] = df[col].bfill().values
        # df[f'{col}_mean_last'] = df[col].bfill().values
        agg_dict[f"{col}_mean"] = "mean"
        # agg_dict[f'{col}_mean_last'] = lambda y: agg_with_min_obs(y, 'last')

    # Apply the groupby with all aggregations
    hydro_df = df.groupby(groupby_cols).agg(agg_dict).reset_index()

    return hydro_df
