"""
Prediction Loading Module

Centralized utilities for loading and processing model predictions from various sources.
Enables database compatibility by decoupling prediction loading from model classes.

This module provides functions to:
- Load predictions from filesystem (CSV files)
- Load predictions from pre-loaded DataFrames (e.g., from databases)
- Apply area-based unit conversions
- Handle duplicate predictions
- Standardize prediction column formats
"""

import logging
import os
from typing import List, Literal, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


def load_predictions_from_filesystem(
    paths: List[str],
    join_type: Literal["inner", "left"] = "inner",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load predictions from CSV files in filesystem.

    Expects CSV files with columns: date, code, Q_{model_name}
    Model name is extracted from the parent directory of the CSV file.

    Args:
        paths: List of file paths to prediction CSV files or directories containing predictions.csv
        join_type: How to merge predictions across models ("inner" or "left")

    Returns:
        Tuple of (predictions_df, model_column_names)
        - predictions_df: DataFrame with columns [date, code, Q_{model1}, Q_{model2}, ...]
        - model_column_names: List of prediction column names (e.g., ['Q_model1', 'Q_model2'])

    Raises:
        FileNotFoundError: If any path doesn't exist
        ValueError: If no valid predictions found

    Example:
        >>> paths = ["/data/model1/predictions.csv", "/data/model2/predictions.csv"]
        >>> preds, cols = load_predictions_from_filesystem(paths)
        >>> print(cols)
        ['Q_model1', 'Q_model2']
    """
    if not paths:
        raise ValueError("At least one path must be provided")

    all_predictions = None
    pred_cols = []

    for path in paths:
        # Handle directory paths - add predictions.csv if not specified
        if not path.endswith(".csv"):
            path = os.path.join(path, "predictions.csv")

        # Extract model name from folder structure (after path normalization)
        model_name = os.path.basename(os.path.dirname(path))

        # Check if file exists
        if not os.path.exists(path):
            logger.warning(f"Prediction file not found: {path}. Skipping.")
            continue

        logger.info(f"Loading predictions from {path} for model {model_name}")

        # Load predictions
        df = pd.read_csv(path)
        df["date"] = pd.to_datetime(df["date"])
        df["code"] = df["code"].astype(int)

        # Identify prediction column
        pred_col = f"Q_{model_name}"

        # Check if prediction column exists
        if pred_col not in df.columns:
            logger.warning(
                f"Prediction column '{pred_col}' not found in {model_name}. "
                f"Available columns: {df.columns.tolist()}. Skipping this model."
            )
            continue

        pred_cols.append(pred_col)

        # Initialize or merge predictions
        if all_predictions is None:
            all_predictions = df[["date", "code", pred_col]].copy()
        else:
            all_predictions = pd.merge(
                all_predictions,
                df[["date", "code", pred_col]],
                on=["date", "code"],
                how=join_type,
            )

    if all_predictions is None or len(pred_cols) == 0:
        raise ValueError(
            f"No valid predictions found in provided paths. Checked {len(paths)} paths."
        )

    logger.info(f"Successfully loaded {len(pred_cols)} prediction models: {pred_cols}")
    logger.debug(
        f"Prediction DataFrame shape: {all_predictions.shape}, "
        f"Date range: {all_predictions['date'].min()} to {all_predictions['date'].max()}"
    )

    return all_predictions, pred_cols


def load_predictions_from_dataframe(
    df: pd.DataFrame,
    model_names: List[str],
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load predictions from a pre-loaded DataFrame (e.g., from database).

    Validates structure and returns in standardized format.

    Args:
        df: DataFrame with columns [date, code, {model_name1}, {model_name2}, ...]
        model_names: List of model column names (without Q_ prefix)

    Returns:
        Tuple of (predictions_df, model_column_names)
        - predictions_df: Validated and standardized DataFrame
        - model_column_names: List of prediction column names with Q_ prefix

    Raises:
        ValueError: If required columns missing or invalid structure

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-02'],
        ...     'code': [1, 1],
        ...     'model1': [100, 110],
        ...     'model2': [95, 105]
        ... })
        >>> preds, cols = load_predictions_from_dataframe(df, ['model1', 'model2'])
        >>> print(cols)
        ['Q_model1', 'Q_model2']
    """
    # Validate required columns
    required_cols = ["date", "code"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"DataFrame missing required columns: {missing_cols}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Ensure proper data types
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["code"].astype(int)

    # Build prediction column names with Q_ prefix
    pred_cols = [
        f"Q_{name}" if not name.startswith("Q_") else name for name in model_names
    ]

    # Check if all model columns exist
    for i, pred_col in enumerate(pred_cols):
        # Handle both Q_model and model naming
        base_name = model_names[i]
        if pred_col not in df.columns and base_name not in df.columns:
            raise ValueError(
                f"Model column '{pred_col}' (or '{base_name}') not found in DataFrame. "
                f"Available columns: {df.columns.tolist()}"
            )
        # Rename if necessary
        if pred_col not in df.columns and base_name in df.columns:
            df = df.rename(columns={base_name: pred_col})

    # Select relevant columns
    result_df = df[["date", "code"] + pred_cols].copy()

    logger.info(
        f"Loaded predictions from DataFrame with {len(pred_cols)} models: {pred_cols}"
    )
    logger.debug(
        f"Prediction DataFrame shape: {result_df.shape}, "
        f"Date range: {result_df['date'].min()} to {result_df['date'].max()}"
    )

    return result_df, pred_cols


def apply_area_conversion(
    predictions: pd.DataFrame,
    static_data: pd.DataFrame,
    pred_cols: List[str],
) -> pd.DataFrame:
    """
    Apply area-based unit conversion to predictions.

    Converts discharge from m³/s to mm/month (or similar) using:
        converted_value = value * area_km2 / 86.4

    Args:
        predictions: DataFrame with predictions (must have 'code' column)
        static_data: DataFrame with basin metadata (must have 'code' and 'area_km2' columns)
        pred_cols: List of prediction column names to convert

    Returns:
        DataFrame with converted predictions (modifies in place)

    Raises:
        ValueError: If required columns missing or codes not found

    Example:
        >>> preds = pd.DataFrame({
        ...     'date': ['2024-01-01'],
        ...     'code': [1],
        ...     'Q_model1': [100.0]
        ... })
        >>> static = pd.DataFrame({'code': [1], 'area_km2': [1000.0]})
        >>> converted = apply_area_conversion(preds, static, ['Q_model1'])
        >>> print(converted['Q_model1'].values[0])  # 100 * 1000 / 86.4
    """
    # Validate inputs
    if "code" not in predictions.columns:
        raise ValueError("Predictions DataFrame must have 'code' column")

    if "code" not in static_data.columns or "area_km2" not in static_data.columns:
        raise ValueError(
            "Static data must have 'code' and 'area_km2' columns. "
            f"Available columns: {static_data.columns.tolist()}"
        )

    # Work on a copy to avoid modifying original
    result = predictions.copy()

    # Apply conversion for each basin code
    for code in result["code"].unique():
        # Check if code exists in static data
        if code not in static_data["code"].values:
            logger.warning(
                f"Code {code} not found in static data. Skipping conversion for this basin."
            )
            continue

        # Get area for this basin
        area = static_data[static_data["code"] == code]["area_km2"].values[0]

        # Apply conversion: value * area / 86.4
        result.loc[result["code"] == code, pred_cols] = (
            result.loc[result["code"] == code, pred_cols] * area / 86.4
        )

        logger.debug(f"Applied area conversion for code {code} (area={area} km²)")

    return result


def handle_duplicate_predictions(
    df: pd.DataFrame,
    pred_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Handle duplicate (date, code) pairs by averaging prediction values.

    Args:
        df: DataFrame with predictions
        pred_cols: List of prediction column names to average.
                  If None, automatically detects Q_ columns (excluding Q_obs)

    Returns:
        DataFrame with duplicates removed (averaged)

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01', '2024-01-01', '2024-01-02'],
        ...     'code': [1, 1, 1],
        ...     'Q_model1': [100, 110, 120]
        ... })
        >>> result = handle_duplicate_predictions(df)
        >>> print(len(result))  # 2 rows (duplicate averaged)
        >>> print(result[result['date'] == '2024-01-01']['Q_model1'].values[0])  # 105
    """
    # Check for duplicates
    n_duplicates = df[["date", "code"]].duplicated().sum()

    if n_duplicates == 0:
        logger.debug("No duplicate (date, code) pairs found")
        return df.copy()

    logger.warning(
        f"Found {n_duplicates} duplicate (date, code) pairs. "
        f"Averaging prediction values for duplicates."
    )

    # Auto-detect prediction columns if not provided
    if pred_cols is None:
        pred_cols = [col for col in df.columns if "Q_" in col and col != "Q_obs"]
        logger.debug(f"Auto-detected prediction columns: {pred_cols}")

    # Create aggregation dictionary for all columns (except groupby keys)
    agg_dict = {}
    for col in df.columns:
        if col not in ["date", "code"]:
            # Average all numeric/prediction columns
            agg_dict[col] = "mean"

    # Average predictions for duplicate (date, code) combinations
    result = df.groupby(["date", "code"], as_index=False).agg(agg_dict)

    logger.info(
        f"After deduplication: {len(result)} unique (date, code) pairs "
        f"(reduced from {len(df)})"
    )

    return result


def standardize_prediction_columns(
    df: pd.DataFrame,
    ensure_prefix: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Standardize prediction column names to Q_{model_name} format.

    Args:
        df: DataFrame with predictions
        ensure_prefix: If True, ensure all prediction columns start with Q_

    Returns:
        Tuple of (standardized_df, prediction_column_names)

    Example:
        >>> df = pd.DataFrame({
        ...     'date': ['2024-01-01'],
        ...     'code': [1],
        ...     'model1': [100],
        ...     'Q_model2': [95]
        ... })
        >>> result, cols = standardize_prediction_columns(df)
        >>> print(cols)
        ['Q_model1', 'Q_model2']
    """
    result = df.copy()
    pred_cols = []

    # Find all prediction-like columns
    for col in df.columns:
        if col in ["date", "code", "Q_obs"]:
            continue

        # Check if it looks like a prediction column
        is_pred_col = False

        if col.startswith("Q_"):
            is_pred_col = True
            pred_cols.append(col)
        elif ensure_prefix:
            # Add Q_ prefix
            new_col = f"Q_{col}"
            result = result.rename(columns={col: new_col})
            pred_cols.append(new_col)
            is_pred_col = True

        if is_pred_col:
            logger.debug(f"Identified prediction column: {col}")

    logger.info(f"Standardized {len(pred_cols)} prediction columns: {pred_cols}")

    return result, pred_cols
