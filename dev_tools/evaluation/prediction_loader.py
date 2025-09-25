"""
Prediction loader module for monthly discharge forecasting evaluation.

This module handles the discovery, loading, and standardization of prediction files
from various model families and formats.
"""

import os
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model family mappings
MODEL_FAMILIES = {
    "BaseCase": ["LR_Base", "GBT"],
    # "SCA_Based": ["LR_Q_SCA", "LR_Q_T_SCA"],
    "SnowMapper_Based": [
        "LR_Snowmapper",
        "LR_Snowmapper_DT",
        "Snow_GBT",
        # "Snow_GBT_old",
        "Snow_GBT_LR",
        "Snow_GBT_Norm",
        # "Snow_HistMeta",
    ],
    # "GlacierMapper_Based": ["Gla_GBT", "Gla_GBT_NormFeat"],
    "Uncertainty": [  # "UncertaintyMixtureMLP",
        "MC_ALD"
    ],
}

# Configuration
EVALUATION_DAY_OF_MONTH = (
    "end"  # 'end' for last day of month, or integer for specific day
)
PREDICTION_FILENAME = "predictions.csv"


def scan_prediction_files(
    results_dir: str = "../monthly_forecasting_results",
) -> Dict[str, List[str]]:
    """
    Scan the results directory to discover all available prediction files.

    Parameters:
    -----------
    results_dir : str
        Path to the monthly forecasting results directory

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping model families to lists of available model paths
    """
    results_path = Path(results_dir)

    if not results_path.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return {}

    discovered_models = {}

    # Scan each model family
    for family_name, expected_models in MODEL_FAMILIES.items():
        family_path = results_path / family_name
        discovered_models[family_name] = []

        if not family_path.exists():
            logger.warning(f"Family directory not found: {family_path}")
            continue

        # Scan for model directories
        for model_name in expected_models:
            model_path = family_path / model_name
            prediction_file = model_path / PREDICTION_FILENAME

            if prediction_file.exists():
                discovered_models[family_name].append(str(model_path))
                logger.info(f"Found predictions for {family_name}/{model_name}")
            else:
                logger.warning(f"No predictions found for {family_name}/{model_name}")

    # Log summary
    total_models = sum(len(models) for models in discovered_models.values())
    logger.info(
        f"Discovered {total_models} models across {len(discovered_models)} families"
    )

    return discovered_models


def _detect_prediction_columns(df: pd.DataFrame, model_name: str) -> List[str]:
    """
    Detect all prediction columns in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with prediction data
    model_name : str
        Name of the model for column naming

    Returns:
    --------
    List[str]
        List of prediction column names
    """
    prediction_cols = []

    # Look for any column starting with Q_ that's not Q_obs
    q_cols = [col for col in df.columns if col.startswith("Q_") and col != "Q_obs"]

    if q_cols:
        prediction_cols = q_cols
        logger.info(
            f"Found {len(prediction_cols)} prediction columns for {model_name}: {prediction_cols}"
        )

    return prediction_cols


def _standardize_prediction_columns(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Standardize prediction column names to a consistent format.
    Note: This function now handles only single prediction column cases.
    For multiple prediction columns, use _create_ensemble_dataframes.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with prediction data
    model_name : str
        Name of the model for column naming

    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized column names
    """
    df = df.copy()

    prediction_cols = _detect_prediction_columns(df, model_name)

    if len(prediction_cols) == 0:
        raise ValueError(f"No prediction column found for {model_name}")

    # For single prediction column, standardize as before
    if len(prediction_cols) == 1:
        prediction_col = prediction_cols[0]
        # Rename prediction column to standardized name
        if prediction_col != "Q_pred":
            df = df.rename(columns={prediction_col: "Q_pred"})
    else:
        # For multiple prediction columns, keep the first one and warn
        # This maintains backward compatibility for existing code
        prediction_col = prediction_cols[0]
        logger.warning(
            f"Multiple prediction columns found for {model_name}. Using {prediction_col} for compatibility."
        )
        if prediction_col != "Q_pred":
            df = df.rename(columns={prediction_col: "Q_pred"})

    # Ensure required columns exist
    required_cols = ["date", "code", "Q_obs", "Q_pred"]
    missing_cols = [col for col in required_cols if col not in df.columns]

    df["code"] = df["code"].astype(int)

    if missing_cols:
        raise ValueError(f"Missing required columns {missing_cols} for {model_name}")

    return df


def _filter_evaluation_dates(
    df: pd.DataFrame, evaluation_day: Union[str, int] = EVALUATION_DAY_OF_MONTH
) -> pd.DataFrame:
    """
    Filter DataFrame to include only evaluation dates.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with date column
    evaluation_day : Union[str, int]
        'end' for last day of month, or integer for specific day

    Returns:
    --------
    pd.DataFrame
        Filtered DataFrame with only evaluation dates
    """
    df = df.copy()

    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], format="mixed")

    if evaluation_day == "end":
        # Filter to last day of each month
        df = df[df["date"].dt.is_month_end]
    elif isinstance(evaluation_day, int):
        # Filter to specific day of month
        df = df[df["date"].dt.day == evaluation_day]
    else:
        raise ValueError(f"Invalid evaluation_day: {evaluation_day}")

    return df


def _create_ensemble_dataframes(
    df: pd.DataFrame, model_name: str, family_name: str, evaluation_day: Union[str, int]
) -> Dict[str, pd.DataFrame]:
    """
    Create separate DataFrames for each ensemble member and an aggregated ensemble.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with prediction data
    model_name : str
        Name of the model
    family_name : str
        Name of the model family
    evaluation_day : Union[str, int]
        Evaluation day configuration

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping ensemble member IDs to DataFrames
    """
    prediction_cols = _detect_prediction_columns(df, model_name)
    ensemble_dataframes = {}

    # Create individual ensemble member DataFrames
    for pred_col in prediction_cols:
        # Create a copy of the dataframe for this ensemble member
        ensemble_df = df.copy()

        # Rename the prediction column to Q_pred for standardization
        ensemble_df = ensemble_df.rename(columns={pred_col: "Q_pred"})

        # Remove other prediction columns
        other_pred_cols = [col for col in prediction_cols if col != pred_col]
        ensemble_df = ensemble_df.drop(columns=other_pred_cols, errors="ignore")

        # Filter to evaluation dates
        ensemble_df = _filter_evaluation_dates(ensemble_df, evaluation_day)

        # Extract ensemble member name from column name
        ensemble_member = pred_col.replace("Q_", "")
        if ensemble_member == model_name:
            ensemble_model_id = f"{family_name}_{model_name}"
            ensemble_member = "ensemble"
        else:
            ensemble_model_id = f"{family_name}_{model_name}_{ensemble_member}"

        # Add model metadata
        ensemble_df["model_id"] = ensemble_model_id
        ensemble_df["family"] = family_name
        ensemble_df["model_name"] = model_name
        ensemble_df["ensemble_member"] = ensemble_member

        # Convert code to int for consistency
        ensemble_df["code"] = ensemble_df["code"].astype(int)

        # Ensure required columns exist
        required_cols = ["date", "code", "Q_obs", "Q_pred"]
        missing_cols = [col for col in required_cols if col not in ensemble_df.columns]

        if missing_cols:
            logger.warning(
                f"Missing required columns {missing_cols} for {ensemble_model_id}"
            )
            continue

        ensemble_dataframes[ensemble_model_id] = ensemble_df
        logger.info(
            f"Created ensemble member {ensemble_model_id} with {len(ensemble_df)} records"
        )

    return ensemble_dataframes


def load_predictions(
    model_paths: Dict[str, List[str]],
    evaluation_day: Union[str, int] = EVALUATION_DAY_OF_MONTH,
    common_codes_only: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Load and standardize prediction data from multiple models.
    Now supports multiple prediction columns per model (ensemble members).

    Parameters:
    -----------
    model_paths : Dict[str, List[str]]
        Dictionary mapping model families to lists of model paths
    evaluation_day : Union[str, int]
        'end' for last day of month, or integer for specific day
    common_codes_only : bool
        If True, filter to common basin codes across all models

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Dictionary mapping model identifiers to prediction DataFrames
        Now includes separate entries for each ensemble member and aggregated ensemble
    """
    loaded_predictions = {}
    all_codes = []

    # Load all prediction files
    for family_name, paths in model_paths.items():
        for model_path in paths:
            model_name = Path(model_path).name
            model_id = f"{family_name}_{model_name}"

            prediction_file = Path(model_path) / PREDICTION_FILENAME

            try:
                # Load CSV file
                df = pd.read_csv(prediction_file)
                logger.info(f"Loaded {len(df)} records from {model_id}")
                df["code"] = df["code"].astype(int)

                # Detect prediction columns
                prediction_cols = _detect_prediction_columns(df, model_name)

                if len(prediction_cols) > 1:
                    # Handle multiple prediction columns (ensemble members)
                    ensemble_dataframes = _create_ensemble_dataframes(
                        df, model_name, family_name, evaluation_day
                    )

                    # Add all ensemble DataFrames to loaded_predictions
                    for ensemble_id, ensemble_df in ensemble_dataframes.items():
                        loaded_predictions[ensemble_id] = ensemble_df
                        all_codes.append(set(ensemble_df["code"].unique()))

                else:
                    # Handle single prediction column (backward compatibility)
                    df = _standardize_prediction_columns(df, model_name)

                    # Filter to evaluation dates
                    df = _filter_evaluation_dates(df, evaluation_day)
                    logger.info(
                        f"Filtered to {len(df)} evaluation records for {model_id}"
                    )

                    # Add model metadata
                    df["model_id"] = model_id
                    df["family"] = family_name
                    df["model_name"] = model_name

                    # Convert code to int for consistency
                    df["code"] = df["code"].astype(int)

                    # Store predictions
                    loaded_predictions[model_id] = df
                    all_codes.append(set(df["code"].unique()))

            except Exception as e:
                logger.error(f"Failed to load predictions for {model_id}: {str(e)}")
                continue

    # Filter to common codes if requested
    if common_codes_only and all_codes:
        common_codes = set.intersection(*all_codes)
        non_common = set.union(*all_codes) - common_codes
        if non_common:
            logger.warning(
                f"Excluding {len(non_common)} non-common basin codes across models"
            )
            logger.debug(f"Non-common codes: {sorted(list(non_common))}")

        common_codes = sorted(list(common_codes))
        logger.info(f"Filtering to {len(common_codes)} common basin codes")

        filtered_predictions = {}
        for model_id, df in loaded_predictions.items():
            filtered_df = df[df["code"].isin(common_codes)].copy()
            filtered_predictions[model_id] = filtered_df
            logger.info(f"Filtered {model_id} to {len(filtered_df)} records")

        loaded_predictions = filtered_predictions

    logger.info(f"Successfully loaded {len(loaded_predictions)} models")
    return loaded_predictions


def get_model_family_mapping(
    loaded_predictions: Dict[str, pd.DataFrame],
) -> Dict[str, List[str]]:
    """
    Get mapping of model families to their constituent models.

    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of loaded prediction DataFrames

    Returns:
    --------
    Dict[str, List[str]]
        Dictionary mapping family names to lists of model IDs
    """
    family_mapping = {}

    for model_id, df in loaded_predictions.items():
        family = df["family"].iloc[0]
        if family not in family_mapping:
            family_mapping[family] = []
        family_mapping[family].append(model_id)

    return family_mapping


def validate_prediction_data(
    loaded_predictions: Dict[str, pd.DataFrame],
) -> Dict[str, Dict]:
    """
    Validate loaded prediction data and return validation summary.

    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of loaded prediction DataFrames

    Returns:
    --------
    Dict[str, Dict]
        Dictionary with validation results for each model
    """
    validation_results = {}

    for model_id, df in loaded_predictions.items():
        results = {
            "total_records": len(df),
            "missing_observations": df["Q_obs"].isna().sum(),
            "missing_predictions": df["Q_pred"].isna().sum(),
            "date_range": (df["date"].min(), df["date"].max()),
            "unique_codes": df["code"].nunique(),
            "codes": sorted(df["code"].unique().tolist()),
        }

        # Check for data quality issues
        results["warnings"] = []

        if results["missing_observations"] > 0:
            results["warnings"].append(
                f"{results['missing_observations']} missing observations"
            )

        if results["missing_predictions"] > 0:
            results["warnings"].append(
                f"{results['missing_predictions']} missing predictions"
            )

        if results["total_records"] == 0:
            results["warnings"].append("No data records found")

        validation_results[model_id] = results

    return validation_results


def load_all_predictions(
    results_dir: str = "../monthly_forecasting_results",
    evaluation_day: Union[str, int] = EVALUATION_DAY_OF_MONTH,
    common_codes_only: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Complete workflow to discover, load, and validate all prediction data.

    Parameters:
    -----------
    results_dir : str
        Path to the monthly forecasting results directory
    evaluation_day : Union[str, int]
        'end' for last day of month, or integer for specific day
    common_codes_only : bool
        If True, filter to common basin codes across all models

    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]
        Tuple of (loaded_predictions, validation_results)
    """
    # Discover prediction files
    logger.info("Scanning for prediction files...")
    model_paths = scan_prediction_files(results_dir)

    # Load predictions
    logger.info("Loading prediction data...")
    loaded_predictions = load_predictions(
        model_paths, evaluation_day, common_codes_only
    )

    # Validate data
    logger.info("Validating prediction data...")
    validation_results = validate_prediction_data(loaded_predictions)

    # Log summary
    logger.info(f"Loaded {len(loaded_predictions)} models successfully")
    total_warnings = sum(
        len(v.get("warnings", [])) for v in validation_results.values()
    )
    if total_warnings > 0:
        logger.warning(f"Found {total_warnings} data quality warnings")

    return loaded_predictions, validation_results


if __name__ == "__main__":
    # Example usage
    predictions, validation = load_all_predictions()

    print("\n=== PREDICTION LOADING SUMMARY ===")
    print(f"Loaded {len(predictions)} models")

    for model_id, results in validation.items():
        print(f"\n{model_id}:")
        print(f"  Records: {results['total_records']}")
        print(f"  Date range: {results['date_range'][0]} to {results['date_range'][1]}")
        print(f"  Unique codes: {results['unique_codes']}")
        if results["warnings"]:
            print(f"  Warnings: {', '.join(results['warnings'])}")
