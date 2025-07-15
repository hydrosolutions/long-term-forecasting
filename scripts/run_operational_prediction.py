import os
import sys
import argparse
import json
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import forecast models
from monthly_forecasting.forecast_models.LINEAR_REGRESSION import LinearRegressionModel
from monthly_forecasting.forecast_models.SciRegressor import SciRegressor
from monthly_forecasting.scr import data_loading as dl


# Setup logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Force ALL logging to INFO level
logging.getLogger().setLevel(logging.INFO)  # Root logger
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.INFO)


# import environment variables saved in .env file
from dotenv import load_dotenv

load_dotenv()

PATH_TO_DISCHARGE = os.getenv("path_discharge")
PATH_TO_FORCING_ERA5 = os.getenv("PATH_TO_FORCING_ERA5")
PATH_TO_FORCING_OPERATIONAL = os.getenv("path_forcing_operational")
PATH_SWE_00003 = os.getenv("path_SWE_00003")
PATH_SWE_500m = os.getenv("PATH_SWE_500m")
PATH_ROF_00003 = os.getenv("path_ROF_00003")
PATH_ROF_500m = os.getenv("PATH_ROF_500m")
PATH_TO_SHP = os.getenv("path_to_shp")
PATH_TO_STATIC = os.getenv("PATH_TO_STATIC")

MODELS_OPERATIONAL = {
    "BaseCase": [
        ("LR", "LR_Q_T_P"),
        ("SciRegressor", "GradBoostTrees"),
    ],
    "SnowMapper_Based": [
        ("LR", "LR_Q_dSWEdt_T_P"),
        ("LR", "LR_Q_SWE_T"),
        ("LR", "LR_Q_T_P_SWE"),
        ("LR", "LR_Q_SWE"),
        ("SciRegressor", "NormBased"),
    ],
}

MODELS_DIR = "../monthly_forecasting_models"


### Load the Data Function
def load_discharge():
    df_discharge = pd.read_csv(PATH_TO_DISCHARGE, parse_dates=["date"])
    df_discharge["date"] = pd.to_datetime(df_discharge["date"], format="%Y-%m-%d")
    df_discharge["code"] = df_discharge["code"].astype(int)
    return df_discharge


def load_forcing():
    operational_T = pd.read_csv(
        os.path.join(PATH_TO_FORCING_OPERATIONAL, "00003_T_control_member.csv")
    )
    operational_P = pd.read_csv(
        os.path.join(PATH_TO_FORCING_OPERATIONAL, "00003_P_control_member.csv")
    )

    operational_T["date"] = pd.to_datetime(operational_T["date"], format="%Y-%m-%d")
    operational_P["date"] = pd.to_datetime(operational_P["date"], format="%Y-%m-%d")
    operational_T["code"] = operational_T["code"].astype(int)
    operational_P["code"] = operational_P["code"].astype(int)

    operational_data = pd.merge(operational_T, operational_P, on=["date", "code"])
    logger.info("Operational forcing data loaded successfully.")

    hindcast_T = pd.read_csv(
        os.path.join(PATH_TO_FORCING_ERA5, "00003_T_reanalysis.csv")
    )
    hindcast_P = pd.read_csv(
        os.path.join(PATH_TO_FORCING_ERA5, "00003_P_reanalysis.csv")
    )
    hindcast_T["date"] = pd.to_datetime(hindcast_T["date"], format="%Y-%m-%d")
    hindcast_P["date"] = pd.to_datetime(hindcast_P["date"], format="%Y-%m-%d")
    hindcast_T["code"] = hindcast_T["code"].astype(int)
    hindcast_P["code"] = hindcast_P["code"].astype(int)
    hindcast_data = pd.merge(hindcast_T, hindcast_P, on=["date", "code"])

    logger.info("Hindcast forcing data loaded successfully.")

    operationl_data = pd.concat([hindcast_data, operational_data], ignore_index=True)
    operationl_data = operationl_data.sort_values(by=["date", "code"]).reset_index(
        drop=True
    )

    # drop duplicates based on date and code - keep last occurrence
    operational_data = operationl_data.drop_duplicates(
        subset=["date", "code"], keep="last"
    )

    return operational_data


def load_snowmapper():
    swe_00003 = pd.read_csv(PATH_SWE_00003, parse_dates=["date"])
    swe_00003["date"] = pd.to_datetime(swe_00003["date"], format="%Y-%m-%d")
    swe_00003["code"] = swe_00003["code"].astype(int)

    logger.info("SWE 00003 columns: %s", swe_00003.columns.tolist())

    swe_500m = pd.read_csv(PATH_SWE_500m, parse_dates=["date"])
    swe_500m["date"] = pd.to_datetime(swe_500m["date"], format="%Y-%m-%d")
    swe_500m["code"] = swe_500m["code"].astype(int)

    logger.info("SWE 500m columns: %s", swe_500m.columns.tolist())

    rof_00003 = pd.read_csv(PATH_ROF_00003, parse_dates=["date"])
    rof_00003["date"] = pd.to_datetime(rof_00003["date"], format="%Y-%m-%d")
    rof_00003["code"] = rof_00003["code"].astype(int)

    logger.info("ROF 00003 columns: %s", rof_00003.columns.tolist())
    # replace RoF with ROF
    rof_00003.rename(columns={"RoF": "ROF"}, inplace=True)

    rof_500m = pd.read_csv(PATH_ROF_500m, parse_dates=["date"])
    rof_500m["date"] = pd.to_datetime(rof_500m["date"], format="%Y-%m-%d")
    rof_500m["code"] = rof_500m["code"].astype(int)

    # replace RoF with ROF
    for col in rof_500m.columns:
        if col.startswith("RoF"):
            new_col_name = col.replace("RoF", "ROF")
            rof_500m.rename(columns={col: new_col_name}, inplace=True)

    logger.info("ROF 500m columns: %s", rof_500m.columns.tolist())

    return swe_00003, swe_500m, rof_00003, rof_500m


def load_static_data():
    df_static = pd.read_csv(PATH_TO_STATIC)

    if "CODE" in df_static.columns:
        df_static.rename(columns={"CODE": "code"}, inplace=True)

    df_static["code"] = df_static["code"].astype(int)

    return df_static


def create_data_frame(config: Dict[str, Any]) -> pd.DataFrame:
    data = load_discharge()

    forcing = load_forcing()

    swe_00003, swe_500m, rof_00003, rof_500m = load_snowmapper()

    static_data = load_static_data()
    logger.info(f"Static Data Columns: {static_data.columns.tolist()}")
    logger.info("Data loading complete. Merging data...")

    # Only keep code which are in static data
    static_codes = static_data["code"].unique()
    data = data[data["code"].isin(static_codes)].copy()

    data = data.merge(forcing, on=["date", "code"], how="left")

    SWE_HRU = config.get("HRU_SWE", None)

    if SWE_HRU == "HRU_00003":
        data = data.merge(swe_00003, on=["date", "code"], how="left")

    elif SWE_HRU == "KGZ500m":
        data = data.merge(swe_500m, on=["date", "code"], how="left")

    else:
        logger.info("No SWE data merged.")

    ROF_HRU = config.get("HRU_ROF", None)
    if ROF_HRU == "HRU_00003":
        data = data.merge(rof_00003, on=["date", "code"], how="left")
    elif ROF_HRU == "KGZ500m":
        data = data.merge(rof_500m, on=["date", "code"], how="left")
    else:
        logger.info("No ROF data merged.")

    return data, static_data


def load_operational_configs(
    model_type: str, family_name: str, model_name: str
) -> Dict[str, Any]:
    """
    Load all configuration files for an operational model.

    Args:
        model_type: Type of model ('LR' or 'SciRegressor')
        model_name: Name of the model configuration

    Returns:
        Dictionary containing all configurations
    """
    # Determine configuration directory based on model type and name
    config_dir = Path(MODELS_DIR) / family_name / model_name

    if not config_dir.exists():
        # Try alternative structure
        raise FileNotFoundError(
            f"Configuration directory not found: {config_dir}. "
            "Please check the model family and name."
        )

    # Load required configuration files
    config_files = {
        "general_config": "general_config.json",
        "model_config": "model_config.json",
        "feature_config": "feature_config.json",
        "data_config": "data_config.json",
        "path_config": "data_paths.json",
    }

    configs = {}

    for config_name, config_file in config_files.items():
        config_path = config_dir / config_file

        if config_path.exists():
            with open(config_path, "r") as f:
                configs[config_name] = json.load(f)
            logger.info(f"Loaded {config_name} from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            configs[config_name] = {}

    # Set model type in general config
    if configs["general_config"]:
        configs["general_config"]["model_type"] = (
            "linear_regression" if model_type == "LR" else "sciregressor"
        )

    return configs


def shift_data_to_current_year(
    data_df: pd.DataFrame, shift_years: int = 1
) -> pd.DataFrame:
    """
    Shift data dates by specified years to mock current year.

    Args:
        data_df: DataFrame with 'date' column
        shift_years: Number of years to shift forward

    Returns:
        DataFrame with shifted dates
    """
    data_df_shifted = data_df.copy()
    data_df_shifted["date"] = data_df_shifted["date"] + pd.DateOffset(years=shift_years)
    return data_df_shifted


def create_model_instance(
    model_type: str,
    model_name: str,
    configs: Dict[str, Any],
    data: pd.DataFrame,
    static_data: pd.DataFrame,
):
    """
    Create the appropriate model instance based on the model type.

    Args:
        model_type: 'LR' or 'SciRegressor'
        model_name: Name of the model configuration
        configs: All configuration dictionaries
        data: Time series data
        static_data: Static basin characteristics

    Returns:
        Model instance
    """
    general_config = configs["general_config"]
    model_config = configs["model_config"]
    feature_config = configs["feature_config"]
    path_config = configs["path_config"]

    # Set model name in general config
    general_config["model_name"] = model_name

    # Create model instance based on type
    if model_type == "LR":
        model = LinearRegressionModel(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    elif model_type == "SciRegressor":
        model = SciRegressor(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def calculate_average_discharge(
    data: pd.DataFrame, valid_from: str, valid_to: str
) -> pd.DataFrame:
    """
    Calculate average discharge for each basin from valid_from to valid_to period.

    Args:
        data: DataFrame with discharge data
        valid_from: Start date for averaging period
        valid_to: End date for averaging period

    Returns:
        DataFrame with basin codes and average discharge
    """
    # Filter data for the specified period
    mask = (data["date"] >= valid_from) & (data["date"] <= valid_to)
    period_data = data[mask]

    # Calculate average discharge per basin
    avg_discharge = period_data.groupby("code")["discharge"].mean().reset_index()
    avg_discharge.columns = ["code", "observed_avg_discharge"]

    return avg_discharge


def evaluate_predictions(
    predictions: pd.DataFrame, observations: pd.DataFrame, model_name: str
) -> Dict[str, Any]:
    """
    Calculate performance metrics for predictions with focus on percentage errors per basin.

    Args:
        predictions: DataFrame with model predictions
        observations: DataFrame with observed values
        model_name: Name of the model for finding the correct prediction column

    Returns:
        Dictionary with performance metrics including per-basin errors
    """
    # Find the prediction column for this model
    q_column_name = f"Q_{model_name}"

    if q_column_name not in predictions.columns:
        logger.warning(f"Prediction column {q_column_name} not found in predictions")
        return {
            "num_predictions": 0,
            "basin_errors": [],
            "mean_absolute_error_pct": np.nan,
            "median_absolute_error_pct": np.nan,
            "basins_over_30pct_error": [],
            "basins_over_50pct_error": [],
        }

    # Merge predictions and observations
    merged = pd.merge(predictions, observations, on="code", how="inner")

    if len(merged) == 0:
        logger.warning("No matching basins found between predictions and observations")
        return {
            "num_predictions": 0,
            "basin_errors": [],
            "mean_absolute_error_pct": np.nan,
            "median_absolute_error_pct": np.nan,
            "basins_over_30pct_error": [],
            "basins_over_50pct_error": [],
        }

    # Remove rows with missing data
    merged.dropna(subset=["observed_avg_discharge", q_column_name], inplace=True)

    if len(merged) == 0:
        logger.warning("No valid data after removing NaN values")
        return {
            "num_predictions": 0,
            "basin_errors": [],
            "mean_absolute_error_pct": np.nan,
            "median_absolute_error_pct": np.nan,
            "basins_over_30pct_error": [],
            "basins_over_50pct_error": [],
        }

    # Calculate percentage error for each basin
    merged["error_pct"] = (
        (merged[q_column_name] - merged["observed_avg_discharge"])
        / merged["observed_avg_discharge"]
        * 100
    )

    merged["abs_error_pct"] = abs(merged["error_pct"])

    # Create basin-level error summary
    basin_errors = []
    for _, row in merged.iterrows():
        basin_errors.append(
            {
                "basin_code": int(row["code"]),
                "observed": round(row["observed_avg_discharge"], 2),
                "predicted": round(row[q_column_name], 2),
                "error_pct": round(row["error_pct"], 1),
                "abs_error_pct": round(row["abs_error_pct"], 1),
            }
        )

    # Sort by absolute error percentage (worst first)
    basin_errors.sort(key=lambda x: x["abs_error_pct"], reverse=True)

    # Calculate summary statistics
    mean_abs_error_pct = merged["abs_error_pct"].mean()
    median_abs_error_pct = merged["abs_error_pct"].median()

    # Identify basins with high errors
    basins_over_30pct = merged[merged["abs_error_pct"] > 30]["code"].tolist()
    basins_over_50pct = merged[merged["abs_error_pct"] > 50]["code"].tolist()

    metrics = {
        "num_predictions": len(merged),
        "basin_errors": basin_errors,
        "mean_absolute_error_pct": round(mean_abs_error_pct, 1),
        "median_absolute_error_pct": round(median_abs_error_pct, 1),
        "basins_over_30pct_error": [int(x) for x in basins_over_30pct],
        "basins_over_50pct_error": [int(x) for x in basins_over_50pct],
    }

    return metrics


def print_model_performance_summary(all_metrics: list):
    """
    Print a nicely formatted summary of model performance.

    Args:
        all_metrics: List of metrics dictionaries from all models
    """
    if not all_metrics:
        logger.info("No performance metrics to display.")
        return

    print("\n" + "=" * 80)
    print("🎯 OPERATIONAL PREDICTION PERFORMANCE SUMMARY")
    print("=" * 80)

    for i, metrics in enumerate(all_metrics):
        model_name = metrics.get("model_name", f"Model {i + 1}")
        family = metrics.get("model_family", "Unknown")
        model_type = metrics.get("model_type", "Unknown")

        print(f"\n📊 {family} → {model_type} → {model_name}")
        print("-" * 60)

        if metrics["num_predictions"] == 0:
            print("   ❌ No predictions available for evaluation")
            continue

        # Summary statistics
        print(f"   📈 Basins evaluated: {metrics['num_predictions']}")
        print(f"   📊 Mean absolute error: {metrics['mean_absolute_error_pct']:.1f}%")
        print(
            f"   🎯 Median absolute error: {metrics['median_absolute_error_pct']:.1f}%"
        )

        # Error categories
        num_over_30 = len(metrics["basins_over_30pct_error"])
        num_over_50 = len(metrics["basins_over_50pct_error"])

        if num_over_30 > 0:
            pct_over_30 = (num_over_30 / metrics["num_predictions"]) * 100
            print(f"   ⚠️  Basins >30% error: {num_over_30} ({pct_over_30:.1f}%)")

        if num_over_50 > 0:
            pct_over_50 = (num_over_50 / metrics["num_predictions"]) * 100
            print(f"   🚨 Basins >50% error: {num_over_50} ({pct_over_50:.1f}%)")

        # Show worst performing basins (top 5)
        if metrics["basin_errors"]:
            print("   🔍 Top 5 worst predictions:")
            for j, basin in enumerate(metrics["basin_errors"][:5]):
                status = (
                    "🚨"
                    if basin["abs_error_pct"] > 50
                    else "⚠️"
                    if basin["abs_error_pct"] > 30
                    else "✅"
                )
                print(
                    f"      {status} Basin {basin['basin_code']:>4}: "
                    f"Obs={basin['observed']:>6.1f} Pred={basin['predicted']:>6.1f} "
                    f"Error={basin['error_pct']:>+6.1f}%"
                )

    # Overall summary
    print(f"\n" + "=" * 80)
    print("📋 OVERALL SUMMARY")
    print("=" * 80)

    total_predictions = sum(
        m["num_predictions"] for m in all_metrics if m["num_predictions"] > 0
    )
    valid_metrics = [
        m for m in all_metrics if not np.isnan(m["mean_absolute_error_pct"])
    ]

    if valid_metrics:
        overall_mean_error = np.mean(
            [m["mean_absolute_error_pct"] for m in valid_metrics]
        )
        overall_median_error = np.median(
            [m["median_absolute_error_pct"] for m in valid_metrics]
        )

        print(f"📊 Total predictions across all models: {total_predictions}")
        print(f"📈 Average mean error across models: {overall_mean_error:.1f}%")
        print(f"🎯 Average median error across models: {overall_median_error:.1f}%")

        # Count models by performance
        good_models = sum(
            1 for m in valid_metrics if m["mean_absolute_error_pct"] <= 20
        )
        ok_models = sum(
            1 for m in valid_metrics if 20 < m["mean_absolute_error_pct"] <= 40
        )
        poor_models = sum(1 for m in valid_metrics if m["mean_absolute_error_pct"] > 40)

        print(f"✅ Models with ≤20% avg error: {good_models}")
        print(f"⚠️  Models with 20-40% avg error: {ok_models}")
        print(f"🚨 Models with >40% avg error: {poor_models}")

    print("=" * 80 + "\n")


def run_operational_prediction(today: datetime.datetime = None) -> Dict[str, Any]:
    """
    Main operational prediction workflow.

    Args:
        today (datetime.datetime, optional): Date to use as "today" for prediction.
            If None, uses current datetime.

    Returns:
        Dictionary with all results and metrics
    """
    if today is None:
        today = datetime.datetime.now()

    logger.info(
        f"Starting operational prediction workflow for {today.strftime('%Y-%m-%d')}..."
    )

    # Initialize results storage
    all_predictions = []
    all_metrics = []
    timing_results = {}

    # Start overall timing
    overall_start_time = datetime.datetime.now()

    # Process each model family
    for family_name, models in MODELS_OPERATIONAL.items():
        logger.info(f"Processing model family: {family_name}")

        for model_type, model_name in models:
            logger.info(f"-" * 50)
            logger.info(f"-" * 50)
            logger.info(f"Processing model: {model_type} - {model_name}")

            # Start model timing
            model_start_time = datetime.datetime.now()

            try:
                # Load configurations
                configs = load_operational_configs(
                    model_type, family_name=family_name, model_name=model_name
                )

                # Load and prepare data
                # Use environment variables for data loading since configs might not have data_config
                data, static_data = create_data_frame(config=configs["path_config"])

                logger.info(f"Data columns after loading: {data.columns.tolist()}")

                # Shift data to current year for operational prediction
                data = shift_data_to_current_year(data)
                today_dt = pd.to_datetime(today.date(), format="%Y-%m-%d")

                today_plus_15 = today_dt + pd.DateOffset(days=15)

                data_model = data[data["date"] <= today_plus_15].copy()
                # set discharge to nan for dates after today
                data_model.loc[data_model["date"] > today_dt, "discharge"] = np.nan

                # Create model instance
                model = create_model_instance(
                    model_type, model_name, configs, data_model, static_data
                )

                # Run operational prediction
                raw_predictions = model.predict_operational(today=today)

                # Process predictions to standardize format
                predictions = process_predictions(raw_predictions, model_name)

                logger.info(
                    f"Prediction columns for {model_name}: {predictions.columns.tolist()}"
                )

                # Store predictions
                predictions["model_family"] = family_name
                predictions["model_type"] = model_type
                predictions["model_name"] = model_name
                all_predictions.append(predictions)

                # Calculate performance metrics if predictions are available
                q_column_name = f"Q_{model_name}"
                if (
                    q_column_name in predictions.columns
                    and "code" in predictions.columns
                ):
                    # Use valid_from and valid_to from predictions if available
                    if (
                        "valid_from" in predictions.columns
                        and "valid_to" in predictions.columns
                    ):
                        # Use the first prediction's validation period for observation calculation
                        first_pred = predictions.iloc[0]
                        start_date = first_pred["valid_from"]
                        end_date = first_pred["valid_to"]
                    else:
                        # Fallback: Use the last 30 days of data for evaluation
                        end_date = data["date"].max()
                        start_date = end_date - pd.DateOffset(days=30)

                    logger.info(
                        f"Calculating observations from {start_date} to {end_date}"
                    )
                    observations = calculate_average_discharge(
                        data, start_date, end_date
                    )

                    # Evaluate predictions
                    model_metrics = evaluate_predictions(
                        predictions, observations, model_name
                    )
                    model_metrics["model_family"] = family_name
                    model_metrics["model_type"] = model_type
                    model_metrics["model_name"] = model_name
                    all_metrics.append(model_metrics)

                # Calculate model timing
                model_end_time = datetime.datetime.now()
                model_duration = (model_end_time - model_start_time).total_seconds()
                timing_results[f"{family_name}_{model_type}_{model_name}"] = (
                    model_duration
                )

                logger.info(
                    f"Completed {model_type} - {model_name} in {model_duration:.2f}s"
                )

            except Exception as e:
                logger.error(f"Error processing {model_type} - {model_name}: {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

    # Calculate overall timing
    overall_end_time = datetime.datetime.now()
    overall_duration = (overall_end_time - overall_start_time).total_seconds()
    timing_results["overall_duration"] = overall_duration

    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
    else:
        combined_predictions = pd.DataFrame()

    logger.info(f"Completed operational prediction workflow in {overall_duration:.2f}s")

    return {
        "predictions": combined_predictions,
        "timing": timing_results,
        "metrics": all_metrics,
    }


def process_predictions(predictions: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Process predictions from model to standardize format.

    Args:
        predictions: DataFrame with model predictions
        model_name: Name of the model for column naming

    Returns:
        DataFrame with standardized predictions
    """
    predictions = predictions.copy()

    # Create the Q_model_name column from Q column
    q_column_name = f"Q_{model_name}"
    if q_column_name in predictions.columns:
        logger.info(f"Using existing prediction column: {q_column_name}")
    else:
        logger.warning(f"No prediction column found in model output for {model_name}")
        return predictions

    # Ensure we have valid_from and valid_to columns
    if "valid_from" not in predictions.columns or "valid_to" not in predictions.columns:
        logger.warning(
            f"Missing valid_from or valid_to columns in predictions for {model_name}"
        )
        # If missing, we can't evaluate properly but we'll still process

    return predictions


def generate_outputs(
    results: Dict[str, Any], output_dir: str = "../monthly_forecasting_results/output"
):
    """
    Generate all required output files.

    Args:
        results: Results dictionary from run_operational_prediction
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Save predictions
    if not results["predictions"].empty:
        predictions_file = output_path / "operational_predictions.csv"
        results["predictions"].to_csv(predictions_file, index=False)
        logger.info(f"Saved predictions to {predictions_file}")

    # Save timing report
    timing_file = output_path / "timing_report.json"
    with open(timing_file, "w") as f:
        json.dump(results["timing"], f, indent=2)
    logger.info(f"Saved timing report to {timing_file}")

    # Save performance metrics
    if results["metrics"]:
        metrics_file = output_path / "performance_metrics.csv"
        pd.DataFrame(results["metrics"]).to_csv(metrics_file, index=False)
        logger.info(f"Saved performance metrics to {metrics_file}")

    # Generate quality report
    quality_file = output_path / "quality_report.txt"
    with open(quality_file, "w") as f:
        f.write("Operational Prediction Quality Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(
            f"Total models processed: {len(results['timing']) - 1}\n"
        )  # -1 for overall_duration
        f.write(
            f"Total processing time: {results['timing']['overall_duration']:.2f}s\n"
        )
        f.write(f"Total predictions: {len(results['predictions'])}\n")
        if results["metrics"]:
            valid_metrics = [
                m
                for m in results["metrics"]
                if not np.isnan(m.get("mean_absolute_error_pct", np.nan))
            ]
            if valid_metrics:
                avg_mean_error = np.mean(
                    [m["mean_absolute_error_pct"] for m in valid_metrics]
                )
                f.write(f"Average mean absolute error: {avg_mean_error:.1f}%\n")
                good_models = sum(
                    1 for m in valid_metrics if m["mean_absolute_error_pct"] <= 20
                )
                f.write(
                    f"Models with ≤20% avg error: {good_models}/{len(valid_metrics)}\n"
                )
    logger.info(f"Saved quality report to {quality_file}")


def create_observed_vs_predicted_plots(
    all_predictions: pd.DataFrame,
    all_metrics: list,
    output_dir: Path,
    target_basins: list = None,
    max_basins: int = 10,
) -> None:
    """
    Create observed vs predicted scatter plots for selected basins.

    Args:
        all_predictions: Combined predictions DataFrame
        all_metrics: List of all metrics
        output_dir: Output directory for plots
        target_basins: List of specific basin codes to plot (None for auto-selection)
        max_basins: Maximum number of basins to plot
    """
    if all_predictions.empty:
        logger.warning("No predictions available for plotting")
        return

    # Get unique models and basins
    models = list(all_predictions["model_name"].unique())

    # Add ensemble model if Q_Ensemble column exists
    if "Q_Ensemble" in all_predictions.columns:
        models.append("Ensemble")
        logger.info("Added Ensemble model to plotting")

    all_basins = all_predictions["code"].unique()

    # Select basins to plot
    if target_basins is None:
        # Auto-select basins based on data availability and variety
        basin_counts = all_predictions["code"].value_counts()
        selected_basins = basin_counts.head(max_basins).index.tolist()
    else:
        # Use specified basins that exist in the data
        selected_basins = [b for b in target_basins if b in all_basins]

    if 16936 not in selected_basins:
        selected_basins.append(16936)

    logger.info(f"Creating plots for {len(selected_basins)} basins: {selected_basins}")

    # Create plots for each basin
    for basin_code in selected_basins:
        basin_data = all_predictions[all_predictions["code"] == basin_code].copy()

        if basin_data.empty:
            continue

        # Create subplot for this basin
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Basin {basin_code}: Observed vs Predicted",
                f"Basin {basin_code}: Model Performance Comparison",
                f"Basin {basin_code}: Error Distribution",
                f"Basin {basin_code}: Performance Over Time",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"secondary_y": False}],
            ],
        )

        colors = px.colors.qualitative.Set3

        # Plot 1: Observed vs Predicted scatter
        for i, model in enumerate(models):
            if model == "Ensemble":
                # Handle ensemble model specially (use all data for this basin)
                model_data = basin_data.copy()
                pred_col = "Q_Ensemble"
            else:
                # Handle regular models
                model_data = basin_data[basin_data["model_name"] == model]
                pred_col = f"Q_{model}"

            if model_data.empty:
                continue

            # Check if prediction column exists
            if pred_col not in model_data.columns:
                continue

            # Get observed and predicted values directly from the data
            obs_values = []
            pred_values = []

            for _, row in model_data.iterrows():
                if pd.notna(row.get("observed_discharge")):
                    pred_value = row.get(pred_col)
                    if pd.notna(pred_value):
                        obs_values.append(row["observed_discharge"])
                        pred_values.append(pred_value)

            if obs_values and pred_values:
                fig.add_trace(
                    go.Scatter(
                        x=obs_values,
                        y=pred_values,
                        mode="markers",
                        name=model,
                        marker=dict(color=colors[i % len(colors)], size=8),
                        text=[
                            f"Date: {date}" for date in model_data["evaluation_date"]
                        ],
                        hovertemplate="<b>%{fullData.name}</b><br>"
                        + "Observed: %{x:.2f}<br>"
                        + "Predicted: %{y:.2f}<br>"
                        + "%{text}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

        # Add 1:1 line
        if obs_values:
            min_val = min(min(obs_values), min(pred_values))
            max_val = max(max(obs_values), max(pred_values))
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode="lines",
                    name="1:1 Line",
                    line=dict(color="black", dash="dash"),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Plot 2: Model performance comparison (bar chart)
        model_errors = []
        model_names = []

        for model in models:
            if model == "Ensemble":
                # Calculate ensemble error manually for this basin
                ensemble_errors = []
                ensemble_data = basin_data[
                    pd.notna(basin_data.get("Q_Ensemble"))
                    & pd.notna(basin_data.get("observed_discharge"))
                ]

                for _, row in ensemble_data.iterrows():
                    observed = row["observed_discharge"]
                    predicted = row["Q_Ensemble"]
                    if observed > 0:  # Avoid division by zero
                        error_pct = abs((predicted - observed) / observed * 100)
                        ensemble_errors.append(error_pct)

                if ensemble_errors:
                    avg_error = np.mean(ensemble_errors)
                    model_errors.append(avg_error)
                    model_names.append(model)
            else:
                # Handle regular models
                model_metrics = [m for m in all_metrics if m["model_name"] == model]
                if model_metrics:
                    # Calculate average error for this basin across all dates
                    basin_errors_for_model = []
                    for metric in model_metrics:
                        basin_errors = metric.get("basin_errors", [])
                        basin_error = next(
                            (
                                be
                                for be in basin_errors
                                if be["basin_code"] == basin_code
                            ),
                            None,
                        )
                        if basin_error:
                            basin_errors_for_model.append(basin_error["abs_error_pct"])

                    if basin_errors_for_model:
                        avg_error = np.mean(basin_errors_for_model)
                        model_errors.append(avg_error)
                        model_names.append(model)

        if model_errors:
            fig.add_trace(
                go.Bar(
                    x=model_names,
                    y=model_errors,
                    name="Mean Absolute Error %",
                    marker_color=colors[: len(model_names)],
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        # Plot 3: Error distribution (histogram)
        all_errors = []
        for model in models:
            if model == "Ensemble":
                # Add ensemble errors
                ensemble_data = basin_data[
                    pd.notna(basin_data.get("Q_Ensemble"))
                    & pd.notna(basin_data.get("observed_discharge"))
                ]

                for _, row in ensemble_data.iterrows():
                    observed = row["observed_discharge"]
                    predicted = row["Q_Ensemble"]
                    if observed > 0:  # Avoid division by zero
                        error_pct = (predicted - observed) / observed * 100
                        all_errors.append(error_pct)
            else:
                # Handle regular models
                model_metrics = [m for m in all_metrics if m["model_name"] == model]
                for metric in model_metrics:
                    basin_errors = metric.get("basin_errors", [])
                    basin_error = next(
                        (be for be in basin_errors if be["basin_code"] == basin_code),
                        None,
                    )
                    if basin_error:
                        all_errors.append(basin_error["error_pct"])

        if all_errors:
            fig.add_trace(
                go.Histogram(
                    x=all_errors,
                    nbinsx=20,
                    name="Error Distribution",
                    marker_color="skyblue",
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

        # Plot 4: Performance over time
        for i, model in enumerate(models):
            dates = []
            errors = []

            if model == "Ensemble":
                # Calculate ensemble performance over time
                unique_dates = basin_data["evaluation_date"].unique()
                for eval_date in unique_dates:
                    date_data = basin_data[basin_data["evaluation_date"] == eval_date]
                    ensemble_data = date_data[
                        pd.notna(date_data.get("Q_Ensemble"))
                        & pd.notna(date_data.get("observed_discharge"))
                    ]

                    if not ensemble_data.empty:
                        # Take the first row for this date (all should have same ensemble value)
                        row = ensemble_data.iloc[0]
                        observed = row["observed_discharge"]
                        predicted = row["Q_Ensemble"]
                        if observed > 0:
                            error_pct = abs((predicted - observed) / observed * 100)
                            dates.append(eval_date)
                            errors.append(error_pct)
            else:
                # Handle regular models
                model_metrics = [m for m in all_metrics if m["model_name"] == model]
                for metric in model_metrics:
                    basin_errors = metric.get("basin_errors", [])
                    basin_error = next(
                        (be for be in basin_errors if be["basin_code"] == basin_code),
                        None,
                    )
                    if basin_error:
                        dates.append(metric["evaluation_date"])
                        errors.append(basin_error["abs_error_pct"])

            if dates and errors:
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=errors,
                        mode="lines+markers",
                        name=f"{model}",
                        line=dict(color=colors[i % len(colors)]),
                        showlegend=False,
                    ),
                    row=2,
                    col=2,
                )

        # Update layout
        fig.update_layout(
            title_text=f"Basin {basin_code} - Prediction Analysis",
            height=800,
            showlegend=True,
            template="plotly_white",
        )

        # Update axes labels
        fig.update_xaxes(title_text="Observed Discharge (m³/s)", row=1, col=1)
        fig.update_yaxes(title_text="Predicted Discharge (m³/s)", row=1, col=1)
        fig.update_xaxes(title_text="Model", row=1, col=2)
        fig.update_yaxes(title_text="Mean Absolute Error (%)", row=1, col=2)
        fig.update_xaxes(title_text="Error (%)", row=2, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_xaxes(title_text="Evaluation Date", row=2, col=2)
        fig.update_yaxes(title_text="Absolute Error (%)", row=2, col=2)

        # Save plot
        plot_file = output_dir / f"basin_{basin_code}_analysis.html"
        fig.write_html(plot_file)
        logger.info(f"Saved basin {basin_code} analysis plot to {plot_file}")


def create_model_comparison_plots(all_metrics: list, output_dir: Path) -> None:
    """
    Create model comparison plots showing performance across all basins and dates.

    Args:
        all_metrics: List of all metrics
        output_dir: Output directory for plots
    """
    if not all_metrics:
        logger.warning("No metrics available for model comparison plots")
        return

    # Convert metrics to DataFrame for easier manipulation
    metrics_df = pd.DataFrame(all_metrics)

    # Create model comparison subplot
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Model Performance Distribution",
            "Model Performance Over Time",
            "Basin Count by Model",
            "Error Categories by Model",
        ),
    )

    models = metrics_df["model_name"].unique()
    colors = px.colors.qualitative.Set3

    # Plot 1: Box plot of model performance
    for i, model in enumerate(models):
        model_data = metrics_df[metrics_df["model_name"] == model]
        fig.add_trace(
            go.Box(
                y=model_data["mean_absolute_error_pct"],
                name=model,
                marker_color=colors[i % len(colors)],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Plot 2: Performance over time (line plot)
    for i, model in enumerate(models):
        model_data = metrics_df[metrics_df["model_name"] == model]
        grouped = (
            model_data.groupby("evaluation_date")["mean_absolute_error_pct"]
            .mean()
            .reset_index()
        )

        fig.add_trace(
            go.Scatter(
                x=grouped["evaluation_date"],
                y=grouped["mean_absolute_error_pct"],
                mode="lines+markers",
                name=model,
                line=dict(color=colors[i % len(colors)]),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    # Plot 3: Basin count by model
    basin_counts = (
        metrics_df.groupby("model_name")["num_predictions"].mean().reset_index()
    )
    fig.add_trace(
        go.Bar(
            x=basin_counts["model_name"],
            y=basin_counts["num_predictions"],
            name="Average Basin Count",
            marker_color=colors[: len(basin_counts)],
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Plot 4: Error categories by model
    error_categories = []
    for model in models:
        model_data = metrics_df[metrics_df["model_name"] == model]
        avg_over_30 = model_data["basins_over_30pct_error"].apply(len).mean()
        avg_over_50 = model_data["basins_over_50pct_error"].apply(len).mean()
        error_categories.append(
            {"model": model, "over_30pct": avg_over_30, "over_50pct": avg_over_50}
        )

    error_df = pd.DataFrame(error_categories)
    fig.add_trace(
        go.Bar(
            x=error_df["model"],
            y=error_df["over_30pct"],
            name="Basins >30% Error",
            marker_color="orange",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Bar(
            x=error_df["model"],
            y=error_df["over_50pct"],
            name="Basins >50% Error",
            marker_color="red",
            showlegend=True,
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig.update_layout(
        title_text="Model Performance Comparison",
        height=800,
        showlegend=True,
        template="plotly_white",
    )

    # Update axes labels
    fig.update_xaxes(title_text="Model", row=1, col=1)
    fig.update_yaxes(title_text="Mean Absolute Error (%)", row=1, col=1)
    fig.update_xaxes(title_text="Evaluation Date", row=1, col=2)
    fig.update_yaxes(title_text="Mean Absolute Error (%)", row=1, col=2)
    fig.update_xaxes(title_text="Model", row=2, col=1)
    fig.update_yaxes(title_text="Average Basin Count", row=2, col=1)
    fig.update_xaxes(title_text="Model", row=2, col=2)
    fig.update_yaxes(title_text="Average Basin Count", row=2, col=2)

    # Save plot
    plot_file = output_dir / "model_comparison.html"
    fig.write_html(plot_file)
    logger.info(f"Saved model comparison plot to {plot_file}")


def create_performance_heatmap(all_metrics: list, output_dir: Path) -> None:
    """
    Create a heatmap showing model performance across evaluation dates.

    Args:
        all_metrics: List of all metrics
        output_dir: Output directory for plots
    """
    if not all_metrics:
        logger.warning("No metrics available for heatmap")
        return

    # Convert to DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    # Create pivot table for heatmap
    heatmap_data = metrics_df.pivot_table(
        values="mean_absolute_error_pct",
        index="model_name",
        columns="evaluation_date",
        aggfunc="mean",
    )

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale="RdYlBu_r",
            colorbar=dict(title="Mean Absolute Error (%)"),
            hovertemplate="<b>%{y}</b><br>"
            + "Date: %{x}<br>"
            + "Error: %{z:.1f}%<extra></extra>",
        )
    )

    fig.update_layout(
        title="Model Performance Heatmap",
        xaxis_title="Evaluation Date",
        yaxis_title="Model",
        template="plotly_white",
        height=600,
    )

    # Save plot
    plot_file = output_dir / "performance_heatmap.html"
    fig.write_html(plot_file)
    logger.info(f"Saved performance heatmap to {plot_file}")


def generate_plots(
    all_predictions: pd.DataFrame,
    all_metrics: list,
    output_dir: Path,
    target_basins: list = None,
    create_plots: bool = True,
) -> None:
    """
    Generate all plots for the synthetic evaluation.

    Args:
        all_predictions: Combined predictions DataFrame
        all_metrics: List of all metrics
        output_dir: Output directory for plots
        target_basins: List of specific basin codes to plot
        create_plots: Whether to create plots
    """
    if not create_plots:
        return

    logger.info("Generating interactive plots...")

    # Create plots directory
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Generate different types of plots
    create_observed_vs_predicted_plots(
        all_predictions, all_metrics, plots_dir, target_basins
    )

    create_model_comparison_plots(all_metrics, plots_dir)

    create_performance_heatmap(all_metrics, plots_dir)

    logger.info(f"All plots saved to {plots_dir}")


def run_synthetic_evaluation(
    start_month: int = 4,
    end_month: int = 9,
    year: int = 2024,
    forecast_days: list = [10, 20, -1],  # -1 represents last day of month
    output_dir: str = "synthetic_test",
    target_basins: list = None,
    create_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run synthetic evaluation over multiple dates to assess model performance.

    Args:
        start_month (int): Starting month (1-12)
        end_month (int): Ending month (1-12)
        year (int): Year for evaluation
        forecast_days (list): List of days to forecast on (-1 for last day)
        output_dir (str): Directory to save evaluation results
        target_basins (list): List of specific basin codes to plot (None for auto-selection)
        create_plots (bool): Whether to create interactive plots

    Returns:
        Dictionary with combined results and performance metrics
    """
    logger.info("=" * 60)
    logger.info("SYNTHETIC EVALUATION WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Evaluating from {start_month:02d}/{year} to {end_month:02d}/{year}")
    logger.info(f"Forecast days: {forecast_days}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize combined results
    all_predictions = []
    all_metrics = []
    evaluation_dates = []

    # Generate evaluation dates
    for month in range(start_month, end_month + 1):
        for day in forecast_days:
            if day == -1:
                # Last day of month
                last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
                eval_date = datetime.datetime(year, month, last_day.day)
            else:
                # Specific day
                try:
                    eval_date = datetime.datetime(year, month, day)
                except ValueError:
                    # Skip invalid dates (e.g., Feb 30th)
                    continue

            evaluation_dates.append(eval_date)

    logger.info(f"Generated {len(evaluation_dates)} evaluation dates")

    # Run predictions for each date
    for i, eval_date in enumerate(evaluation_dates):
        logger.info(
            f"Processing date {i + 1}/{len(evaluation_dates)}: {eval_date.strftime('%Y-%m-%d')}"
        )

        try:
            # Run prediction for this date
            results = run_operational_prediction(today=eval_date)

            # Add evaluation date to predictions
            if not results["predictions"].empty:
                results["predictions"]["evaluation_date"] = eval_date.strftime(
                    "%Y-%m-%d"
                )
                all_predictions.append(results["predictions"])

            # Add evaluation date to metrics
            for metric in results["metrics"]:
                metric["evaluation_date"] = eval_date.strftime("%Y-%m-%d")
                all_metrics.append(metric)

        except Exception as e:
            logger.error(
                f"Error processing date {eval_date.strftime('%Y-%m-%d')}: {str(e)}"
            )
            continue

    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        logger.info(
            f"Combined {len(combined_predictions)} predictions from {len(evaluation_dates)} dates"
        )
    else:
        combined_predictions = pd.DataFrame()
        logger.warning("No predictions generated during synthetic evaluation")

    # Save metrics
    if all_metrics:
        metrics_file = output_path / "synthetic_metrics.csv"
        pd.DataFrame(all_metrics).to_csv(metrics_file, index=False)
        logger.info(f"Saved synthetic metrics to {metrics_file}")

    # Calculate aggregated performance metrics
    aggregated_metrics = calculate_aggregated_metrics(all_metrics)

    # Generate synthetic evaluation report
    generate_synthetic_evaluation_report(
        all_metrics, aggregated_metrics, evaluation_dates, output_path
    )

    # Add observed discharge data to predictions for plotting
    if not combined_predictions.empty and all_metrics:
        # Create a mapping of basin codes to observed discharge from metrics
        observed_data = {}
        for metric in all_metrics:
            if "basin_errors" in metric and metric["basin_errors"]:
                for basin_error in metric["basin_errors"]:
                    basin_code = basin_error["basin_code"]
                    observed = basin_error["observed"]
                    eval_date = metric["evaluation_date"]
                    observed_data[(basin_code, eval_date)] = observed

        # Add observed discharge column to predictions
        combined_predictions["observed_discharge"] = combined_predictions.apply(
            lambda row: observed_data.get((row["code"], row["evaluation_date"]), None),
            axis=1,
        )

        logger.info(
            f"Added observed discharge data for {len(observed_data)} basin-date combinations"
        )

    # Calculate ensemble predictions (mean of all model predictions)
    if not combined_predictions.empty:
        # Find all Q_ columns (prediction columns)
        prediction_columns = [
            col for col in combined_predictions.columns if col.startswith("Q_")
        ]

        if prediction_columns:
            # Calculate ensemble by grouping by basin and evaluation date
            ensemble_results = []

            # Group by code and evaluation_date to get predictions for the same basin-date combination
            for (code, eval_date), group in combined_predictions.groupby(
                ["code", "evaluation_date"]
            ):
                # Collect predictions from each model for this basin-date combination
                model_predictions = []

                # For each prediction column, get the single non-null value if it exists
                for col in prediction_columns:
                    col_values = group[col].dropna()
                    if len(col_values) > 0:
                        # Take the first (and should be only) non-null value for this model
                        model_predictions.append(col_values.iloc[0])

                # Calculate ensemble mean across different models
                if model_predictions:
                    ensemble_value = np.mean(model_predictions)
                    # Add ensemble value to all rows for this basin-date combination
                    for idx in group.index:
                        ensemble_results.append((idx, ensemble_value))

            # Apply ensemble values to the dataframe
            combined_predictions["Q_Ensemble"] = np.nan
            for idx, ensemble_value in ensemble_results:
                combined_predictions.loc[idx, "Q_Ensemble"] = ensemble_value

            logger.info(
                f"Added ensemble predictions using {len(prediction_columns)} models: {prediction_columns}"
            )
            num_combinations = len(
                combined_predictions.groupby(["code", "evaluation_date"])
            )
            logger.info(
                f"Calculated ensemble for {num_combinations} basin-date combinations"
            )
        else:
            logger.warning("No prediction columns found for ensemble calculation")

    # Save results (after observed data and ensemble are added)
    if not combined_predictions.empty:
        predictions_file = output_path / "synthetic_predictions.csv"
        combined_predictions.to_csv(predictions_file, index=False)
        logger.info(f"Saved synthetic predictions to {predictions_file}")

    # Generate plots
    generate_plots(
        combined_predictions, all_metrics, output_path, target_basins, create_plots
    )

    print_synthetic_evaluation_summary(aggregated_metrics, evaluation_dates)

    return {
        "predictions": combined_predictions,
        "metrics": all_metrics,
        "aggregated_metrics": aggregated_metrics,
        "evaluation_dates": evaluation_dates,
    }


def calculate_aggregated_metrics(all_metrics: list) -> Dict[str, Any]:
    """
    Calculate aggregated performance metrics across all evaluation dates.
    """
    if not all_metrics:
        return {}

    # Group metrics by model
    model_metrics = {}
    for metric in all_metrics:
        model_key = (
            f"{metric['model_family']}_{metric['model_type']}_{metric['model_name']}"
        )
        if model_key not in model_metrics:
            model_metrics[model_key] = {
                "model_family": metric["model_family"],
                "model_type": metric["model_type"],
                "model_name": metric["model_name"],
                "evaluation_dates": [],
                "mean_errors": [],
                "median_errors": [],
                "num_predictions": [],
            }

        if not np.isnan(metric.get("mean_absolute_error_pct", np.nan)):
            model_metrics[model_key]["evaluation_dates"].append(
                metric["evaluation_date"]
            )
            model_metrics[model_key]["mean_errors"].append(
                metric["mean_absolute_error_pct"]
            )
            model_metrics[model_key]["median_errors"].append(
                metric["median_absolute_error_pct"]
            )
            model_metrics[model_key]["num_predictions"].append(
                metric["num_predictions"]
            )

    # Calculate aggregated statistics
    aggregated = {}
    for model_key, data in model_metrics.items():
        if data["mean_errors"]:
            aggregated[model_key] = {
                "model_family": data["model_family"],
                "model_type": data["model_type"],
                "model_name": data["model_name"],
                "num_evaluation_dates": len(data["evaluation_dates"]),
                "avg_mean_error": np.mean(data["mean_errors"]),
                "std_mean_error": np.std(data["mean_errors"]),
                "avg_median_error": np.mean(data["median_errors"]),
                "avg_num_predictions": np.mean(data["num_predictions"]),
                "total_predictions": sum(data["num_predictions"]),
            }

    return aggregated


def generate_synthetic_evaluation_report(
    all_metrics: list,
    aggregated_metrics: Dict[str, Any],
    evaluation_dates: list,
    output_path: Path,
):
    """
    Generate a comprehensive synthetic evaluation report.
    """
    report_file = output_path / "synthetic_evaluation_report.txt"

    with open(report_file, "w") as f:
        f.write("SYNTHETIC EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(
            f"Evaluation Period: {evaluation_dates[0].strftime('%Y-%m-%d')} to {evaluation_dates[-1].strftime('%Y-%m-%d')}\n"
        )
        f.write(f"Number of evaluation dates: {len(evaluation_dates)}\n")
        f.write(f"Total models evaluated: {len(aggregated_metrics)}\n\n")

        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-" * 30 + "\n")

        for model_key, metrics in aggregated_metrics.items():
            f.write(
                f"\n{metrics['model_family']} → {metrics['model_type']} → {metrics['model_name']}\n"
            )
            f.write(f"  Evaluation dates: {metrics['num_evaluation_dates']}\n")
            f.write(
                f"  Average mean error: {metrics['avg_mean_error']:.1f}% ± {metrics['std_mean_error']:.1f}%\n"
            )
            f.write(f"  Average median error: {metrics['avg_median_error']:.1f}%\n")
            f.write(f"  Total predictions: {metrics['total_predictions']}\n")

    logger.info(f"Saved synthetic evaluation report to {report_file}")


def print_synthetic_evaluation_summary(
    aggregated_metrics: Dict[str, Any], evaluation_dates: list
):
    """
    Print a summary of synthetic evaluation results.
    """
    print("\n" + "=" * 70)
    print("🔬 SYNTHETIC EVALUATION SUMMARY")
    print("=" * 70)

    print(
        f"📅 Evaluation period: {evaluation_dates[0].strftime('%Y-%m-%d')} to {evaluation_dates[-1].strftime('%Y-%m-%d')}"
    )
    print(f"📊 Evaluation dates: {len(evaluation_dates)}")
    print(f"🤖 Models evaluated: {len(aggregated_metrics)}")

    if aggregated_metrics:
        print(f"\n📈 MODEL PERFORMANCE RANKING:")

        # Sort models by average mean error
        sorted_models = sorted(
            aggregated_metrics.items(), key=lambda x: x[1]["avg_mean_error"]
        )

        for i, (model_key, metrics) in enumerate(sorted_models):
            rank_emoji = (
                "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else f"{i + 1}."
            )
            print(
                f"  {rank_emoji} {metrics['model_family']} → {metrics['model_name']}: {metrics['avg_mean_error']:.1f}% ± {metrics['std_mean_error']:.1f}%"
            )

    print("=" * 70 + "\n")


def run_operational(today: datetime.datetime = None):
    """
    Main entry point for operational prediction workflow.

    Args:
        today (datetime.datetime, optional): Date to use as "today" for prediction.
            If None, uses current datetime.
    """
    try:
        logger.info("=" * 50)
        logger.info("OPERATIONAL PREDICTION WORKFLOW")
        logger.info("=" * 50)

        # Run the prediction workflow
        results = run_operational_prediction(today=today)

        # Display performance summary
        print_model_performance_summary(results["metrics"])

        # Generate outputs
        generate_outputs(results)

        logger.info("Operational prediction workflow completed successfully!")

    except Exception as e:
        logger.error(f"Error in operational prediction workflow: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run operational prediction workflow")
    parser.add_argument(
        "--mode",
        choices=["operational", "synthetic"],
        default="operational",
        help="Mode to run: operational (single date) or synthetic (multiple dates)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../monthly_forecasting_results/synthetic_test",
        help="Output directory for synthetic evaluation",
    )
    parser.add_argument(
        "--target-basins",
        nargs="+",
        type=int,
        default=None,
        help="Specific basin codes to plot (default: auto-select top 10)",
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        default=True,
        help="Create interactive HTML plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip plot generation",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=4,
        help="Starting month for synthetic evaluation (default: 4 for April)",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=9,
        help="Ending month for synthetic evaluation (default: 9 for September)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year for synthetic evaluation (default: 2025)",
    )
    parser.add_argument(
        "--forecast-days",
        nargs="+",
        type=str,
        default=["-1"],
        help="Forecast days for synthetic evaluation (default: 10 20 -1)",
    )

    args = parser.parse_args()

    # Convert forecast_days from strings to integers, handling -1 for end of month
    forecast_days = []
    for day in args.forecast_days:
        forecast_days.append(int(day))

    year = args.year
    start_month = args.start_month
    end_month = args.end_month

    # Parse today argument if provided

    today = datetime.datetime.now()  # Default to current date
    try:
        today = pd.to_datetime(
            today, format="%Y-%m-%d"
        )  # Ensure today is a datetime object
    except ValueError:
        logger.error(f"Invalid date format: {today}. Use YYYY-MM-DD format.")
        sys.exit(1)

    if args.mode == "operational":
        run_operational(today=today)
    elif args.mode == "synthetic":
        create_plots = args.create_plots and not args.no_plots
        run_synthetic_evaluation(
            start_month=start_month,
            end_month=end_month,
            year=year,
            forecast_days=forecast_days,
            output_dir=args.output_dir,
            target_basins=args.target_basins,
            create_plots=create_plots,
        )
