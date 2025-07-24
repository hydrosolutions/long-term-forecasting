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
        ("SciRegressor", "GBT"),
    ],
    "SnowMapper_Based": [
        ("LR", "LR_Q_dSWEdt_T_P"),
        #("LR", "LR_Q_SWE_T"),
        ("LR", "LR_Q_T_P_SWE"),
        ("LR", "LR_Q_SWE"),
        ("SciRegressor", "Snow_GBT_Norm"),
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

    # count number of valid observations per basin
    valid_counts = period_data.groupby("code")["discharge"].count().reset_index()

    # if a basin has less than 25 valid observations, we skip it
    basin_to_nan = valid_counts[valid_counts["discharge"] < 25]["code"]

    # Calculate average discharge per basin
    avg_discharge = period_data.groupby("code")["discharge"].mean().reset_index()
    avg_discharge.columns = ["code", "observed_avg_discharge"]

    avg_discharge.loc[avg_discharge["code"].isin(basin_to_nan), "observed_avg_discharge"] = np.nan

    return avg_discharge


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
    all_predictions = {}
    all_metrics = []
    timing_results = {}

    ensemble_predictions = None

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

                logger.info(
                    f"Raw predictions for {model_name}:\n{raw_predictions.head()}"
                )

                pred_cols = [
                    col for col in raw_predictions.columns if col.startswith("Q_")
                ]

                valid_from = raw_predictions["valid_from"].min()
                valid_to = raw_predictions["valid_to"].max()

                logger.info(
                    f"Valid period for predictions: {valid_from} to {valid_to}"
                )

                # use original data - we have "future data"
                observed_avg_discharge = calculate_average_discharge(
                    data=data, valid_from=valid_from, valid_to=valid_to
                )

                raw_predictions = raw_predictions.merge(
                    observed_avg_discharge, on="code", how="left"
                )

                for pred_col in pred_cols:
                    
                    exact_type = pred_col.split("_")[-1]
                    this_model_name = f"{family_name}_{model_name}_{exact_type}"

                    all_predictions[this_model_name] = raw_predictions[
                        [ "code", pred_col, "valid_from", "valid_to", "observed_avg_discharge"]
                    ].rename(columns={pred_col: f"Q_pred"})

                    if ensemble_predictions is None:
                        ensemble_predictions = all_predictions[this_model_name].copy()
                    else:
                        ensemble_predictions = pd.merge(
                            ensemble_predictions,
                            all_predictions[this_model_name],
                            on=[ "code"],
                            how="outer",
                            suffixes=("", f"_{this_model_name}")
                        )

            except Exception as e:
                logger.error(f"Error processing {model_type} - {model_name}: {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue


    ensemble_pred_cols = [
        col for col in ensemble_predictions.columns if col.startswith("Q_pred")
    ]

    ensemble_predictions["Q_Ensemble"] = ensemble_predictions[ensemble_pred_cols].mean(axis=1)

    ensemble_predictions.drop(columns=ensemble_pred_cols, inplace=True)

    all_predictions["Ensemble"] = ensemble_predictions.rename(
        columns={"Q_Ensemble": "Q_pred"}
    )

    # Calculate overall timing
    overall_end_time = datetime.datetime.now()
    overall_duration = (overall_end_time - overall_start_time).total_seconds()
    timing_results["overall_duration"] = overall_duration

    metrics_df = pd.DataFrame()
    for model_name, predictions in all_predictions.items():
        if predictions.empty:
            logger.warning(f"No predictions for model: {model_name}")
            continue

        # Calculate metrics for this model
        metrics = calculate_metrics(predictions)

        if metrics.empty:
            logger.warning(f"No valid metrics calculated for model: {model_name}")
            continue

        # Add model name to metrics
        metrics["model"] = model_name

        # Append to overall metrics DataFrame
        metrics_df = pd.concat([metrics_df, metrics], ignore_index=True)

    logger.info(f"Completed operational prediction workflow in {overall_duration:.2f}s")

    return {
        "predictions": all_predictions,
        "timing": timing_results,
        "metrics": metrics_df,
    }


def relative_error(predicted: pd.Series, observed: pd.Series) -> float:
    """
    Calculate the absolute relative error between predicted and observed values.

    Args:
        predicted (pd.Series): Predicted values.
        observed (pd.Series): Observed values.

    Returns:
        pd.Series: Relative error values.
    """
    return np.abs((predicted - observed) / (observed + 1e-4))

def absolute_mean_error(predicted: pd.Series, observed: pd.Series) -> float:
    """
    Calculate the absolute mean error between predicted and observed values.

    Args:
        predicted (pd.Series): Predicted values.
        observed (pd.Series): Observed values.

    Returns:
        float: Absolute mean error.
    """
    return np.mean(np.abs(predicted - observed))

def calculate_metrics(
    df: pd.DataFrame,
    pred_col: str = "Q_pred",
    obs_col: str = "observed_avg_discharge",
) -> pd.DataFrame:
    """
    Calculate performance metrics for each basin in the DataFrame.
    
    Args:
        df: DataFrame containing predictions and observations
        pred_col: Column name for predictions
        obs_col: Column name for observations
        
    Returns:
        DataFrame with metrics for each basin
    """
    metrics_df = pd.DataFrame()
    
    for code in df["code"].unique():
        basin_data = df[df["code"] == code].copy()

        if basin_data.empty:
            continue

        # Get prediction and observation values
        predicted = basin_data[pred_col].iloc[0] if len(basin_data) > 0 else np.nan
        observed = basin_data[obs_col].iloc[0] if len(basin_data) > 0 else np.nan
        
        # Skip if either value is NaN
        if pd.isna(predicted) or pd.isna(observed):
            logger.warning(f"Skipping basin {code}: predicted={predicted}, observed={observed}")
            continue

        # Calculate relative error (scalar value)
        relative_error_value = relative_error(
            pd.Series([predicted]), pd.Series([observed])
        ).iloc[0]

        # Calculate absolute mean error (scalar value)
        abs_mean_error = absolute_mean_error(
            pd.Series([predicted]), pd.Series([observed])
        )

        # Create metrics row with scalar values
        metrics_row = {
            "code": code,
            "abs_mean_error": abs_mean_error,
            "relative_error_basin": relative_error_value,
        }

        metrics_row = pd.DataFrame([metrics_row], index=[0])
        metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)

    return metrics_df

def plot_metric_boxplot(
        metrics_df: pd.DataFrame,
        metric_col: str = "abs_mean_error",
):

    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=metrics_df,
        x = 'model',
        y = metric_col,
        hue='model',
        palette="Set2",
    )

    plt.title(f"Boxplot of {metric_col} by Model")
    plt.xlabel("Model")
    plt.ylabel(metric_col.replace("_", " ").title())
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def print_performance_model(df: pd.DataFrame, model_name: str, metric_col: str = "relative_error_basin") -> None:
    """
    Print performance metrics for a specific model.

    Args:
        df (pd.DataFrame): DataFrame containing metrics.
        model_name (str): Name of the model to filter by.
    """
    model_metrics = df[df["model"] == model_name].copy()

    if model_metrics.empty:
        logger.warning(f"No metrics found for model: {model_name}")
        return
    
    logger.info(f"-" * 50)
    logger.info(f"Performance metrics for model: {model_name}")

    # sort by the metric column 
    model_metrics = model_metrics.sort_values(by=metric_col, ascending=True)

    # print the top 10 basins with the lowest relative error
    logger.info(f"Top 10 basins with lowest {metric_col}:")
    logger.info(model_metrics.head(10))

    # print the worst 10 basins with the highest relative error
    logger.info(f"Worst 10 basins with highest {metric_col}:")
    logger.info(model_metrics.tail(10))

    mean_metric = model_metrics[metric_col].mean()
    median_metric = model_metrics[metric_col].median()
    max_metric = model_metrics[metric_col].max()
    min_metric = model_metrics[metric_col].min()
    std_metric = model_metrics[metric_col].std()    


    logger.info(f"Performance metrics for model {model_name}:")
    logger.info(f"Mean {metric_col}: {mean_metric:.4f}")
    logger.info(f"Median {metric_col}: {median_metric:.4f}")
    logger.info(f"Max {metric_col}: {max_metric:.4f}")
    logger.info(f"Min {metric_col}: {min_metric:.4f}")
    logger.info(f"Standard Deviation {metric_col}: {std_metric:.4f}")


    logger.info(f"-" * 50)

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

        for model, model_metrics in results["metrics"].groupby("model"):
            print_performance_model(
                model_metrics, model_name=model, metric_col="relative_error_basin"
            )

            # print also the observed vs predicted values
            predictions = results["predictions"][model]
            if not predictions.empty:
                # sort by code 
                predictions = predictions.sort_values(by="code", ascending=False).reset_index(drop=True)
                logger.info(f"Observed vs Predicted for model {model}:")
                logger.info(predictions[["code", "Q_pred", "observed_avg_discharge"]].head(10))


        plot_metric_boxplot(
            results["metrics"],
            metric_col="relative_error_basin",
        )

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
