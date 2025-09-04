"""
Calibration and Hindcasting Script for Monthly Forecasting Models

This script orchestrates the complete calibration process for monthly discharge
forecasting models, including Leave-One-Year-Out cross-validation evaluation
and generating standardized metrics and visualizations.

Usage:
    python calibrate_hindcast.py --config_dir path/to/model/config --model_name ModelName [options]

Example:
    python calibrate_hindcast.py --config_dir monthly_forecasting_models/XGBoost_AllFeatures --model_name XGBoost_AllFeatures --output_dir results/
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import forecast models
from monthly_forecasting.forecast_models.LINEAR_REGRESSION import LinearRegressionModel
from monthly_forecasting.forecast_models.SciRegressor import SciRegressor
from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)
from monthly_forecasting.forecast_models.deep_models.uncertainty_mixture import (
    UncertaintyMixtureModel,
)
from monthly_forecasting.scr import data_loading as dl
from dev_tools.eval_scr import eval_helper, metric_functions

# Setup logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def load_configuration(config_dir: str) -> Dict[str, Any]:
    """
    Load all configuration files from the model directory.

    Args:
        config_dir: Path to the model configuration directory

    Returns:
        Dictionary containing all configurations
    """
    config_dir = Path(config_dir)

    if not config_dir.exists():
        raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

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

        # Try alternative locations for path_config
        if not config_path.exists() and config_name == "path_config":
            alternative_paths = [
                config_dir.parent / "config" / "data_paths.json",
                config_dir.parent / "data_paths.json",
                Path("config") / "data_paths.json",
            ]

            for alt_path in alternative_paths:
                if alt_path.exists():
                    config_path = alt_path
                    break

        if config_path.exists():
            with open(config_path, "r") as f:
                configs[config_name] = json.load(f)
            logger.info(f"Loaded {config_name} from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            configs[config_name] = {}

    return configs


def load_data(data_config: Dict[str, Any], path_config: Dict[str, Any]) -> tuple:
    """
    Load data using the data loading utilities.

    Args:
        data_config: Data configuration
        path_config: Path configuration

    Returns:
        Tuple of (data, static_data)
    """
    # -------------- 1. Load Data ------------------------------
    hydro_ca, static_df = dl.load_data(
        path_discharge=path_config["path_discharge"],
        path_forcing=path_config["path_forcing"],
        path_static_data=path_config["path_static_data"],
        path_to_sca=path_config["path_to_sca"],
        path_to_swe=path_config["path_to_swe"],
        path_to_hs=path_config["path_to_hs"],
        path_to_rof=path_config["path_to_rof"],
        HRU_SWE=path_config["HRU_SWE"],
        HRU_HS=path_config["HRU_HS"],
        HRU_ROF=path_config["HRU_ROF"],
        path_to_sla=path_config.get("path_to_sla", None),
    )

    # if log_discharge in columns - drop
    if "log_discharge" in hydro_ca.columns:
        hydro_ca.drop(columns=["log_discharge"], inplace=True)

    hydro_ca = hydro_ca.sort_values("date")

    hydro_ca["code"] = hydro_ca["code"].astype(int)

    if "CODE" in static_df.columns:
        static_df.rename(columns={"CODE": "code"}, inplace=True)
    static_df["code"] = static_df["code"].astype(int)

    return hydro_ca, static_df


def create_model(
    model_name: str,
    configs: Dict[str, Any],
    data: pd.DataFrame,
    static_data: pd.DataFrame,
):
    """
    Create the appropriate model instance based on the model name.

    Args:
        model_name: Name of the model
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

    # Determine model type based on configuration or name
    model_type = general_config.get("model_type", "linear_regression")

    # Create model instance
    if model_type == "linear_regression":
        model = LinearRegressionModel(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    elif model_type == "sciregressor":
        model = SciRegressor(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    elif model_type == "historical_meta_learner":
        model = HistoricalMetaLearner(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    elif model_type == "UncertaintyMixtureModel":
        model = UncertaintyMixtureModel(
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


def run_calibration_and_hindcast(model, output_dir: str) -> pd.DataFrame:
    """
    Run the calibration and hindcasting process.

    Args:
        model: Model instance
        output_dir: Directory to save results

    Returns:
        DataFrame with hindcast predictions
    """
    logger.info(f"Starting calibration and hindcasting for {model.name}")

    # Run calibration and hindcasting
    hindcast_df = model.calibrate_model_and_hindcast()

    if len(hindcast_df) == 0:
        logger.warning("No predictions generated during calibration")
        return hindcast_df

    logger.info(
        f"Generated {len(hindcast_df)} predictions for {len(hindcast_df['code'].unique())} basins"
    )

    # Save predictions
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.csv"
    hindcast_df.to_csv(predictions_path, index=False)
    logger.info(f"Predictions saved to {predictions_path}")

    # Save model if calibration was successful
    if hasattr(model, "fitted_models") and len(model.fitted_models) > 0:
        try:
            model.save_model()
            logger.info("Model saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save model: {e}")

    return hindcast_df


def calculate_metrics(
    hindcast_df: pd.DataFrame, output_dir: str, model_name: str
) -> pd.DataFrame:
    """
    Calculate evaluation metrics for the hindcast predictions.

    Args:
        hindcast_df: DataFrame with predictions
        output_dir: Directory to save metrics

    Returns:
        DataFrame with metrics
    """
    logger.info("Calculating evaluation metrics")

    if len(hindcast_df) == 0:
        logger.warning("No matching observations found for predictions")
        return pd.DataFrame()

    logger.info(f"Found {len(hindcast_df)} prediction-observation pairs")

    # Calculate metrics per model and basin
    metrics_list = []
    prediction_col = f"Q_{model_name}"
    for code, group in hindcast_df.groupby(["code"]):
        if len(group) < 5:  # Need sufficient data points
            continue

        obs = group["Q_obs"].values
        pred = group[prediction_col].values
        code = group["code"].iloc[0]  # Get the basin code

        # Calculate various metrics
        try:
            metrics = {
                "code": code,
                "n_predictions": len(group),
                "r2": metric_functions.r2_score(obs, pred),
                "rmse": metric_functions.rmse(obs, pred),
                "mae": metric_functions.mae(obs, pred),
                "bias": metric_functions.bias(obs, pred),
                "nse": metric_functions.nse(obs, pred)
                if hasattr(metric_functions, "nse")
                else np.nan,
            }
            if code == 16936:
                logger.debug(f"Metrics for code {code}: {metrics}")
            metrics_list.append(metrics)
        except Exception as e:
            logger.warning(f"Failed to calculate metrics for - {code}: {e}")

    if not metrics_list:
        logger.warning("No metrics calculated")
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics_list)

    # Save metrics
    output_dir = Path(output_dir)
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    logger.info(f"Metrics saved to {metrics_path}")

    # Calculate summary statistics
    summary_stats = {}
    for metric_col in ["r2", "rmse", "mae", "bias", "kge", "nse"]:
        if metric_col in metrics_df.columns:
            valid_values = metrics_df[metric_col].dropna()
            if len(valid_values) > 0:
                summary_stats[f"{metric_col}_mean"] = valid_values.mean()
                summary_stats[f"{metric_col}_median"] = valid_values.median()
                summary_stats[f"{metric_col}_std"] = valid_values.std()

    summary_stats["n_basins"] = len(metrics_df["code"].unique())
    summary_stats["total_predictions"] = metrics_df["n_predictions"].sum()

    # print summary statistics
    logger.info("Summary Statistics:")
    for key, value in summary_stats.items():
        logger.info(
            f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}"
        )

    # Log key results
    logger.info("Calibration Results Summary:")
    logger.info(f"  Basins evaluated: {summary_stats['n_basins']}")
    logger.info(f"  Total predictions: {summary_stats['total_predictions']}")
    if "r2_mean" in summary_stats:
        logger.info(f"  Mean R²: {summary_stats['r2_mean']:.3f}")
    if "rmse_mean" in summary_stats:
        logger.info(f"  Mean RMSE: {summary_stats['rmse_mean']:.3f}")

    return metrics_df


def main():
    """Main function for calibration and hindcasting script."""

    parser = argparse.ArgumentParser(
        description="Run calibration and hindcasting for monthly forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Calibrate Linear Regression model
  python calibrate_hindcast.py --config_dir monthly_forecasting_models/LinearRegression_BasicFeatures --model_name LinearRegression_BasicFeatures
  
  # Calibrate SciRegressor with custom output directory
  python calibrate_hindcast.py --config_dir monthly_forecasting_models/XGBoost_AdvancedFeatures --model_name XGBoost_AdvancedFeatures --output_dir results/xgb_calibration/
  
  # Skip metrics calculation (predictions only)
  python calibrate_hindcast.py --config_dir models/test_model --model_name test_model --skip_metrics
        """,
    )

    # Required arguments
    parser.add_argument(
        "--config_dir", required=True, help="Path to model configuration directory"
    )
    parser.add_argument(
        "--model_name", required=True, help="Name of the model to calibrate"
    )
    parser.add_argument(
        "--input_family",
        required=True,
        help="Family of the model input data: BaseCase (No Snow), SnowMapper_Based, GlacierMapper_Based ... etc",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory for results (default: monthly_forecasting_results/model_name/)",
    )
    parser.add_argument(
        "--skip_metrics",
        action="store_true",
        help="Skip metrics calculation (only generate predictions)",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Set default output directory
    if args.output_dir is None:
        parent_out = "../monthly_forecasting_results"
        parent_out = os.path.join(parent_out, args.input_family)
        if not os.path.exists(parent_out):
            os.makedirs(parent_out)

        output_dir = os.path.join(parent_out, args.model_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        args.output_dir = output_dir

    try:
        # Load configurations
        logger.info(f"Loading configuration from {args.config_dir}")
        configs = load_configuration(args.config_dir)

        # Load data
        logger.info("Loading data...")
        data, static_data = load_data(configs["data_config"], configs["path_config"])

        # Create model
        logger.info(f"Creating model: {args.model_name}")
        model = create_model(args.model_name, configs, data, static_data)

        # Run calibration and hindcasting
        hindcast_df = run_calibration_and_hindcast(model, args.output_dir)

        if len(hindcast_df) == 0:
            logger.warning("No predictions generated. Exiting.")
            return

        # Calculate metrics if not skipped
        metrics_df = pd.DataFrame()
        if not args.skip_metrics:
            metrics_df = calculate_metrics(
                hindcast_df, args.output_dir, args.model_name
            )

        metrics_df.to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
        logger.info("Calibration and hindcasting completed successfully!")
        logger.info(f"Results saved to {args.output_dir}")

    except KeyboardInterrupt:
        logger.info("Calibration interrupted by user")
        sys.exit(1)
    except Exception as e:
        # Log full traceback at ERROR level so shell script surfaces root cause
        import traceback

        tb = traceback.format_exc()
        logger.error(f"Calibration script failed: {e}\nFull traceback:\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
