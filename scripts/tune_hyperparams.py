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
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def main():
    """Main function for hyperparameter tuning script."""

    parser = argparse.ArgumentParser(
        description="Tune hyperparameters for monthly forecasting models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune Linear Regression model
  python tune_hyperparams.py --config_dir monthly_forecasting_models/LinearRegression_BasicFeatures --model_name LinearRegression_BasicFeatures
  
  # Tune SciRegressor with XGBoost ensemble
  python tune_hyperparams.py --config_dir monthly_forecasting_models/XGBoost_AdvancedFeatures --model_name XGBoost_AdvancedFeatures --trials 200
  
  # Quick tuning with fewer trials
  python tune_hyperparams.py --config_dir models/test_model --model_name test_model --trials 20 --tuning_years 2
        """,
    )

    # Required arguments
    parser.add_argument(
        "--config_dir", required=True, help="Path to model configuration directory"
    )
    parser.add_argument("--model_name", required=True, help="Name of the model to tune")

    # Optional arguments
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of Optuna trials (default: from config or 50/100)",
    )
    parser.add_argument(
        "--tuning_years",
        type=int,
        default=None,
        help="Number of years to use for hyperparameter validation (default: 3)",
    )
    parser.add_argument(
        "--save_config",
        action="store_true",
        help="Save updated configuration with best hyperparameters",
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

        # Run hyperparameter tuning
        success, message = model.tune_hyperparameters()

        if success:
            logger.info("Hyperparameter tuning completed successfully!")
            logger.info(f"Results: {message}")

        else:
            logger.error("Hyperparameter tuning failed!")
            logger.error(f"Error: {message}")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("Hyperparameter tuning interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hyperparameter tuning script failed: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
