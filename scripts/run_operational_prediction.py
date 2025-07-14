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
from monthly_forecasting.scr import data_loading as dl


# Setup logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# import environment variables saved in .env file
from dotenv import load_dotenv

load_dotenv()

PATH_TO_DISCHARGE = os.getenv("path_discharge")
PATH_TO_FORCING_ERA5 = os.getenv("path_forcing_era5")
PATH_TO_FORCING_OPERATIONAL = os.getenv("path_forcing_operational")
PATH_SWE_00003 = os.getenv("path_SWE_00003")
PATH_SWE_500m = os.getenv("path_SWE_500m")
PATH_ROF_00003 = os.getenv("path_ROF_00003")
PATH_ROF_500m = os.getenv("path_ROF_500m")
PATH_TO_SHP = os.getenv("path_to_shp")
PATH_TO_STATIC = os.getenv("path_to_static")

MODELS_OPERATIONAL = {
    "BaseCase": [
        ("LR", "LR_Q_T_P"),
        ("SciRegressor", "ShortTerm_Features"),
        ("SciRegressor", "NormBased"),
    ],
    "SnowMapper_Based": [
        ("LR", "LR_Q_dSWEdt_T_P"),
        ("LR", "LR_Q_SWE_T"),
        ("LR", "LR_Q_T_P_SWE"),
        ("LR", "LR_Q_SWE"),
        ("SciRegressor", "NormBased"),
        ("SciRegressor", "ShortTermLR"),
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

    swe_500m = pd.read_csv(PATH_SWE_500m, parse_dates=["date"])
    swe_500m["date"] = pd.to_datetime(swe_500m["date"], format="%Y-%m-%d")
    swe_500m["code"] = swe_500m["code"].astype(int)

    rof_00003 = pd.read_csv(PATH_ROF_00003, parse_dates=["date"])
    rof_00003["date"] = pd.to_datetime(rof_00003["date"], format="%Y-%m-%d")
    rof_00003["code"] = rof_00003["code"].astype(int)

    rof_500m = pd.read_csv(PATH_ROF_500m, parse_dates=["date"])
    rof_500m["date"] = pd.to_datetime(rof_500m["date"], format="%Y-%m-%d")
    rof_500m["code"] = rof_500m["code"].astype(int)

    return swe_00003, swe_500m, rof_00003, rof_500m


def load_static_data():
    df_static = pd.read_csv(PATH_TO_STATIC, parse_dates=["date"])

    if "CODE" in df_static.columns:
        df_static.rename(columns={"CODE": "code"}, inplace=True)
    df_static["code"] = df_static["code"].astype(int)

    return df_static


def create_data_frame(config: Dict[str, Any]) -> pd.DataFrame:
    data = load_discharge()
    forcing = load_forcing()
    swe_00003, swe_500m, rof_00003, rof_500m = load_snowmapper()
    static_data = load_static_data()

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


def load_operational_configs(model_type: str, model_name: str) -> Dict[str, Any]:
    """
    Load all configuration files for an operational model.
    
    Args:
        model_type: Type of model ('LR' or 'SciRegressor')
        model_name: Name of the model configuration
        
    Returns:
        Dictionary containing all configurations
    """
    # Determine configuration directory based on model type and name
    config_dir = Path(MODELS_DIR) / model_name
    
    if not config_dir.exists():
        # Try alternative structure
        config_dir = Path(project_root) / "example_config" / "DUMMY_MODEL"
        logger.warning(f"Using dummy configuration from {config_dir}")
    
    # Load required configuration files
    config_files = {
        "general_config": "general_config.json",
        "model_config": "model_config.json", 
        "feature_config": "feature_config.json",
        "data_config": "data_config.json",
        "path_config": "data_paths.json"
    }
    
    configs = {}
    
    for config_name, config_file in config_files.items():
        config_path = config_dir / config_file
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                configs[config_name] = json.load(f)
            logger.info(f"Loaded {config_name} from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            configs[config_name] = {}
    
    # Set model type in general config
    if configs["general_config"]:
        configs["general_config"]["model_type"] = "linear_regression" if model_type == "LR" else "sciregressor"
    
    return configs


def shift_data_to_current_year(data_df: pd.DataFrame, shift_years: int = 1) -> pd.DataFrame:
    """
    Shift data dates by specified years to mock current year.
    
    Args:
        data_df: DataFrame with 'date' column
        shift_years: Number of years to shift forward
        
    Returns:
        DataFrame with shifted dates
    """
    data_df_shifted = data_df.copy()
    data_df_shifted['date'] = data_df_shifted['date'] + pd.DateOffset(years=shift_years)
    return data_df_shifted


def create_model_instance(model_type: str, model_name: str, configs: Dict[str, Any], 
                         data: pd.DataFrame, static_data: pd.DataFrame):
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


def calculate_average_discharge(data: pd.DataFrame, valid_from: str, valid_to: str) -> pd.DataFrame:
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
    mask = (data['date'] >= valid_from) & (data['date'] <= valid_to)
    period_data = data[mask]
    
    # Calculate average discharge per basin
    avg_discharge = period_data.groupby('code')['discharge'].mean().reset_index()
    avg_discharge.columns = ['code', 'observed_avg_discharge']
    
    return avg_discharge


def evaluate_predictions(predictions: pd.DataFrame, observations: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate performance metrics for predictions.
    
    Args:
        predictions: DataFrame with model predictions
        observations: DataFrame with observed values
        
    Returns:
        Dictionary with performance metrics
    """
    # Merge predictions and observations
    merged = pd.merge(predictions, observations, on='code', how='inner')
    
    # Calculate R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(merged['observed_avg_discharge'], merged['predicted_avg_discharge'])
    
    # Calculate relative error
    merged['relative_error'] = abs(merged['predicted_avg_discharge'] - merged['observed_avg_discharge']) / merged['observed_avg_discharge']
    
    # Identify poor predictions (>30% error)
    poor_predictions = merged[merged['relative_error'] > 0.3]
    
    metrics = {
        'overall_r2': r2,
        'num_predictions': len(merged),
        'num_poor_predictions': len(poor_predictions),
        'poor_prediction_rate': len(poor_predictions) / len(merged) if len(merged) > 0 else 0,
        'poor_prediction_basins': poor_predictions['code'].tolist()
    }
    
    return metrics


def run_operational_prediction() -> Dict[str, Any]:
    """
    Main operational prediction workflow.
    
    Returns:
        Dictionary with all results and metrics
    """
    logger.info("Starting operational prediction workflow...")
    
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
            logger.info(f"Processing model: {model_type} - {model_name}")
            
            # Start model timing
            model_start_time = datetime.datetime.now()
            
            try:
                # Load configurations
                configs = load_operational_configs(model_type, model_name)
                
                # Load and prepare data
                data, static_data = create_data_frame(configs.get("data_config", {}))
                
                # Shift data to current year
                data = shift_data_to_current_year(data, shift_years=1)
                
                # Create model instance
                model = create_model_instance(model_type, model_name, configs, data, static_data)
                
                # Run prediction (this would need to be implemented in the model classes)
                # For now, we'll simulate predictions
                predictions = simulate_predictions(model, data)
                
                # Store predictions
                predictions['model_family'] = family_name
                predictions['model_type'] = model_type
                predictions['model_name'] = model_name
                all_predictions.append(predictions)
                
                # Calculate model timing
                model_end_time = datetime.datetime.now()
                model_duration = (model_end_time - model_start_time).total_seconds()
                timing_results[f"{family_name}_{model_type}_{model_name}"] = model_duration
                
                logger.info(f"Completed {model_type} - {model_name} in {model_duration:.2f}s")
                
            except Exception as e:
                logger.error(f"Error processing {model_type} - {model_name}: {str(e)}")
                continue
    
    # Calculate overall timing
    overall_end_time = datetime.datetime.now()
    overall_duration = (overall_end_time - overall_start_time).total_seconds()
    timing_results['overall_duration'] = overall_duration
    
    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
    else:
        combined_predictions = pd.DataFrame()
    
    logger.info(f"Completed operational prediction workflow in {overall_duration:.2f}s")
    
    return {
        'predictions': combined_predictions,
        'timing': timing_results,
        'metrics': all_metrics
    }


def simulate_predictions(model, data: pd.DataFrame) -> pd.DataFrame:
    """
    Simulate predictions for testing purposes.
    This should be replaced with actual model prediction logic.
    
    Args:
        model: Model instance
        data: Input data
        
    Returns:
        DataFrame with predictions
    """
    # Get unique basin codes
    basin_codes = data['code'].unique()
    
    # Simulate predictions (random values for testing)
    np.random.seed(42)  # For reproducible results
    predictions = pd.DataFrame({
        'code': basin_codes,
        'predicted_avg_discharge': np.random.uniform(10, 100, len(basin_codes))
    })
    
    return predictions


def generate_outputs(results: Dict[str, Any], output_dir: str = "output"):
    """
    Generate all required output files.
    
    Args:
        results: Results dictionary from run_operational_prediction
        output_dir: Directory to save outputs
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Save predictions
    if not results['predictions'].empty:
        predictions_file = output_path / "operational_predictions.csv"
        results['predictions'].to_csv(predictions_file, index=False)
        logger.info(f"Saved predictions to {predictions_file}")
    
    # Save timing report
    timing_file = output_path / "timing_report.json"
    with open(timing_file, 'w') as f:
        json.dump(results['timing'], f, indent=2)
    logger.info(f"Saved timing report to {timing_file}")
    
    # Save performance metrics
    if results['metrics']:
        metrics_file = output_path / "performance_metrics.csv"
        pd.DataFrame(results['metrics']).to_csv(metrics_file, index=False)
        logger.info(f"Saved performance metrics to {metrics_file}")
    
    # Generate quality report
    quality_file = output_path / "quality_report.txt"
    with open(quality_file, 'w') as f:
        f.write("Operational Prediction Quality Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total models processed: {len(results['timing']) - 1}\n")  # -1 for overall_duration
        f.write(f"Total processing time: {results['timing']['overall_duration']:.2f}s\n")
        f.write(f"Total predictions: {len(results['predictions'])}\n")
        if results['metrics']:
            f.write(f"Average R² score: {np.mean([m['overall_r2'] for m in results['metrics']]):.3f}\n")
    logger.info(f"Saved quality report to {quality_file}")


def run_operational():
    """
    Main entry point for operational prediction workflow.
    """
    try:
        logger.info("=" * 50)
        logger.info("OPERATIONAL PREDICTION WORKFLOW")
        logger.info("=" * 50)
        
        # Run the prediction workflow
        results = run_operational_prediction()
        
        # Generate outputs
        generate_outputs(results)
        
        logger.info("Operational prediction workflow completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in operational prediction workflow: {str(e)}")
        raise


if __name__ == "__main__":
    run_operational()
