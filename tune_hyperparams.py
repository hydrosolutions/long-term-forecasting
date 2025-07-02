#!/usr/bin/env python3
"""
Standalone Hyperparameter Tuning Script for Monthly Forecasting Models

This script provides a command-line interface for tuning hyperparameters of
both LinearRegressionModel and SciRegressor models using Optuna optimization.

Usage:
    python tune_hyperparams.py --config_dir path/to/model/config --model_name ModelName [options]

Example:
    python tune_hyperparams.py --config_dir monthly_forecasting_models/LinearRegression_BasicFeatures --model_name LinearRegression_BasicFeatures --trials 100
"""

import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import forecast models
from forecast_models.LINEAR_REGRESSION import LinearRegressionModel
from forecast_models.SciRegressor import SciRegressor
from scr import data_loading as dl

# Setup logging
from log_config import setup_logging
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
        'general_config': 'general_config.json',
        'model_config': 'model_config.json', 
        'feature_config': 'feature_config.json',
        'data_config': 'data_config.json',
        'path_config': '../config/data_paths.json'  # Usually in parent config directory
    }
    
    configs = {}
    
    for config_name, config_file in config_files.items():
        config_path = config_dir / config_file
        
        # Try alternative locations for path_config
        if not config_path.exists() and config_name == 'path_config':
            alternative_paths = [
                config_dir.parent / 'config' / 'data_paths.json',
                config_dir.parent / 'data_paths.json',
                Path('config') / 'data_paths.json'
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    config_path = alt_path
                    break
        
        if config_path.exists():
            with open(config_path, 'r') as f:
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
    try:
        # Try to use the existing data loading function
        if 'data_loader_function' in data_config:
            loader_func = data_config['data_loader_function']
            # This would need to be implemented based on the specific data loading approach
            logger.warning(f"Custom data loader {loader_func} not implemented. Using default approach.")
        
        # Default data loading approach
        data_path = path_config.get('hydro_data_path', 'data/hydro_data.csv')
        static_path = path_config.get('static_data_path', 'data/static_data.csv')
        
        if os.path.exists(data_path):
            data = pd.read_csv(data_path)
            logger.info(f"Loaded data from {data_path}: {len(data)} rows")
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        if os.path.exists(static_path):
            static_data = pd.read_csv(static_path)
            logger.info(f"Loaded static data from {static_path}: {len(static_data)} rows")
        else:
            logger.warning(f"Static data file not found: {static_path}. Using empty DataFrame.")
            static_data = pd.DataFrame()
        
        # Ensure data has required columns
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        else:
            raise ValueError("Data must contain a 'date' column")
        
        if 'target' not in data.columns:
            raise ValueError("Data must contain a 'target' column")
        
        if 'code' not in data.columns:
            raise ValueError("Data must contain a 'code' column")
        
        return data, static_data
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def create_model(model_name: str, configs: Dict[str, Any], data: pd.DataFrame, static_data: pd.DataFrame):
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
    general_config = configs['general_config']
    model_config = configs['model_config']
    feature_config = configs['feature_config']
    path_config = configs['path_config']
    
    # Determine model type based on configuration or name
    model_type = general_config.get('model_type', 'auto')
    
    if model_type == 'auto':
        # Infer model type from name or configuration
        if 'linear' in model_name.lower() or 'lr' in model_name.lower():
            model_type = 'linear_regression'
        elif any(ml_type in model_name.lower() for ml_type in ['xgb', 'lgbm', 'catboost', 'tree', 'ml']):
            model_type = 'sciregressor'
        else:
            # Check if models are specified in general_config
            if 'models' in general_config and len(general_config['models']) > 0:
                model_type = 'sciregressor'
            else:
                model_type = 'linear_regression'
        
        logger.info(f"Auto-detected model type: {model_type}")
    
    # Create model instance
    if model_type == 'linear_regression':
        model = LinearRegressionModel(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config
        )
    elif model_type == 'sciregressor':
        model = SciRegressor(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def tune_hyperparameters(model, args):
    """
    Run hyperparameter tuning for the model.
    
    Args:
        model: Model instance
        args: Command line arguments
    """
    logger.info(f"Starting hyperparameter tuning for {model.name}")
    logger.info(f"Model type: {type(model).__name__}")
    
    # Update configuration with command line arguments
    if args.trials:
        model.general_config['hyperparam_tuning_trials'] = args.trials
    
    if args.tuning_years:
        model.general_config['hyperparam_tuning_years'] = args.tuning_years
    
    # Run hyperparameter tuning
    success, message = model.tune_hyperparameters(model.data)
    
    if success:
        logger.info("Hyperparameter tuning completed successfully!")
        logger.info(f"Results: {message}")
        
        # Save updated model configuration
        if args.save_config:
            config_dir = Path(args.config_dir)
            updated_config_path = config_dir / 'model_config_tuned.json'
            
            with open(updated_config_path, 'w') as f:
                json.dump(model.model_config, f, indent=4)
            
            logger.info(f"Updated model configuration saved to {updated_config_path}")
    else:
        logger.error("Hyperparameter tuning failed!")
        logger.error(f"Error: {message}")
        sys.exit(1)


def main():
    """Main function for hyperparameter tuning script."""
    
    parser = argparse.ArgumentParser(
        description='Tune hyperparameters for monthly forecasting models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tune Linear Regression model
  python tune_hyperparams.py --config_dir monthly_forecasting_models/LinearRegression_BasicFeatures --model_name LinearRegression_BasicFeatures
  
  # Tune SciRegressor with XGBoost ensemble
  python tune_hyperparams.py --config_dir monthly_forecasting_models/XGBoost_AdvancedFeatures --model_name XGBoost_AdvancedFeatures --trials 200
  
  # Quick tuning with fewer trials
  python tune_hyperparams.py --config_dir models/test_model --model_name test_model --trials 20 --tuning_years 2
        """
    )
    
    # Required arguments
    parser.add_argument('--config_dir', required=True,
                       help='Path to model configuration directory')
    parser.add_argument('--model_name', required=True,
                       help='Name of the model to tune')
    
    # Optional arguments
    parser.add_argument('--trials', type=int, default=None,
                       help='Number of Optuna trials (default: from config or 50/100)')
    parser.add_argument('--tuning_years', type=int, default=None,
                       help='Number of years to use for hyperparameter validation (default: 3)')
    parser.add_argument('--save_config', action='store_true',
                       help='Save updated configuration with best hyperparameters')
    parser.add_argument('--log_level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    try:
        # Load configurations
        logger.info(f"Loading configuration from {args.config_dir}")
        configs = load_configuration(args.config_dir)
        
        # Load data
        logger.info("Loading data...")
        data, static_data = load_data(configs['data_config'], configs['path_config'])
        
        # Create model
        logger.info(f"Creating model: {args.model_name}")
        model = create_model(args.model_name, configs, data, static_data)
        
        # Run hyperparameter tuning
        tune_hyperparameters(model, args)
        
        logger.info("Hyperparameter tuning script completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Hyperparameter tuning interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Hyperparameter tuning script failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == '__main__':
    main()