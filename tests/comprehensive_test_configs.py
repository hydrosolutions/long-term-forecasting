#!/usr/bin/env python3
"""
Comprehensive test configuration constants for SciRegressor models.

This module defines all configuration constants needed for comprehensive testing
of SciRegressor models (XGBoost, LightGBM, CatBoost) with different preprocessing
methods and workflow components.
"""

# Model configurations for testing
MODEL_CONFIGS = {
    "xgb": {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "reg:squarederror",
        "random_state": 42
    },
    "lgbm": {
        "n_estimators": 20,
        "max_depth": 3,
        "learning_rate": 0.1,
        "objective": "regression",
        "random_state": 42,
        "verbose": -1
    },
    "catboost": {
        "iterations": 20,
        "depth": 3,
        "learning_rate": 0.1,
        "loss_function": "RMSE",
        "random_state": 42,
        "verbose": False
    }
}

# Preprocessing method configurations
PREPROCESSING_CONFIGS = {
    "no_normalization": {
        "normalization_type": "none",
        "normalize_per_basin": False,
        "handle_na": "drop"
    },
    "global_normalization": {
        "normalization_type": "standard",
        "normalize_per_basin": False,
        "handle_na": "mean"
    },
    "per_basin_normalization": {
        "normalization_type": "standard",
        "normalize_per_basin": True,
        "handle_na": "mean"
    },
    "long_term_mean_scaling": {
        "normalization_type": "long_term_mean",
        "normalize_per_basin": False,
        "handle_na": "long_term_mean"
    }
}

# Base general configuration
BASE_GENERAL_CONFIG = {
    "prediction_horizon": 30,
    "offset": 30,
    "num_features": 5,
    "base_features": ["discharge", "P", "T"],
    "snow_vars": [],
    "forecast_days": ["end"],
    "filter_years": None,
    "target": "target",
    "use_pca": False,
    "use_lr_predictors": False,
    "use_temporal_features": True,
    "use_static_features": True,
    "cat_features": ["code_str"],
    "feature_cols": ["discharge", "P", "T"],
    "static_features": ["area", "elevation"],
    "rivers_to_exclude": [],
    "test_years": 3,
    "num_elevation_zones": 5,
    "n_trials": 1,  # Single trial for fast testing
    "hparam_tuning_years": 3,
    "early_stopping_val_fraction": 0.1,
    "num_test_years": 2
}

# Feature configuration
FEATURE_CONFIG = {
    "discharge": [
        {
            "operation": "mean",
            "windows": [15, 30],
            "lags": {}
        }
    ],
    "P": [
        {
            "operation": "sum",
            "windows": [30],
            "lags": {}
        }
    ],
    "T": [
        {
            "operation": "mean",
            "windows": [15],
            "lags": {}
        }
    ]
}

# Path configuration
PATH_CONFIG = {
    "path_discharge": "/fake/path/discharge",
    "path_forcing": "/fake/path/forcing",
    "path_static_data": "/fake/path/static",
    "path_to_sca": "/fake/path/sca",
    "path_to_swe": "/fake/path/swe",
    "path_to_hs": "/fake/path/hs",
    "path_to_rof": "/fake/path/rof",
    "HRU_SWE": "/fake/path/hru_swe",
    "HRU_HS": "/fake/path/hru_hs",
    "HRU_ROF": "/fake/path/hru_rof",
    "path_to_hru_shp": None,
    "model_home_path": "tests_output"
}

# Data configuration
DATA_CONFIG = {
    "start_year": 2018,
    "end_year": 2023,
    "basins": [16936, 16940, 16942]
}

# Test constants
TEST_CONSTANTS = {
    "basin_codes": [16936, 16940, 16942],
    "start_date": "2018-01-01",
    "noise_level": 0.1,
    "random_seed": 42,
    "min_predictions_for_metrics": 5,
    "expected_columns": {
        "hindcast": ["date", "code", "Q_obs"],
        "forecast": ["code", "forecast_date", "valid_from", "valid_to"],
        "time_series": ["date", "code", "discharge", "P", "T"],
        "static": ["code", "area", "elevation", "area_km2"]
    }
}

# Model types and preprocessing methods for parameterized testing
MODEL_TYPES = ["xgb", "lgbm", "catboost"]
PREPROCESSING_METHODS = ["no_normalization", "global_normalization", "per_basin_normalization", "long_term_mean_scaling"]

# Workflow components
WORKFLOW_COMPONENTS = ["hyperparameter_tuning", "calibration", "hindcast", "operational_prediction"]

# Validation criteria
VALIDATION_CRITERIA = {
    "r2_min": -1.0,  # R² can be negative for very bad models
    "r2_max": 1.0,   # R² should not exceed 1.0
    "rmse_min": 0.0, # RMSE should be non-negative
    "discharge_min": 0.0,  # Discharge should be non-negative
    "discharge_max": 1000.0,  # Reasonable upper bound for discharge
    "min_data_points": 10,  # Minimum data points for meaningful tests
    "max_missing_ratio": 0.5,  # Maximum ratio of missing values allowed
}

# Test timeouts (in seconds)
TEST_TIMEOUTS = {
    "hyperparameter_tuning": 300,  # 5 minutes
    "calibration": 180,  # 3 minutes
    "hindcast": 120,  # 2 minutes
    "operational_prediction": 60,  # 1 minute
    "complete_workflow": 600,  # 10 minutes
    "multi_model_ensemble": 300  # 5 minutes
}

# Expected performance benchmarks (for synthetic data)
PERFORMANCE_BENCHMARKS = {
    "xgb": {
        "r2_min": 0.1,
        "rmse_max": 100.0
    },
    "lgbm": {
        "r2_min": 0.1,
        "rmse_max": 100.0
    },
    "catboost": {
        "r2_min": 0.1,
        "rmse_max": 100.0
    }
}

# Test configuration templates
def get_test_config(model_type: str, preprocessing_method: str, model_name: str = None) -> dict:
    """
    Get a complete test configuration for a specific model and preprocessing method.
    
    Args:
        model_type: Type of model (xgb, lgbm, catboost)
        preprocessing_method: Preprocessing method to use
        model_name: Optional custom model name
        
    Returns:
        Dictionary containing complete test configuration
    """
    if model_name is None:
        model_name = f"Test_{model_type}_{preprocessing_method}"
    
    # Create general config
    general_config = BASE_GENERAL_CONFIG.copy()
    general_config["model_name"] = model_name
    general_config["models"] = [model_type]
    
    # Add preprocessing-specific config
    preprocessing_config = PREPROCESSING_CONFIGS[preprocessing_method]
    general_config.update(preprocessing_config)
    
    # Create model config
    model_config = {model_type: MODEL_CONFIGS[model_type].copy()}
    
    return {
        "general_config": general_config,
        "model_config": model_config,
        "feature_config": FEATURE_CONFIG.copy(),
        "path_config": PATH_CONFIG.copy(),
        "data_config": DATA_CONFIG.copy()
    }

# Test data generation parameters
TEST_DATA_PARAMS = {
    "num_years": 5,  # 2018-2023
    "num_basins": 3,
    "seasonal_amplitude": 15.0,  # Temperature seasonal variation
    "precip_base": 10.0,  # Base precipitation
    "discharge_base": 50.0,  # Base discharge
    "noise_factors": {
        "temperature": 2.0,
        "precipitation": 0.3,
        "discharge": 0.1
    },
    "extreme_event_probability": 0.05,  # 5% chance of extreme events
    "basin_scaling_factor": 0.2,  # Inter-basin variability
    "temporal_correlation": 0.7,  # Temporal correlation in time series
    "elevation_effect": 0.5  # Elevation effect on temperature
}