#!/usr/bin/env python3
"""
Comprehensive test utility functions for SciRegressor models.

This module provides utility functions for comprehensive testing of SciRegressor
models with different preprocessing methods and workflow components.
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import logging
import datetime
from unittest.mock import patch, MagicMock

# Import test configurations
from comprehensive_test_configs import (
    MODEL_CONFIGS, PREPROCESSING_CONFIGS, BASE_GENERAL_CONFIG, FEATURE_CONFIG,
    PATH_CONFIG, DATA_CONFIG, TEST_CONSTANTS, VALIDATION_CRITERIA,
    TEST_DATA_PARAMS, get_test_config
)

# Setup logging
logger = logging.getLogger(__name__)

class ComprehensiveTestDataGenerator:
    """Enhanced test data generator for comprehensive SciRegressor testing."""
    
    @staticmethod
    def generate_comprehensive_timeseries_data(
        start_date: str = "2000-01-01",
        basin_codes: List[int] = None,
        noise_level: float = 0.1,
        preprocessing_method: str = "global_normalization"
    ) -> pd.DataFrame:
        """
        Generate comprehensive synthetic time series data for testing.
        
        Args:
            start_date: Start date for data generation
            basin_codes: List of basin codes to generate data for
            noise_level: Amount of random noise to add
            preprocessing_method: Preprocessing method to optimize data for
            
        Returns:
            DataFrame with synthetic time series data
        """
        if basin_codes is None:
            basin_codes = TEST_CONSTANTS["basin_codes"]
            
        end_date = pd.Timestamp.today()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data_list = []
        
        np.random.seed(TEST_CONSTANTS["random_seed"])
        
        for code in basin_codes:
            for date in date_range:
                # Generate seasonal patterns
                day_of_year = date.timetuple().tm_yday
                year = date.year
                
                # Enhanced temperature with seasonal cycle, trend, and basin effects
                T = (5 + TEST_DATA_PARAMS["seasonal_amplitude"] * 
                     np.sin(2 * np.pi * day_of_year / 365.25))
                T += 0.02 * (year - 2000)  # Climate change trend
                T += TEST_DATA_PARAMS["elevation_effect"] * (code - 16938)  # Elevation effect
                T += np.random.normal(0, TEST_DATA_PARAMS["noise_factors"]["temperature"])
                
                # Enhanced precipitation with seasonality, extremes, and basin effects
                P = max(0, TEST_DATA_PARAMS["precip_base"] + 
                       10 * np.sin(2 * np.pi * (day_of_year + 90) / 365.25))
                P += np.random.exponential(5)
                
                # Add extreme precipitation events
                if np.random.random() < TEST_DATA_PARAMS["extreme_event_probability"]:
                    P += np.random.exponential(20)
                
                # Basin-specific precipitation scaling
                basin_factor = 1 + TEST_DATA_PARAMS["basin_scaling_factor"] * (code - 16938) / 4
                P *= basin_factor
                P = max(0, P + np.random.normal(0, P * TEST_DATA_PARAMS["noise_factors"]["precipitation"]))
                
                # Complex discharge model with multiple influences
                base_discharge = (TEST_DATA_PARAMS["discharge_base"] + 
                                30 * np.sin(2 * np.pi * (day_of_year + 120) / 365.25))
                
                # Precipitation effect with delay
                precip_effect = P * 0.5
                if len(data_list) > 0:
                    recent_data = [d for d in data_list[-7:] if d['code'] == code]
                    if recent_data:
                        delayed_precip = sum([d['P'] for d in recent_data]) * 0.1
                        precip_effect += delayed_precip
                
                # Temperature effect (snowmelt)
                temp_effect = max(0, T - 0) * 2
                
                # Seasonal baseflow variation
                baseflow_seasonal = 20 * np.sin(2 * np.pi * (day_of_year + 200) / 365.25)
                
                discharge = base_discharge + precip_effect + temp_effect + baseflow_seasonal
                discharge *= basin_factor
                discharge = max(0.1, discharge + np.random.normal(0, discharge * TEST_DATA_PARAMS["noise_factors"]["discharge"]))
                
                # Add preprocessing-specific characteristics
                if preprocessing_method == "per_basin_normalization":
                    # Add more basin-specific variability
                    discharge *= (1 + 0.5 * np.sin(code * 0.1))
                    T *= (1 + 0.2 * np.cos(code * 0.1))
                    P *= (1 + 0.3 * np.sin(code * 0.2))
                elif preprocessing_method == "long_term_mean_scaling":
                    # Add more seasonal variability
                    seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * day_of_year / 365.25)
                    discharge *= seasonal_factor
                    T *= (1 + 0.1 * seasonal_factor)
                    P *= (1 + 0.2 * seasonal_factor)
                
                data_list.append({
                    'date': date,
                    'code': code,
                    'discharge': round(discharge, 2),
                    'T': round(T, 2),
                    'P': round(P, 2)
                })
        
        df = pd.DataFrame(data_list)
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        
        logger.info(f"Generated comprehensive synthetic data: {len(df)} records for {len(basin_codes)} basins")
        logger.info(f"Preprocessing method: {preprocessing_method}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        return df
    
    @staticmethod
    def generate_comprehensive_static_data(
        basin_codes: List[int] = None,
        preprocessing_method: str = "global_normalization"
    ) -> pd.DataFrame:
        """
        Generate comprehensive synthetic static basin data.
        
        Args:
            basin_codes: List of basin codes
            preprocessing_method: Preprocessing method to optimize data for
            
        Returns:
            DataFrame with synthetic static data
        """
        if basin_codes is None:
            basin_codes = TEST_CONSTANTS["basin_codes"]
            
        np.random.seed(TEST_CONSTANTS["random_seed"])
        
        static_data = []
        for code in basin_codes:
            # Base characteristics
            area = 100 + np.random.uniform(50, 500)
            elevation = 1000 + np.random.uniform(500, 2000)
            
            # Add preprocessing-specific characteristics
            if preprocessing_method == "per_basin_normalization":
                # Add more variability between basins
                area *= (1 + 0.5 * (code - 16938) / 4)
                elevation *= (1 + 0.3 * (code - 16938) / 4)
            elif preprocessing_method == "long_term_mean_scaling":
                # Add characteristics that would benefit from long-term scaling
                area *= (1 + 0.2 * np.sin(code * 0.1))
                elevation *= (1 + 0.1 * np.cos(code * 0.1))
            
            static_data.append({
                'code': code,
                'area': area,
                'area_km2': area,  # For SciRegressor compatibility
                'elevation': elevation,
                'slope': np.random.uniform(5, 30),
                'forest_cover': np.random.uniform(0.1, 0.8),
                'latitude': 40 + np.random.uniform(-5, 5),
                'longitude': 70 + np.random.uniform(-10, 10),
            })
        
        df = pd.DataFrame(static_data)
        logger.info(f"Generated comprehensive static data for {len(basin_codes)} basins")
        logger.info(f"Preprocessing method: {preprocessing_method}")
        
        return df

class ComprehensiveTestValidator:
    """Comprehensive test validation utilities."""
    
    @staticmethod
    def validate_model_initialization(model, model_type: str, preprocessing_method: str) -> bool:
        """
        Validate that a model was initialized correctly.
        
        Args:
            model: SciRegressor model instance
            model_type: Expected model type
            preprocessing_method: Expected preprocessing method
            
        Returns:
            True if validation passes
        """
        try:
            # Check basic attributes
            assert hasattr(model, 'name'), "Model should have name attribute"
            assert hasattr(model, 'data'), "Model should have data attribute"
            assert hasattr(model, 'static_data'), "Model should have static_data attribute"
            assert hasattr(model, 'models'), "Model should have models attribute"
            assert hasattr(model, 'fitted_models'), "Model should have fitted_models attribute"
            
            # Check model type
            assert model_type in model.models, f"Model type {model_type} not in models list"
            
            # Check preprocessing configuration
            preprocessing_config = PREPROCESSING_CONFIGS[preprocessing_method]
            for key, expected_value in preprocessing_config.items():
                if key in model.general_config:
                    actual_value = model.general_config[key]
                    assert actual_value == expected_value, f"Config mismatch for {key}: expected {expected_value}, got {actual_value}"
            
            logger.info(f"Model initialization validation passed for {model_type} with {preprocessing_method}")
            return True
            
        except AssertionError as e:
            logger.error(f"Model initialization validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in model initialization validation: {e}")
            return False
    
    @staticmethod
    def validate_predictions(predictions: pd.DataFrame, model_name: str, workflow_component: str) -> bool:
        """
        Validate prediction results.
        
        Args:
            predictions: DataFrame with predictions
            model_name: Name of the model
            workflow_component: Workflow component being tested
            
        Returns:
            True if validation passes
        """
        try:
            # Check basic structure
            assert len(predictions) > 0, "Predictions should not be empty"
            assert 'code' in predictions.columns, "Predictions should have code column"
            
            # Check prediction column exists
            pred_col = f"Q_{model_name}"
            assert pred_col in predictions.columns, f"Prediction column {pred_col} not found"
            
            # Check for valid predictions
            pred_values = predictions[pred_col].dropna()
            assert len(pred_values) > 0, "No valid predictions found"
            
            # Check prediction ranges
            min_val = pred_values.min()
            max_val = pred_values.max()
            assert min_val >= VALIDATION_CRITERIA["discharge_min"], f"Predictions too low: {min_val}"
            assert max_val <= VALIDATION_CRITERIA["discharge_max"], f"Predictions too high: {max_val}"
            
            # Component-specific validation
            if workflow_component in ["calibration", "hindcast"]:
                assert 'Q_obs' in predictions.columns, "Calibration/hindcast should have Q_obs column"
                assert 'date' in predictions.columns, "Calibration/hindcast should have date column"
            elif workflow_component == "operational_prediction":
                assert 'forecast_date' in predictions.columns, "Operational prediction should have forecast_date"
                assert 'valid_from' in predictions.columns, "Operational prediction should have valid_from"
                assert 'valid_to' in predictions.columns, "Operational prediction should have valid_to"
            
            logger.info(f"Prediction validation passed for {model_name} {workflow_component}")
            return True
            
        except AssertionError as e:
            logger.error(f"Prediction validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in prediction validation: {e}")
            return False
    
    @staticmethod
    def validate_performance_metrics(metrics: Dict[str, float], model_type: str) -> bool:
        """
        Validate performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            model_type: Type of model
            
        Returns:
            True if validation passes
        """
        try:
            # Check required metrics
            required_metrics = ['r2', 'rmse', 'mae']
            for metric in required_metrics:
                assert metric in metrics, f"Missing required metric: {metric}"
                assert not np.isnan(metrics[metric]), f"Metric {metric} is NaN"
            
            # Check R² bounds
            r2 = metrics['r2']
            assert r2 >= VALIDATION_CRITERIA["r2_min"], f"R² too low: {r2}"
            assert r2 <= VALIDATION_CRITERIA["r2_max"], f"R² too high: {r2}"
            
            # Check RMSE bounds
            rmse = metrics['rmse']
            assert rmse >= VALIDATION_CRITERIA["rmse_min"], f"RMSE negative: {rmse}"
            
            # Check against model-specific benchmarks if available
            if model_type in ["xgb", "lgbm", "catboost"]:
                # For synthetic data, we expect reasonable performance
                # but not perfect due to noise and complexity
                assert r2 >= -0.5, f"R² too low for synthetic data: {r2}"
                assert rmse <= 200.0, f"RMSE too high for synthetic data: {rmse}"
            
            logger.info(f"Performance metrics validation passed for {model_type}: R²={r2:.3f}, RMSE={rmse:.3f}")
            return True
            
        except AssertionError as e:
            logger.error(f"Performance metrics validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error in performance metrics validation: {e}")
            return False

class ComprehensiveTestEnvironment:
    """Comprehensive test environment management."""
    
    def __init__(self, model_type: str, preprocessing_method: str):
        """
        Initialize test environment.
        
        Args:
            model_type: Type of model to test
            preprocessing_method: Preprocessing method to use
        """
        self.model_type = model_type
        self.preprocessing_method = preprocessing_method
        self.test_dir = None
        self.data = None
        self.static_data = None
        self.configs = None
        
    def setup(self):
        """Setup test environment."""
        try:
            # Create temporary directory
            self.test_dir = tempfile.mkdtemp(prefix=f"comprehensive_test_{self.model_type}_{self.preprocessing_method}_")
            logger.info(f"Created test directory: {self.test_dir}")
            
            # Generate test data
            self.data = ComprehensiveTestDataGenerator.generate_comprehensive_timeseries_data(
                preprocessing_method=self.preprocessing_method
            )
            self.static_data = ComprehensiveTestDataGenerator.generate_comprehensive_static_data(
                preprocessing_method=self.preprocessing_method
            )
            
            # Get test configuration
            self.configs = get_test_config(self.model_type, self.preprocessing_method)
            
            # Update path config with test directory
            self.configs['path_config']['model_home_path'] = self.test_dir
            
            # Create config files
            self._create_config_files()
            
            logger.info(f"Test environment setup complete for {self.model_type} with {self.preprocessing_method}")
            
        except Exception as e:
            logger.error(f"Failed to setup test environment: {e}")
            raise
    
    def teardown(self):
        """Teardown test environment."""
        try:
            if self.test_dir and os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir)
                logger.info(f"Cleaned up test directory: {self.test_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup test environment: {e}")
    
    def _create_config_files(self):
        """Create configuration files in test directory."""
        config_dir = Path(self.test_dir) / "config"
        config_dir.mkdir(exist_ok=True)
        
        # Save configuration files
        config_files = {
            'general_config': self.configs['general_config'],
            'model_config': self.configs['model_config'],
            'feature_config': self.configs['feature_config'],
            'data_config': self.configs['data_config']
        }
        
        for config_name, config_data in config_files.items():
            config_file = config_dir / f"{config_name}.json"
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
        
        logger.info(f"Created configuration files in {config_dir}")

def create_mock_patches():
    """Create mock patches for external dependencies."""
    patches = []
    
    # Mock external data processing functions
    patches.append(patch('scr.data_utils.glacier_mapper_features', side_effect=lambda df, static: df))
    patches.append(patch.object(type(None), '__preprocess_data__', return_value=None))
    
    return patches

def run_with_timeout(func, timeout_seconds: int):
    """
    Run a function with timeout.
    
    Args:
        func: Function to run
        timeout_seconds: Timeout in seconds
        
    Returns:
        Function result or raises TimeoutError
    """
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = func()
        signal.alarm(0)  # Cancel timeout
        return result
    except TimeoutError:
        signal.alarm(0)  # Cancel timeout
        raise
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        raise e

def calculate_comprehensive_metrics(predictions: pd.DataFrame, pred_col: str, obs_col: str = 'Q_obs') -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        predictions: DataFrame with predictions and observations
        pred_col: Name of prediction column
        obs_col: Name of observation column
        
    Returns:
        Dictionary of performance metrics
    """
    try:
        # Filter valid data
        valid_data = predictions[[pred_col, obs_col]].dropna()
        
        if len(valid_data) < TEST_CONSTANTS["min_predictions_for_metrics"]:
            logger.warning(f"Not enough valid data points for metrics: {len(valid_data)}")
            return {}
        
        pred = valid_data[pred_col].values
        obs = valid_data[obs_col].values
        
        # Calculate metrics
        metrics = {}
        
        # R²
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # RMSE
        metrics['rmse'] = np.sqrt(np.mean((obs - pred) ** 2))
        
        # MAE
        metrics['mae'] = np.mean(np.abs(obs - pred))
        
        # Bias
        metrics['bias'] = np.mean(pred - obs)
        
        # Additional metrics
        metrics['n_predictions'] = len(valid_data)
        metrics['pred_mean'] = np.mean(pred)
        metrics['pred_std'] = np.std(pred)
        metrics['obs_mean'] = np.mean(obs)
        metrics['obs_std'] = np.std(obs)
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {}

def format_test_results(test_name: str, success: bool, metrics: Dict[str, float] = None, 
                       execution_time: float = None) -> Dict[str, Any]:
    """
    Format test results for reporting.
    
    Args:
        test_name: Name of the test
        success: Whether test passed
        metrics: Performance metrics (optional)
        execution_time: Test execution time (optional)
        
    Returns:
        Formatted test results
    """
    results = {
        'test_name': test_name,
        'success': success,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    
    if metrics:
        results['metrics'] = metrics
    
    if execution_time:
        results['execution_time'] = execution_time
    
    return results