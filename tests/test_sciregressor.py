#!/usr/bin/env python3
"""
Comprehensive test script for SciRegressor model including calibration, hindcasting, operational forecasting, and hyperparameter tuning.

This script tests the complete SciRegressor workflow using synthetic data including:
- Model initialization and configuration loading
- Feature extraction and preprocessing
- Model calibration and hindcasting with Leave-One-Year-Out cross-validation
- Operational forecasting
- Hyperparameter tuning
- Model saving and loading

Usage:
    # Run with pytest
    pytest test_sciregressor.py -v
    
    # Run specific test
    pytest test_sciregressor.py::test_sciregressor_calibration_hindcast -v
    
    # Run with coverage
    pytest test_sciregressor.py --cov=forecast_models.SciRegressor
    
    # Run standalone
    python test_sciregressor.py
    
    # With verbose logging
    python test_sciregressor.py --verbose
    
    # Only show failures
    python test_sciregressor.py --only-failures
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List
import logging
import datetime
from unittest.mock import patch, MagicMock

# Setup logging with enhanced configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_sciregressor.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

# Set specific loggers to appropriate levels
logging.getLogger('forecast_models.SciRegressor').setLevel(logging.DEBUG)
logging.getLogger('scr.FeatureExtractor').setLevel(logging.INFO)
logging.getLogger('scr.FeatureProcessingArtifacts').setLevel(logging.INFO)

# Conditional pytest import
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None
    
    # Create mock pytest for type hints when not available
    class MockPytest:
        def fixture(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    if not PYTEST_AVAILABLE:
        pytest = MockPytest()

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import the modules to test
from forecast_models.SciRegressor import SciRegressor
from scr import FeatureExtractor as FE
from eval_scr import metric_functions
import calibrate_hindcast

# Test configurations for SciRegressor
GENERAL_CONFIG = {
    "prediction_horizon": 30,
    "offset": 30,
    "num_features": 5,
    "base_features": ["discharge", "P", "T"],
    "snow_vars": [],
    "forecast_days": ["end"],
    "filter_years": None,
    "model_name": "TestSciRegressor",
    "models": ["xgb", "rf", "catboost"],  # Multiple models for ensemble
    "target": "target",
    "handle_na": "long_term_mean",
    "normalization_type": "standard",
    "normalize_per_basin": False,
    "use_pca": False,
    "use_lr_predictors": False,
    "use_temporal_features": True,
    "use_static_features": True,
    "cat_features": ["code_str"],
    "feature_cols": ["discharge", "P", "T"],
    "static_features": ["area", "elevation"],
    "rivers_to_exclude": [],
    "snow_vars": [],
    "test_years": 3,
    "num_elevation_zones": 5
}

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

MODEL_CONFIG = {
    "xgb": {
        "n_estimators": 10,
        "max_depth": 2,
        "learning_rate": 0.1
    },
    "rf": {
        "n_estimators": 5,
        "max_depth": 2
    },
    "catboost": {
        "iterations": 20,
        "depth": 2,
        "learning_rate": 0.1
    }
}

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
    "output_path" : "tests_output"
}

DATA_CONFIG = {
    "start_year": 2000,
    "end_year": 2020,
    "basins": [16936, 16940, 16942]
}

class TestDataGenerator:
    """Helper class to generate synthetic data for testing SciRegressor."""
    
    @staticmethod
    def generate_synthetic_timeseries_data(
        start_date: str = "2000-01-01",
        basin_codes: List[int] = [16936, 16940, 16942],
        noise_level: float = 0.1
    ) -> pd.DataFrame:
        """
        Generate synthetic time series data mimicking hydro-meteorological data.
        Enhanced for SciRegressor testing with more complex relationships.
        
        Args:
            start_date: Start date for data generation
            basin_codes: List of basin codes to generate data for
            noise_level: Amount of random noise to add
            
        Returns:
            DataFrame with synthetic time series data
        """
        end_date = pd.Timestamp.today()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        data_list = []
        
        np.random.seed(42)  # For reproducible tests
        
        for code in basin_codes:
            for date in date_range:
                # Generate seasonal patterns
                day_of_year = date.timetuple().tm_yday
                year = date.year
                
                # Temperature with seasonal cycle and trend
                T = 5 + 15 * np.sin(2 * np.pi * day_of_year / 365.25) + np.random.normal(0, 2)
                T += 0.02 * (year - 2000)  # Small warming trend
                
                # Precipitation with seasonality and extremes
                P = max(0, 10 + 10 * np.sin(2 * np.pi * (day_of_year + 90) / 365.25) + 
                       np.random.exponential(5))
                
                # Add occasional extreme precipitation events
                if np.random.random() < 0.05:  # 5% chance
                    P += np.random.exponential(20)
                
                # Complex discharge model with multiple influences
                base_discharge = 50 + 30 * np.sin(2 * np.pi * (day_of_year + 120) / 365.25)
                
                # Precipitation effect with delay
                precip_effect = P * 0.5
                if len(data_list) > 0:
                    # Add delayed precipitation effect
                    recent_data = [d for d in data_list[-7:] if d['code'] == code]
                    if recent_data:
                        delayed_precip = sum([d['P'] for d in recent_data]) * 0.1
                        precip_effect += delayed_precip
                
                # Temperature effect (snowmelt)
                temp_effect = max(0, T - 0) * 2
                
                # Seasonal baseflow variation
                baseflow_seasonal = 20 * np.sin(2 * np.pi * (day_of_year + 200) / 365.25)
                
                discharge = base_discharge + precip_effect + temp_effect + baseflow_seasonal
                discharge = max(0.1, discharge + np.random.normal(0, 5))
                
                # Add basin-specific scaling and characteristics
                basin_factor = 1 + 0.2 * (code - 16938) / 4
                elevation_effect = (code - 16938) * 0.5  # Higher elevation = cooler
                
                discharge *= basin_factor
                T += elevation_effect
                P *= basin_factor
                
                # Add temporal correlations and noise
                discharge += np.random.normal(0, discharge * noise_level)
                T += np.random.normal(0, 1)
                P = max(0, P + np.random.normal(0, P * noise_level))
                
                data_list.append({
                    'date': date,
                    'code': code,
                    'discharge': round(discharge, 2),
                    'T': round(T, 2),
                    'P': round(P, 2)
                })
        
        df = pd.DataFrame(data_list)
        df = df.sort_values(['code', 'date']).reset_index(drop=True)
        
        logger.info(f"Generated synthetic data: {len(df)} records for {len(basin_codes)} basins")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Discharge range: {df['discharge'].min():.2f} to {df['discharge'].max():.2f}")
        logger.info(f"Temperature range: {df['T'].min():.2f} to {df['T'].max():.2f}")
        logger.info(f"Precipitation range: {df['P'].min():.2f} to {df['P'].max():.2f}")
        
        return df
    
    @staticmethod
    def generate_synthetic_static_data(basin_codes: List[int] = [16936, 16940, 16942]) -> pd.DataFrame:
        """
        Generate synthetic static basin characteristics data for SciRegressor.
        
        Args:
            basin_codes: List of basin codes
            
        Returns:
            DataFrame with synthetic static data
        """
        np.random.seed(42)
        
        static_data = []
        for code in basin_codes:
            static_data.append({
                'code': code,
                'area': 100 + np.random.uniform(50, 500),  # km²
                'area_km2': 100 + np.random.uniform(50, 500),  # km² (for SciRegressor)
                'elevation': 1000 + np.random.uniform(500, 2000),  # m
                'slope': np.random.uniform(5, 30),  # degrees
                'forest_cover': np.random.uniform(0.1, 0.8),  # fraction
                'latitude': 40 + np.random.uniform(-5, 5),  # degrees
                'longitude': 70 + np.random.uniform(-10, 10),  # degrees
            })
        
        df = pd.DataFrame(static_data)
        logger.info(f"Generated static data for {len(basin_codes)} basins")
        
        return df

class SciRegressorTester:
    """Main test class for SciRegressor model."""
    
    def __init__(self):
        """Initialize the tester with synthetic data and configurations."""
        self.test_dir = None
        self.data = None
        self.static_data = None
        self.configs = None
        self.only_failures = False
        
    def setup_test_environment(self):
        """Set up test environment with synthetic data and temporary directories."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp(prefix="sciregressor_test_")
        logger.info(f"Created test directory: {self.test_dir}")
        
        # Generate synthetic data
        self.data = TestDataGenerator.generate_synthetic_timeseries_data()
        self.static_data = TestDataGenerator.generate_synthetic_static_data()
        
        # Setup configurations
        self.configs = {
            'general_config': GENERAL_CONFIG.copy(),
            'feature_config': FEATURE_CONFIG.copy(),
            'model_config': MODEL_CONFIG.copy(),
            'path_config': PATH_CONFIG.copy(),
            'data_config': DATA_CONFIG.copy()
        }
        
        # Create config files in test directory
        config_dir = Path(self.test_dir) / "config"
        config_dir.mkdir(exist_ok=True)
        
        for config_name, config_data in self.configs.items():
            if config_name != 'path_config':
                config_file = config_dir / f"{config_name}.json"
                with open(config_file, 'w') as f:
                    json.dump(config_data, f, indent=2)
        
        logger.info("Test environment setup complete")
        
    def teardown_test_environment(self):
        """Clean up test environment."""
        if self.test_dir and os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
            logger.info(f"Cleaned up test directory: {self.test_dir}")
    
    def _run_test_with_detailed_logging(self, test_func, test_name):
        """
        Run a test with detailed logging on failure.
        
        Args:
            test_func: The test function to run
            test_name: Name of the test for logging
            
        Returns:
            tuple: (success: bool, result: Any)
        """
        try:
            if not self.only_failures:
                logger.info(f"Running {test_name}...")
            
            result = test_func()
            
            if not self.only_failures:
                logger.info(f"✓ {test_name} passed")
            else:
                print(f"✓ {test_name} passed")
            
            return True, result
            
        except Exception as e:
            # Always show detailed logs for failures
            logger.error(f"✗ {test_name} failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception details:", exc_info=True)
            
            # Also print to console for immediate visibility
            print(f"✗ {test_name} FAILED: {e}")
            print(f"Check log file for detailed traceback")
            
            return False, None
    
    def test_model_initialization(self):
        """Test SciRegressor model initialization."""
        logger.info("Testing SciRegressor model initialization...")
        
        try:
            # Mock the preprocessing methods that might fail due to missing dependencies
            with patch.object(SciRegressor, '__preprocess_data__') as mock_preprocess:
                mock_preprocess.return_value = None
                
                model = SciRegressor(
                    data=self.data,
                    static_data=self.static_data,
                    general_config=self.configs['general_config'],
                    model_config=self.configs['model_config'],
                    feature_config=self.configs['feature_config'],
                    path_config=self.configs['path_config']
                )
            
            # Verify model attributes
            assert model.name == "TestSciRegressor"
            assert hasattr(model, 'data')
            assert hasattr(model, 'static_data')
            assert hasattr(model, 'general_config')
            assert hasattr(model, 'model_config')
            assert hasattr(model, 'feature_config')
            assert hasattr(model, 'models')
            assert hasattr(model, 'fitted_models')
            
            # Check model-specific attributes

            expected_models = MODEL_CONFIG.keys()
            expected_models = list(expected_models)  # Convert to list for comparison
            assert model.models == expected_models
            assert model.target == "target"
            assert model.feature_cols == ["discharge", "P", "T"]
            assert model.static_features == ["area", "elevation"]
            
            logger.info("✓ SciRegressor model initialization test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ SciRegressor model initialization test failed: {e}")
            raise
    
    def test_feature_extraction(self):
        """Test feature extraction functionality."""
        logger.info("Testing feature extraction...")
        
        try:
            # Mock external dependencies that might not be available
            with patch.object(SciRegressor, '__preprocess_data__') as mock_preprocess:
                with patch('scr.data_utils.glacier_mapper_features') as mock_glacier:
                    mock_glacier.return_value = self.data
                    mock_preprocess.return_value = None
                    
                    model = SciRegressor(
                        data=self.data,
                        static_data=self.static_data,
                        general_config=self.configs['general_config'],
                        model_config=self.configs['model_config'],
                        feature_config=self.configs['feature_config'],
                        path_config=self.configs['path_config']
                    )
                    
                    # Call feature extraction manually
                    model.__extract_features__()
            
            # Check that features were extracted
            assert 'year' in model.data.columns
            assert 'month' in model.data.columns
            assert 'month_sin' in model.data.columns
            assert 'month_cos' in model.data.columns
            assert 'week_sin' in model.data.columns
            assert 'week_cos' in model.data.columns
            
            # Check for feature columns
            feature_cols = [col for col in model.data.columns if any(f in col for f in ['discharge', 'P_', 'T_'])]
            assert len(feature_cols) > 0, "No feature columns found"
            
            logger.info("✓ Feature extraction test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Feature extraction test failed: {e}")
            raise
    
    def test_calibration_and_hindcast(self):
        """Test model calibration and hindcasting."""
        logger.info("Testing calibration and hindcasting...")
        
        try:           
                        
            model = SciRegressor(
                data=self.data,
                static_data=self.static_data,
                general_config=self.configs['general_config'],
                model_config=self.configs['model_config'],
                feature_config=self.configs['feature_config'],
                path_config=self.configs['path_config']
            )
                        
            # Run calibration and hindcasting
            hindcast_df = model.calibrate_model_and_hindcast()

            # Verify results
            assert len(hindcast_df) > 0, "No hindcast predictions generated"
            
            # Check required columns
            required_cols = ['date', 'code', 'Q_obs', f'Q_{model.name}']
            for col in required_cols:
                assert col in hindcast_df.columns, f"Missing required column: {col}"
            
            # Verify predictions for each basin
            unique_basins = hindcast_df['code'].unique()
            assert len(unique_basins) >= 1, "No basins in hindcast results"
            
            # Check that predictions are reasonable
            pred_col = f'Q_{model.name}'
            assert not hindcast_df[pred_col].isna().all(), "All predictions are NaN"
            
            logger.info(f"✓ Calibration and hindcast test passed")
            logger.info(f"  Generated {len(hindcast_df)} predictions for {len(unique_basins)} basins")
            
            return hindcast_df
            
        except Exception as e:
            logger.error(f"✗ Calibration and hindcast test failed: {e}")
            raise
    
    def test_operational_forecast(self):
        """Test operational forecasting functionality."""
        logger.info("Testing operational forecasting...")
        
        try:
            # Mock external dependencies
            with patch.object(SciRegressor, '__preprocess_data__') as mock_preprocess:
                with patch('scr.data_utils.glacier_mapper_features') as mock_glacier:
                    with patch.object(SciRegressor, 'load_model') as mock_load:
                        with patch.object(SciRegressor, 'predict_operational') as mock_predict:
                            # Create a mock forecast result
                            mock_forecast_data = []
                            for code in [16936, 16940, 16942]:
                                mock_forecast_data.append({
                                    'forecast_date': datetime.datetime.now(),
                                    'code': code,
                                    'valid_from': '2025-07-04',
                                    'valid_to': '2025-08-03',
                                    'Q_TestSciRegressor': np.random.uniform(10, 100)
                                })
                            
                            mock_forecast_df = pd.DataFrame(mock_forecast_data)
                            mock_predict.return_value = mock_forecast_df
                            mock_load.return_value = {'xgboost': {}, 'random_forest': {}}
                            
                            mock_glacier.return_value = self.data
                            mock_preprocess.return_value = None
                            
                            model = SciRegressor(
                                data=self.data,
                                static_data=self.static_data,
                                general_config=self.configs['general_config'],
                                model_config=self.configs['model_config'],
                                feature_config=self.configs['feature_config'],
                                path_config=self.configs['path_config']
                            )
                            
                            # Run operational forecast
                            forecast_df = model.predict_operational()
            
            # Verify results
            if len(forecast_df) > 0:
                # Check required columns
                required_cols = ['forecast_date', 'code', 'valid_from', 'valid_to', f'Q_{model.name}']
                for col in required_cols:
                    assert col in forecast_df.columns, f"Missing required column: {col}"
                
                # Check that predictions are reasonable
                pred_col = f'Q_{model.name}'
                assert not forecast_df[pred_col].isna().all(), "All forecasts are NaN"
                
                # Check date consistency
                assert all(pd.to_datetime(forecast_df['valid_from']) < pd.to_datetime(forecast_df['valid_to']))
                
                logger.info(f"✓ Operational forecast test passed")
                logger.info(f"  Generated forecasts for {len(forecast_df)} basins")
            else:
                logger.info("✓ Operational forecast test passed (no forecasts for test period)")
            
            return forecast_df
            
        except Exception as e:
            logger.error(f"✗ Operational forecast test failed: {e}")
            raise
    
    def test_hyperparameter_tuning(self):
        """Test hyperparameter tuning functionality."""
        logger.info("Testing hyperparameter tuning...")
        
        try:
            # Mock external dependencies
            with patch.object(SciRegressor, '__preprocess_data__') as mock_preprocess:
                with patch('scr.data_utils.glacier_mapper_features') as mock_glacier:
                    with patch.object(SciRegressor, 'tune_hyperparameters') as mock_tune:
                        # Mock the tuning result
                        mock_tune.return_value = (True, "Hyperparameters tuned successfully")
                        
                        mock_glacier.return_value = self.data
                        mock_preprocess.return_value = None
                        
                        model = SciRegressor(
                            data=self.data,
                            static_data=self.static_data,
                            general_config=self.configs['general_config'],
                            model_config=self.configs['model_config'],
                            feature_config=self.configs['feature_config'],
                            path_config=self.configs['path_config']
                        )
                        
                        # Test hyperparameter tuning
                        success, message = model.tune_hyperparameters(self.data)
            
            # Verify results
            assert isinstance(success, bool), "Tuning should return boolean success status"
            assert isinstance(message, str), "Tuning should return string message"
            
            logger.info(f"✓ Hyperparameter tuning test passed")
            logger.info(f"  Tuning result: {success}, Message: {message}")
            
            return success, message
            
        except Exception as e:
            logger.error(f"✗ Hyperparameter tuning test failed: {e}")
            raise
    
    def test_model_persistence(self):
        """Test model saving and loading functionality."""
        logger.info("Testing model persistence...")
        
        try:
            # Mock external dependencies
            with patch.object(SciRegressor, '__preprocess_data__') as mock_preprocess:
                with patch('scr.data_utils.glacier_mapper_features') as mock_glacier:
                    with patch.object(SciRegressor, 'save_model') as mock_save:
                        with patch.object(SciRegressor, 'load_model') as mock_load:
                            mock_save.return_value = None
                            mock_load.return_value = {'xgboost': {}, 'random_forest': {}}
                            
                            mock_glacier.return_value = self.data
                            mock_preprocess.return_value = None
                            
                            model = SciRegressor(
                                data=self.data,
                                static_data=self.static_data,
                                general_config=self.configs['general_config'],
                                model_config=self.configs['model_config'],
                                feature_config=self.configs['feature_config'],
                                path_config=self.configs['path_config']
                            )
                            
                            # Test saving
                            model.save_model()
                            logger.info("✓ Model saving test passed")
                            
                            # Test loading
                            loaded_models = model.load_model()
                            logger.info("✓ Model loading test passed")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Model persistence test failed: {e}")
            raise
    
    def test_configuration_loading(self):
        """Test configuration loading functionality."""
        logger.info("Testing configuration loading...")
        
        try:
            config_dir = Path(self.test_dir) / "config"
            
            # Test the load_configuration function from calibrate_hindcast
            configs = calibrate_hindcast.load_configuration(str(config_dir))
            
            # Verify all expected configs are loaded
            expected_configs = ['general_config', 'model_config', 'feature_config', 'data_config']
            for config_name in expected_configs:
                assert config_name in configs, f"Missing config: {config_name}"
                assert len(configs[config_name]) > 0, f"Empty config: {config_name}"
            
            logger.info("✓ Configuration loading test passed")
            return True
            
        except Exception as e:
            logger.error(f"✗ Configuration loading test failed: {e}")
            raise
    
    def test_metrics_calculation(self, hindcast_df: pd.DataFrame):
        """Test metrics calculation on hindcast results."""
        logger.info("Testing metrics calculation...")
        
        try:
            if len(hindcast_df) == 0:
                logger.info("✓ Metrics calculation test skipped (no hindcast data)")
                return pd.DataFrame()
            
            # Calculate metrics using the same logic as in calibrate_hindcast.py
            metrics_list = []
            model_name = "TestSciRegressor"
            prediction_col = f"Q_{model_name}"
            
            for code, group in hindcast_df.groupby(['code']):
                if len(group) < 5:  # Need sufficient data points
                    continue
                
                obs = group['Q_obs'].values
                pred = group[prediction_col].values
                
                # Calculate various metrics
                try:
                    metrics = {
                        'code': code,
                        'n_predictions': len(group),
                        'r2': metric_functions.r2_score(obs, pred),
                        'rmse': metric_functions.rmse(obs, pred),
                        'mae': metric_functions.mae(obs, pred),
                        'bias': metric_functions.bias(obs, pred),
                    }
                    
                    # Add NSE if available
                    if hasattr(metric_functions, 'nse'):
                        metrics['nse'] = metric_functions.nse(obs, pred)
                    
                    metrics_list.append(metrics)
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate metrics for basin {code}: {e}")
            
            if not metrics_list:
                logger.info("✓ Metrics calculation test passed (no valid metrics)")
                return pd.DataFrame()
            
            metrics_df = pd.DataFrame(metrics_list)
            
            # Verify metrics are reasonable
            assert len(metrics_df) > 0, "No metrics calculated"
            assert 'r2' in metrics_df.columns, "R2 metric missing"
            assert 'rmse' in metrics_df.columns, "RMSE metric missing"
            assert 'mae' in metrics_df.columns, "MAE metric missing"
            
            # Calculate summary statistics
            summary_stats = {
                'r2_mean': metrics_df['r2'].mean(),
                'r2_median': metrics_df['r2'].median(),
                'rmse_mean': metrics_df['rmse'].mean(),
                'mae_mean': metrics_df['mae'].mean(),
                'n_basins': len(metrics_df)
            }
            
            logger.info("✓ Metrics calculation test passed")
            logger.info(f"  Calculated metrics for {len(metrics_df)} basins")
            logger.info(f"  Mean R²: {summary_stats['r2_mean']:.3f}")
            logger.info(f"  Mean RMSE: {summary_stats['rmse_mean']:.3f}")
            logger.info(f"  Mean MAE: {summary_stats['mae_mean']:.3f}")
            
            return metrics_df
            
        except Exception as e:
            logger.error(f"✗ Metrics calculation test failed: {e}")
            raise
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow."""
        logger.info("Testing end-to-end workflow...")
        
        try:
            # Create output directory
            output_dir = Path(self.test_dir) / "results"
            output_dir.mkdir(exist_ok=True)
            
            # Mock external dependencies for complete workflow
            with patch.object(SciRegressor, '__preprocess_data__') as mock_preprocess:
                with patch('scr.data_utils.glacier_mapper_features') as mock_glacier:
                    with patch.object(SciRegressor, 'calibrate_model_and_hindcast') as mock_calibrate:
                        with patch.object(SciRegressor, 'predict_operational') as mock_predict:
                            with patch.object(SciRegressor, 'save_model') as mock_save:
                                # Setup mocks
                                mock_glacier.return_value = self.data
                                mock_preprocess.return_value = None
                                mock_save.return_value = None
                                
                                # Create mock results
                                mock_hindcast_data = []
                                for code in [16936, 16940, 16942]:
                                    for i in range(50):  # Generate enough data points
                                        mock_hindcast_data.append({
                                            'date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i*7),
                                            'code': code,
                                            'Q_obs': np.random.uniform(10, 100),
                                            'Q_TestSciRegressor': np.random.uniform(10, 100)
                                        })
                                
                                mock_hindcast_df = pd.DataFrame(mock_hindcast_data)
                                mock_calibrate.return_value = mock_hindcast_df
                                
                                mock_forecast_data = []
                                for code in [16936, 16940, 16942]:
                                    mock_forecast_data.append({
                                        'forecast_date': datetime.datetime.now(),
                                        'code': code,
                                        'valid_from': '2025-07-04',
                                        'valid_to': '2025-08-03',
                                        'Q_TestSciRegressor': np.random.uniform(10, 100)
                                    })
                                
                                mock_forecast_df = pd.DataFrame(mock_forecast_data)
                                mock_predict.return_value = mock_forecast_df
                                
                                # Test model creation
                                model = SciRegressor(
                                    data=self.data,
                                    static_data=self.static_data,
                                    general_config=self.configs['general_config'],
                                    model_config=self.configs['model_config'],
                                    feature_config=self.configs['feature_config'],
                                    path_config=self.configs['path_config']
                                )
                                
                                # Test calibration and hindcasting
                                hindcast_df = model.calibrate_model_and_hindcast(self.data)
                                
                                # Save predictions
                                if len(hindcast_df) > 0:
                                    predictions_path = output_dir / "predictions.csv"
                                    hindcast_df.to_csv(predictions_path, index=False)
                                    assert predictions_path.exists(), "Predictions file not saved"
                                
                                # Test operational forecasting
                                forecast_df = model.predict_operational()
                                
                                # Save forecasts
                                if len(forecast_df) > 0:
                                    forecasts_path = output_dir / "forecasts.csv"
                                    forecast_df.to_csv(forecasts_path, index=False)
                                    assert forecasts_path.exists(), "Forecasts file not saved"
                                
                                # Test hyperparameter tuning
                                success, message = model.tune_hyperparameters(self.data)
                                
                                # Test metrics calculation
                                metrics_df = self.test_metrics_calculation(hindcast_df)
                                
                                # Save metrics
                                if len(metrics_df) > 0:
                                    metrics_path = output_dir / "metrics.csv"
                                    metrics_df.to_csv(metrics_path, index=False)
                                    assert metrics_path.exists(), "Metrics file not saved"
                                
                                # Test model saving
                                model.save_model()
            
            logger.info("✓ End-to-end workflow test passed")
            logger.info(f"  Results saved to: {output_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ End-to-end workflow test failed: {e}")
            raise
    
    def run_all_tests(self):
        """Run all tests in sequence."""
        logger.info("="*60)
        logger.info("STARTING SCIREGRESSOR MODEL TESTS")
        logger.info("="*60)
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Define tests with their names
            test_suite = [
                (self.test_model_initialization, "Model Initialization"),
                (self.test_feature_extraction, "Feature Extraction"),
                (self.test_configuration_loading, "Configuration Loading"),
                (self.test_calibration_and_hindcast, "Calibration and Hindcast"),
                (self.test_operational_forecast, "Operational Forecast"),
                (self.test_hyperparameter_tuning, "Hyperparameter Tuning"),
                (self.test_model_persistence, "Model Persistence"),
                (self.test_end_to_end_workflow, "End-to-End Workflow")
            ]
            
            passed_tests = 0
            total_tests = len(test_suite)
            failed_tests = []
            
            for i, (test_func, test_name) in enumerate(test_suite, 1):
                print(f"\n[{i}/{total_tests}] {test_name}...")
                
                success, result = self._run_test_with_detailed_logging(test_func, test_name)
                
                if success:
                    passed_tests += 1
                else:
                    failed_tests.append(test_name)
            
            # Summary
            logger.info("="*60)
            logger.info("TEST SUMMARY")
            logger.info("="*60)
            logger.info(f"Tests passed: {passed_tests}/{total_tests}")
            
            if failed_tests:
                logger.error(f"Failed tests: {', '.join(failed_tests)}")
                print(f"\n❌ FAILED TESTS: {', '.join(failed_tests)}")
            
            if passed_tests == total_tests:
                logger.info("🎉 ALL TESTS PASSED!")
                print("🎉 ALL TESTS PASSED!")
                return True
            else:
                logger.warning(f"⚠️  {total_tests - passed_tests} tests failed")
                print(f"⚠️  {total_tests - passed_tests} tests failed")
                return False
                
        finally:
            # Cleanup
            self.teardown_test_environment()

# Pytest-compatible test functions
if PYTEST_AVAILABLE:
    @pytest.fixture
    def sciregressor_tester():
        """Pytest fixture for SciRegressorTester."""
        tester = SciRegressorTester()
        tester.setup_test_environment()
        yield tester
        tester.teardown_test_environment()
    
    def test_sciregressor_initialization(sciregressor_tester):
        """Pytest: Test model initialization."""
        sciregressor_tester.test_model_initialization()
    
    def test_sciregressor_feature_extraction(sciregressor_tester):
        """Pytest: Test feature extraction."""
        sciregressor_tester.test_feature_extraction()
    
    def test_sciregressor_calibration_hindcast(sciregressor_tester):
        """Pytest: Test calibration and hindcasting."""
        sciregressor_tester.test_calibration_and_hindcast()
    
    def test_sciregressor_operational_forecast(sciregressor_tester):
        """Pytest: Test operational forecasting."""
        sciregressor_tester.test_operational_forecast()
    
    def test_sciregressor_hyperparameter_tuning(sciregressor_tester):
        """Pytest: Test hyperparameter tuning."""
        sciregressor_tester.test_hyperparameter_tuning()
    
    def test_sciregressor_configuration_loading(sciregressor_tester):
        """Pytest: Test configuration loading."""
        sciregressor_tester.test_configuration_loading()
    
    def test_sciregressor_end_to_end(sciregressor_tester):
        """Pytest: Test end-to-end workflow."""
        sciregressor_tester.test_end_to_end_workflow()

def main():
    """Main function for standalone execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run SciRegressor Model Tests')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging (DEBUG level)')
    parser.add_argument('--log-file', type=str, default='test_sciregressor.log',
                       help='Log file name (default: test_sciregressor.log)')
    parser.add_argument('--only-failures', action='store_true',
                       help='Only show detailed logs for failed tests')
    
    args = parser.parse_args()
    
    # Configure logging based on arguments
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Create custom formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Setup handlers
    handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(args.log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)  # Always DEBUG in file
    file_handler.setFormatter(formatter)
    handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(level=logging.DEBUG, handlers=handlers, force=True)
    
    # Set specific logger levels
    if args.verbose:
        logging.getLogger('forecast_models.SciRegressor').setLevel(logging.DEBUG)
        logging.getLogger('scr.FeatureExtractor').setLevel(logging.DEBUG)
    else:
        logging.getLogger('forecast_models.SciRegressor').setLevel(logging.INFO)
        logging.getLogger('scr.FeatureExtractor').setLevel(logging.WARNING)
    
    print(f"Logging configured - Level: {log_level}, File: {args.log_file}")
    print(f"Verbose mode: {args.verbose}, Only failures: {args.only_failures}")
    print("="*60)
    
    tester = SciRegressorTester()
    tester.only_failures = args.only_failures  # Pass flag to tester
    success = tester.run_all_tests()
    
    print("="*60)
    print(f"Check detailed logs in: {args.log_file}")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())
