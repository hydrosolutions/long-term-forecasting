#!/usr/bin/env python3
"""
Simplified functionality tests for Meta-Learning models.

This script tests the basic functionality of the meta-learning framework
by testing the public interface and basic configuration.
"""

import os
import sys
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
import logging
import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Conditional pytest import
try:
    import pytest

    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False
    pytest = None

    class MockPytest:
        def fixture(self, *args, **kwargs):
            def decorator(func):
                return func

            return decorator

    if not PYTEST_AVAILABLE:
        pytest = MockPytest()

# Add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import the modules to test
from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)


class TestHistoricalMetaLearnerSimple:
    """Simple functionality tests for HistoricalMetaLearner."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create 100 days of data for multiple basins
        dates = pd.date_range("2020-01-01", "2022-12-31", freq="D")
        np.random.seed(42)

        data = []
        for basin_code in [1, 2, 3]:
            for date in dates:
                # Create synthetic discharge data
                day_of_year = date.timetuple().tm_yday
                seasonal_factor = 1 + 0.5 * np.sin(2 * np.pi * day_of_year / 365)
                base_discharge = {1: 50, 2: 75, 3: 100}[basin_code]
                discharge = base_discharge * seasonal_factor + np.random.normal(0, 10)
                discharge = max(5, discharge)

                data.append(
                    {
                        "date": date,
                        "code": basin_code,
                        "Q_obs": discharge,
                    }
                )

        return pd.DataFrame(data)

    @pytest.fixture
    def sample_static_data(self):
        """Create sample static data for testing."""
        return pd.DataFrame(
            {
                "code": [1, 2, 3],
                "area": [100, 200, 300],
                "elevation": [1000, 1500, 2000],
            }
        )

    @pytest.fixture
    def sample_configs(self, temp_dir):
        """Create sample configuration dictionaries."""
        general_config = {
            "model_name": "TestHistoricalMetaLearner",
            "prediction_horizon": 30,
            "offset": 30,
            "target_column": "Q_obs",
        }

        model_config = {
            "num_samples_val": 10,
            "metric": "nmse",
        }

        feature_config = {
            "discharge": [{"operation": "mean", "windows": [15, 30], "lags": {}}],
            "features": ["discharge"],
            "target": "Q_obs",
        }

        path_config = {
            "model_home_path": temp_dir,
            "path_to_base_predictors": [],
        }

        return general_config, model_config, feature_config, path_config

    @pytest.fixture
    def meta_learner(self, sample_data, sample_static_data, sample_configs):
        """Create a HistoricalMetaLearner instance for testing."""
        general_config, model_config, feature_config, path_config = sample_configs
        return HistoricalMetaLearner(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

    def test_initialization(self, meta_learner):
        """Test that HistoricalMetaLearner initializes correctly."""
        logger.info("Testing HistoricalMetaLearner initialization")

        # Check that the model is initialized with correct parameters
        assert meta_learner.num_samples_val == 10
        assert meta_learner.metric == "nmse"
        assert meta_learner.invert_metric is True
        assert hasattr(meta_learner, "data")
        assert hasattr(meta_learner, "static_data")

        logger.info("✓ HistoricalMetaLearner initialization successful")

    def test_configuration_validation(self, meta_learner):
        """Test configuration validation."""
        logger.info("Testing configuration validation")

        # Test that configuration is valid
        assert meta_learner.metric in ["nmse", "r2", "nrmse", "nmae"]
        assert meta_learner.num_samples_val >= 5
        assert meta_learner.general_config is not None
        assert meta_learner.model_config is not None
        assert meta_learner.feature_config is not None
        assert meta_learner.path_config is not None

        logger.info("✓ Configuration validation successful")

    def test_public_interface(self, meta_learner):
        """Test that required public methods are available."""
        logger.info("Testing public interface")

        # Test that the public interface methods exist
        assert hasattr(meta_learner, "calibrate_model_and_hindcast")
        assert hasattr(meta_learner, "predict_operational")
        assert hasattr(meta_learner, "tune_hyperparameters")
        assert hasattr(meta_learner, "save_model")
        assert hasattr(meta_learner, "load_model")

        # Test that methods are callable
        assert callable(meta_learner.calibrate_model_and_hindcast)
        assert callable(meta_learner.predict_operational)
        assert callable(meta_learner.tune_hyperparameters)
        assert callable(meta_learner.save_model)
        assert callable(meta_learner.load_model)

        logger.info("✓ Public interface validation successful")

    def test_model_save_load(self, meta_learner):
        """Test model saving and loading functionality."""
        logger.info("Testing model save and load")

        # Test model saving
        meta_learner.save_model()

        # Verify save directory and files were created
        save_dir = os.path.join(
            meta_learner.path_config["model_home_path"], meta_learner.name
        )
        assert os.path.exists(save_dir)
        assert os.path.exists(os.path.join(save_dir, "metadata.json"))
        assert os.path.exists(os.path.join(save_dir, "config.pkl"))

        # Test model loading
        meta_learner.load_model()

        logger.info("✓ Model save and load successful")

    def test_different_metrics(self, sample_data, sample_static_data, sample_configs):
        """Test meta-learner with different metric configurations."""
        logger.info("Testing different metrics configuration")

        metrics_to_test = ["nmse", "r2", "nrmse", "nmae"]

        for metric in metrics_to_test:
            logger.info(f"Testing with metric: {metric}")

            # Update configuration
            general_config, model_config, feature_config, path_config = sample_configs
            model_config["metric"] = metric

            # Create meta-learner
            meta_learner = HistoricalMetaLearner(
                data=sample_data,
                static_data=sample_static_data,
                general_config=general_config,
                model_config=model_config,
                feature_config=feature_config,
                path_config=path_config,
            )

            # Test basic functionality
            assert meta_learner.metric == metric

            # Check invert_metric logic
            if metric in ["nmse", "nmae", "nrmse"]:
                assert meta_learner.invert_metric is True
            else:
                assert meta_learner.invert_metric is False

            logger.info(f"✓ Metric {metric} configuration successful")

    def test_data_access(self, meta_learner):
        """Test that meta-learner can access its data properly."""
        logger.info("Testing data access")

        # Test data access
        assert meta_learner.data is not None
        assert isinstance(meta_learner.data, pd.DataFrame)
        assert len(meta_learner.data) > 0

        # Test static data access
        assert meta_learner.static_data is not None
        assert isinstance(meta_learner.static_data, pd.DataFrame)
        assert len(meta_learner.static_data) > 0

        # Test that required columns exist
        assert "date" in meta_learner.data.columns
        assert "code" in meta_learner.data.columns
        assert "Q_obs" in meta_learner.data.columns

        assert "code" in meta_learner.static_data.columns

        logger.info("✓ Data access validation successful")

    def test_error_handling(self, sample_data, sample_static_data, sample_configs):
        """Test error handling for invalid configurations."""
        logger.info("Testing error handling")

        # Test invalid metric
        general_config, model_config, feature_config, path_config = sample_configs
        model_config["metric"] = "invalid_metric"

        with pytest.raises(
            ValueError, match="Metric 'invalid_metric' is not supported"
        ):
            HistoricalMetaLearner(
                data=sample_data,
                static_data=sample_static_data,
                general_config=general_config,
                model_config=model_config,
                feature_config=feature_config,
                path_config=path_config,
            )

        logger.info("✓ Error handling validation successful")


def main():
    """Run all tests when script is executed directly."""
    logger.info("Starting HistoricalMetaLearner simplified functionality tests")

    try:
        # Create test instance
        test_instance = TestHistoricalMetaLearnerSimple()

        # Create fixtures
        with tempfile.TemporaryDirectory() as temp_dir:
            sample_data = test_instance.sample_data()
            sample_static_data = test_instance.sample_static_data()
            sample_configs = test_instance.sample_configs(temp_dir)
            meta_learner = test_instance.meta_learner(
                sample_data, sample_static_data, sample_configs
            )

            # Run basic tests
            test_methods = [
                test_instance.test_initialization,
                test_instance.test_configuration_validation,
                test_instance.test_public_interface,
                test_instance.test_model_save_load,
                test_instance.test_data_access,
            ]

            for test_method in test_methods:
                try:
                    test_method(meta_learner)
                except Exception as e:
                    logger.error(f"Test {test_method.__name__} failed: {str(e)}")
                    raise

            # Test different metrics
            test_instance.test_different_metrics(
                sample_data, sample_static_data, sample_configs
            )

            # Test error handling
            test_instance.test_error_handling(
                sample_data, sample_static_data, sample_configs
            )

        logger.info(
            "✓ All HistoricalMetaLearner simplified functionality tests passed!"
        )

    except Exception as e:
        logger.error(f"✗ HistoricalMetaLearner functionality tests failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
