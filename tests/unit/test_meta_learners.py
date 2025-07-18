"""
Simplified unit tests for meta-learning models.

This module contains basic unit tests for the HistoricalMetaLearner
that focus on the public interface and basic functionality.
"""

import pytest
import pandas as pd
import numpy as np
import datetime
import tempfile
import os

from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)


class TestHistoricalMetaLearnerBasic:
    """Basic tests for HistoricalMetaLearner class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        data = pd.DataFrame(
            {
                "date": dates,
                "code": np.random.choice([1, 2], size=50),
                "Q_obs": np.random.uniform(10, 100, size=50),
            }
        )
        return data

    @pytest.fixture
    def sample_static_data(self):
        """Create sample static data for testing."""
        return pd.DataFrame(
            {
                "code": [1, 2],
                "area": [100, 200],
                "elevation": [1000, 1500],
            }
        )

    @pytest.fixture
    def sample_configs(self):
        """Create sample configuration dictionaries."""
        general_config = {
            "model_name": "test_historical_meta_learner",
            "target_column": "Q_obs",
        }
        model_config = {
            "num_samples_val": 10,
            "metric": "nmse",
        }
        feature_config = {
            "features": ["Q_obs"],
            "target": "Q_obs",
        }
        path_config = {
            "model_home_path": "/tmp/test_models",
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

    def test_initialization_valid_config(self, meta_learner):
        """Test successful initialization with valid configuration."""
        assert meta_learner.num_samples_val == 10
        assert meta_learner.metric == "nmse"
        assert meta_learner.invert_metric is True

    def test_initialization_invalid_metric(self, sample_data, sample_static_data, sample_configs):
        """Test initialization with invalid metric raises error."""
        general_config, model_config, feature_config, path_config = sample_configs
        model_config["metric"] = "invalid_metric"
        
        with pytest.raises(ValueError, match="Metric 'invalid_metric' is not supported"):
            HistoricalMetaLearner(
                data=sample_data,
                static_data=sample_static_data,
                general_config=general_config,
                model_config=model_config,
                feature_config=feature_config,
                path_config=path_config,
            )

    def test_initialization_default_values(self, sample_data, sample_static_data, sample_configs):
        """Test initialization with default values."""
        general_config, model_config, feature_config, path_config = sample_configs
        # Remove optional parameters to test defaults
        del model_config["num_samples_val"]
        del model_config["metric"]
        
        learner = HistoricalMetaLearner(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
        
        assert learner.num_samples_val == 10  # default value
        assert learner.metric == "nmse"  # default value

    def test_invert_metric_logic(self, sample_data, sample_static_data, sample_configs):
        """Test that invert_metric is set correctly for different metrics."""
        general_config, model_config, feature_config, path_config = sample_configs
        
        # Test error metric (should be inverted)
        model_config["metric"] = "nmse"
        learner = HistoricalMetaLearner(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
        assert learner.invert_metric is True
        
        # Test accuracy metric (should not be inverted)
        model_config["metric"] = "r2"
        learner = HistoricalMetaLearner(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
        assert learner.invert_metric is False

    def test_available_methods(self, meta_learner):
        """Test that required methods are available."""
        # Test that the public interface methods exist
        assert hasattr(meta_learner, 'calibrate_model_and_hindcast')
        assert hasattr(meta_learner, 'predict_operational')
        assert hasattr(meta_learner, 'tune_hyperparameters')
        assert hasattr(meta_learner, 'save_model')
        assert hasattr(meta_learner, 'load_model')

    def test_configuration_properties(self, meta_learner):
        """Test configuration properties."""
        assert meta_learner.num_samples_val == 10
        assert meta_learner.metric == "nmse"
        assert meta_learner.invert_metric is True
        assert hasattr(meta_learner, 'data')
        assert hasattr(meta_learner, 'static_data')
        assert hasattr(meta_learner, 'general_config')
        assert hasattr(meta_learner, 'model_config')
        assert hasattr(meta_learner, 'feature_config')
        assert hasattr(meta_learner, 'path_config')

    def test_model_save_load_basic(self, meta_learner):
        """Test basic model save and load functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update path config to use temp directory
            meta_learner.path_config["model_home_path"] = temp_dir
            meta_learner.name = "test_model"
            
            # Save model
            meta_learner.save_model()
            
            # Verify files were created
            model_dir = os.path.join(temp_dir, "test_model")
            assert os.path.exists(model_dir)
            assert os.path.exists(os.path.join(model_dir, "metadata.json"))
            assert os.path.exists(os.path.join(model_dir, "config.pkl"))
            
            # Load model
            meta_learner.load_model()

    def test_metric_options(self, sample_data, sample_static_data, sample_configs):
        """Test different metric options."""
        general_config, model_config, feature_config, path_config = sample_configs
        
        metrics_to_test = ["nmse", "r2", "nrmse", "nmae"]
        
        for metric in metrics_to_test:
            model_config["metric"] = metric
            learner = HistoricalMetaLearner(
                data=sample_data,
                static_data=sample_static_data,
                general_config=general_config,
                model_config=model_config,
                feature_config=feature_config,
                path_config=path_config,
            )
            assert learner.metric == metric
            
            # Check invert_metric logic
            if metric in ["nmse", "nmae", "nrmse"]:
                assert learner.invert_metric is True
            else:
                assert learner.invert_metric is False