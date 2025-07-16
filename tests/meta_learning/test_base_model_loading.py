"""
Tests for base model loading functionality in meta-learning framework.
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import Mock, patch

from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)


class TestBaseModelLoading:
    """Test base model loading functionality."""

    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic data
        np.random.seed(42)

        # Create simple test data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        self.data = pd.DataFrame(
            {
                "date": dates,
                "code": [1] * 50,
                "Q": np.random.randn(50) * 10 + 50,
                "T": np.random.randn(50) * 5 + 15,
                "P": np.random.randn(50) * 20 + 100,
            }
        )

        self.static_data = pd.DataFrame(
            {"code": [1], "area": [1000], "elevation": [500]}
        )

        # Create configurations
        self.general_config = {
            "model_name": "test_base_loading_model",
            "target_column": "Q",
            "date_column": "date",
            "code_column": "code",
        }

        self.model_config = {
            "meta_learning": {
                "ensemble_method": "weighted_mean",
                "weighting_strategy": "performance_based",
                "performance_metric": "rmse",
                "basin_specific": True,
                "temporal_weighting": True,
                "min_samples_per_basin": 5,
                "weight_smoothing": 0.1,
            }
        }

        self.feature_config = {"feature_columns": ["T", "P"], "lag_features": [1, 2, 3]}

        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()

        # Create sample prediction files
        self.create_sample_prediction_files()

    def create_sample_prediction_files(self):
        """Create sample prediction files for testing."""
        # Create model directories
        model1_dir = os.path.join(self.temp_dir, "XGBoost")
        model2_dir = os.path.join(self.temp_dir, "LightGBM")
        os.makedirs(model1_dir)
        os.makedirs(model2_dir)

        # Create sample prediction data
        dates = pd.date_range("2020-01-01", periods=50, freq="D")

        # XGBoost predictions
        xgb_predictions = pd.DataFrame(
            {
                "date": dates,
                "code": [1] * 50,
                "Q_obs": np.random.randn(50) * 10 + 50,
                "Q_XGBoost": np.random.randn(50) * 10 + 50,
            }
        )
        xgb_predictions.to_csv(os.path.join(model1_dir, "predictions.csv"), index=False)

        # LightGBM predictions
        lgb_predictions = pd.DataFrame(
            {
                "date": dates,
                "code": [1] * 50,
                "Q_obs": np.random.randn(50) * 10 + 50,
                "Q_LightGBM": np.random.randn(50) * 10 + 50,
            }
        )
        lgb_predictions.to_csv(os.path.join(model2_dir, "predictions.csv"), index=False)

        # Create direct CSV file (alternative format)
        catboost_predictions = pd.DataFrame(
            {
                "date": dates,
                "code": [1] * 50,
                "Q_obs": np.random.randn(50) * 10 + 50,
                "Q_CatBoost": np.random.randn(50) * 10 + 50,
            }
        )
        catboost_predictions.to_csv(
            os.path.join(self.temp_dir, "CatBoost_predictions.csv"), index=False
        )

        # Store paths for configuration
        self.base_model_paths = [
            model1_dir,
            model2_dir,
            os.path.join(self.temp_dir, "CatBoost_predictions.csv"),
        ]

    def test_base_model_loading_from_directories(self):
        """Test loading base model predictions from directory paths."""
        path_config = {
            "model_home_path": self.temp_dir,
            "path_to_base_models": self.base_model_paths[:2],  # Only directories
        }

        # Initialize meta-learner with path configuration
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
        )

        # Verify that base model predictions were loaded
        assert len(meta_learner.base_model_predictions) == 2
        assert "XGBoost" in meta_learner.base_model_predictions
        assert "LightGBM" in meta_learner.base_model_predictions

        # Verify structure of loaded predictions
        xgb_preds = meta_learner.base_model_predictions["XGBoost"]
        assert list(xgb_preds.columns) == ["date", "code", "Q_pred", "Q_obs", "model"]
        assert len(xgb_preds) == 50
        assert xgb_preds["model"].iloc[0] == "XGBoost"

    def test_base_model_loading_from_csv_files(self):
        """Test loading base model predictions from CSV files."""
        path_config = {
            "model_home_path": self.temp_dir,
            "path_to_base_models": [self.base_model_paths[2]],  # Only CSV file
        }

        # Initialize meta-learner with path configuration
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
        )

        # Verify that base model predictions were loaded
        assert len(meta_learner.base_model_predictions) == 1
        assert "CatBoost_predictions" in meta_learner.base_model_predictions

        # Verify structure of loaded predictions
        catboost_preds = meta_learner.base_model_predictions["CatBoost_predictions"]
        assert list(catboost_preds.columns) == [
            "date",
            "code",
            "Q_pred",
            "Q_obs",
            "model",
        ]
        assert len(catboost_preds) == 50
        assert catboost_preds["model"].iloc[0] == "CatBoost_predictions"

    def test_base_model_loading_mixed_paths(self):
        """Test loading base model predictions from mixed directory and file paths."""
        path_config = {
            "model_home_path": self.temp_dir,
            "path_to_base_models": self.base_model_paths,  # All paths
        }

        # Initialize meta-learner with path configuration
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
        )

        # Verify that all base model predictions were loaded
        assert len(meta_learner.base_model_predictions) == 3
        assert "XGBoost" in meta_learner.base_model_predictions
        assert "LightGBM" in meta_learner.base_model_predictions
        assert "CatBoost_predictions" in meta_learner.base_model_predictions

        # Verify all predictions have correct structure
        for model_id, predictions in meta_learner.base_model_predictions.items():
            assert list(predictions.columns) == [
                "date",
                "code",
                "Q_pred",
                "Q_obs",
                "model",
            ]
            assert len(predictions) == 50
            assert predictions["model"].iloc[0] == model_id

    def test_base_model_loading_no_paths(self):
        """Test initialization without base model paths."""
        path_config = {
            "model_home_path": self.temp_dir
            # No path_to_base_models specified
        }

        # Initialize meta-learner without path configuration
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
        )

        # Verify that no base model predictions were loaded
        assert len(meta_learner.base_model_predictions) == 0

    def test_base_model_loading_with_manual_predictions(self):
        """Test that manually provided predictions are preserved along with loaded ones."""
        # Create manual predictions
        manual_predictions = {
            "Manual_Model": pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                    "code": [1] * 30,
                    "Q_obs": np.random.randn(30) * 10 + 50,
                    "Q_pred": np.random.randn(30) * 10 + 50,
                    "model": ["Manual_Model"] * 30,
                }
            )
        }

        path_config = {
            "model_home_path": self.temp_dir,
            "path_to_base_models": [self.base_model_paths[0]],  # Only XGBoost
        }

        # Initialize meta-learner with both manual and path-based predictions
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
            base_model_predictions=manual_predictions,
        )

        # Verify that both manual and loaded predictions exist
        assert len(meta_learner.base_model_predictions) == 2
        assert "Manual_Model" in meta_learner.base_model_predictions
        assert "XGBoost" in meta_learner.base_model_predictions

        # Verify manual predictions are preserved
        manual_preds = meta_learner.base_model_predictions["Manual_Model"]
        assert len(manual_preds) == 30
        assert manual_preds["model"].iloc[0] == "Manual_Model"

    def test_base_model_loading_error_handling(self):
        """Test error handling for invalid paths."""
        path_config = {
            "model_home_path": self.temp_dir,
            "path_to_base_models": [
                "/nonexistent/path",
                os.path.join(self.temp_dir, "nonexistent.csv"),
                self.base_model_paths[0],  # Valid path
            ],
        }

        # Initialize meta-learner with some invalid paths
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
        )

        # Should only load the valid path
        assert len(meta_learner.base_model_predictions) == 1
        assert "XGBoost" in meta_learner.base_model_predictions

    def test_base_model_loading_multiple_q_columns(self):
        """Test handling of prediction files with multiple Q_ columns."""
        # Create a prediction file with multiple Q_ columns
        multi_q_dir = os.path.join(self.temp_dir, "MultiQ_Model")
        os.makedirs(multi_q_dir)

        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        multi_q_predictions = pd.DataFrame(
            {
                "date": dates,
                "code": [1] * 50,
                "Q_obs": np.random.randn(50) * 10 + 50,
                "Q_MultiQ_Model": np.random.randn(50) * 10 + 50,  # Matches model name
                "Q_alternative": np.random.randn(50) * 10 + 50,
                "Q_another": np.random.randn(50) * 10 + 50,
            }
        )
        multi_q_predictions.to_csv(
            os.path.join(multi_q_dir, "predictions.csv"), index=False
        )

        path_config = {
            "model_home_path": self.temp_dir,
            "path_to_base_models": [multi_q_dir],
        }

        # Initialize meta-learner
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=path_config,
        )

        # Should load predictions using the column that matches model name
        assert len(meta_learner.base_model_predictions) == 1
        assert "MultiQ_Model" in meta_learner.base_model_predictions

        # Verify correct column was used
        predictions = meta_learner.base_model_predictions["MultiQ_Model"]
        assert len(predictions) == 50
        # The Q_pred values should come from Q_MultiQ_Model column
        assert not predictions["Q_pred"].isna().any()


if __name__ == "__main__":
    pytest.main([__file__])
