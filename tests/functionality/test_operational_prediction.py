"""
Tests for operational prediction workflow functionality.

This test suite verifies that the operational prediction workflow works correctly
with all components including configuration loading, data processing, model
execution, and performance evaluation.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import json

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.run_operational_prediction import (
    load_operational_configs,
    shift_data_to_current_year,
    create_model_instance,
    calculate_average_discharge,
    evaluate_predictions,
    process_predictions,
    run_operational_prediction,
    generate_outputs,
    MODELS_OPERATIONAL,
)


class TestConfigurationLoading:
    """Test configuration loading functionality."""

    def test_load_operational_configs_dummy_model(self):
        """Test loading configurations with dummy model."""
        configs = load_operational_configs(
            "LR",
            "BaseCase",
            "LR_Q_T_P",
        )

        assert "general_config" in configs
        assert "model_config" in configs
        assert "feature_config" in configs
        assert "data_config" in configs
        assert "path_config" in configs

        # Check that model type is set correctly
        assert configs["general_config"]["model_type"] == "linear_regression"

    def test_load_operational_configs_sciregressor(self):
        """Test loading configurations for SciRegressor model."""
        configs = load_operational_configs(
            "SciRegressor", "SnowMapper_Based", "NormBased"
        )

        assert "general_config" in configs
        assert configs["general_config"]["model_type"] == "sciregressor"


class TestDataProcessing:
    """Test data processing functionality."""

    def test_shift_data_to_current_year(self):
        """Test date shifting functionality."""
        # Create test data
        test_data = pd.DataFrame(
            {
                "date": pd.date_range("2022-01-01", periods=10, freq="D"),
                "code": [1] * 10,
                "discharge": np.random.randn(10),
            }
        )

        # Shift data by 1 year
        shifted_data = shift_data_to_current_year(test_data, shift_years=1)

        # Check that dates are shifted correctly
        assert shifted_data["date"].min() == pd.Timestamp("2023-01-01")
        assert shifted_data["date"].max() == pd.Timestamp("2023-01-10")

        # Check that other columns are unchanged
        assert shifted_data["code"].equals(test_data["code"])
        assert shifted_data["discharge"].equals(test_data["discharge"])

    def test_calculate_average_discharge(self):
        """Test discharge averaging functionality."""
        # Create test data with alternating codes
        dates = pd.date_range("2023-01-01", periods=60, freq="D")
        codes = [1, 2] * 30  # Alternating codes
        test_data = pd.DataFrame(
            {
                "date": dates,
                "code": codes,
                "discharge": [10.0 if code == 1 else 20.0 for code in codes],
            }
        )

        # Calculate averages for last 30 days
        avg_discharge = calculate_average_discharge(
            test_data, "2023-01-31", "2023-03-01"
        )

        # Check results
        assert len(avg_discharge) == 2
        assert (
            avg_discharge[avg_discharge["code"] == 1]["observed_avg_discharge"].iloc[0]
            == 10.0
        )
        assert (
            avg_discharge[avg_discharge["code"] == 2]["observed_avg_discharge"].iloc[0]
            == 20.0
        )

    def test_process_predictions(self):
        """Test prediction processing functionality."""
        # Test with 'Q' column
        predictions_q = pd.DataFrame(
            {
                "code": [1, 2, 3],
                "Q_LR_Q_T_P": [10.0, 20.0, 30.0],
                "valid_from": ["2023-01-01", "2023-01-01", "2023-01-01"],
                "valid_to": ["2023-01-31", "2023-01-31", "2023-01-31"],
            }
        )

        processed = process_predictions(predictions_q, "LR_Q_T_P")
        assert "Q_LR_Q_T_P" in processed.columns
        assert processed["Q_LR_Q_T_P"].tolist() == [10.0, 20.0, 30.0]


class TestPerformanceEvaluation:
    """Test performance evaluation functionality."""

    def test_evaluate_predictions(self):
        """Test prediction evaluation functionality."""
        # Create test predictions
        predictions = pd.DataFrame(
            {
                "code": [1, 2, 3],
                "Q_TestModel": [10.0, 20.0, 30.0],
                "valid_from": ["2023-01-01", "2023-01-01", "2023-01-01"],
                "valid_to": ["2023-01-31", "2023-01-31", "2023-01-31"],
            }
        )

        # Create test observations
        observations = pd.DataFrame(
            {"code": [1, 2, 3], "observed_avg_discharge": [12.0, 18.0, 32.0]}
        )

        # Evaluate predictions
        metrics = evaluate_predictions(predictions, observations, "TestModel")

        # Check that metrics are calculated
        assert "num_predictions" in metrics
        assert "basin_errors" in metrics
        assert "mean_absolute_error_pct" in metrics
        assert "median_absolute_error_pct" in metrics
        assert "basins_over_30pct_error" in metrics
        assert "basins_over_50pct_error" in metrics

        # Check basic values
        assert metrics["num_predictions"] == 3
        assert isinstance(metrics["mean_absolute_error_pct"], float)
        assert isinstance(metrics["median_absolute_error_pct"], float)
        assert isinstance(metrics["basin_errors"], list)
        assert len(metrics["basin_errors"]) == 3

    def test_evaluate_predictions_poor_performance(self):
        """Test evaluation with poor predictions (>30% error)."""
        # Create predictions with high error
        predictions = pd.DataFrame(
            {
                "code": [1, 2, 3],
                "Q_TestModel": [10.0, 20.0, 100.0],  # Third prediction is way off
                "valid_from": ["2023-01-01", "2023-01-01", "2023-01-01"],
                "valid_to": ["2023-01-31", "2023-01-31", "2023-01-31"],
            }
        )

        observations = pd.DataFrame(
            {"code": [1, 2, 3], "observed_avg_discharge": [12.0, 18.0, 30.0]}
        )

        metrics = evaluate_predictions(predictions, observations, "TestModel")

        # Check that high errors are identified
        assert len(metrics["basins_over_30pct_error"]) > 0
        assert 3 in metrics["basins_over_30pct_error"]  # Basin 3 should be flagged
        # Basin 3 has 100 predicted vs 30 observed = 233% error, so should be >50% too
        assert 3 in metrics["basins_over_50pct_error"]


class TestModelExecution:
    """Test model execution functionality."""

    @patch("scripts.run_operational_prediction.create_data_frame")
    def test_create_model_instance_lr(self, mock_create_data_frame):
        """Test creating LinearRegression model instance."""
        # Mock data
        mock_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=10),
                "code": [1] * 10,
                "discharge": np.random.randn(10),
            }
        )
        mock_static_data = pd.DataFrame({"code": [1], "area": [100]})

        # Mock configs with required parameters
        configs = {
            "general_config": {
                "model_name": "test_model",
                "model_type": "linear_regression",
                "prediction_horizon": 30,
                "offset": None,
                "feature_cols": ["discharge", "T", "P"],
                "static_features": ["area"],
                "handle_na": "long_term_mean",
                "normalize": True,
                "num_test_years": 2,
            },
            "model_config": {"model_type": "linear"},
            "feature_config": {
                "discharge": [
                    {
                        "operation": "mean",
                        "windows": [15, 30],
                        "lags": {"30": [30, 365]},
                    }
                ]
            },
            "path_config": {"model_home_path": "/tmp/test_models"},
        }

        # Create model instance
        model = create_model_instance(
            "LR", "test_model", configs, mock_data, mock_static_data
        )

        # Check that correct model type is created
        assert model.__class__.__name__ == "LinearRegressionModel"
        assert model.name == "test_model"


class TestWorkflowIntegration:
    """Test full workflow integration."""

    def test_models_operational_structure(self):
        """Test that MODELS_OPERATIONAL has the correct structure."""
        assert "BaseCase" in MODELS_OPERATIONAL
        assert "SnowMapper_Based" in MODELS_OPERATIONAL

        # Check BaseCase models
        base_case = MODELS_OPERATIONAL["BaseCase"]
        assert len(base_case) == 2
        assert ("LR", "LR_Q_T_P") in base_case
        assert ("SciRegressor", "GradBoostTrees") in base_case

        # Check SnowMapper_Based models
        snow_based = MODELS_OPERATIONAL["SnowMapper_Based"]
        assert len(snow_based) == 5
        assert ("LR", "LR_Q_dSWEdt_T_P") in snow_based
        assert ("SciRegressor", "NormBased") in snow_based

    def test_generate_outputs(self):
        """Test output generation functionality."""
        # Create temporary directory for outputs
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test results
            test_predictions = pd.DataFrame(
                {
                    "code": [1, 2, 3],
                    "Q_LR_Q_T_P": [10.0, 20.0, 30.0],
                    "model_family": ["BaseCase"] * 3,
                    "model_type": ["LR"] * 3,
                    "model_name": ["LR_Q_T_P"] * 3,
                    "valid_from": ["2023-01-01"] * 3,
                    "valid_to": ["2023-01-31"] * 3,
                }
            )

            test_metrics = [
                {
                    "mean_absolute_error_pct": 15.0,
                    "median_absolute_error_pct": 12.0,
                    "num_predictions": 3,
                    "model_name": "LR_Q_T_P",
                    "basin_errors": [],
                    "basins_over_30pct_error": [],
                    "basins_over_50pct_error": [],
                }
            ]

            test_timing = {"BaseCase_LR_LR_Q_T_P": 1.5, "overall_duration": 5.0}

            results = {
                "predictions": test_predictions,
                "metrics": test_metrics,
                "timing": test_timing,
            }

            # Generate outputs
            generate_outputs(results, temp_dir)

            # Check that files are created
            assert (Path(temp_dir) / "operational_predictions.csv").exists()
            assert (Path(temp_dir) / "timing_report.json").exists()
            assert (Path(temp_dir) / "performance_metrics.csv").exists()
            assert (Path(temp_dir) / "quality_report.txt").exists()

            # Check file contents
            predictions_df = pd.read_csv(Path(temp_dir) / "operational_predictions.csv")
            assert len(predictions_df) == 3
            assert "Q_LR_Q_T_P" in predictions_df.columns

            with open(Path(temp_dir) / "timing_report.json") as f:
                timing_data = json.load(f)
                assert "overall_duration" in timing_data
                assert timing_data["overall_duration"] == 5.0


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_process_predictions_no_prediction_column(self):
        """Test processing predictions without prediction column."""
        predictions = pd.DataFrame(
            {"code": [1, 2, 3], "some_other_column": [10.0, 20.0, 30.0]}
        )

        processed = process_predictions(predictions, "TestModel")

        # Should return unchanged if no prediction column found
        assert "Q_TestModel" not in processed.columns

    def test_evaluate_predictions_empty_data(self):
        """Test evaluation with empty data."""
        empty_predictions = pd.DataFrame()
        empty_observations = pd.DataFrame()

        # Should handle empty data gracefully
        metrics = evaluate_predictions(
            empty_predictions, empty_observations, "TestModel"
        )
        assert metrics["num_predictions"] == 0
        assert np.isnan(metrics["mean_absolute_error_pct"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
