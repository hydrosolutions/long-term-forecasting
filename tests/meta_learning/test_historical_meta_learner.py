"""
Tests for historical meta-learner module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)


class TestHistoricalMetaLearner:
    """Test historical meta-learner functionality."""

    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic data
        np.random.seed(42)
        self.n_samples = 100

        # Create test data
        dates = pd.date_range("2020-01-01", periods=self.n_samples, freq="D")
        codes = np.tile([1, 2], (self.n_samples // 2) + 1)[: self.n_samples]

        self.data = pd.DataFrame(
            {
                "date": dates,
                "code": codes,
                "Q": np.random.randn(self.n_samples) * 10 + 50,
                "T": np.random.randn(self.n_samples) * 5 + 15,
                "P": np.random.randn(self.n_samples) * 20 + 100,
            }
        )

        self.static_data = pd.DataFrame(
            {
                "code": [1, 2],
                "area": [1000, 1500],
                "elevation": [500, 800],
            }
        )

        # Create configurations
        self.general_config = {
            "model_name": "test_historical_meta_learner",
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

        self.path_config = {
            "model_dir": "/tmp/test_models",
            "output_dir": "/tmp/test_output",
        }

        # Create base model predictions with multi-year data
        self.base_model_predictions = self._create_multiyear_predictions()

    def _create_multiyear_predictions(self):
        """Create multi-year synthetic base model predictions for LOOCV."""
        predictions = {}

        # Create predictions for 3 base models over 3 years
        for i, model_name in enumerate(["XGB", "LGBM", "CatBoost"]):
            dates = pd.date_range("2020-01-01", periods=365 * 3, freq="D")
            n_samples = len(dates)
            codes = np.tile([1, 2], (n_samples // 2) + 1)[:n_samples]

            # Create seasonal pattern
            day_of_year = dates.dayofyear
            seasonal_pattern = 10 * np.sin(2 * np.pi * day_of_year / 365)

            # Add some model-specific bias/noise
            obs = 50 + seasonal_pattern + np.random.randn(n_samples) * 5
            pred = obs + np.random.randn(n_samples) * (
                1 + i * 0.5
            )  # Different noise levels

            predictions[model_name] = pd.DataFrame(
                {
                    "date": dates,
                    "code": codes,
                    "Q_obs": obs,
                    "Q_pred": pred,
                    "model": model_name,
                }
            )

        return predictions

    def test_initialization(self):
        """Test historical meta-learner initialization."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        assert meta_learner.name == "test_historical_meta_learner"
        assert meta_learner.basin_specific == True
        assert meta_learner.temporal_weighting == True
        assert meta_learner.min_samples_per_basin == 5
        assert meta_learner.weight_smoothing == 0.1
        assert len(meta_learner.base_model_predictions) == 3

    def test_calculate_historical_performance(self):
        """Test calculation of historical performance."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        performance = meta_learner.calculate_historical_performance()

        # Check that all models have performance data
        assert len(performance) == 3
        assert "XGB" in performance
        assert "LGBM" in performance
        assert "CatBoost" in performance

        # Check performance structure
        for model_id, perf_data in performance.items():
            assert "overall" in perf_data
            assert "basin" in perf_data  # Basin-specific enabled
            assert "temporal" in perf_data  # Temporal weighting enabled

            # Check overall metrics
            overall_metrics = perf_data["overall"]
            assert "r2" in overall_metrics
            assert "rmse" in overall_metrics
            assert "nse" in overall_metrics

            # Check basin-specific metrics
            basin_metrics = perf_data["basin"]
            assert 1 in basin_metrics
            assert 2 in basin_metrics

            # Check temporal metrics
            temporal_metrics = perf_data["temporal"]
            assert len(temporal_metrics) == 12  # 12 months

    def test_compute_performance_weights(self):
        """Test computation of performance weights."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Calculate historical performance first
        meta_learner.calculate_historical_performance()

        # Test with error metric (RMSE)
        performance_data = {
            "XGB": {"rmse": 2.0},
            "LGBM": {"rmse": 3.0},
            "CatBoost": {"rmse": 4.0},
        }

        weights = meta_learner.compute_performance_weights(
            performance_data, metric="rmse"
        )

        # Check that weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # Check that better performance (lower RMSE) gets higher weight
        assert weights["XGB"] > weights["LGBM"]
        assert weights["LGBM"] > weights["CatBoost"]

    def test_compute_performance_weights_skill_metric(self):
        """Test computation of performance weights with skill metric."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Test with skill metric (NSE)
        performance_data = {
            "XGB": {"nse": 0.8},
            "LGBM": {"nse": 0.7},
            "CatBoost": {"nse": 0.6},
        }

        weights = meta_learner.compute_performance_weights(
            performance_data, metric="nse", invert_metric=False
        )

        # Check that weights sum to 1
        assert abs(sum(weights.values()) - 1.0) < 1e-6

        # Check that better performance (higher NSE) gets higher weight
        assert weights["XGB"] > weights["LGBM"]
        assert weights["LGBM"] > weights["CatBoost"]

    def test_compute_basin_specific_weights(self):
        """Test computation of basin-specific weights."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Calculate historical performance first
        meta_learner.calculate_historical_performance()

        # Test basin-specific weights
        basin_weights = meta_learner.compute_basin_specific_weights(1)

        # Check that weights sum to 1
        assert abs(sum(basin_weights.values()) - 1.0) < 1e-6

        # Check that all models have weights
        assert "XGB" in basin_weights
        assert "LGBM" in basin_weights
        assert "CatBoost" in basin_weights

    def test_compute_temporal_weights(self):
        """Test computation of temporal weights."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Calculate historical performance first
        meta_learner.calculate_historical_performance()

        # Test temporal weights for January first 10 days
        temporal_weights = meta_learner.compute_temporal_weights("1-10")

        # Check that weights sum to 1
        assert abs(sum(temporal_weights.values()) - 1.0) < 1e-6

        # Check that all models have weights
        assert "XGB" in temporal_weights
        assert "LGBM" in temporal_weights
        assert "CatBoost" in temporal_weights

    def test_compute_weights_combined(self):
        """Test computation of combined basin-specific and temporal weights."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Calculate historical performance first
        meta_learner.calculate_historical_performance()

        # Test combined weights
        combined_weights = meta_learner.compute_weights(basin_code=1, period="1-10")

        # Check that weights sum to 1 (with reasonable tolerance)
        assert abs(sum(combined_weights.values()) - 1.0) < 1e-6

        # Check that all models have weights
        assert "XGB" in combined_weights
        assert "LGBM" in combined_weights
        assert "CatBoost" in combined_weights

    def test_train_meta_model(self):
        """Test training of meta-model."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Train meta-model
        meta_learner.train_meta_model()

        # Check that historical performance was calculated
        assert len(meta_learner.historical_performance) > 0

        # Check that performance weights were cached
        assert len(meta_learner.performance_weights) > 0

    def test_calibrate_model_and_hindcast(self):
        """Test calibration and hindcast using LOOCV."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Perform calibration and hindcast
        hindcast_df = meta_learner.calibrate_model_and_hindcast()

        # Check hindcast results
        assert len(hindcast_df) > 0
        assert "date" in hindcast_df.columns
        assert "code" in hindcast_df.columns
        assert "Q_obs" in hindcast_df.columns
        assert "Q_pred" in hindcast_df.columns
        assert "model" in hindcast_df.columns
        assert "test_year" in hindcast_df.columns

        # Check that predictions are reasonable
        assert not hindcast_df["Q_pred"].isna().all()

        # Check that multiple years are represented
        assert hindcast_df["test_year"].nunique() > 1

    def test_predict_operational(self):
        """Test operational prediction."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Train meta-model first
        meta_learner.train_meta_model()

        # Test operational prediction
        operational_df = meta_learner.predict_operational(today=datetime(2022, 6, 15))

        # Check operational results
        assert len(operational_df) > 0
        assert "date" in operational_df.columns
        assert "code" in operational_df.columns
        assert "Q_obs" in operational_df.columns
        assert "Q_pred" in operational_df.columns
        assert "model" in operational_df.columns

        # Check that predictions are reasonable
        assert not operational_df["Q_pred"].isna().all()

    def test_tune_hyperparameters(self):
        """Test hyperparameter tuning."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Mock calibrate_model_and_hindcast to speed up testing
        def mock_calibrate():
            # Return a simple hindcast DataFrame
            return pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=100, freq="D"),
                    "code": [1] * 100,
                    "Q_obs": np.random.randn(100) * 10 + 50,
                    "Q_pred": np.random.randn(100) * 10 + 50,
                    "model": "test_meta_learner",
                }
            )

        meta_learner.calibrate_model_and_hindcast = mock_calibrate

        # Test hyperparameter tuning
        success, message = meta_learner.tune_hyperparameters()

        assert success == True
        assert "Hyperparameter tuning completed" in message

        # Check that hyperparameters were set
        assert hasattr(meta_learner, "weight_smoothing")
        assert hasattr(meta_learner, "min_samples_per_basin")

    def test_weight_smoothing(self):
        """Test weight smoothing functionality."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Test with high smoothing
        meta_learner.weight_smoothing = 0.5

        # Test with extreme performance difference
        performance_data = {
            "XGB": {"rmse": 1.0},
            "LGBM": {"rmse": 10.0},
            "CatBoost": {"rmse": 100.0},
        }

        weights = meta_learner.compute_performance_weights(
            performance_data, metric="rmse"
        )

        # With high smoothing, weights should be more uniform
        weight_values = list(weights.values())
        weight_range = max(weight_values) - min(weight_values)

        # Should be less extreme due to smoothing
        assert weight_range < 0.7  # Without smoothing, this would be much larger

    def test_no_basin_specific_weighting(self):
        """Test with basin-specific weighting disabled."""
        # Disable basin-specific weighting
        config = self.model_config.copy()
        config["meta_learning"]["basin_specific"] = False

        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        assert meta_learner.basin_specific == False

        # Calculate historical performance
        performance = meta_learner.calculate_historical_performance()

        # Should not have basin-specific performance
        for model_id, perf_data in performance.items():
            assert "overall" in perf_data
            assert "basin" not in perf_data or len(perf_data["basin"]) == 0

    def test_no_temporal_weighting(self):
        """Test with temporal weighting disabled."""
        # Disable temporal weighting
        config = self.model_config.copy()
        config["meta_learning"]["temporal_weighting"] = False

        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        assert meta_learner.temporal_weighting == False

        # Calculate historical performance
        performance = meta_learner.calculate_historical_performance()

        # Should not have temporal performance
        for model_id, perf_data in performance.items():
            assert "overall" in perf_data
            assert "temporal" not in perf_data or len(perf_data["temporal"]) == 0

    def test_insufficient_data_for_loocv(self):
        """Test error handling with insufficient data for LOOCV."""
        # Create predictions with only one year of data
        single_year_predictions = {}
        for model_name in ["XGB", "LGBM", "CatBoost"]:
            dates = pd.date_range("2020-01-01", periods=100, freq="D")
            single_year_predictions[model_name] = pd.DataFrame(
                {
                    "date": dates,
                    "code": [1] * 100,
                    "Q_obs": np.random.randn(100) * 10 + 50,
                    "Q_pred": np.random.randn(100) * 10 + 50,
                    "model": model_name,
                }
            )

        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=single_year_predictions,
        )

        # Should raise error for insufficient data
        with pytest.raises(ValueError, match="At least 2 years of data required"):
            meta_learner.calibrate_model_and_hindcast()

    def test_missing_performance_data(self):
        """Test handling of missing performance data."""
        meta_learner = HistoricalMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions,
        )

        # Test with no performance data
        weights = meta_learner.compute_performance_weights({})
        assert weights == {}

        # Test with invalid performance data
        invalid_data = {
            "XGB": {"rmse": np.nan},
            "LGBM": {"rmse": np.inf},
            "CatBoost": {"rmse": -1.0},
        }

        weights = meta_learner.compute_performance_weights(invalid_data)
        # Should fall back to uniform weights
        assert abs(sum(weights.values()) - 1.0) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
