"""
Tests for evaluation utilities module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from monthly_forecasting.scr.evaluation_utils import (
    r2_score,
    rmse,
    mae,
    nse,
    kge,
    bias,
    nrmse,
    mape,
    pbias,
    calculate_all_metrics,
    evaluate_predictions,
)


class TestEvaluationUtils:
    """Test evaluation utility functions."""

    def setup_method(self):
        """Set up test data."""
        # Create synthetic data for testing
        np.random.seed(42)
        self.n_samples = 100

        # Perfect predictions
        self.obs_perfect = np.random.randn(self.n_samples) * 10 + 50
        self.pred_perfect = self.obs_perfect.copy()

        # Noisy predictions
        self.obs_noisy = np.random.randn(self.n_samples) * 10 + 50
        self.pred_noisy = self.obs_noisy + np.random.randn(self.n_samples) * 2

        # Predictions with bias
        self.obs_biased = np.random.randn(self.n_samples) * 10 + 50
        self.pred_biased = self.obs_biased + 5  # Constant bias

        # Predictions with NaN values
        self.obs_nan = self.obs_noisy.copy()
        self.pred_nan = self.pred_noisy.copy()
        self.obs_nan[10:15] = np.nan
        self.pred_nan[20:25] = np.nan

    def test_r2_score_perfect(self):
        """Test R2 score with perfect predictions."""
        score = r2_score(self.obs_perfect, self.pred_perfect)
        assert abs(score - 1.0) < 1e-10

    def test_r2_score_noisy(self):
        """Test R2 score with noisy predictions."""
        score = r2_score(self.obs_noisy, self.pred_noisy)
        assert 0.8 < score < 1.0  # Should be high but not perfect

    def test_r2_score_with_nan(self):
        """Test R2 score with NaN values."""
        score = r2_score(self.obs_nan, self.pred_nan)
        assert not np.isnan(score)  # Should handle NaN values

    def test_rmse_perfect(self):
        """Test RMSE with perfect predictions."""
        score = rmse(self.obs_perfect, self.pred_perfect)
        assert abs(score) < 1e-10

    def test_rmse_noisy(self):
        """Test RMSE with noisy predictions."""
        score = rmse(self.obs_noisy, self.pred_noisy)
        assert 1.0 < score < 5.0  # Should be reasonable for our noise level

    def test_mae_perfect(self):
        """Test MAE with perfect predictions."""
        score = mae(self.obs_perfect, self.pred_perfect)
        assert abs(score) < 1e-10

    def test_mae_biased(self):
        """Test MAE with biased predictions."""
        score = mae(self.obs_biased, self.pred_biased)
        assert abs(score - 5.0) < 0.1  # Should be close to bias magnitude

    def test_nse_perfect(self):
        """Test NSE with perfect predictions."""
        score = nse(self.obs_perfect, self.pred_perfect)
        assert abs(score - 1.0) < 1e-10

    def test_nse_noisy(self):
        """Test NSE with noisy predictions."""
        score = nse(self.obs_noisy, self.pred_noisy)
        assert 0.8 < score < 1.0  # Should be high but not perfect

    def test_kge_perfect(self):
        """Test KGE with perfect predictions."""
        score = kge(self.obs_perfect, self.pred_perfect)
        assert abs(score - 1.0) < 1e-10

    def test_kge_noisy(self):
        """Test KGE with noisy predictions."""
        score = kge(self.obs_noisy, self.pred_noisy)
        assert 0.8 < score < 1.0  # Should be high but not perfect

    def test_bias_perfect(self):
        """Test bias with perfect predictions."""
        score = bias(self.obs_perfect, self.pred_perfect)
        assert abs(score) < 1e-10

    def test_bias_biased(self):
        """Test bias with biased predictions."""
        score = bias(self.obs_biased, self.pred_biased)
        assert abs(score - 5.0) < 0.1  # Should be close to bias value

    def test_nrmse_perfect(self):
        """Test NRMSE with perfect predictions."""
        score = nrmse(self.obs_perfect, self.pred_perfect)
        assert abs(score) < 1e-10

    def test_mape_perfect(self):
        """Test MAPE with perfect predictions."""
        score = mape(self.obs_perfect, self.pred_perfect)
        assert abs(score) < 1e-10

    def test_pbias_perfect(self):
        """Test PBIAS with perfect predictions."""
        score = pbias(self.obs_perfect, self.pred_perfect)
        assert abs(score) < 1e-10

    def test_pbias_biased(self):
        """Test PBIAS with biased predictions."""
        score = pbias(self.obs_biased, self.pred_biased)
        expected_pbias = 5.0 / np.mean(self.obs_biased) * 100
        assert abs(score - expected_pbias) < 1.0

    def test_calculate_all_metrics(self):
        """Test calculation of all metrics."""
        metrics = calculate_all_metrics(self.obs_noisy, self.pred_noisy)

        expected_metrics = [
            "r2",
            "rmse",
            "nrmse",
            "mae",
            "mape",
            "nse",
            "kge",
            "bias",
            "pbias",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric])

    def test_evaluate_predictions_simple(self):
        """Test prediction evaluation without grouping."""
        # Create test DataFrame
        test_df = pd.DataFrame(
            {
                "Q_obs": self.obs_noisy,
                "Q_pred": self.pred_noisy,
                "date": pd.date_range("2020-01-01", periods=self.n_samples, freq="D"),
                "code": "TEST01",
            }
        )

        results = evaluate_predictions(test_df)

        assert len(results) == 1
        assert "r2" in results.columns
        assert "rmse" in results.columns
        assert "nse" in results.columns

    def test_evaluate_predictions_grouped(self):
        """Test prediction evaluation with grouping."""
        # Create test DataFrame with multiple codes
        test_df = pd.DataFrame(
            {
                "Q_obs": np.concatenate([self.obs_noisy, self.obs_biased]),
                "Q_pred": np.concatenate([self.pred_noisy, self.pred_biased]),
                "date": pd.date_range(
                    "2020-01-01", periods=2 * self.n_samples, freq="D"
                ),
                "code": ["TEST01"] * self.n_samples + ["TEST02"] * self.n_samples,
            }
        )

        results = evaluate_predictions(test_df, group_cols=["code"])

        assert len(results) == 2
        assert "code" in results.columns
        assert "r2" in results.columns
        assert set(results["code"]) == {"TEST01", "TEST02"}

    def test_error_handling_empty_arrays(self):
        """Test error handling with empty arrays."""
        empty_obs = np.array([])
        empty_pred = np.array([])

        score = r2_score(empty_obs, empty_pred)
        assert np.isnan(score)

    def test_error_handling_all_nan(self):
        """Test error handling with all NaN arrays."""
        nan_obs = np.full(10, np.nan)
        nan_pred = np.full(10, np.nan)

        score = r2_score(nan_obs, nan_pred)
        assert np.isnan(score)

    def test_error_handling_mismatched_shapes(self):
        """Test error handling with mismatched array shapes."""
        obs_short = np.array([1, 2, 3])
        pred_long = np.array([1, 2, 3, 4, 5])

        # Should return NaN for mismatched shapes, not raise an exception
        result = r2_score(obs_short, pred_long)
        assert np.isnan(result)

    def test_error_handling_zero_variance(self):
        """Test error handling with zero variance observations."""
        obs_constant = np.full(10, 5.0)
        pred_constant = np.full(10, 5.0)

        # R2 should be NaN for constant observations
        score = r2_score(obs_constant, pred_constant)
        assert np.isnan(score)

        # NSE should be NaN for constant observations
        score = nse(obs_constant, pred_constant)
        assert np.isnan(score)

    def test_pandas_series_input(self):
        """Test that functions work with pandas Series input."""
        obs_series = pd.Series(self.obs_noisy)
        pred_series = pd.Series(self.pred_noisy)

        score = r2_score(obs_series, pred_series)
        assert not np.isnan(score)
        assert 0.8 < score < 1.0

    def test_mape_with_zero_observations(self):
        """Test MAPE handling with zero observations."""
        obs_with_zeros = np.array([0, 1, 2, 3, 4])
        pred_with_zeros = np.array([0.1, 1.1, 2.1, 3.1, 4.1])

        score = mape(obs_with_zeros, pred_with_zeros)
        # Should handle zeros gracefully
        assert not np.isnan(score)

    def test_nrmse_with_zero_mean(self):
        """Test NRMSE handling with zero mean observations."""
        obs_zero_mean = np.array([-1, 0, 1])
        pred_zero_mean = np.array([-1.1, 0.1, 1.1])

        score = nrmse(obs_zero_mean, pred_zero_mean)
        # Should be NaN when mean is zero
        assert np.isnan(score)

    def test_metric_consistency(self):
        """Test consistency between different metric implementations."""
        # Test that our implementations are consistent
        obs = self.obs_noisy
        pred = self.pred_noisy

        # Calculate metrics
        r2 = r2_score(obs, pred)
        rmse_val = rmse(obs, pred)
        nse_val = nse(obs, pred)

        # NSE and R2 should be very similar for linear relationships
        assert abs(r2 - nse_val) < 0.1

        # RMSE should be positive
        assert rmse_val > 0

    def test_legacy_compatibility(self):
        """Test legacy compatibility functions."""
        from monthly_forecasting.scr.evaluation_utils import calculate_metrics_dict

        metrics = calculate_metrics_dict(self.obs_noisy, self.pred_noisy)

        expected_metrics = [
            "r2",
            "rmse",
            "nrmse",
            "mae",
            "mape",
            "nse",
            "kge",
            "bias",
            "pbias",
        ]

        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric])


if __name__ == "__main__":
    pytest.main([__file__])
