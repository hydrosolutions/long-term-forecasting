"""
Tests for enhanced long-term mean scaling with period granularity.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from lt_forecasting.scr import data_utils as du
from lt_forecasting.scr.FeatureProcessingArtifacts import (
    FeatureProcessingArtifacts,
    _normalization_training,
    post_process_predictions,
)


class TestPeriodCalculation:
    """Test period calculation functionality."""

    def test_get_periods_creates_36_unique_periods(self):
        """Test that get_periods creates exactly 36 unique periods per year."""
        # Create a full year of daily data
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.randn(len(dates))})

        # Apply get_periods
        df_with_periods = du.get_periods(df)

        # Check unique periods
        unique_periods = df_with_periods["period"].unique()
        assert len(unique_periods) == 36, (
            f"Expected 36 unique periods, got {len(unique_periods)}"
        )

    def test_get_periods_handles_leap_year(self):
        """Test that get_periods correctly handles February in leap years."""
        # Test leap year (2020)
        leap_dates = pd.date_range(start="2020-02-01", end="2020-02-29", freq="D")
        df_leap = pd.DataFrame(
            {"date": leap_dates, "value": np.random.randn(len(leap_dates))}
        )
        df_leap = du.get_periods(df_leap)

        # Check that Feb 29 is marked as "end"
        feb_29 = df_leap[df_leap["date"] == "2020-02-29"]
        assert feb_29["period"].iloc[0] == "2-end"

        # Test non-leap year (2021)
        non_leap_dates = pd.date_range(start="2021-02-01", end="2021-02-28", freq="D")
        df_non_leap = pd.DataFrame(
            {"date": non_leap_dates, "value": np.random.randn(len(non_leap_dates))}
        )
        df_non_leap = du.get_periods(df_non_leap)

        # Check that Feb 28 is marked as "end"
        feb_28 = df_non_leap[df_non_leap["date"] == "2021-02-28"]
        assert feb_28["period"].iloc[0] == "2-end"

    def test_period_format(self):
        """Test that periods follow the correct format: month-day or month-end."""
        dates = pd.date_range(start="2020-03-01", end="2020-03-31", freq="D")
        df = pd.DataFrame({"date": dates, "value": np.random.randn(len(dates))})
        df = du.get_periods(df)

        # Check specific dates
        assert df[df["date"] == "2020-03-10"]["period"].iloc[0] == "3-10"
        assert df[df["date"] == "2020-03-20"]["period"].iloc[0] == "3-20"
        assert df[df["date"] == "2020-03-31"]["period"].iloc[0] == "3-end"


class TestLongTermMeanCalculation:
    """Test long-term mean and std calculation with periods."""

    def test_long_term_stats_calculation(self):
        """Test that long-term stats are calculated correctly per period."""
        # Create sample data with multiple years
        dates = pd.date_range(start="2018-01-01", end="2020-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "code": np.random.choice(["A", "B"], size=len(dates)),
                "feature1": np.random.randn(len(dates)),
                "feature2": np.random.randn(len(dates)) * 2 + 5,
            }
        )

        # Calculate long-term stats
        features = ["feature1", "feature2"]
        long_term_stats = du.get_long_term_mean_per_basin(df, features)

        # Check structure
        assert "code" in long_term_stats.columns
        assert "period" in long_term_stats.columns
        assert ("feature1", "mean") in long_term_stats.columns
        assert ("feature1", "std") in long_term_stats.columns
        assert ("feature2", "mean") in long_term_stats.columns
        assert ("feature2", "std") in long_term_stats.columns

        # Check that we have stats for each code-period combination
        unique_codes = df["code"].unique()
        # We should have approximately 36 periods per code (some might be missing)
        for code in unique_codes:
            code_stats = long_term_stats[long_term_stats["code"] == code]
            assert len(code_stats) <= 36, f"Too many periods for code {code}"
            assert len(code_stats) >= 30, f"Too few periods for code {code}"

    def test_zero_std_handling(self):
        """Test that zero std is replaced with 1.0."""
        # Create data with constant values (std = 0)
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "constant_feature": 5.0,  # Constant value -> std = 0
                "variable_feature": np.random.randn(len(dates)),
            }
        )

        features = ["constant_feature", "variable_feature"]
        long_term_stats = du.get_long_term_mean_per_basin(df, features)

        # Check that std for constant feature is 1.0
        constant_std = long_term_stats[("constant_feature", "std")]
        assert all(constant_std == 1.0), "Zero std should be replaced with 1.0"


class TestSelectiveFeatureScaling:
    """Test selective feature scaling based on pattern matching."""

    def test_get_relative_scaling_features(self):
        """Test pattern matching for relative scaling features."""
        features = [
            "SWE_1",
            "SWE_2",
            "SWE_Perc_Elev_1",
            "P_1",
            "P_sum_30",
            "T_mean_15",
            "discharge",
            "elevation",
        ]
        relative_scaling_vars = ["SWE", "discharge"]

        relative_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )

        # Check that all SWE-related features are included
        assert "SWE_1" in relative_features
        assert "SWE_2" in relative_features
        assert "SWE_Perc_Elev_1" in relative_features

        # Check that discharge is included
        assert "discharge" in relative_features

        # Check that non-matching features are excluded
        assert "P_1" not in relative_features
        assert "T_mean_15" not in relative_features
        assert "elevation" not in relative_features

    def test_empty_relative_scaling_vars(self):
        """Test that empty relative_scaling_vars returns empty list."""
        features = ["SWE_1", "P_1", "T_1"]
        relative_features = du.get_relative_scaling_features(features, [])
        assert relative_features == []


class TestScalingFormula:
    """Test the standardization formula implementation."""

    def test_standardization_formula(self):
        """Test that (x - mean) / std is correctly applied."""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "feature": np.random.randn(len(dates)) * 3 + 10,  # mean~10, std~3
            }
        )

        # Calculate stats and apply scaling
        features = ["feature"]
        long_term_stats = du.get_long_term_mean_per_basin(df, features)
        df_scaled = du.apply_long_term_mean_scaling(df, long_term_stats, features)

        # Check that scaled values have mean ~0 and std ~1
        scaled_mean = df_scaled["feature"].mean()
        scaled_std = df_scaled["feature"].std()

        assert abs(scaled_mean) < 0.1, f"Scaled mean should be ~0, got {scaled_mean}"
        assert abs(scaled_std - 1.0) < 0.1, f"Scaled std should be ~1, got {scaled_std}"

    def test_selective_scaling(self):
        """Test that only specified features are scaled."""
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "feature1": np.random.randn(len(dates)) * 2 + 5,
                "feature2": np.random.randn(len(dates)) * 3 + 10,
            }
        )

        # Calculate stats for all features
        all_features = ["feature1", "feature2"]
        long_term_stats = du.get_long_term_mean_per_basin(df, all_features)

        # Scale only feature1
        features_to_scale = ["feature1"]
        df_scaled = du.apply_long_term_mean_scaling(
            df, long_term_stats, all_features, features_to_scale=features_to_scale
        )

        # Check that feature1 is scaled
        assert df_scaled["feature1"].mean() < 1.0  # Should be ~0

        # Check that feature2 is NOT scaled
        assert abs(df_scaled["feature2"].mean() - 10) < 1.0  # Should still be ~10


class TestInverseTransformation:
    """Test inverse transformation functionality."""

    def test_inverse_scaling_basic(self):
        """Test basic inverse scaling: x_original = x_scaled * std + mean."""
        # Create sample data
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
        # Create repeating pattern that matches the length of dates
        pattern = [5.0, 10.0, 15.0, 20.0]
        feature_values = (pattern * (len(dates) // len(pattern) + 1))[: len(dates)]
        df_original = pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "feature": feature_values,
            }
        )

        # Apply scaling
        features = ["feature"]
        long_term_stats = du.get_long_term_mean_per_basin(df_original, features)
        df_scaled = du.apply_long_term_mean_scaling(
            df_original.copy(), long_term_stats, features
        )

        # Apply inverse scaling
        df_inverse = du.apply_inverse_long_term_mean_scaling(
            df_scaled.copy(), long_term_stats, features
        )

        # Check that we recover original values
        np.testing.assert_array_almost_equal(
            df_original["feature"].values,
            df_inverse["feature"].values,
            decimal=5,
        )

    def test_inverse_scaling_predictions(self):
        """Test prediction-specific inverse scaling."""
        # Create sample data with predictions
        dates = pd.date_range(start="2020-01-01", end="2020-01-31", freq="D")
        df = pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "prediction": np.random.randn(len(dates)),  # Scaled predictions
            }
        )

        # Create mock long-term stats for the target
        long_term_stats = pd.DataFrame(
            {
                "code": ["A"] * 1,
                "period": ["1-10"],
                ("target", "mean"): [10.0],
                ("target", "std"): [2.0],
            }
        )
        # Set MultiIndex columns
        long_term_stats.columns = pd.MultiIndex.from_tuples(
            [("code", ""), ("period", ""), ("target", "mean"), ("target", "std")]
        )

        # Apply inverse scaling
        df_inverse = du.apply_inverse_long_term_mean_scaling_predictions(
            df.copy(), long_term_stats, "prediction", "target"
        )

        # Check that predictions are scaled back
        # For a standard normal value, after inverse: x * 2 + 10
        # So values should be around 10 +/- 6 (3 std)
        assert df_inverse["prediction"].mean() > 5
        assert df_inverse["prediction"].mean() < 15


class TestFeatureProcessingArtifactsIntegration:
    """Test integration with FeatureProcessingArtifacts."""

    def test_mixed_normalization_training(self):
        """Test training with mixed normalization approach."""
        # Create sample data
        dates = pd.date_range(start="2018-01-01", end="2020-12-31", freq="D")
        np.random.seed(42)
        df = pd.DataFrame(
            {
                "date": dates,
                "code": np.random.choice(["A", "B"], size=len(dates)),
                "SWE_1": np.random.randn(len(dates)),
                "P_1": np.random.randn(len(dates)),
                "T_1": np.random.randn(len(dates)),
                "target": np.random.randn(len(dates)),
            }
        )

        # Create artifacts and config
        artifacts = FeatureProcessingArtifacts()
        artifacts.num_features = ["SWE_1", "P_1", "T_1"]
        artifacts.selected_features = ["SWE_1", "P_1", "T_1"]
        artifacts.static_features = []
        artifacts.cat_features = []  # Add empty cat_features
        artifacts.target_col = "target"

        experiment_config = {
            "normalize": True,
            "normalization_type": "global",
            "relative_scaling_vars": ["SWE"],
            "use_relative_target": True,
        }

        # Apply normalization
        df_normalized, artifacts = _normalization_training(
            df, "target", experiment_config, artifacts
        )

        # Check artifacts
        assert artifacts.relative_features is not None
        assert "SWE_1" in artifacts.relative_features
        assert "target" in artifacts.relative_features
        assert "P_1" not in artifacts.relative_features
        assert artifacts.long_term_stats is not None
        assert artifacts.use_relative_target is True

    def test_post_process_predictions_with_relative_target(self):
        """Test post-processing predictions when target uses relative scaling."""
        # Create sample predictions
        dates = pd.date_range(start="2021-01-01", end="2021-01-31", freq="D")
        df_predictions = pd.DataFrame(
            {
                "date": dates,
                "code": "A",
                "predictions": np.random.randn(len(dates)),  # Scaled predictions
            }
        )

        # Create artifacts with relative target scaling
        artifacts = FeatureProcessingArtifacts()
        artifacts.use_relative_target = True
        artifacts.target_col = "target"

        # Create mock long-term stats
        periods = [f"1-{d}" for d in ["10", "20", "end"]]
        stats_data = []
        for period in periods:
            stats_data.append(
                {
                    "code": "A",
                    "period": period,
                    ("target", "mean"): 100.0,
                    ("target", "std"): 20.0,
                }
            )
        artifacts.long_term_stats = pd.DataFrame(stats_data)
        artifacts.long_term_stats.columns = pd.MultiIndex.from_tuples(
            [("code", ""), ("period", "")] + [("target", "mean"), ("target", "std")]
        )

        experiment_config = {"normalize": True}

        # Apply post-processing
        df_denorm = post_process_predictions(
            df_predictions, artifacts, experiment_config, "predictions", "target"
        )

        # Check that predictions are denormalized
        # Original scale should be around 100 +/- 60 (3 std)
        assert df_denorm["predictions"].mean() > 40
        assert df_denorm["predictions"].mean() < 160


class TestEndToEndPipeline:
    """Test the complete pipeline from scaling to inverse scaling."""

    def test_full_pipeline_r2_consistency(self):
        """Test that R2 scores are consistent between transformed and original space."""
        # Create synthetic data with known relationship
        np.random.seed(42)
        # Use single basin and full year to avoid missing periods
        dates = pd.date_range(start="2019-01-01", end="2019-12-31", freq="D")

        # Create features with known relationship to target
        df = pd.DataFrame(
            {
                "date": dates,
                "code": "A",  # Single basin to ensure all periods exist
                "feature1": np.random.randn(len(dates)),
                "feature2": np.random.randn(len(dates)),
            }
        )
        # Target is a linear combination of features plus noise
        df["target"] = (
            2 * df["feature1"] + 3 * df["feature2"] + np.random.randn(len(dates)) * 0.1
        )

        # Apply scaling to data
        features = ["feature1", "feature2", "target"]
        long_term_stats = du.get_long_term_mean_per_basin(df, features)
        df_scaled = du.apply_long_term_mean_scaling(
            df.copy(), long_term_stats, features
        )

        # Simulate predictions (add small noise to test robustness)
        np.random.seed(123)
        df_scaled["predictions"] = (
            df_scaled["target"] + np.random.randn(len(df_scaled)) * 0.01
        )

        # Apply inverse scaling to predictions
        df_predictions = du.apply_inverse_long_term_mean_scaling_predictions(
            df_scaled.copy(), long_term_stats, "predictions", "target"
        )

        # Calculate R2 in both spaces
        from sklearn.metrics import r2_score

        # R2 in scaled space
        r2_scaled = r2_score(df_scaled["target"], df_scaled["predictions"])

        # R2 in original space
        # First inverse scale the target for comparison
        df_original = du.apply_inverse_long_term_mean_scaling(
            df_scaled[["date", "code", "target"]].copy(), long_term_stats, ["target"]
        )

        r2_original = r2_score(df_original["target"], df_predictions["predictions"])

        # Check that R2 scores are very close (within 0.05)
        assert abs(r2_scaled - r2_original) < 0.05, (
            f"R2 scores differ too much: scaled={r2_scaled:.3f}, original={r2_original:.3f}"
        )

        # Both should be very high since we used perfect predictions
        assert r2_scaled > 0.95
        assert r2_original > 0.95


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
