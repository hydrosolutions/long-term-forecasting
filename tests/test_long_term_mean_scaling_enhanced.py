"""
Test suite for enhanced long-term mean scaling with day-of-year granularity
and selective feature scaling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scr import data_utils as du
from scr.FeatureProcessingArtifacts import (
    FeatureProcessingArtifacts,
    _normalization_training,
)


class TestEnhancedLongTermMeanScaling:
    """Test the enhanced long-term mean scaling functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple basins and features."""
        # Create date range covering multiple years
        dates = pd.date_range(start="2020-01-01", end="2022-12-31", freq="D")

        # Create data for 2 basins
        data = []
        for basin in ["basin1", "basin2"]:
            for date in dates:
                # Create features with seasonal patterns
                day_of_year = date.dayofyear
                seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)

                row = {
                    "date": date,
                    "code": basin,
                    "discharge": 100 + 50 * seasonal_factor + np.random.normal(0, 5),
                    "SWE_mean": 50 + 30 * seasonal_factor + np.random.normal(0, 3),
                    "SWE_max": 80 + 40 * seasonal_factor + np.random.normal(0, 4),
                    "T_mean": 15 + 10 * seasonal_factor + np.random.normal(0, 2),
                    "T_max": 20 + 12 * seasonal_factor + np.random.normal(0, 2),
                    "P_sum": 30 + 20 * seasonal_factor + np.random.normal(0, 3),
                    "elevation": 1500 if basin == "basin1" else 2000,  # static feature
                    "area": 100 if basin == "basin1" else 150,  # static feature
                    "target": 120 + 60 * seasonal_factor + np.random.normal(0, 6),
                }
                data.append(row)

        return pd.DataFrame(data)

    def test_get_long_term_mean_per_basin_day_of_year(self, sample_data):
        """Test that long-term mean is calculated per day of year."""
        features = ["discharge", "SWE_mean", "T_mean"]

        # Calculate long-term means
        ltm = du.get_long_term_mean_per_basin(sample_data, features)

        # Check structure
        assert "code" in ltm.columns
        assert "day_of_year" in ltm.columns

        # Check that we have 365 or 366 unique day_of_year values per basin
        for basin in ltm["code"].unique():
            basin_data = ltm[ltm["code"] == basin]
            unique_days = basin_data["day_of_year"].nunique()
            assert unique_days in [365, 366], (
                f"Expected 365/366 days, got {unique_days}"
            )

        # Check that means are calculated correctly
        # For a specific basin and day_of_year
        test_basin = "basin1"
        test_day = 100

        expected_mean = sample_data[
            (sample_data["code"] == test_basin)
            & (sample_data["date"].dt.dayofyear == test_day)
        ]["discharge"].mean()

        actual_mean = ltm[
            (ltm["code"] == test_basin) & (ltm["day_of_year"] == test_day)
        ][("discharge", "mean")].iloc[0]

        assert np.isclose(expected_mean, actual_mean), (
            f"Expected mean {expected_mean}, got {actual_mean}"
        )

    def test_get_relative_scaling_features(self):
        """Test feature separation based on patterns."""
        features = [
            "discharge_lag1",
            "SWE_mean",
            "SWE_max",
            "T_mean",
            "T_max",
            "P_sum",
            "elevation",
            "area",
            "other_feature",
        ]
        relative_scaling_vars = ["SWE", "T", "discharge"]

        relative_features, per_basin_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )

        # Check relative features (features containing pattern "{var}_")
        expected_relative = ["discharge_lag1", "SWE_mean", "SWE_max", "T_mean", "T_max"]
        assert set(relative_features) == set(expected_relative)

        # Check per-basin features
        expected_per_basin = ["P_sum", "elevation", "area", "other_feature"]
        assert set(per_basin_features) == set(expected_per_basin)

    def test_get_relative_scaling_features_empty(self):
        """Test with no relative scaling vars."""
        features = ["discharge", "SWE_mean", "T_mean"]
        relative_scaling_vars = []

        relative_features, per_basin_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )

        assert relative_features == []
        assert per_basin_features == features

    def test_apply_long_term_mean_scaling_selective(self, sample_data):
        """Test selective scaling with mixed feature types."""
        features = ["SWE_mean", "T_mean", "P_sum", "elevation"]
        relative_scaling_vars = ["SWE", "T"]

        # Calculate long-term means for relative features
        relative_features, per_basin_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )

        ltm = du.get_long_term_mean_per_basin(sample_data, relative_features)

        # Calculate per-basin scaler for other features
        if per_basin_features:
            per_basin_scaler = du.get_normalization_params_per_basin(
                sample_data, per_basin_features, "target"
            )
        else:
            per_basin_scaler = None

        # Apply scaling
        df_scaled = du.apply_long_term_mean_scaling(
            sample_data,
            long_term_mean=ltm,
            features=features,
            relative_scaling_vars=relative_scaling_vars,
            per_basin_scaler=per_basin_scaler,
        )

        # Check that relative features are scaled by long-term mean
        # For relative features, the mean should be close to 1
        for feat in ["SWE_mean", "T_mean"]:
            for basin in df_scaled["code"].unique():
                basin_data = df_scaled[df_scaled["code"] == basin]
                mean_val = basin_data[feat].mean()
                assert np.isclose(mean_val, 1.0, atol=0.1), (
                    f"Mean of {feat} for {basin} should be ~1, got {mean_val}"
                )

        # Check that per-basin features are normalized (mean ~0, std ~1)
        for feat in ["P_sum"]:
            for basin in df_scaled["code"].unique():
                basin_data = df_scaled[df_scaled["code"] == basin]
                mean_val = basin_data[feat].mean()
                std_val = basin_data[feat].std()
                assert np.isclose(mean_val, 0.0, atol=0.1), (
                    f"Mean of {feat} for {basin} should be ~0, got {mean_val}"
                )
                assert np.isclose(std_val, 1.0, atol=0.1), (
                    f"Std of {feat} for {basin} should be ~1, got {std_val}"
                )

    def test_apply_inverse_long_term_mean_scaling_selective(self, sample_data):
        """Test inverse scaling with mixed feature types."""
        features = ["SWE_mean", "T_mean", "P_sum"]
        relative_scaling_vars = ["SWE", "T"]
        target = "target"
        use_relative_target = False  # Target uses per-basin scaling

        # Setup scaling
        relative_features, per_basin_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )

        # Only calculate long-term means for relative features (not target)
        ltm = du.get_long_term_mean_per_basin(sample_data, relative_features)

        # Calculate per-basin scaler including target
        per_basin_scaler = du.get_normalization_params_per_basin(
            sample_data, per_basin_features, target
        )

        # Apply forward scaling
        df_scaled = du.apply_long_term_mean_scaling(
            sample_data.copy(),
            long_term_mean=ltm,
            features=features,
            relative_scaling_vars=relative_scaling_vars,
            per_basin_scaler=per_basin_scaler,
        )

        # Apply per-basin scaling to target separately
        df_scaled = du.apply_normalization_per_basin(
            df_scaled, per_basin_scaler, [target]
        )

        # Create predictions (just copy scaled target for testing)
        df_scaled["prediction"] = df_scaled[target]

        # Apply inverse scaling - since target used per-basin scaling
        df_inverse = du.apply_inverse_normalization_per_basin(
            df_scaled,
            scaler=per_basin_scaler,
            var_to_scale="prediction",
            var_used_for_scaling=target,
        )

        # Check that predictions are restored to original scale
        assert np.allclose(df_inverse["prediction"], sample_data[target], rtol=1e-5)

    def test_feature_processing_artifacts_with_selective_scaling(self, sample_data):
        """Test FeatureProcessingArtifacts with new attributes."""
        artifacts = FeatureProcessingArtifacts()
        artifacts.num_features = ["SWE_mean", "T_mean", "P_sum"]
        artifacts.static_features = ["elevation", "area"]
        artifacts.selected_features = artifacts.num_features + artifacts.static_features
        artifacts.target_col = "target"

        experiment_config = {
            "normalize": True,
            "normalization_type": "long_term_mean",
            "relative_scaling_vars": ["SWE", "T"],
            "use_relative_target": True,
        }

        # Apply normalization
        df_normalized, artifacts = _normalization_training(
            sample_data.copy(), "target", experiment_config, artifacts
        )

        # Check that artifacts have new attributes
        assert artifacts.relative_scaling_vars == ["SWE", "T"]
        assert artifacts.use_relative_target == True
        assert "SWE_mean" in artifacts.relative_features
        assert "T_mean" in artifacts.relative_features
        assert "P_sum" in artifacts.per_basin_features
        assert artifacts.per_basin_scaler is not None
        assert artifacts.long_term_means is not None

    def test_leap_year_handling(self):
        """Test that leap years are handled correctly."""
        # Create data including leap year
        dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")

        data = []
        for date in dates:
            data.append(
                {
                    "date": date,
                    "code": "basin1",
                    "discharge": 100 + np.random.normal(0, 5),
                }
            )

        df = pd.DataFrame(data)

        # Calculate long-term means
        ltm = du.get_long_term_mean_per_basin(df, ["discharge"])

        # Check that day 366 exists for leap year
        assert 366 in ltm["day_of_year"].values

        # Check total number of unique days
        assert ltm["day_of_year"].nunique() == 366

    def test_backward_compatibility(self, sample_data):
        """Test that old behavior is preserved when relative_scaling_vars is None."""
        features = ["discharge", "SWE_mean", "T_mean"]

        # Calculate long-term means
        ltm = du.get_long_term_mean_per_basin(sample_data, features)

        # Apply scaling without relative_scaling_vars (old behavior)
        df_scaled = du.apply_long_term_mean_scaling(
            sample_data, long_term_mean=ltm, features=features
        )

        # All features should be scaled by long-term mean
        for feat in features:
            for basin in df_scaled["code"].unique():
                basin_data = df_scaled[df_scaled["code"] == basin]
                mean_val = basin_data[feat].mean()
                assert np.isclose(mean_val, 1.0, atol=0.1), (
                    f"Mean of {feat} for {basin} should be ~1, got {mean_val}"
                )

    def test_save_load_artifacts(self, sample_data, tmp_path):
        """Test saving and loading artifacts with new attributes."""
        artifacts = FeatureProcessingArtifacts()
        artifacts.relative_features = ["SWE_mean", "T_mean"]
        artifacts.per_basin_features = ["P_sum"]
        artifacts.relative_scaling_vars = ["SWE", "T"]
        artifacts.use_relative_target = True
        artifacts.per_basin_scaler = {"basin1": {"P_sum": (30.0, 5.0)}}
        artifacts.final_features = ["SWE_mean", "T_mean", "P_sum"]

        # Save artifacts
        save_path = tmp_path / "test_artifacts"
        artifacts.save(save_path, format="hybrid")

        # Load artifacts
        loaded_artifacts = FeatureProcessingArtifacts.load(save_path, format="hybrid")

        # Check that new attributes are preserved
        assert loaded_artifacts.relative_features == artifacts.relative_features
        assert loaded_artifacts.per_basin_features == artifacts.per_basin_features
        assert loaded_artifacts.relative_scaling_vars == artifacts.relative_scaling_vars
        assert loaded_artifacts.use_relative_target == artifacts.use_relative_target
        assert loaded_artifacts.per_basin_scaler == artifacts.per_basin_scaler


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
