"""
Test to verify that relative target scaling works correctly, especially when
the target column name doesn't match any patterns in relative_scaling_vars.
"""

import pytest
import pandas as pd
import numpy as np
from scr import data_utils as du
from scr.FeatureProcessingArtifacts import (
    FeatureProcessingArtifacts,
    _normalization_training,
    post_process_predictions,
)


def test_relative_target_scaling_with_non_matching_name():
    """Test that target is scaled correctly even when its name doesn't match patterns."""
    # Create sample data
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    data = []
    for date in dates:
        day_of_year = date.dayofyear
        seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)

        data.append(
            {
                "date": date,
                "code": "basin1",
                "SWE_mean": 50 + 30 * seasonal_factor + np.random.normal(0, 1),
                "T_mean": 15 + 10 * seasonal_factor + np.random.normal(0, 1),
                "P_sum": 30 + 20 * seasonal_factor + np.random.normal(0, 1),
                "my_target_column": 100
                + 50 * seasonal_factor
                + np.random.normal(
                    0, 2
                ),  # Target with name that doesn't match patterns
            }
        )

    df = pd.DataFrame(data)

    # Setup artifacts
    artifacts = FeatureProcessingArtifacts()
    artifacts.num_features = ["SWE_mean", "T_mean", "P_sum"]
    artifacts.static_features = []
    artifacts.selected_features = artifacts.num_features
    artifacts.target_col = "my_target_column"

    # Configure with relative scaling for SWE and T, but target name doesn't match
    experiment_config = {
        "normalize": True,
        "normalization_type": "long_term_mean",
        "relative_scaling_vars": [
            "SWE",
            "T",
        ],  # Note: 'my_target_column' doesn't match these patterns
        "use_relative_target": True,  # But we want to use relative scaling for target
    }

    # Store original target values
    original_target = df["my_target_column"].copy()

    # Apply normalization
    df_normalized, artifacts = _normalization_training(
        df.copy(), "my_target_column", experiment_config, artifacts
    )

    # Verify that target was scaled (mean should be close to 1 for relative scaling)
    assert np.abs(df_normalized["my_target_column"].mean() - 1.0) < 0.1, (
        f"Target should have mean ~1 after relative scaling, got {df_normalized['my_target_column'].mean()}"
    )

    # Verify artifacts are set correctly
    assert artifacts.use_relative_target == True
    assert "my_target_column" in artifacts.relative_features, (
        "Target should be in relative_features when use_relative_target=True"
    )

    # Create fake predictions (just copy scaled target)
    df_predictions = df_normalized.copy()
    df_predictions["predictions"] = df_predictions["my_target_column"]

    # Apply inverse transformation
    df_inverse = post_process_predictions(
        df_predictions,
        artifacts,
        experiment_config,
        prediction_column="predictions",
        target="my_target_column",
    )

    # Check that predictions are restored to original scale
    predictions_restored = df_inverse["predictions"]

    # Calculate metrics
    mae = np.abs(predictions_restored - original_target).mean()
    max_error = np.abs(predictions_restored - original_target).max()
    relative_error = np.abs(
        (predictions_restored - original_target) / original_target
    ).mean()

    print(f"MAE: {mae}")
    print(f"Max error: {max_error}")
    print(f"Mean relative error: {relative_error * 100:.2f}%")

    # The predictions should be very close to original values (since we just copied them)
    assert mae < 0.01, f"MAE should be very small, got {mae}"
    assert relative_error < 0.0001, (
        f"Relative error should be very small, got {relative_error}"
    )


def test_mixed_scaling_consistency():
    """Test that mixed scaling (some features relative, others per-basin) works correctly."""
    # Create sample data with multiple basins
    dates = pd.date_range(start="2020-01-01", end="2020-06-30", freq="D")
    data = []

    for basin in ["basin1", "basin2"]:
        for date in dates:
            day_of_year = date.dayofyear
            seasonal_factor = np.sin(2 * np.pi * day_of_year / 365)
            basin_offset = 10 if basin == "basin1" else 20

            data.append(
                {
                    "date": date,
                    "code": basin,
                    "SWE_mean": basin_offset
                    + 30 * seasonal_factor
                    + np.random.normal(0, 1),
                    "T_mean": basin_offset / 2
                    + 10 * seasonal_factor
                    + np.random.normal(0, 0.5),
                    "P_sum": basin_offset * 2
                    + 20 * seasonal_factor
                    + np.random.normal(0, 1),
                    "discharge": basin_offset * 5
                    + 50 * seasonal_factor
                    + np.random.normal(0, 2),
                }
            )

    df = pd.DataFrame(data)

    # Setup artifacts
    artifacts = FeatureProcessingArtifacts()
    artifacts.num_features = ["SWE_mean", "T_mean", "P_sum"]
    artifacts.static_features = []
    artifacts.selected_features = artifacts.num_features
    artifacts.target_col = "discharge"

    # Configure with selective scaling
    experiment_config = {
        "normalize": True,
        "normalization_type": "long_term_mean",
        "relative_scaling_vars": [
            "SWE",
            "T",
        ],  # Only SWE and T features use relative scaling
        "use_relative_target": False,  # Target uses per-basin scaling
    }

    # Apply normalization
    df_normalized, artifacts = _normalization_training(
        df.copy(), "discharge", experiment_config, artifacts
    )

    # Verify that relative features have mean ~1 per basin
    for basin in ["basin1", "basin2"]:
        basin_data = df_normalized[df_normalized["code"] == basin]

        # SWE and T should have mean ~1 (relative scaling)
        assert np.abs(basin_data["SWE_mean"].mean() - 1.0) < 0.1
        assert np.abs(basin_data["T_mean"].mean() - 1.0) < 0.1

        # P_sum should have mean ~0 and std ~1 (per-basin scaling)
        assert np.abs(basin_data["P_sum"].mean()) < 0.1
        assert np.abs(basin_data["P_sum"].std() - 1.0) < 0.1

        # Target should have mean ~0 and std ~1 (per-basin scaling)
        assert np.abs(basin_data["discharge"].mean()) < 0.1
        assert np.abs(basin_data["discharge"].std() - 1.0) < 0.1

    print("Mixed scaling test passed!")


if __name__ == "__main__":
    test_relative_target_scaling_with_non_matching_name()
    test_mixed_scaling_consistency()
    print("\nAll tests passed!")
