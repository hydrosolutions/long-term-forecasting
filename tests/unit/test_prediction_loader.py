"""
Unit tests for prediction_loader module.

Tests centralized prediction loading utilities for:
- Loading from filesystem
- Loading from DataFrames
- Area conversions
- Duplicate handling
- Column standardization
"""

import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from lt_forecasting.scr.prediction_loader import (
    apply_area_conversion,
    handle_duplicate_predictions,
    load_predictions_from_dataframe,
    load_predictions_from_filesystem,
    standardize_prediction_columns,
)


@pytest.fixture
def sample_prediction_csv(tmp_path):
    """Create a sample prediction CSV file for testing."""
    model_dir = tmp_path / "model1"
    model_dir.mkdir()

    data = {
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "code": [1, 1, 2],
        "Q_model1": [100.0, 110.0, 120.0],
    }
    df = pd.DataFrame(data)

    csv_path = model_dir / "predictions.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_static_data():
    """Create sample static data with basin areas."""
    return pd.DataFrame({"code": [1, 2, 3], "area_km2": [1000.0, 2000.0, 1500.0]})


@pytest.fixture
def sample_predictions_df():
    """Create sample predictions DataFrame."""
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "code": [1, 1, 2],
            "Q_model1": [100.0, 110.0, 120.0],
            "Q_model2": [95.0, 105.0, 115.0],
        }
    )


class TestLoadPredictionsFromFilesystem:
    """Tests for load_predictions_from_filesystem function."""

    def test_load_single_model(self, sample_prediction_csv):
        """Test loading predictions from a single model."""
        paths = [str(sample_prediction_csv)]
        predictions, pred_cols = load_predictions_from_filesystem(paths)

        assert isinstance(predictions, pd.DataFrame)
        assert len(pred_cols) == 1
        assert "Q_model1" in pred_cols
        assert "date" in predictions.columns
        assert "code" in predictions.columns
        assert len(predictions) == 3

    def test_load_multiple_models(self, tmp_path):
        """Test loading predictions from multiple models."""
        # Create two model directories
        model1_dir = tmp_path / "model1"
        model2_dir = tmp_path / "model2"
        model1_dir.mkdir()
        model2_dir.mkdir()

        # Create prediction files
        data1 = {
            "date": ["2024-01-01", "2024-01-02"],
            "code": [1, 1],
            "Q_model1": [100.0, 110.0],
        }
        data2 = {
            "date": ["2024-01-01", "2024-01-02"],
            "code": [1, 1],
            "Q_model2": [95.0, 105.0],
        }

        pd.DataFrame(data1).to_csv(model1_dir / "predictions.csv", index=False)
        pd.DataFrame(data2).to_csv(model2_dir / "predictions.csv", index=False)

        paths = [str(model1_dir / "predictions.csv"), str(model2_dir / "predictions.csv")]
        predictions, pred_cols = load_predictions_from_filesystem(paths)

        assert len(pred_cols) == 2
        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert len(predictions) == 2
        assert all(col in predictions.columns for col in pred_cols)

    def test_load_with_directory_path(self, tmp_path):
        """Test loading when passing directory path (should auto-append predictions.csv)."""
        model_dir = tmp_path / "model1"
        model_dir.mkdir()

        data = {
            "date": ["2024-01-01"],
            "code": [1],
            "Q_model1": [100.0],
        }
        pd.DataFrame(data).to_csv(model_dir / "predictions.csv", index=False)

        # Pass directory path, not file path
        paths = [str(model_dir)]
        predictions, pred_cols = load_predictions_from_filesystem(paths)

        assert len(pred_cols) == 1
        assert "Q_model1" in pred_cols

    def test_load_with_inner_join(self, tmp_path):
        """Test that inner join only keeps common date-code pairs."""
        model1_dir = tmp_path / "model1"
        model2_dir = tmp_path / "model2"
        model1_dir.mkdir()
        model2_dir.mkdir()

        # Model1 has dates 1, 2, 3
        data1 = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "code": [1, 1, 1],
            "Q_model1": [100.0, 110.0, 120.0],
        }
        # Model2 only has dates 2, 3, 4
        data2 = {
            "date": ["2024-01-02", "2024-01-03", "2024-01-04"],
            "code": [1, 1, 1],
            "Q_model2": [95.0, 105.0, 115.0],
        }

        pd.DataFrame(data1).to_csv(model1_dir / "predictions.csv", index=False)
        pd.DataFrame(data2).to_csv(model2_dir / "predictions.csv", index=False)

        paths = [str(model1_dir / "predictions.csv"), str(model2_dir / "predictions.csv")]
        predictions, _ = load_predictions_from_filesystem(paths, join_type="inner")

        # Only dates 2 and 3 should be present (intersection)
        assert len(predictions) == 2
        assert predictions["date"].min() == pd.Timestamp("2024-01-02")
        assert predictions["date"].max() == pd.Timestamp("2024-01-03")

    def test_load_with_left_join(self, tmp_path):
        """Test that left join keeps all dates from first model."""
        model1_dir = tmp_path / "model1"
        model2_dir = tmp_path / "model2"
        model1_dir.mkdir()
        model2_dir.mkdir()

        # Model1 has dates 1, 2, 3
        data1 = {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "code": [1, 1, 1],
            "Q_model1": [100.0, 110.0, 120.0],
        }
        # Model2 only has dates 2, 3
        data2 = {
            "date": ["2024-01-02", "2024-01-03"],
            "code": [1, 1],
            "Q_model2": [95.0, 105.0],
        }

        pd.DataFrame(data1).to_csv(model1_dir / "predictions.csv", index=False)
        pd.DataFrame(data2).to_csv(model2_dir / "predictions.csv", index=False)

        paths = [str(model1_dir / "predictions.csv"), str(model2_dir / "predictions.csv")]
        predictions, _ = load_predictions_from_filesystem(paths, join_type="left")

        # All 3 dates from model1 should be present
        assert len(predictions) == 3
        assert predictions["date"].min() == pd.Timestamp("2024-01-01")

    def test_missing_prediction_column(self, tmp_path):
        """Test handling of missing Q_{model_name} column."""
        model_dir = tmp_path / "model1"
        model_dir.mkdir()

        # Create CSV without Q_model1 column
        data = {
            "date": ["2024-01-01"],
            "code": [1],
            "wrong_column": [100.0],
        }
        pd.DataFrame(data).to_csv(model_dir / "predictions.csv", index=False)

        paths = [str(model_dir / "predictions.csv")]

        # Should raise ValueError when no valid predictions found
        with pytest.raises(ValueError, match="No valid predictions found"):
            load_predictions_from_filesystem(paths)

    def test_nonexistent_file(self, tmp_path):
        """Test handling of nonexistent file."""
        paths = [str(tmp_path / "nonexistent" / "predictions.csv")]

        # Should raise ValueError when no valid files found
        with pytest.raises(ValueError, match="No valid predictions found"):
            load_predictions_from_filesystem(paths)

    def test_empty_paths_list(self):
        """Test that empty paths list raises error."""
        with pytest.raises(ValueError, match="At least one path must be provided"):
            load_predictions_from_filesystem([])


class TestLoadPredictionsFromDataframe:
    """Tests for load_predictions_from_dataframe function."""

    def test_load_with_q_prefix(self):
        """Test loading DataFrame with Q_ prefixed columns."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "Q_model1": [100.0, 110.0],
                "Q_model2": [95.0, 105.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(df, ["model1", "model2"])

        assert len(pred_cols) == 2
        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert len(predictions) == 2

    def test_load_without_q_prefix(self):
        """Test loading DataFrame without Q_ prefix (should add it)."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "model1": [100.0, 110.0],
                "model2": [95.0, 105.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(df, ["model1", "model2"])

        assert len(pred_cols) == 2
        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        # Check columns were renamed
        assert "Q_model1" in predictions.columns
        assert "Q_model2" in predictions.columns

    def test_missing_required_columns(self):
        """Test that missing date/code columns raises error."""
        df = pd.DataFrame({"model1": [100.0, 110.0]})

        with pytest.raises(ValueError, match="missing required columns"):
            load_predictions_from_dataframe(df, ["model1"])

    def test_missing_model_column(self):
        """Test that missing model column raises error."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_model1": [100.0],
            }
        )

        with pytest.raises(ValueError, match="not found in DataFrame"):
            load_predictions_from_dataframe(df, ["model1", "model2"])

    def test_date_conversion(self):
        """Test that date strings are converted to datetime."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "model1": [100.0, 110.0],
            }
        )

        predictions, _ = load_predictions_from_dataframe(df, ["model1"])

        assert pd.api.types.is_datetime64_any_dtype(predictions["date"])


class TestApplyAreaConversion:
    """Tests for apply_area_conversion function."""

    def test_basic_conversion(self, sample_predictions_df, sample_static_data):
        """Test basic area conversion."""
        pred_cols = ["Q_model1", "Q_model2"]
        result = apply_area_conversion(
            sample_predictions_df, sample_static_data, pred_cols
        )

        # Check conversion formula: value * area / 86.4
        # For code 1: area = 1000, so multiply by 1000/86.4 ≈ 11.574
        code1_original = sample_predictions_df[sample_predictions_df["code"] == 1][
            "Q_model1"
        ].iloc[0]
        code1_converted = result[result["code"] == 1]["Q_model1"].iloc[0]

        expected = code1_original * 1000.0 / 86.4
        assert abs(code1_converted - expected) < 0.01

    def test_different_areas_per_basin(self, sample_static_data):
        """Test that different basins get different conversions."""
        predictions = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "code": [1, 2],
                "Q_model1": [100.0, 100.0],  # Same input value
            }
        )

        result = apply_area_conversion(predictions, sample_static_data, ["Q_model1"])

        # Code 1 has area 1000, code 2 has area 2000
        # So code 2 should have double the conversion factor
        code1_value = result[result["code"] == 1]["Q_model1"].iloc[0]
        code2_value = result[result["code"] == 2]["Q_model1"].iloc[0]

        assert code2_value == pytest.approx(code1_value * 2.0)

    def test_missing_code_in_static_data(self, sample_predictions_df):
        """Test handling of codes not in static data."""
        incomplete_static = pd.DataFrame({"code": [1], "area_km2": [1000.0]})

        # Should not raise error, but warn (code 2 will be skipped)
        result = apply_area_conversion(
            sample_predictions_df, incomplete_static, ["Q_model1"]
        )

        # Code 1 should be converted, code 2 should remain unchanged
        assert len(result) == 3

    def test_missing_required_columns(self, sample_predictions_df):
        """Test that missing required columns raises error."""
        bad_static = pd.DataFrame({"code": [1]})  # Missing area_km2

        with pytest.raises(ValueError, match="must have 'code' and 'area_km2'"):
            apply_area_conversion(sample_predictions_df, bad_static, ["Q_model1"])


class TestHandleDuplicatePredictions:
    """Tests for handle_duplicate_predictions function."""

    def test_no_duplicates(self, sample_predictions_df):
        """Test that DataFrames without duplicates are unchanged."""
        result = handle_duplicate_predictions(sample_predictions_df)

        assert len(result) == len(sample_predictions_df)
        pd.testing.assert_frame_equal(result, sample_predictions_df)

    def test_with_duplicates(self):
        """Test averaging of duplicate predictions."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
                "code": [1, 1, 1],
                "Q_model1": [100.0, 110.0, 120.0],  # First two should average to 105
            }
        )

        result = handle_duplicate_predictions(df, pred_cols=["Q_model1"])

        # Should have 2 rows after deduplication
        assert len(result) == 2

        # Check that duplicate values were averaged
        date1_value = result[result["date"] == "2024-01-01"]["Q_model1"].iloc[0]
        assert date1_value == 105.0  # (100 + 110) / 2

    def test_auto_detect_pred_cols(self):
        """Test automatic detection of prediction columns."""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "code": [1, 1],
                "Q_model1": [100.0, 110.0],
                "Q_model2": [90.0, 100.0],
                "Q_obs": [95.0, 95.0],  # Should be excluded
            }
        )

        result = handle_duplicate_predictions(df)  # Don't specify pred_cols

        assert len(result) == 1
        assert result["Q_model1"].iloc[0] == 105.0
        assert result["Q_model2"].iloc[0] == 95.0
        assert result["Q_obs"].iloc[0] == 95.0  # Should not be averaged


class TestStandardizePredictionColumns:
    """Tests for standardize_prediction_columns function."""

    def test_already_standardized(self):
        """Test DataFrame with already standardized columns."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_model1": [100.0],
                "Q_model2": [95.0],
            }
        )

        result, pred_cols = standardize_prediction_columns(df)

        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert len(pred_cols) == 2

    def test_add_prefix(self):
        """Test adding Q_ prefix to columns."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "model1": [100.0],
                "model2": [95.0],
            }
        )

        result, pred_cols = standardize_prediction_columns(df, ensure_prefix=True)

        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert "Q_model1" in result.columns
        assert "Q_model2" in result.columns
        assert "model1" not in result.columns

    def test_exclude_special_columns(self):
        """Test that date, code, Q_obs are excluded."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_obs": [98.0],
                "Q_model1": [100.0],
            }
        )

        result, pred_cols = standardize_prediction_columns(df)

        assert "Q_model1" in pred_cols
        assert "Q_obs" not in pred_cols
        assert "date" not in pred_cols
        assert "code" not in pred_cols
        assert len(pred_cols) == 1

    def test_without_prefix_requirement(self):
        """Test without requiring Q_ prefix."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "model1": [100.0],
                "Q_model2": [95.0],
            }
        )

        result, pred_cols = standardize_prediction_columns(df, ensure_prefix=False)

        # Only Q_model2 should be detected
        assert len(pred_cols) == 1
        assert "Q_model2" in pred_cols
