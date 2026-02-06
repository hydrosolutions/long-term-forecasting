"""
Backward compatibility tests for prediction_loader refactoring.

Tests ensure that:
1. Column naming conventions work with existing code patterns
2. Model constructors accept new optional parameters
3. Old code patterns continue to work without breaking changes
"""

import pandas as pd
import pytest

from lt_forecasting.forecast_models.deep_models.uncertainty_mixture import (
    UncertaintyMixtureModel,
)
from lt_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)
from lt_forecasting.forecast_models.SciRegressor import SciRegressor
from lt_forecasting.scr.prediction_loader import load_predictions_from_dataframe


@pytest.fixture
def sample_data():
    """Create sample time series data for model initialization."""
    return pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=30, freq="D"),
            "code": [1] * 30,
            "discharge": [100.0 + i for i in range(30)],
            "P": [5.0] * 30,
            "T": [10.0] * 30,
        }
    )


@pytest.fixture
def sample_static_data():
    """Create sample static basin data."""
    return pd.DataFrame(
        {
            "code": [1, 2],
            "area_km2": [1000.0, 2000.0],
            "lat": [42.0, 43.0],
            "lon": [70.0, 71.0],
        }
    )


@pytest.fixture
def minimal_configs():
    """Create minimal configuration dictionaries for model initialization."""
    general_config = {
        "model_name": "test_model",
        "target": "discharge",
        "test_years": [2023],
        "hparam_tuning_years": [2020],
        "models": ["xgboost"],
    }

    model_config = {
        "num_samples_val": 10,
        "metric": "nmse",
    }

    feature_config = {}

    path_config = {
        "model_home_path": "/tmp/test_models",
    }

    return general_config, model_config, feature_config, path_config


class TestColumnNamingBackwardCompatibility:
    """Test that column naming conventions work with existing code patterns."""

    def test_q_prefix_columns_work(self):
        """Verify predictions with Q_ prefix are handled correctly."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "Q_model1": [100.0, 110.0],
                "Q_model2": [95.0, 105.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(
            df, ["model1", "model2"]
        )

        # Verify Q_ prefix is preserved
        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert len(pred_cols) == 2

        # Verify DataFrame contains expected columns
        assert "Q_model1" in predictions.columns
        assert "Q_model2" in predictions.columns

    def test_no_prefix_columns_converted_to_q_prefix(self):
        """Verify columns without Q_ prefix are automatically prefixed."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "model1": [100.0, 110.0],
                "model2": [95.0, 105.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(
            df, ["model1", "model2"]
        )

        # Verify Q_ prefix was added
        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert len(pred_cols) == 2

        # Verify original columns were renamed
        assert "Q_model1" in predictions.columns
        assert "Q_model2" in predictions.columns
        assert "model1" not in predictions.columns
        assert "model2" not in predictions.columns

    def test_mixed_prefix_columns_handled_correctly(self):
        """Verify mixed prefix columns (some with Q_, some without) work."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "Q_model1": [100.0, 110.0],  # Already has prefix
                "model2": [95.0, 105.0],  # No prefix
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(
            df, ["Q_model1", "model2"]
        )

        # Verify both columns have Q_ prefix in output
        assert "Q_model1" in pred_cols
        assert "Q_model2" in pred_cols
        assert len(pred_cols) == 2

        # Verify DataFrame structure
        assert "Q_model1" in predictions.columns
        assert "Q_model2" in predictions.columns

    def test_dataframe_loading_with_and_without_prefix_produces_same_result(self):
        """Verify loading with/without prefix produces equivalent results."""
        # DataFrame with Q_ prefix
        df_with_prefix = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "Q_model1": [100.0, 110.0],
                "Q_model2": [95.0, 105.0],
            }
        )

        # DataFrame without Q_ prefix
        df_without_prefix = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "model1": [100.0, 110.0],
                "model2": [95.0, 105.0],
            }
        )

        # Load both
        preds1, cols1 = load_predictions_from_dataframe(
            df_with_prefix, ["model1", "model2"]
        )
        preds2, cols2 = load_predictions_from_dataframe(
            df_without_prefix, ["model1", "model2"]
        )

        # Verify same columns produced
        assert cols1 == cols2

        # Verify same data (after accounting for column names)
        pd.testing.assert_frame_equal(preds1, preds2)

    def test_q_obs_excluded_from_predictions(self):
        """Verify Q_obs column is not treated as a prediction column."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "Q_obs": [98.0, 108.0],  # Observations, not predictions
                "Q_model1": [100.0, 110.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(df, ["model1"])

        # Verify Q_obs is NOT in prediction columns
        assert "Q_obs" not in pred_cols
        assert len(pred_cols) == 1
        assert "Q_model1" in pred_cols


class TestModelConstructorCompatibility:
    """Test that all meta learner models accept base_predictors parameter."""

    def test_sci_regressor_accepts_base_predictors_none(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify SciRegressor accepts base_predictors=None without error."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Should not raise any errors
        model = SciRegressor(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=None,
            base_model_names=None,
        )

        # Verify model was created
        assert model is not None
        assert model._external_base_predictors is None
        assert model._external_base_model_names is None

    def test_sci_regressor_backward_compatible_without_base_predictors(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify SciRegressor works without base_predictors parameter (old API)."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Old API - without base_predictors parameter
        model = SciRegressor(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        # Verify model was created successfully
        assert model is not None
        assert hasattr(model, "_external_base_predictors")
        assert model._external_base_predictors is None

    def test_sci_regressor_accepts_external_predictions(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify SciRegressor accepts external predictions."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Create external predictions
        external_preds = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                "code": [1] * 30,
                "Q_model1": [100.0 + i for i in range(30)],
                "Q_model2": [95.0 + i for i in range(30)],
            }
        )

        model = SciRegressor(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=external_preds,
            base_model_names=["Q_model1", "Q_model2"],
        )

        # Verify external predictions were stored
        assert model._external_base_predictors is not None
        assert len(model._external_base_predictors) == 30
        assert model._external_base_model_names == ["Q_model1", "Q_model2"]

    def test_historical_meta_learner_backward_compatible(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify HistoricalMetaLearner works without base_predictors (old API)."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Old API - without base_predictors parameter
        # Note: HistoricalMetaLearner doesn't have base_predictors in its __init__
        # but inherits from BaseMetaLearner which does
        model = HistoricalMetaLearner(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        # Verify model was created successfully
        assert model is not None
        # HistoricalMetaLearner should have _external_base_predictors from parent
        assert hasattr(model, "_external_base_predictors")

    def test_uncertainty_mixture_accepts_base_predictors_none(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify UncertaintyMixtureModel accepts base_predictors=None."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Should not raise any errors
        model = UncertaintyMixtureModel(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=None,
            base_model_names=None,
        )

        # Verify model was created
        # UncertaintyMixtureModel stores these in parent class as _external_*
        assert model is not None
        assert model._external_base_predictors is None
        assert model._external_base_model_names is None

    def test_uncertainty_mixture_backward_compatible_without_base_predictors(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify UncertaintyMixtureModel works without base_predictors (old API)."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Old API - without base_predictors parameter
        model = UncertaintyMixtureModel(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        # Verify model was created successfully
        # UncertaintyMixtureModel stores these in parent class as _external_*
        assert model is not None
        assert hasattr(model, "_external_base_predictors")
        assert model._external_base_predictors is None

    def test_uncertainty_mixture_accepts_external_predictions(
        self, sample_data, sample_static_data, minimal_configs
    ):
        """Verify UncertaintyMixtureModel accepts external predictions."""
        general_config, model_config, feature_config, path_config = minimal_configs

        # Create external predictions
        external_preds = pd.DataFrame(
            {
                "date": pd.date_range("2020-01-01", periods=30, freq="D"),
                "code": [1] * 30,
                "Q_model1": [100.0 + i for i in range(30)],
                "Q_model2": [95.0 + i for i in range(30)],
            }
        )

        model = UncertaintyMixtureModel(
            data=sample_data,
            static_data=sample_static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=external_preds,
            base_model_names=["Q_model1", "Q_model2"],
        )

        # Verify external predictions were stored
        # UncertaintyMixtureModel stores these in parent class as _external_*
        assert model._external_base_predictors is not None
        assert len(model._external_base_predictors) == 30
        assert model._external_base_model_names == ["Q_model1", "Q_model2"]


class TestEdgeCasesBackwardCompatibility:
    """Test edge cases that may occur in production."""

    def test_empty_model_names_list(self):
        """Verify handling of empty model names list."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_model1": [100.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(df, [])

        # Should return empty prediction columns
        assert len(pred_cols) == 0
        assert "date" in predictions.columns
        assert "code" in predictions.columns

    def test_model_name_with_q_prefix_in_input(self):
        """Verify handling of model names that already include Q_ prefix."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_model1": [100.0],
            }
        )

        # Pass model name with Q_ prefix
        predictions, pred_cols = load_predictions_from_dataframe(df, ["Q_model1"])

        # Should not add extra Q_ prefix
        assert "Q_model1" in pred_cols
        assert len(pred_cols) == 1
        assert "Q_Q_model1" not in pred_cols  # Should NOT double prefix

    def test_duplicate_model_names_handled(self):
        """Verify handling of duplicate model names."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_model1": [100.0],
            }
        )

        # Request same model twice
        predictions, pred_cols = load_predictions_from_dataframe(
            df, ["model1", "model1"]
        )

        # Should have 2 entries (even if duplicate)
        assert len(pred_cols) == 2
        assert pred_cols == ["Q_model1", "Q_model1"]

    def test_special_characters_in_model_names(self):
        """Verify handling of special characters in model names."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_model-v1.0": [100.0],
                "Q_model_test_2": [95.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(
            df, ["model-v1.0", "model_test_2"]
        )

        # Should handle special characters correctly
        assert "Q_model-v1.0" in pred_cols
        assert "Q_model_test_2" in pred_cols
        assert len(pred_cols) == 2

    def test_numeric_model_names(self):
        """Verify handling of numeric model names."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "Q_1": [100.0],
                "Q_2": [95.0],
            }
        )

        predictions, pred_cols = load_predictions_from_dataframe(df, ["1", "2"])

        # Should handle numeric names correctly
        assert "Q_1" in pred_cols
        assert "Q_2" in pred_cols
        assert len(pred_cols) == 2


class TestDataTypeCompatibility:
    """Test that data types are handled consistently with old code."""

    def test_date_string_converted_to_datetime(self):
        """Verify date strings are converted to datetime."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": [1, 1],
                "model1": [100.0, 110.0],
            }
        )

        predictions, _ = load_predictions_from_dataframe(df, ["model1"])

        # Verify date column is datetime
        assert pd.api.types.is_datetime64_any_dtype(predictions["date"])

    def test_code_converted_to_int(self):
        """Verify code is converted to integer type."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "code": ["1", "2"],  # String codes
                "model1": [100.0, 110.0],
            }
        )

        predictions, _ = load_predictions_from_dataframe(df, ["model1"])

        # Verify code column is integer
        assert pd.api.types.is_integer_dtype(predictions["code"])

    def test_float_predictions_preserved(self):
        """Verify float prediction values are preserved correctly."""
        df = pd.DataFrame(
            {
                "date": ["2024-01-01"],
                "code": [1],
                "model1": [123.456789],
            }
        )

        predictions, _ = load_predictions_from_dataframe(df, ["model1"])

        # Verify float precision is preserved
        assert predictions["Q_model1"].iloc[0] == 123.456789
        assert pd.api.types.is_float_dtype(predictions["Q_model1"])
