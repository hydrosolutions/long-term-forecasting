"""
Comprehensive tests for sci_utils module.

This module tests all scientific utility functions including model creation,
training, hyperparameter optimization, and feature importance extraction.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from monthly_forecasting.scr import sci_utils
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor


class TestGetModel:
    """Test the get_model function for creating different model types."""

    def test_get_model_xgb(self):
        """Test XGBoost model creation."""
        params = {"n_estimators": 100, "max_depth": 6}
        model = sci_utils.get_model("xgb", params)

        assert isinstance(model, XGBRegressor)
        assert model.n_estimators == 100
        assert model.max_depth == 6

    def test_get_model_lgbm(self):
        """Test LightGBM model creation."""
        params = {"n_estimators": 200, "num_leaves": 31}
        model = sci_utils.get_model("lgbm", params)

        assert isinstance(model, LGBMRegressor)
        assert model.n_estimators == 200
        assert model.num_leaves == 31
        assert model.objective == "regression"
        assert model.metric == "rmse"

    def test_get_model_catboost(self):
        """Test CatBoost model creation."""
        params = {"iterations": 100, "depth": 6}
        cat_features = ["feature1", "feature2"]
        model = sci_utils.get_model("catboost", params, cat_features)

        assert isinstance(model, CatBoostRegressor)
        # CatBoost parameters are accessible through get_params()
        model_params = model.get_params()
        assert model_params["iterations"] == 100
        assert model_params["depth"] == 6
        # Test creation was successful
        assert model is not None

    def test_get_model_rf(self):
        """Test Random Forest model creation."""
        params = {"n_estimators": 100, "max_depth": 10}
        model = sci_utils.get_model("rf", params)

        assert isinstance(model, RandomForestRegressor)
        assert model.n_estimators == 100
        assert model.max_depth == 10

    def test_get_model_mlp(self):
        """Test MLP model creation."""
        params = {"hidden_layer_sizes": (100, 50), "max_iter": 1000}
        model = sci_utils.get_model("mlp", params)

        assert isinstance(model, MLPRegressor)
        assert model.hidden_layer_sizes == (100, 50)
        assert model.max_iter == 1000

    def test_get_model_invalid_type(self):
        """Test error handling for invalid model type."""
        with pytest.raises(ValueError, match="Invalid model type"):
            sci_utils.get_model("invalid_model", {})

    def test_get_model_empty_params(self):
        """Test model creation with empty parameters."""
        model = sci_utils.get_model("xgb", {})
        assert isinstance(model, XGBRegressor)


class TestFitModel:
    """Test the fit_model function for training models."""

    def create_sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_fit_model_basic(self):
        """Test basic model fitting."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("xgb", {"n_estimators": 10})

        fitted_model = sci_utils.fit_model(model, X, y)

        assert fitted_model is not None
        assert hasattr(fitted_model, "predict")

        # Test that model can make predictions
        predictions = fitted_model.predict(X)
        assert len(predictions) == len(y)

    def test_fit_model_with_validation(self):
        """Test model fitting with validation split."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("rf", {"n_estimators": 10})

        fitted_model = sci_utils.fit_model(model, X, y, val_fraction=0.2)

        assert fitted_model is not None
        predictions = fitted_model.predict(X)
        assert len(predictions) == len(y)

    @patch("monthly_forecasting.scr.sci_utils.logger")
    def test_fit_model_logs_performance(self, mock_logger):
        """Test that model fitting logs performance metrics."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("xgb", {"n_estimators": 10})

        sci_utils.fit_model(model, X, y)

        # Check that logger.info was called with R² information
        mock_logger.info.assert_called()
        call_args = [call[0][0] for call in mock_logger.info.call_args_list]
        assert any("Model R²" in arg for arg in call_args)


class TestGetFeatureImportance:
    """Test the get_feature_importance function."""

    def create_sample_data(self):
        """Create sample data for testing feature importance."""
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y = pd.Series(np.random.randn(100))
        return X, y

    def test_get_feature_importance_xgb(self):
        """Test feature importance extraction for XGBoost."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("xgb", {"n_estimators": 10})
        model.fit(X, y)

        importance_df = sci_utils.get_feature_importance(model, X.columns.tolist())

        assert isinstance(importance_df, pd.DataFrame)
        assert "feature" in importance_df.columns
        assert "importance" in importance_df.columns
        assert len(importance_df) == 3
        # Check that all features are present, but not order since it's sorted by importance
        assert set(importance_df["feature"].tolist()) == {
            "feature1",
            "feature2",
            "feature3",
        }

    def test_get_feature_importance_lgbm(self):
        """Test feature importance extraction for LightGBM."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("lgbm", {"n_estimators": 10})
        model.fit(X, y)

        importance_df = sci_utils.get_feature_importance(model, X.columns.tolist())

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 3
        assert all(col in importance_df.columns for col in ["feature", "importance"])

    def test_get_feature_importance_rf(self):
        """Test feature importance extraction for Random Forest."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("rf", {"n_estimators": 10})
        model.fit(X, y)

        importance_df = sci_utils.get_feature_importance(model, X.columns.tolist())

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 3
        assert all(col in importance_df.columns for col in ["feature", "importance"])

    def test_get_feature_importance_none_model(self):
        """Test feature importance extraction with None model."""
        importance_df = sci_utils.get_feature_importance(None)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 0

    def test_get_feature_importance_unsupported_model(self):
        """Test feature importance extraction with unsupported model."""

        # Create a simple class without feature importance attributes
        class UnsupportedModel:
            pass

        unsupported_model = UnsupportedModel()

        importance_df = sci_utils.get_feature_importance(unsupported_model)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 0

    def test_get_feature_importance_without_feature_names(self):
        """Test feature importance extraction without providing feature names."""
        X, y = self.create_sample_data()
        model = sci_utils.get_model("xgb", {"n_estimators": 10})
        model.fit(X, y)

        importance_df = sci_utils.get_feature_importance(model)

        assert isinstance(importance_df, pd.DataFrame)
        assert len(importance_df) == 3
        # Should have generated feature names
        assert all(importance_df["feature"].str.startswith("feature"))


class TestOptimizeHyperparams:
    """Test the optimize_hyperparams function."""

    def create_sample_data(self):
        """Create sample data for hyperparameter optimization."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randn(100))
        X_val = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            }
        )
        y_val = pd.Series(np.random.randn(50))
        return X_train, y_train, X_val, y_val

    @patch("monthly_forecasting.scr.sci_utils.optuna.create_study")
    def test_optimize_hyperparams_xgb(self, mock_create_study):
        """Test hyperparameter optimization for XGBoost."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock optuna study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.params = {"n_estimators": 100, "max_depth": 6}
        mock_study.best_trial = mock_trial
        mock_create_study.return_value = mock_study

        result = sci_utils.optimize_hyperparams(
            X_train, y_train, X_val, y_val, model_type="xgb", n_trials=5
        )

        assert result == {"n_estimators": 100, "max_depth": 6}
        mock_create_study.assert_called_once_with(direction="maximize")
        mock_study.optimize.assert_called_once()

    @patch("monthly_forecasting.scr.sci_utils.optuna.create_study")
    def test_optimize_hyperparams_lgbm(self, mock_create_study):
        """Test hyperparameter optimization for LightGBM."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock optuna study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.params = {"n_estimators": 200, "num_leaves": 31}
        mock_study.best_trial = mock_trial
        mock_create_study.return_value = mock_study

        result = sci_utils.optimize_hyperparams(
            X_train, y_train, X_val, y_val, model_type="lgbm", n_trials=5
        )

        assert result == {"n_estimators": 200, "num_leaves": 31}

    @patch("monthly_forecasting.scr.sci_utils.optuna.create_study")
    def test_optimize_hyperparams_catboost(self, mock_create_study):
        """Test hyperparameter optimization for CatBoost."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock optuna study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.params = {"iterations": 100, "depth": 6}
        mock_study.best_trial = mock_trial
        mock_create_study.return_value = mock_study

        result = sci_utils.optimize_hyperparams(
            X_train,
            y_train,
            X_val,
            y_val,
            model_type="catboost",
            cat_features=["feature1"],
            n_trials=5,
        )

        assert result == {"iterations": 100, "depth": 6}

    @patch("monthly_forecasting.scr.sci_utils.optuna.create_study")
    def test_optimize_hyperparams_mlp(self, mock_create_study):
        """Test hyperparameter optimization for MLP."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock optuna study
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.params = {"hidden_layer_sizes": (100,), "alpha": 0.001}
        mock_study.best_trial = mock_trial
        mock_create_study.return_value = mock_study

        result = sci_utils.optimize_hyperparams(
            X_train, y_train, X_val, y_val, model_type="mlp", n_trials=5
        )

        assert result == {"hidden_layer_sizes": (100,), "alpha": 0.001}

    def test_optimize_hyperparams_invalid_model(self):
        """Test error handling for invalid model type."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        with pytest.raises(
            ValueError, match="Hyperparameter optimization not supported"
        ):
            sci_utils.optimize_hyperparams(
                X_train, y_train, X_val, y_val, model_type="invalid_model"
            )

    @patch("monthly_forecasting.scr.sci_utils.optuna.create_study")
    @patch("monthly_forecasting.scr.sci_utils.os.makedirs")
    def test_optimize_hyperparams_with_save_path(
        self, mock_makedirs, mock_create_study
    ):
        """Test hyperparameter optimization with save path."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock optuna study and visualization
        mock_study = Mock()
        mock_trial = Mock()
        mock_trial.params = {"n_estimators": 100}
        mock_study.best_trial = mock_trial
        mock_create_study.return_value = mock_study

        with patch(
            "monthly_forecasting.scr.sci_utils.optuna.visualization"
        ) as mock_viz:
            mock_fig = Mock()
            mock_viz.plot_optimization_history.return_value = mock_fig
            mock_viz.plot_param_importances.return_value = mock_fig
            mock_viz.plot_contour.return_value = mock_fig

            result = sci_utils.optimize_hyperparams(
                X_train,
                y_train,
                X_val,
                y_val,
                model_type="xgb",
                n_trials=5,
                save_path="/tmp/test",
            )

            mock_makedirs.assert_called_once_with("/tmp/test", exist_ok=True)
            mock_viz.plot_optimization_history.assert_called_once()
            mock_viz.plot_param_importances.assert_called_once()


class TestObjectiveFunctions:
    """Test the objective functions for hyperparameter optimization."""

    def create_sample_data(self):
        """Create sample data for testing objective functions."""
        np.random.seed(42)
        X_train = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        y_train = pd.Series(np.random.randn(100))
        X_val = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
                "feature3": np.random.randn(50),
            }
        )
        y_val = pd.Series(np.random.randn(50))
        return X_train, y_train, X_val, y_val

    def test_objective_xgb(self):
        """Test XGBoost objective function."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = [
            100,
            6,
            1,
        ]  # n_estimators, max_depth, min_child_weight
        mock_trial.suggest_float.side_effect = [
            0.1,
            0.8,
            0.8,
            0.1,
            0.1,
            0.1,
        ]  # learning_rate, subsample, colsample_bytree, gamma, lambda, alpha

        r2_score = sci_utils._objective_xgb(mock_trial, X_train, y_train, X_val, y_val)

        assert isinstance(r2_score, float)
        assert -1 <= r2_score <= 1  # R² score should be between -1 and 1

    def test_objective_lgbm(self):
        """Test LightGBM objective function."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = [
            100,
            31,
            6,
            5,
        ]  # n_estimators, num_leaves, max_depth, min_child_samples
        mock_trial.suggest_float.side_effect = [
            0.1,
            0.8,
            0.8,
            0.1,
            0.1,
        ]  # learning_rate, subsample, colsample_bytree, lambda_l1, lambda_l2

        r2_score = sci_utils._objective_lgbm(mock_trial, X_train, y_train, X_val, y_val)

        assert isinstance(r2_score, float)
        assert -1 <= r2_score <= 1

    def test_objective_catboost(self):
        """Test CatBoost objective function."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = [
            100,
            6,
            128,
        ]  # iterations, depth, border_count
        mock_trial.suggest_float.side_effect = [
            0.1,
            0.5,
            0.5,
        ]  # learning_rate, l2_leaf_reg, bagging_temperature

        r2_score = sci_utils._objective_catboost(
            mock_trial, X_train, y_train, X_val, y_val, []
        )

        assert isinstance(r2_score, float)
        assert -1 <= r2_score <= 1

    def test_objective_mlp(self):
        """Test MLP objective function."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = [100, 500]  # hidden_layer_sizes, max_iter
        mock_trial.suggest_categorical.side_effect = [
            "relu",
            "adam",
            "constant",
        ]  # activation, solver, learning_rate
        mock_trial.suggest_float.return_value = 0.001  # alpha

        r2_score = sci_utils._objective_mlp(mock_trial, X_train, y_train, X_val, y_val)

        assert isinstance(r2_score, float)
        assert -1 <= r2_score <= 1

    def test_objective_functions_with_normalization(self):
        """Test objective functions with normalization artifacts."""
        X_train, y_train, X_val, y_val = self.create_sample_data()

        # Mock artifacts and config for normalization
        mock_artifacts = Mock()
        mock_artifacts.scaler = {"target": (0.0, 1.0)}
        experiment_config = {"normalize": True, "normalization_type": "global"}

        # Mock trial
        mock_trial = Mock()
        mock_trial.suggest_int.side_effect = [100, 6, 1]
        mock_trial.suggest_float.side_effect = [0.1, 0.8, 0.8, 0.1, 0.1, 0.1]

        with patch(
            "monthly_forecasting.scr.FeatureProcessingArtifacts.post_process_predictions"
        ) as mock_post_process:
            # Mock post_process_predictions to return a DataFrame
            mock_post_process.return_value = pd.DataFrame(
                {"prediction": np.random.randn(len(y_val)), "target": y_val.values}
            )

            r2_score = sci_utils._objective_xgb(
                mock_trial,
                X_train,
                y_train,
                X_val,
                y_val,
                artifacts=mock_artifacts,
                experiment_config=experiment_config,
                target="target",
            )

            assert isinstance(r2_score, float)
            mock_post_process.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
