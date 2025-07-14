"""
Modular SciRegressor utilities for forecasting models.

This module provides utilities for training and evaluating various regression models
including XGBoost, LightGBM, CatBoost, Random Forest, and others. It supports
leave-one-year-out cross-validation, hyperparameter optimization, and feature processing.
"""

import os
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import datetime

# ML libraries
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Tree-based models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor

# Hyperparameter optimization
import optuna

# Custom modules
from . import data_utils as du

# Shared logging
import logging
from ..log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


def get_model(
    model_type: str, params: Dict[str, Any], cat_features: Optional[List[str]] = None
):
    """
    Create a model instance based on type and parameters.

    Args:
        model_type: Type of model ('xgb', 'lgbm', 'catboost', 'rf', etc.)
        params: Model parameters
        cat_features: Categorical features for CatBoost

    Returns:
        Initialized model instance

    Raises:
        ValueError: If model_type is not supported
    """
    model_mapping = {
        "xgb": lambda p: XGBRegressor(**p),
        "lgbm": lambda p: LGBMRegressor(
            objective="regression",
            metric="rmse",
            boosting_type="gbdt",
            random_state=42,
            verbose=-1,
            **p,
        ),
        "svr": lambda p: SVR(**p),
        "catboost": lambda p: CatBoostRegressor(
            **p,
            cat_features=cat_features or [],
            verbose=0,  # Suppress CatBoost output),
        ),
        "rf": lambda p: RandomForestRegressor(**p),
        "gradient_boosting": lambda p: GradientBoostingRegressor(**p),
        "mlp": lambda p: MLPRegressor(**p),
    }

    if model_type not in model_mapping:
        raise ValueError(
            f"Invalid model type: {model_type}. Supported types: {list(model_mapping.keys())}"
        )

    return model_mapping[model_type](params)


def fit_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = "xgb",
    val_fraction: float = 0.1,
) -> Any:
    """
    Fit a model and log training performance.

    Args:
        model: Model instance
        X: Features
        y: Target values

    Returns:
        Fitted model
    """
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_fraction, random_state=42
    )

    # Fit model without early stopping - iterations are tuned directly
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    logger.info(f"Model R²: {r2_score(y_val, y_pred):.4f}")
    return model


def get_feature_importance(
    model, feature_names: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Extract and visualize feature importance from a trained model.

    Args:
        model: Trained model with feature importance
        feature_names: Optional list of feature names
        save_path: Optional path to save the plot

    Returns:
        DataFrame with feature importance sorted by importance
    """
    if model is None:
        logger.warning("Model is None. Returning empty DataFrame.")
        return pd.DataFrame()

    # Extract feature importance
    if isinstance(model, XGBRegressor):
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = model.get_booster().feature_names
    elif isinstance(model, LGBMRegressor):
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = model.booster_.feature_name()
    elif hasattr(model, "get_feature_importance") and hasattr(model, "feature_names_"):
        # CatBoost models
        importance = model.get_feature_importance()
        if feature_names is None:
            feature_names = model.feature_names_
    elif hasattr(model, "feature_importances_"):
        # Generic models with feature_importances_ attribute
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importance))]
    else:
        logger.warning(
            f"Model type {type(model)} not supported for feature importance extraction."
        )
        return pd.DataFrame()

    # Create DataFrame
    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": importance}
    ).sort_values("importance", ascending=False)

    return feature_importance


def optimize_hyperparams(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str,
    cat_features: Optional[List[str]] = [],
    n_trials: int = 100,
    save_path: Optional[str] = None,
    artifacts: Optional[Any] = None,
    experiment_config: Optional[Dict[str, Any]] = None,
    target: Optional[str] = None,
    basin_codes: Optional[pd.Series] = None,
    val_dates: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna without early stopping.

    Args:
        X_train: Training features
        y_train: Training target values
        X_val: Validation features
        y_val: Validation target values
        model_type: Type of model to optimize ('xgb', 'lgbm', 'catboost', 'mlp')
        cat_features: List of categorical features for CatBoost
        n_trials: Number of optimization trials
        save_path: Optional path to save optimization results and plots
        artifacts: Optional FeatureProcessingArtifacts for inverse scaling
        experiment_config: Optional experiment configuration for normalization settings
        target: Optional target column name for inverse scaling
        basin_codes: Optional Series of basin codes for per-basin normalization
        val_dates: Optional Series of dates for validation data (needed for long_term_mean normalization)

    Returns:
        Dictionary of best hyperparameters
    """

    X_train = X_train.copy()
    X_val = X_val.copy()
    y_train = y_train.copy()
    y_val = y_val.copy()

    # Define objective function based on model type
    def objective(trial):
        if model_type == "xgb":
            return _objective_xgb(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                artifacts,
                experiment_config,
                target,
                basin_codes,
                val_dates,
            )
        elif model_type == "lgbm":
            return _objective_lgbm(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                artifacts,
                experiment_config,
                target,
                basin_codes,
                val_dates,
            )
        elif model_type == "catboost":
            return _objective_catboost(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                cat_features,
                artifacts,
                experiment_config,
                target,
                basin_codes,
                val_dates,
            )
        elif model_type == "mlp":
            return _objective_mlp(
                trial,
                X_train,
                y_train,
                X_val,
                y_val,
                artifacts,
                experiment_config,
                target,
                basin_codes,
                val_dates,
            )
        else:
            raise ValueError(
                f"Hyperparameter optimization not supported for model type: {model_type}"
            )

    # Run optimization
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    # Log results
    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info("  Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")

    # Save optimization plots if path provided
    if save_path:
        os.makedirs(save_path, exist_ok=True)

        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_html(os.path.join(save_path, "optimization_history.html"))

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(save_path, "param_importances.html"))

        try:
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(os.path.join(save_path, "contour.html"))
        except Exception as e:
            logger.warning(f"Could not create contour plot: {e}")

    return study.best_trial.params


# Objective functions for different models
def _objective_xgb(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    artifacts=None,
    experiment_config=None,
    target=None,
    basin_codes=None,
    val_dates=None,
):
    """Optuna objective function for XGBoost without early stopping."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "lambda": trial.suggest_float("lambda", 1e-3, 10.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "n_jobs": -1,
        "verbosity": 0,
    }

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_val)

    # If normalization is enabled and artifacts are provided, inverse transform for R2 calculation
    if (
        experiment_config
        and experiment_config.get("normalize", False)
        and artifacts is not None
        and target is not None
    ):
        # Import here to avoid circular imports
        from .FeatureProcessingArtifacts import post_process_predictions

        # Create temporary DataFrame for post-processing
        df_temp = pd.DataFrame(
            {
                "prediction": y_pred_scaled,
                target: y_val.values if hasattr(y_val, "values") else y_val,
            }
        )

        if basin_codes is not None:
            # Add code column if needed for per-basin normalization
            df_temp["code"] = (
                basin_codes.values if hasattr(basin_codes, "values") else basin_codes
            )

        if val_dates is not None:
            # Add date column for long_term_mean normalization
            df_temp["date"] = (
                val_dates.values if hasattr(val_dates, "values") else val_dates
            )

        # Apply inverse transformation
        df_temp = post_process_predictions(
            df_predictions=df_temp,
            artifacts=artifacts,
            experiment_config=experiment_config,
            prediction_column="prediction",
            target=target,
        )

        # Calculate R2 on original scale
        return r2_score(df_temp[target], df_temp["prediction"])
    else:
        # No normalization, calculate R2 directly on scaled values
        return r2_score(y_val, y_pred_scaled)


def _objective_lgbm(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    artifacts=None,
    experiment_config=None,
    target=None,
    basin_codes=None,
    val_dates=None,
):
    """Optuna objective function for LightGBM without early stopping."""
    params = {
        "objective": "regression",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 50, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 1, 30),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "n_jobs": -1,
        "verbose": -1,
    }

    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_val)

    # If normalization is enabled and artifacts are provided, inverse transform for R2 calculation
    if (
        experiment_config
        and experiment_config.get("normalize", False)
        and artifacts is not None
        and target is not None
    ):
        # Import here to avoid circular imports
        from .FeatureProcessingArtifacts import post_process_predictions

        # Create temporary DataFrame for post-processing
        df_temp = pd.DataFrame(
            {
                "prediction": y_pred_scaled,
                target: y_val.values if hasattr(y_val, "values") else y_val,
            }
        )

        if basin_codes is not None:
            # Add code column if needed for per-basin normalization
            df_temp["code"] = (
                basin_codes.values if hasattr(basin_codes, "values") else basin_codes
            )

        if val_dates is not None:
            # Add date column for long_term_mean normalization
            df_temp["date"] = (
                val_dates.values if hasattr(val_dates, "values") else val_dates
            )

        # Apply inverse transformation
        df_temp = post_process_predictions(
            df_predictions=df_temp,
            artifacts=artifacts,
            experiment_config=experiment_config,
            prediction_column="prediction",
            target=target,
        )

        # Calculate R2 on original scale
        return r2_score(df_temp[target], df_temp["prediction"])
    else:
        # No normalization, calculate R2 directly on scaled values
        return r2_score(y_val, y_pred_scaled)


def _objective_catboost(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    cat_features,
    artifacts=None,
    experiment_config=None,
    target=None,
    basin_codes=None,
    val_dates=None,
):
    """Optuna objective function for CatBoost without early stopping."""
    params = {
        "iterations": trial.suggest_int("iterations", 50, 1000),
        "depth": trial.suggest_int("depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-1, 10.0, log=True),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 1e-1, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "allow_writing_files": False,
        "verbose": False,
    }

    model = CatBoostRegressor(**params, cat_features=cat_features)
    model.fit(X_train, y_train)

    y_pred_scaled = model.predict(X_val)

    # If normalization is enabled and artifacts are provided, inverse transform for R2 calculation
    if (
        experiment_config
        and experiment_config.get("normalize", False)
        and artifacts is not None
        and target is not None
    ):
        # Import here to avoid circular imports
        from .FeatureProcessingArtifacts import post_process_predictions

        # Create temporary DataFrame for post-processing
        df_temp = pd.DataFrame(
            {
                "prediction": y_pred_scaled,
                target: y_val.values if hasattr(y_val, "values") else y_val,
            }
        )

        if basin_codes is not None:
            # Add code column if needed for per-basin normalization
            df_temp["code"] = (
                basin_codes.values if hasattr(basin_codes, "values") else basin_codes
            )

        if val_dates is not None:
            # Add date column for long_term_mean normalization
            df_temp["date"] = (
                val_dates.values if hasattr(val_dates, "values") else val_dates
            )

        # Apply inverse transformation
        df_temp = post_process_predictions(
            df_predictions=df_temp,
            artifacts=artifacts,
            experiment_config=experiment_config,
            prediction_column="prediction",
            target=target,
        )

        # Calculate R2 on original scale
        return r2_score(df_temp[target], df_temp["prediction"])
    else:
        # No normalization, calculate R2 directly on scaled values
        return r2_score(y_val, y_pred_scaled)


def _objective_mlp(
    trial,
    X_train,
    y_train,
    X_val,
    y_val,
    artifacts=None,
    experiment_config=None,
    target=None,
    basin_codes=None,
    val_dates=None,
):
    """Optuna objective function for MLP without early stopping."""
    params = {
        "hidden_layer_sizes": (trial.suggest_int("hidden_layer_sizes", 50, 200),),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "solver": trial.suggest_categorical("solver", ["adam", "lbfgs"]),
        "alpha": trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
        "learning_rate": trial.suggest_categorical(
            "learning_rate", ["constant", "adaptive"]
        ),
        "max_iter": trial.suggest_int("max_iter", 100, 1000),
        "random_state": 42,
    }

    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    y_pred_scaled = model.predict(X_val)

    # If normalization is enabled and artifacts are provided, inverse transform for R2 calculation
    if (
        experiment_config
        and experiment_config.get("normalize", False)
        and artifacts is not None
        and target is not None
    ):
        # Import here to avoid circular imports
        from .FeatureProcessingArtifacts import post_process_predictions

        # Create temporary DataFrame for post-processing
        df_temp = pd.DataFrame(
            {
                "prediction": y_pred_scaled,
                target: y_val.values if hasattr(y_val, "values") else y_val,
            }
        )

        if basin_codes is not None:
            # Add code column if needed for per-basin normalization
            df_temp["code"] = (
                basin_codes.values if hasattr(basin_codes, "values") else basin_codes
            )

        if val_dates is not None:
            # Add date column for long_term_mean normalization
            df_temp["date"] = (
                val_dates.values if hasattr(val_dates, "values") else val_dates
            )

        # Apply inverse transformation
        df_temp = post_process_predictions(
            df_predictions=df_temp,
            artifacts=artifacts,
            experiment_config=experiment_config,
            prediction_column="prediction",
            target=target,
        )

        # Calculate R2 on original scale
        return r2_score(df_temp[target], df_temp["prediction"])
    else:
        # No normalization, calculate R2 directly on scaled values
        return r2_score(y_val, y_pred_scaled)
