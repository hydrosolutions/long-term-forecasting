"""
Modular SciRegressor utilities for forecasting models.

This module provides utilities for training and evaluating various regression models
including XGBoost, LightGBM, CatBoost, Random Forest, and others. It supports
leave-one-year-out cross-validation, hyperparameter optimization, and feature processing.
"""

import os
import logging
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

# Tree-based models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# Hyperparameter optimization
import optuna

# Custom modules
from scr import data_utils as du

logger = logging.getLogger(__name__)



def get_model(model_type: str, params: Dict[str, Any], cat_features: Optional[List[str]] = None):
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
        'xgb': lambda p: XGBRegressor(**p),
        'lgbm': lambda p: LGBMRegressor(**p),
        'svr': lambda p: SVR(**p),
        'catboost': lambda p: CatBoostRegressor(**p, cat_features=cat_features or []),
        'rf': lambda p: RandomForestRegressor(**p),
        'gradient_boosting': lambda p: GradientBoostingRegressor(**p),
        'mlp': lambda p: MLPRegressor(**p),
    }
    
    if model_type not in model_mapping:
        raise ValueError(f"Invalid model type: {model_type}. Supported types: {list(model_mapping.keys())}")
    
    return model_mapping[model_type](params)


def fit_model(model, X: pd.DataFrame, y: pd.Series):
    """
    Fit a model and log training performance.
    
    Args:
        model: Model instance
        X: Features
        y: Target values
        
    Returns:
        Fitted model
    """
    model.fit(X, y)
    y_pred = model.predict(X)
    logger.info(f"Model R²: {r2_score(y, y_pred):.4f}")
    return model


def process_features(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: List[str],
    target: str,
    experiment_config: Dict[str, Any],
    pca_groups: Optional[Dict[str, List[str]]] = None,
    variance_threshold: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Optional[Any]]:
    """
    Process features including handling missing values, normalization, and feature selection.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        features: List of feature names
        target: Target variable name
        experiment_config: Configuration dictionary containing processing parameters
        pca_groups: Optional PCA grouping configuration
        variance_threshold: Variance threshold for PCA
        
    Returns:
        Tuple of (processed_train, processed_test, final_features, scaler)
    """
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Handle missing values
    handle_na = experiment_config.get('handle_na', 'drop')
    
    if handle_na == 'drop':
        all_cols = list(set(features + [target]))
        df_train = df_train.dropna(subset=all_cols)
        df_test = df_test.dropna(subset=all_cols)
        
    elif handle_na == 'long_term_mean':
        long_term_mean = du.get_long_term_mean_per_basin(df_train, features=features)
        df_train = du.apply_long_term_mean(df_train, long_term_mean=long_term_mean, features=features)
        df_test = du.apply_long_term_mean(df_test, long_term_mean=long_term_mean, features=features)
        
    elif handle_na == 'impute':
        impute_cols = [col for col in features if df_train[col].dtype.kind in 'ifc']
        
        impute_method = experiment_config.get('impute_method', 'mean')
        if impute_method == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            imputer = SimpleImputer(strategy=impute_method)
        
        if df_train[impute_cols].isna().any().any():
            imputed_train = imputer.fit_transform(df_train[impute_cols])
            df_train[impute_cols] = pd.DataFrame(imputed_train, columns=impute_cols, index=df_train.index)
        if df_test[impute_cols].isna().any().any():
            imputed_test = imputer.transform(df_test[impute_cols])
            df_test[impute_cols] = pd.DataFrame(imputed_test, columns=impute_cols, index=df_test.index)

    # Separate numeric and categorical features
    features_num = [col for col in features if df_train[col].dtype.kind in 'ifc']
    cat_features = [col for col in features if col not in features_num]

    # Normalization
    scaler = None
    if experiment_config.get('normalize', False):
        if experiment_config.get('normalize_per_basin', False):
            df_train, df_test, scaler = du.normalize_features_per_basin(
                df_train, df_test, features_num, target
            )
        else:
            df_train, df_test, scaler = du.normalize_features(
                df_train, df_test, features_num, target
            )

    # PCA (if configured)
    if pca_groups:
        # Note: This would require implementing pca_utils.apply_pca_groups
        logger.warning("PCA functionality not implemented in this refactored version")

    # Feature selection using mutual information
    if experiment_config.get('use_mutual_info', False):
        X_train = df_train[features_num]
        y_train = df_train[target]

        # Remove highly correlated features
        if experiment_config.get('remove_correlated_features', False):
            corr_matrix = X_train.corr().abs()
            upper_triangle = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            highly_correlated = [
                column for column in upper_triangle.columns 
                if any(upper_triangle[column] > 0.95)
            ]
            X_train = X_train.drop(columns=highly_correlated)
            logger.info(f"Removed highly correlated features: {highly_correlated}")
        
        # Select best features using mutual information
        n_features = experiment_config.get('number_of_features', 10)
        selector = SelectKBest(mutual_info_regression, k=n_features)
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        logger.info(f"Selected features from mutual information: {selected_features}")

        # Add categorical features back
        features = list(set(selected_features) | set(cat_features))

    return df_train, df_test, features, scaler


def post_process_target(
    df_predictions: pd.DataFrame,
    target: str,
    scaler: Optional[Any],
    experiment_config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Reverse normalization of predictions if needed.
    
    Args:
        df_predictions: DataFrame with predictions
        target: Target variable name
        scaler: Fitted scaler object
        experiment_config: Configuration dictionary
        
    Returns:
        DataFrame with denormalized predictions
    """
    df_predictions = df_predictions.copy()
    
    if experiment_config.get('normalize', False) and scaler is not None:
        if experiment_config.get('normalize_per_basin', False):
            for code in df_predictions.code.unique():
                mean_, std_ = scaler[code][target]
                mask = df_predictions['code'] == code
                df_predictions.loc[mask, 'Q_pred'] = (
                    df_predictions.loc[mask, 'Q_pred'] * std_ + mean_
                )
        else:
            mean_, std_ = scaler[target]
            df_predictions['Q_pred'] = df_predictions['Q_pred'] * std_ + mean_
    
    return df_predictions


def loo_cv(
    df: pd.DataFrame,
    features: List[str],
    target: str,
    model_type: str,
    experiment_config: Dict[str, Any],
    model_config: Dict[str, Any],
    params: Optional[Dict[str, Any]] = None,
    pca_groups: Optional[Dict[str, List[str]]] = None,
    variance_threshold: float = 0.95
) -> pd.DataFrame:
    """
    Perform Leave-One-Year-Out cross-validation for time series data.
    
    Args:
        df: DataFrame containing all data
        features: List of feature names
        target: Target variable name
        model_type: Type of model to use
        experiment_config: Experiment configuration
        model_config: Model-specific configuration
        params: Optional model parameters (overrides model_config)
        pca_groups: Optional PCA configuration
        variance_threshold: Variance threshold for PCA
        
    Returns:
        DataFrame containing predictions with columns ['date', 'code', 'Q_obs', 'Q_pred']
    """
    df = df.copy()
    df['year'] = df['date'].dt.year

    years = df['year'].unique()
    df_predictions = pd.DataFrame()
    
    for year in tqdm(years, desc="Processing years", leave=True):
        df_train = df[df['year'] != year].dropna(subset=[target])
        df_test = df[df['year'] == year].dropna(subset=[target])
        df_predictions_year = df_test[['date', 'code', target]].copy()
        
        # Process features
        df_train_proc, df_test_proc, final_features, scaler = process_features(
            df_train, df_test, features, target, experiment_config, 
            pca_groups, variance_threshold
        )

        
        # Prepare data
        X_train = df_train_proc[final_features]
        y_train = df_train_proc[target]
        X_test = df_test_proc[final_features]
        
        # Create model
        if params:
            model = get_model(model_type, params, experiment_config.get('cat_features', []))
        else:
            model_params = model_config.get(model_type, {})
            model = get_model(model_type, model_params, experiment_config.get('cat_features', []))

        # Train and predict
        model = fit_model(model, X_train, y_train)
        y_pred = model.predict(X_test)

        df_predictions_year['Q_pred'] = y_pred
        df_predictions_year = post_process_target(
            df_predictions_year, target, scaler, experiment_config
        )
        
        df_predictions = pd.concat([df_predictions, df_predictions_year])

    # Rename columns
    df_predictions.rename(columns={target: 'Q_obs'}, inplace=True)
    return df_predictions


def get_feature_importance(
    model,
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
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
    elif hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
    else:
        logger.warning(f"Model type {type(model)} not supported for feature importance extraction.")
        return pd.DataFrame()

    # Create DataFrame
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
    plt.title('Top 20 Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=250, bbox_inches='tight')
    plt.show()
    
    return feature_importance


def optimize_hyperparams(
    df: pd.DataFrame,
    features: List[str],
    model_type: str,
    experiment_config: Dict[str, Any],
    target: str = 'target',
    n_trials: int = 100,
    pca_groups: Optional[Dict[str, List[str]]] = None,
    variance_threshold: float = 0.95,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters using Optuna.
    
    Args:
        df: Input DataFrame
        features: List of feature names
        model_type: Type of model to optimize
        experiment_config: Experiment configuration
        target: Target column name
        n_trials: Number of optimization trials
        pca_groups: Optional PCA configuration
        variance_threshold: Variance threshold for PCA
        save_path: Optional path to save optimization plots
        
    Returns:
        Dictionary of best hyperparameters
    """
    df = df.copy()
    df.dropna(subset=[target], inplace=True)
    df['year'] = df['date'].dt.year
    
    # Limit years for hyperparameter tuning
    year_limit = experiment_config.get('hyperparam_tuning_year_limit', df['year'].max())
    df = df[df['year'] < year_limit]
    
    years = sorted(df['year'].unique())
    num_train_years = int(len(years) * 0.8)
    train_years = years[:num_train_years]
    val_years = years[num_train_years:]
    
    logger.info(f"Train years: {train_years}")
    logger.info(f"Validation years: {val_years}")
    
    # Split data
    train_data = df[df['year'].isin(train_years)]
    val_data = df[df['year'].isin(val_years)]

    train_data, val_data, final_features, _ = process_features(
        train_data, val_data, features, target, 
        experiment_config, pca_groups, variance_threshold
    )
    
    X_train = train_data[final_features]
    y_train = train_data[target]
    X_val = val_data[final_features]
    y_val = val_data[target]

    # Define objective function based on model type
    def objective(trial):
        if model_type == 'xgb':
            return _objective_xgb(trial, X_train, y_train, X_val, y_val)
        elif model_type == 'lgbm':
            return _objective_lgbm(trial, X_train, y_train, X_val, y_val)
        elif model_type == 'catboost':
            cat_features = experiment_config.get('cat_features', [])
            return _objective_catboost(trial, X_train, y_train, X_val, y_val, cat_features)
        elif model_type == 'mlp':
            return _objective_mlp(trial, X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Hyperparameter optimization not supported for model type: {model_type}")

    # Run optimization
    study = optuna.create_study(direction='maximize')
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
        fig.write_html(os.path.join(save_path, 'optimization_history.html'))
        
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_html(os.path.join(save_path, 'param_importances.html'))
        
        try:
            fig = optuna.visualization.plot_contour(study)
            fig.write_html(os.path.join(save_path, 'contour.html'))
        except Exception as e:
            logger.warning(f"Could not create contour plot: {e}")
    
    return study.best_trial.params


# Objective functions for different models
def _objective_xgb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for XGBoost."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-4, 10.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-4, 10.0, log=True),
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)


def _objective_lgbm(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for LightGBM."""
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 30),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-2, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-2, 10.0, log=True),
        'n_jobs': -1,
        'verbose': -1
    }
    
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)


def _objective_catboost(trial, X_train, y_train, X_val, y_val, cat_features):
    """Optuna objective function for CatBoost."""
    params = {
        'iterations': trial.suggest_int('iterations', 100, 2000),
        'depth': trial.suggest_int('depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
        'border_count': 254,
        'bagging_temperature': trial.suggest_float('bagging_temperature', 1e-4, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-4, 10.0, log=True),
        'allow_writing_files': False,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params, cat_features=cat_features)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)


def _objective_mlp(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for MLP."""
    params = {
        'hidden_layer_sizes': trial.suggest_int('hidden_layer_sizes', 50, 200),
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
        'learning_rate': trial.suggest_categorical('learning_rate', ['constant', 'adaptive']),
        'max_iter': trial.suggest_int('max_iter', 100, 1000),
        'random_state': 42
    }
    
    model = MLPRegressor(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    return r2_score(y_val, y_pred)