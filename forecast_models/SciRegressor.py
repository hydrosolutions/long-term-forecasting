import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import json
import joblib
import datetime
import warnings
warnings.filterwarnings('ignore')

from forecast_models.base_class import BaseForecastModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

from scr import FeatureExtractor as FE
from scr import tree_utils
from scr import data_utils as du

# Shared logging
import logging
from log_config import setup_logging
setup_logging()  

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


class SciRegressor(BaseForecastModel):
    """
    A regressor class for ensemble models which can be fitted using "Sci-Kit Learn style" (fit/predict) methods with tabular data., 
    XGBoost, LightGBM, CatBoost, etc.
    Uses GLOBAL fitting approach where a model is trained on all basins and periods simultaneously.
    """

    def __init__(self, 
                data: pd.DataFrame,
                static_data: pd.DataFrame,
                general_config: Dict[str, Any], 
                model_config: Dict[str, Any],
                feature_config: Dict[str, Any],
                path_config: Dict[str, Any]) -> None:
        """
        Initialize the SciRegressor model with a configuration dictionary.

        Args:
            data (pd.DataFrame): Time series data
                columns should atleast include: ['date', 'code', 'discharge']
            static_data (pd.DataFrame): Static basin characteristics
            general_config (Dict[str, Any]): General configuration for the model.
            model_config (Dict[str, Any]): Model-specific  - hyperparameters.
            feature_config (Dict[str, Any]): Feature engineering configuration.
            path_config (Dict[str, Any]): Path configuration for saving/loading data.
        """
        super().__init__(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        # Initialize model-specific attributes
        self.models = self.general_config.get('models', ['xgboost'])  # List of model types
        self.fitted_models = {}  # Will store fitted model objects per period
        self.scalers = {}  # Will store scalers per period
        self.feature_sets = {}  # Will store feature sets per period
        
        # Get preprocessing configuration
        self.missing_value_handling = self.model_config.get('missing_value_handling', 'drop')
        self.normalization_type = self.model_config.get('normalization_type', None)
        self.normalize_per_basin = self.model_config.get('normalize_per_basin', False)
        self.use_pca = self.model_config.get('use_pca', False)
        self.use_lr_predictors = self.model_config.get('use_lr_predictors', False)
        self.use_temporal_features = self.model_config.get('use_temporal_features', True)
        self.use_static_features = self.model_config.get('use_static_features', True)
        self.cat_features = self.model_config.get('cat_features', ['code_str'])
        
        logger.debug('Preprocessing data for SciRegressor model')
        self.__preprocess_data__()
        logger.debug('Data preprocessed successfully')
        
        logger.debug('Extracting features for SciRegressor model')
        self.__extract_features__()
        logger.debug('Features extracted successfully')

    def __preprocess_data__(self):
        """
        Preprocess the data by adding position and other derived features.
        """
        # Add position name (equivalent to du.get_position_name)
        if 'position' not in self.data.columns:
            # Simple position encoding based on data availability
            self.data['position'] = self.data.apply(
                lambda row: f"pos_{row['code']}_{row['date'].month}", axis=1
            )

    def __extract_features__(self):
        """
        Extract features from the data using FeatureExtractor and prepare for global fitting.
        """
        # Use FeatureExtractor for time series features
        extractor = FE.StreamflowFeatureExtractor(
            feature_configs=self.feature_config,
            prediction_horizon=self.general_config['prediction_horizon'],
            offset=self.general_config.get('offset', 0),
        )

        self.data = extractor.create_all_features(self.data)
        
        # Add temporal features
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month
        self.data['day'] = self.data['date'].dt.day
        
        # Add basin identity features for global training
        self.data['basin'] = self.data['code']
        self.data['code_str'] = self.data['code'].astype(str)
        
        if self.use_temporal_features:
            logger.debug("Adding temporal features")
            # Add cyclical encoding for temporal features
            self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
            self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
            
            # Add week features if available
            self.data['week'] = self.data['date'].dt.isocalendar().week
            self.data['week_sin'] = np.sin(2 * np.pi * self.data['week'] / 52)
            self.data['week_cos'] = np.cos(2 * np.pi * self.data['week'] / 52)

    def __prepare_global_features__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for global training across all basins.
        
        Args:
            df: Input dataframe
            
        Returns:
            DataFrame with dummy variables and merged static features
        """
        # Create dummy variables for basins and months (key for global fitting)
        df_with_dummies = pd.get_dummies(df, columns=['month', 'basin'], dtype=int)
        
        # Merge static features if configured
        if self.use_static_features and self.static_data is not None:
            logger.debug("Merging static features for global training")
            static_features = self.general_config.get('static_features', [])
            if static_features and 'code' in self.static_data.columns:
                static_df_feat = self.static_data[['code'] + static_features]
                df_with_dummies = pd.merge(df_with_dummies, static_df_feat, on='code', how='inner')
                logger.debug(f"Added {len(static_features)} static features")
        
        return df_with_dummies

    def __get_global_feature_sets__(self, df_with_dummies: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Get feature sets for global training following the ML_MONTHLY_FORECAST pattern.
        
        Args:
            df_with_dummies: DataFrame with dummy variables
            
        Returns:
            Tuple of (features_1, features_2) lists
        """
        # Features 1: All dummy variables (month and basin dummies)
        features_1 = [col for col in df_with_dummies.columns 
                     if ('month' in col or 'basin' in col) and '_sin' not in col and '_cos' not in col]
        
        # Base feature columns
        feature_cols = self.general_config.get('base_features', ['discharge', 'precip', 'temp'])
        snow_vars = self.general_config.get('snow_vars', ['SWE', 'SCA'])
        all_base_features = feature_cols + snow_vars
        
        # LR features if using linear regression predictors
        lr_features = []
        if self.use_lr_predictors:
            lr_all_cols = [col for col in df_with_dummies.columns if 'LR' in col]
            lr_features = [col for col in lr_all_cols if any([f in col for f in all_base_features])]
            if 'LR Benchmark' in df_with_dummies.columns:
                lr_features.append('LR Benchmark')
        
        # Static features
        static_features = self.general_config.get('static_features', [])
        
        # Features 2: Dynamic features + temporal + static + LR
        feature_set_2 = [col + "_" for col in all_base_features] + \
                       ['week_sin', 'week_cos', 'month_sin', 'month_cos'] + \
                       static_features + lr_features
        
        # Add categorical features for tree models
        for model_type in self.models:
            if model_type == 'catboost':
                for cat_feature in self.cat_features:
                    if cat_feature not in feature_set_2:
                        feature_set_2.append(cat_feature)
        
        # Filter features that actually exist in the dataframe
        features_2 = [col for col in df_with_dummies.columns
                     if any([f in col for f in feature_set_2])]
        
        logger.debug(f"Features 1 (dummies): {len(features_1)} features")
        logger.debug(f"Features 2 (dynamic): {len(features_2)} features")
        
        return features_1, features_2

    def predict_operational(self) -> pd.DataFrame:
        """
        Predict in operational mode using global trained models.

        Returns:
            forecast (pd.DataFrame): DataFrame containing the forecasted values.
        """
        logger.info(f"Starting operational prediction for {self.name}")
        
        today = datetime.datetime.now()
        today_day = today.day
        today_month = today.month
        
        # Add day column if not present
        if 'day' not in self.data.columns:
            self.data['day'] = self.data['date'].dt.day
            
        period_name = f"{today_month}-{today_day}"
        self.data['period'] = self.data['month'].astype(str) + '-' + self.data['day'].astype(str)
        
        # Filter the data for the current period
        operational_data = self.data[self.data['period'] == period_name].copy()
        
        if len(operational_data) == 0:
            logger.warning(f"No data available for period {period_name}")
            return pd.DataFrame(columns=['forecast_date', 'model_name', 'code', 'valid_from', 'valid_to', 'Q'])
        
        # Check if we have fitted models for this period
        if period_name not in self.fitted_models:
            logger.warning(f"No fitted models found for period {period_name}")
            return pd.DataFrame(columns=['forecast_date', 'model_name', 'code', 'valid_from', 'valid_to', 'Q'])
        
        # Prepare global features
        today_normalized = today.replace(hour=0, minute=0, second=0, microsecond=0)
        prediction_data = operational_data[operational_data['date'].dt.normalize() == today_normalized].copy()
        
        if len(prediction_data) == 0:
            logger.warning(f"No prediction data for {today}")
            return pd.DataFrame(columns=['forecast_date', 'model_name', 'code', 'valid_from', 'valid_to', 'Q'])
        
        # Prepare features for global prediction
        prediction_data_global = self.__prepare_global_features__(prediction_data)
        features_1, features_2 = self.feature_sets[period_name]
        
        forecast = pd.DataFrame()
        
        # Calculate valid period
        if not self.general_config.get('offset'):
            self.general_config['offset'] = self.general_config['prediction_horizon']
        shift = self.general_config['offset'] - self.general_config['prediction_horizon']
        valid_from = today + datetime.timedelta(days=1) + datetime.timedelta(days=shift)
        valid_to = valid_from + datetime.timedelta(days=self.general_config['prediction_horizon'])
        
        valid_from_str = valid_from.strftime('%Y-%m-%d')
        valid_to_str = valid_to.strftime('%Y-%m-%d')
        
        # Make predictions with ensemble of models
        for model_type in self.models:
            model_key = f"{period_name}_{model_type}"
            if model_key in self.fitted_models[period_name]:
                model = self.fitted_models[period_name][model_key]
                scaler = self.scalers[period_name][model_key] if model_key in self.scalers[period_name] else None
                
                try:
                    # Apply same preprocessing as training
                    if scaler and self.normalization_type:
                        if self.normalize_per_basin:
                            # Per-basin normalization - would need calibration data for proper scaling
                            logger.warning("Per-basin normalization in operational mode requires calibration data")
                            X_pred = prediction_data_global[features_2].fillna(0)
                        else:
                            # Global normalization
                            X_pred = prediction_data_global[features_2].fillna(0)
                            # Apply scaling if available (simplified)
                    else:
                        X_pred = prediction_data_global[features_2].fillna(0)
                    
                    # Make predictions
                    predictions = model.predict(X_pred)
                    predictions = np.maximum(predictions, 0)  # Ensure non-negative
                    
                    # Create forecast dataframe for this model
                    model_forecast = pd.DataFrame({
                        'forecast_date': [today] * len(predictions),
                        'model_name': [f"{self.name}_{model_type}"] * len(predictions),
                        'code': prediction_data['code'].values,
                        'valid_from': [valid_from_str] * len(predictions),
                        'valid_to': [valid_to_str] * len(predictions),
                        'Q': [round(p, 2) for p in predictions],
                    })
                    
                    forecast = pd.concat([forecast, model_forecast], ignore_index=True)
                    
                except Exception as e:
                    logger.warning(f"Prediction failed for {model_type}: {e}")
        
        # If we have multiple models, also create ensemble predictions
        if len(self.models) > 1 and len(forecast) > 0:
            ensemble_forecast = forecast.groupby('code').agg({
                'forecast_date': 'first',
                'valid_from': 'first', 
                'valid_to': 'first',
                'Q': 'mean'
            }).reset_index()
            ensemble_forecast['model_name'] = self.name
            ensemble_forecast['Q'] = ensemble_forecast['Q'].round(2)
            
            forecast = pd.concat([forecast, ensemble_forecast], ignore_index=True)
        
        return forecast
    
    def calibrate_model_and_hindcast(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate the global ensemble models using Leave-One-Year-Out cross-validation.
        Follows the ML_MONTHLY_FORECAST.py global training approach.

        Args:
            data (pd.DataFrame): DataFrame containing the calibration data.

        Returns:
            hindcast (pd.DataFrame): DataFrame containing the hindcasted values.
        """
        logger.info(f"Starting GLOBAL calibration and hindcasting for {self.name} with models: {self.models}")
        
        # Add day column if not present
        if 'day' not in self.data.columns:
            self.data['day'] = self.data['date'].dt.day
        
        # Get configuration parameters
        test_years = self.general_config.get('test_years', 3)
        forecast_days = self.general_config.get('forecast_days', [5, 10, 15, 20, 25, 'end'])
        
        # Prepare global features
        data_global = self.__prepare_global_features__(self.data)
        
        # Get unique years
        years = sorted(data_global['year'].unique())
        
        # Split into LOOCV years and final test years
        if len(years) <= test_years:
            logger.warning(f"Not enough years ({len(years)}) for test split of {test_years} years. Using all data for LOOCV.")
            loocv_years = years
            test_years_list = []
        else:
            loocv_years = years[:-test_years]
            test_years_list = years[-test_years:]
            logger.info(f"LOOCV years: {loocv_years[0]}-{loocv_years[-1]}, Test years: {test_years_list}")
        
        # Initialize results container
        all_predictions = []
        
        # Process each period GLOBALLY (not per basin)
        for month in range(1, 13):
            for day in forecast_days:
                # Handle 'end' of month
                if day == 'end':
                    month_data = data_global[data_global['month'] == month]
                    if len(month_data) == 0:
                        continue
                    last_days = month_data.groupby('year')['day'].max()
                    period_data = []
                    for year, last_day in last_days.items():
                        year_month_data = data_global[(data_global['year'] == year) & 
                                                     (data_global['month'] == month) & 
                                                     (data_global['day'] == last_day)]
                        period_data.append(year_month_data)
                    if period_data:
                        period_df = pd.concat(period_data, ignore_index=True)
                    else:
                        continue
                    period_name = f"{month}-end"
                else:
                    period_df = data_global[(data_global['month'] == month) & (data_global['day'] == day)].copy()
                    period_name = f"{month}-{day}"
                
                if len(period_df) == 0:
                    logger.debug(f"No data for period {period_name}")
                    continue
                
                period_df['period'] = period_name
                logger.debug(f"Processing period {period_name} with {len(period_df)} samples across all basins")
                
                # Get feature sets for this period
                features_1, features_2 = self.__get_global_feature_sets__(period_df)
                self.feature_sets[period_name] = (features_1, features_2)
                
                # Initialize fitted models for this period
                if period_name not in self.fitted_models:
                    self.fitted_models[period_name] = {}
                    self.scalers[period_name] = {}
                
                # GLOBAL Leave-One-Year-Out CV (all basins together)
                for test_year in loocv_years:
                    # Global training: ALL basins except test year
                    train_data = period_df[
                        (period_df['year'].isin(loocv_years)) & 
                        (period_df['year'] != test_year)
                    ].copy()
                    
                    # Global testing: ALL basins for test year
                    test_data = period_df[period_df['year'] == test_year].copy()
                    
                    if len(train_data) < 20 or len(test_data) == 0:  # Need sufficient global samples
                        continue
                    
                    try:
                        # Train and predict with each model type GLOBALLY
                        for model_type in self.models:
                            # Get model configuration
                            model_params = self.model_config.get(f'{model_type}_params', {})
                            
                            # Apply feature processing using tree_utils
                            train_processed, test_processed, selected_features, scalers = tree_utils.process_features(
                                train_data[features_2 + ['target']].copy(),
                                test_data[features_2 + (['target'] if 'target' in test_data.columns else [])].copy(),
                                'target',
                                {
                                    'missing_value_handling': self.missing_value_handling,
                                    'normalization_type': self.normalization_type,
                                    'normalize_per_basin': self.normalize_per_basin,
                                    'use_pca': self.use_pca,
                                    'n_features': len(features_2),
                                }
                            )
                            
                            # Create and train GLOBAL model
                            model = tree_utils.get_model(model_type, model_params)
                            
                            # Prepare training data
                            X_train = train_processed[selected_features] if selected_features else train_processed[features_2]
                            y_train = train_processed['target']
                            X_test = test_processed[selected_features] if selected_features else test_processed[features_2]
                            
                            # Special handling for CatBoost with categorical features
                            if model_type == 'catboost':
                                cat_features = []
                                for cat_feat in self.cat_features:
                                    if cat_feat in X_train.columns:
                                        cat_features.append(cat_feat)
                                
                                if cat_features:
                                    model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                                else:
                                    model.fit(X_train, y_train, verbose=False)
                            else:
                                model.fit(X_train, y_train)
                            
                            # Make predictions for all basins in test year
                            predictions = model.predict(X_test)
                            predictions = np.maximum(predictions, 0)  # Ensure non-negative
                            
                            # Store predictions for all basins
                            for i, (idx, row) in enumerate(test_data.iterrows()):
                                all_predictions.append({
                                    'date': row['date'],
                                    'model': f"{self.name}_{model_type}",
                                    'code': row['code'],
                                    'Q_pred': predictions[i],
                                    'period': period_name,
                                    'cv_type': 'loocv'
                                })
                    
                    except Exception as e:
                        logger.warning(f"Failed to process period {period_name}, year {test_year}: {e}")
                        continue
                
                # Train final GLOBAL models on all LOOCV years for test set prediction and operational use
                if test_years_list:
                    train_data = period_df[period_df['year'].isin(loocv_years)].copy()
                    test_data = period_df[period_df['year'].isin(test_years_list)].copy()
                    
                    if len(train_data) >= 20 and len(test_data) > 0:
                        try:
                            for model_type in self.models:
                                model_params = self.model_config.get(f'{model_type}_params', {})
                                
                                # Apply feature processing
                                train_processed, test_processed, selected_features, scalers = tree_utils.process_features(
                                    train_data[features_2 + ['target']].copy(),
                                    test_data[features_2 + (['target'] if 'target' in test_data.columns else [])].copy(),
                                    'target',
                                    {
                                        'missing_value_handling': self.missing_value_handling,
                                        'normalization_type': self.normalization_type,
                                        'normalize_per_basin': self.normalize_per_basin,
                                        'use_pca': self.use_pca,
                                        'n_features': len(features_2),
                                    }
                                )
                                
                                # Create and train final GLOBAL model
                                model = tree_utils.get_model(model_type, model_params)
                                
                                X_train = train_processed[selected_features] if selected_features else train_processed[features_2]
                                y_train = train_processed['target']
                                X_test = test_processed[selected_features] if selected_features else test_processed[features_2]
                                
                                model.fit(X_train, y_train)
                                predictions = model.predict(X_test)
                                predictions = np.maximum(predictions, 0)
                                
                                # Store the fitted GLOBAL model for operational use
                                model_key = f"{period_name}_{model_type}"
                                self.fitted_models[period_name][model_key] = model
                                self.scalers[period_name][model_key] = scalers
                                
                                # Store test predictions
                                for i, (idx, row) in enumerate(test_data.iterrows()):
                                    all_predictions.append({
                                        'date': row['date'],
                                        'model': f"{self.name}_{model_type}",
                                        'code': row['code'],
                                        'Q_pred': predictions[i],
                                        'period': period_name,
                                        'cv_type': 'test'
                                    })
                        
                        except Exception as e:
                            logger.warning(f"Failed to process test years for period {period_name}: {e}")
        
        # Convert to DataFrame and create ensemble predictions
        if all_predictions:
            hindcast_df = pd.DataFrame(all_predictions)
            
            # Create ensemble predictions by averaging across models
            if len(self.models) > 1:
                ensemble_predictions = []
                for (date, code), group in hindcast_df.groupby(['date', 'code']):
                    if len(group) > 1:  # Multiple models available
                        ensemble_pred = {
                            'date': date,
                            'model': self.name,
                            'code': code,
                            'Q_pred': group['Q_pred'].mean(),
                            'period': group['period'].iloc[0],
                            'cv_type': group['cv_type'].iloc[0]
                        }
                        ensemble_predictions.append(ensemble_pred)
                
                if ensemble_predictions:
                    ensemble_df = pd.DataFrame(ensemble_predictions)
                    hindcast_df = pd.concat([hindcast_df, ensemble_df], ignore_index=True)
            
            # Return in required format
            hindcast_df = hindcast_df[['date', 'model', 'code', 'Q_pred']]
            hindcast_df = hindcast_df.sort_values(['date', 'code', 'model']).reset_index(drop=True)
            logger.info(f"Global calibration complete. Generated {len(hindcast_df)} predictions for {len(hindcast_df['code'].unique())} codes")
        else:
            logger.warning("No predictions generated during calibration")
            hindcast_df = pd.DataFrame(columns=['date', 'model', 'code', 'Q_pred'])
        
        return hindcast_df
    
    def tune_hyperparameters(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """
        Tune hyperparameters for all models using Optuna (global approach).
        Uses time series cross-validation on the last N years before test set.

        Args:
            data (pd.DataFrame): DataFrame containing the data for hyperparameter tuning.

        Returns:
            bool: True if hyperparameters were tuned successfully, False otherwise.
            str: Message indicating the result of the tuning process.
        """
        try:
            import optuna
        except ImportError:
            return False, "Optuna not available for hyperparameter tuning"
        
        logger.info(f"Starting hyperparameter tuning for {self.name} with models: {self.models}")
        
        # Get hyperparameter tuning configuration
        n_trials = self.general_config.get('hyperparam_tuning_trials', 100)
        tuning_years = self.general_config.get('hyperparam_tuning_years', 3)
        test_years = self.general_config.get('test_years', 3)
        
        # Prepare global features
        data_global = self.__prepare_global_features__(self.data)
        
        # Get years for tuning (exclude final test years)
        years = sorted(data_global['year'].unique())
        if len(years) <= tuning_years + test_years:
            logger.warning(f"Not enough years for hyperparameter tuning. Using available years.")
            tuning_years_list = years[:-test_years] if test_years < len(years) else years
        else:
            # Use N years before the test years for hyperparameter tuning
            tuning_years_list = years[-(tuning_years + test_years):-test_years]
        
        tuning_data = data_global[data_global['year'].isin(tuning_years_list)].copy()
        logger.info(f"Using years {tuning_years_list} for hyperparameter tuning")
        
        if len(tuning_data) < 50:
            return False, "Insufficient data for hyperparameter tuning"
        
        # Select a representative period for tuning (to reduce computational cost)
        # Use one period per season
        tuning_periods = ['3-15', '6-15', '9-15', '12-15']  # Spring, Summer, Fall, Winter
        
        best_params_all_models = {}
        
        for model_type in self.models:
            logger.info(f"Tuning hyperparameters for {model_type}")
            
            def objective(trial):
                """Optuna objective function for tree-based model hyperparameters"""
                
                # Define parameter spaces for different models
                if model_type == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 30),
                        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                        'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                        'verbosity': 0,
                        'random_state': 42
                    }
                elif model_type == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 2000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                        'max_depth': trial.suggest_int('max_depth', 3, 12),
                        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
                        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
                        'verbosity': -1,
                        'random_state': 42
                    }
                elif model_type == 'catboost':
                    params = {
                        'iterations': trial.suggest_int('iterations', 100, 2000),
                        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                        'depth': trial.suggest_int('depth', 3, 10),
                        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0, log=True),
                        'random_strength': trial.suggest_float('random_strength', 1e-4, 10.0, log=True),
                        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
                        'verbose': False,
                        'random_seed': 42
                    }
                elif model_type == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                        'max_depth': trial.suggest_int('max_depth', 3, 20),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
                        'random_state': 42
                    }
                else:
                    return -999  # Unknown model type
                
                period_scores = []
                
                # Evaluate on selected periods
                for period_str in tuning_periods:
                    month, day = period_str.split('-')
                    month = int(month)
                    day = int(day)
                    
                    period_df = tuning_data[(tuning_data['month'] == month) & (tuning_data['day'] == day)].copy()
                    
                    if len(period_df) < 20:  # Need sufficient global samples
                        continue
                    
                    period_df['period'] = period_str
                    
                    # Get feature sets for this period
                    features_1, features_2 = self.__get_global_feature_sets__(period_df)
                    
                    # Time series cross-validation on this period
                    period_years = sorted(period_df['year'].unique())
                    if len(period_years) < 3:
                        continue
                    
                    # Use last 2 years as validation for each period
                    n_val_years = min(2, len(period_years) // 2)
                    train_years = period_years[:-n_val_years]
                    val_years = period_years[-n_val_years:]
                    
                    train_data = period_df[period_df['year'].isin(train_years)].copy()
                    val_data = period_df[period_df['year'].isin(val_years)].copy()
                    
                    if len(train_data) < 15 or len(val_data) < 5:
                        continue
                    
                    try:
                        # Apply feature processing
                        train_processed, val_processed, selected_features, scalers = tree_utils.process_features(
                            train_data[features_2 + ['target']].copy(),
                            val_data[features_2 + ['target']].copy(),
                            'target',
                            {
                                'missing_value_handling': self.missing_value_handling,
                                'normalization_type': self.normalization_type,
                                'normalize_per_basin': self.normalize_per_basin,
                                'use_pca': self.use_pca,
                                'n_features': len(features_2),
                            }
                        )
                        
                        # Create and train model with trial parameters
                        model = tree_utils.get_model(model_type, params)
                        
                        X_train = train_processed[selected_features] if selected_features else train_processed[features_2]
                        y_train = train_processed['target']
                        X_val = val_processed[selected_features] if selected_features else val_processed[features_2]
                        y_val = val_processed['target']
                        
                        # Handle categorical features for CatBoost
                        if model_type == 'catboost':
                            cat_features = []
                            for cat_feat in self.cat_features:
                                if cat_feat in X_train.columns:
                                    cat_features.append(cat_feat)
                            
                            if cat_features:
                                model.fit(X_train, y_train, cat_features=cat_features, verbose=False)
                            else:
                                model.fit(X_train, y_train, verbose=False)
                        else:
                            model.fit(X_train, y_train)
                        
                        # Make predictions and evaluate
                        y_pred = model.predict(X_val)
                        score = r2_score(y_val, y_pred)
                        
                        if not np.isnan(score) and score > -10:  # Reasonable score
                            period_scores.append(score)
                    
                    except Exception as e:
                        logger.debug(f"Failed to evaluate period {period_str}: {e}")
                        continue
                
                # Return mean score across periods
                if period_scores:
                    mean_score = np.mean(period_scores)
                    logger.debug(f"Trial {model_type}, mean R2={mean_score:.4f}")
                    return mean_score
                else:
                    return -999
            
            # Create and run Optuna study for this model type
            study_name = f"{self.name}_{model_type}"
            study = optuna.create_study(
                direction='maximize',
                pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
                study_name=study_name
            )
            
            logger.info(f"Running {n_trials} trials for {model_type}")
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
            
            # Store best parameters
            best_params = study.best_params
            best_score = study.best_value
            
            logger.info(f"Best hyperparameters for {model_type}:")
            logger.info(f"  Score: {best_score:.4f}")
            for key, value in best_params.items():
                logger.info(f"  {key}: {value}")
            
            best_params_all_models[model_type] = {
                'params': best_params,
                'score': best_score,
                'n_trials': n_trials
            }
            
            # Update model configuration
            param_key = f'{model_type}_params'
            self.model_config[param_key] = best_params
        
        # Save all hyperparameters to file
        model_dir = self.path_config.get('model_dir', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        hyperparams_path = os.path.join(model_dir, f"{self.name}_hyperparams.json")
        hyperparams_data = {
            'model_types': self.models,
            'best_params_all_models': best_params_all_models,
            'tuning_config': {
                'n_trials': n_trials,
                'tuning_years': tuning_years_list,
                'tuning_periods': tuning_periods,
            },
            'tuning_timestamp': datetime.datetime.now().isoformat()
        }
        
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams_data, f, indent=4)
        
        logger.info(f"Hyperparameters saved to {hyperparams_path}")
        
        # Create summary message
        summary_lines = [f"Successfully tuned hyperparameters for {len(self.models)} models:"]
        for model_type, results in best_params_all_models.items():
            summary_lines.append(f"  {model_type}: R2={results['score']:.4f}")
        
        return True, "\n".join(summary_lines)

    
    def save_model(self) -> None:
        """
        Save all fitted global models and preprocessing artifacts.
        """
        logger.info(f"Saving {self.name} global models")
        
        # Create model directory
        model_dir = self.path_config.get('model_dir', 'models')
        model_subdir = os.path.join(model_dir, self.name)
        os.makedirs(model_subdir, exist_ok=True)
        
        # Save each fitted model by period
        for period_name, period_models in self.fitted_models.items():
            period_dir = os.path.join(model_subdir, period_name.replace('/', '_'))
            os.makedirs(period_dir, exist_ok=True)
            
            for model_key, model in period_models.items():
                model_path = os.path.join(period_dir, f"{model_key}.joblib")
                joblib.dump(model, model_path)
        
        # Save scalers
        scalers_path = os.path.join(model_subdir, 'scalers.json')
        scalers_serializable = {}
        for period, scaler_dict in self.scalers.items():
            scalers_serializable[period] = {}
            for model_key, scaler in scaler_dict.items():
                if scaler is not None:
                    # Store scaler information (simplified)
                    scalers_serializable[period][model_key] = str(scaler)
        
        with open(scalers_path, 'w') as f:
            json.dump(scalers_serializable, f, indent=4)
        
        # Save feature sets
        features_path = os.path.join(model_subdir, 'feature_sets.json')
        with open(features_path, 'w') as f:
            json.dump(self.feature_sets, f, indent=4)
        
        # Save model configuration
        config_path = os.path.join(model_subdir, 'model_config.json')
        config_data = {
            'general_config': self.general_config,
            'model_config': self.model_config,
            'feature_config': self.feature_config,
            'models': self.models,
        }
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=4)
        
        logger.info(f"Global models saved to {model_subdir}")
    
    def load_model(self) -> None:
        """
        Load all fitted global models and preprocessing artifacts.
        """
        logger.info(f"Loading {self.name} global models")
        
        model_dir = self.path_config.get('model_dir', 'models')
        model_subdir = os.path.join(model_dir, self.name)
        
        if not os.path.exists(model_subdir):
            logger.warning(f"Model directory not found: {model_subdir}")
            return
        
        # Load configuration
        config_path = os.path.join(model_subdir, 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                self.models = config_data.get('models', self.models)
        
        # Load fitted models
        self.fitted_models = {}
        for period_name in os.listdir(model_subdir):
            period_dir = os.path.join(model_subdir, period_name)
            if os.path.isdir(period_dir):
                period_key = period_name.replace('_', '-')
                self.fitted_models[period_key] = {}
                for model_file in os.listdir(period_dir):
                    if model_file.endswith('.joblib'):
                        model_key = model_file.replace('.joblib', '')
                        model_path = os.path.join(period_dir, model_file)
                        self.fitted_models[period_key][model_key] = joblib.load(model_path)
        
        # Load feature sets
        features_path = os.path.join(model_subdir, 'feature_sets.json')
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_sets = json.load(f)
        
        # Load scalers
        scalers_path = os.path.join(model_subdir, 'scalers.json')
        if os.path.exists(scalers_path):
            with open(scalers_path, 'r') as f:
                self.scalers = json.load(f)
        
        logger.info(f"Loaded global models for {len(self.fitted_models)} periods")