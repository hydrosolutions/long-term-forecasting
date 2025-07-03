import os
import pandas as pd
import numpy as np
import geopandas as gpd
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
from scr import sci_utils

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
        self.feature_cols = self.feature_config.get('feature_cols', ['discharge', 'P', 'T',])
        self.static_features = self.feature_config.get('static_features', [])
        self.rivers_to_exclude = self.general_config.get('rivers_to_exclude', [])
        self.snow_vars = self.general_config.get('snow_vars', ['SWE'])


    def __preprocess_data__(self):
        """
        Preprocess the data by adding position and other derived features.
        """
        try:
            self.data = du.glacier_mapper_features(
                df=self.data,
                static=self.static_data,
            )
        except Exception as e:
            logger.error(f"Error in glacier_mapper_features: {e}")

        #remove log_discharge if it exists
        if "log_discharge" in self.data.columns:
            self.data.drop(columns=["log_discharge"], inplace=True)

        # Sort by date
        self.data.sort_values(by="date", inplace=True)

        cols_to_keep = [col for col in self.data.columns if any([feature in col for feature in self.feature_cols])]
        self.data = self.data[['date', 'code'] + cols_to_keep]
        
        # -------------- 2. Preprocess Discharge ------------------------------
        self.data = self.data[~self.data['code'].isin(self.rivers_to_exclude)].copy()

        for code in self.data.code.unique():
            area = self.static_data[self.static_data['code'] == code]['area_km2'].values[0]
            #transform from m3/s to mm/day
            self.data.loc[self.data['code'] == code, 'discharge'] = self.data.loc[self.data['code'] == code, 'discharge'] * 86.4 / area

        # -------------- 3. Snow Data to equal percentage area ------------------------------
        elevation_band_shp = gpd.read_file(self.path_config['path_to_hru_shp'])
        #rename CODE to code
        elevation_band_shp.rename(columns={'CODE': 'code'}, inplace=True)
        elevation_band_shp['code'] = elevation_band_shp['code'].astype(int)

        for snow_var in self.snow_vars:
            self.data = du.calculate_percentile_snow_bands(
                self.data, elevation_band_shp,
                num_bands=self.experiment_config["num_elevation_zones"], col_name=snow_var)
            snow_vars_drop = [col for col in self.data.columns if snow_var in col and 'P' not in col]
            self.data = self.data.drop(columns=snow_vars_drop)

        logger.debug('Data preprocessing completed. Data shape: %s', self.data.shape)

        # -------------- 4. Feature Extraction ------------------------------
        self.__extract_features__()

        # -------------- 5. Load LR predictors if configured ------------------------------
        if self.general_config['use_lr_predictors']:
            lr_predictors = self.__load_lr_predictors__()
            self.data = pd.merge(self.data, lr_predictors, on=['date', 'code'], how='inner')

        # -------------- 6. Dummy encoding for categorical features ------------------------------
        self.data['basin'] = self.data['code']
        self.data['code_str'] = self.data['code'].astype(str)
        self.data_with_dummies = pd.get_dummies(self.data, columns=['month', 'basin'], dtype=int)


        # -------------- 7. Merge with static features ------------------------------
        static_df_feat = self.static_data[['code'] + self.static_features].copy()
        self.data_with_dummies = pd.merge(self.data_with_dummies, static_df_feat, on='code', how='inner')

        # -------------- 8. Prepare feature sets ------------------------------
        lr_features = [col for col in self.data_with_dummies.columns if 'LR' in col]
        self.feature_set = [col + "_" for col in self.feature_cols] + \
        ['week_sin', 'week_cos','month_sin', 'month_cos'] + \
        self.static_features + lr_features

    def __extract_features__(self):
        """
        Extract features from the data using FeatureExtractor and prepare for global fitting.
        """

        keys_to_remove = [key for key in self.feature_config.keys() if key not in self.feature_cols]
        for key in keys_to_remove:
            self.feature_config.pop(key)

        logger.debug('Extracting features using FeatureExtractor')
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
    
        # Add cyclical encoding for temporal features
        self.data['month_sin'] = np.sin(2 * np.pi * self.data['month'] / 12)
        self.data['month_cos'] = np.cos(2 * np.pi * self.data['month'] / 12)
        
        # Add week features if available
        self.data['week'] = self.data['date'].dt.isocalendar().week
        self.data['week_sin'] = np.sin(2 * np.pi * self.data['week'] / 52)
        self.data['week_cos'] = np.cos(2 * np.pi * self.data['week'] / 52)

        logger.debug('Feature extraction completed. Data shape: %s', self.data.shape)

    def __load_lr_predictors__(self) -> pd.DataFrame:
        pass
    def __post_process_data__(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    def predict_operational(self) -> pd.DataFrame:
        """
        Predict in operational mode using global trained models.

        Returns:
            forecast (pd.DataFrame): DataFrame containing the forecasted values.
        """
        logger.info(f"Starting operational prediction for {self.name}")
        
        today = datetime.datetime.now()
        today_year = today.year

        # Load models
        models = self.load_model()

        # Prepare the prediction data, use only the last 2 years of data (for fast processing)
        self.data = self.data[self.data['date'] >= (today - pd.DateOffset(years=2))].copy()
        self.__preprocess_data__()


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
            continue

        
        return forecast
    
    def calibrate_model_and_hindcast(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calibrate the ensemble models using Leave-One-Year-Out cross-validation.

        Args:
            data (pd.DataFrame): DataFrame containing the calibration data.

        Returns:
            hindcast (pd.DataFrame): DataFrame containing the hindcasted values.
        """
        logger.info(f"Starting calibration and hindcasting for {self.name} with models: {self.models}")
        
        # Add day column if not present
        if 'day' not in self.data.columns:
            self.data['day'] = self.data['date'].dt.day
        
        # Get configuration parameters
        test_years = self.general_config.get('test_years', 3)
        forecast_days = self.general_config.get('forecast_days', [5, 10, 15, 20, 25, 'end'])
        
 
        
        return hindcast_df
    
    def tune_hyperparameters(self, data: pd.DataFrame) -> Tuple[bool, str]:
        pass

    
    def save_model(self) -> None:
        """
        Save all fitted models and preprocessing artifacts.
        """
        logger.info(f"Saving {self.name} models")

    def load_model(self) -> None:
        """
        Load all fitted models and preprocessing artifacts.
        """
        logger.info(f"Loading {self.name}  models")