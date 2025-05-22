import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import matplotlib.pyplot as plt


from forecast_models.base_class import BaseForecastModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

from scr import FeatureExtractor as FE


# Shared logging
# Ensure the logs directory exists
import datetime
import logging
from log_config import setup_logging
setup_logging()  

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


class SciRegressor(BaseForecastModel):
    """
    A regressor class for models which can be fitted using Sci-Kit Learn, XGBoost, LightGBM, etc.
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
            general_config (Dict[str, Any]): General configuration for the model.
            model_config (Dict[str, Any]): Model-specific configuration.
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

        self.models = self.general_config['models']

        logger.debug('Preprocessing data for SciRegressor model')
        self.__preprocess_data__()
        logger.debug('Data preprocessed successfully')
        logger.debug('Extracting features for SciRegressor model')
        self.__extract_features__()
        logger.debug('Features extracted successfully')

    def __preprocess_data__(self):
        """
        Preprocess the data by handling missing values and scaling features.
        """
        # Implement preprocessing logic here
        self.data.fillna(0, inplace=True)

    def __extract_features__(self):
        """
        Extract features from the data.

        Args:
            data (pd.DataFrame): DataFrame containing the data.
        """
        # Implement feature extraction logic here
        extractor = FE.StreamflowFeatureExtractor(
            feature_configs=self.feature_config,
            prediction_horizon=self.general_config['prediction_horizon'],
            offset=self.general_config['offset'],
        )

        self.data = extractor.create_all_features(self.data)
        self.data['year'] = self.data['date'].dt.year
        self.data['month'] = self.data['date'].dt.month


    def predict_operational(self, data):
        return super().predict_operational(data)
    
    def calibrate_model_and_hindcast(self, data):

        return super().calibrate_model_and_hindcast(data)
    
    def tune_hyperparameters(self):
        return super().tune_hyperparameters()