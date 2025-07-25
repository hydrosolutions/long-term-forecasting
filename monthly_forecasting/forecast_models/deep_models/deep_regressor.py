import os
import pandas as pd
import numpy as np
import datetime
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm as progress_bar

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from monthly_forecasting.forecast_models.base_class import BaseForecastModel
from monthly_forecasting.scr import FeatureExtractor as FE
from monthly_forecasting.scr.FeatureProcessingArtifacts import (
    process_training_data,
    process_test_data,
    post_process_predictions,
    FeatureProcessingArtifacts,
)
from monthly_forecasting.scr import data_utils as du

# Import deep learning components
from .utils.data_utils import DeepLearningDataset, create_deep_learning_dataloader
from .utils.lightning_base import LightningForecastBase
from .architectures.uncertainty_models import UncertaintyNet
from .architectures.lstm_models import LSTMForecaster

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class DeepRegressor(BaseForecastModel):
    """
    Deep learning regressor for monthly discharge forecasting.
    
    Supports various neural network architectures (LSTM, CNN-LSTM, TiDE, TSMixer, Mamba)
    with unified interface matching SciRegressor for seamless integration with existing
    evaluation pipeline.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
    ) -> None:
        """
        Initialize the DeepRegressor model.

        Args:
            data: Time series data with columns ['date', 'code', 'discharge', ...]
            static_data: Static basin characteristics
            general_config: General configuration
            model_config: Deep learning model hyperparameters
            feature_config: Feature engineering configuration
            path_config: Path configuration for saving/loading
        """
        super().__init__(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        # Deep learning specific configuration
        self.architecture = self.model_config.get("architecture", "lstm")
        self.hidden_size = self.model_config.get("hidden_size", 64)
        self.num_layers = self.model_config.get("num_layers", 2)
        self.dropout = self.model_config.get("dropout", 0.1)
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 32)
        self.max_epochs = self.model_config.get("max_epochs", 100)
        self.patience = self.model_config.get("patience", 10)
        
        # Data processing parameters
        self.lookback_steps = self.model_config.get("lookback_steps", 365)
        self.future_steps = self.model_config.get("future_steps", 30)
        self.use_nan_mask = self.model_config.get("use_nan_mask", True)
        
        # Training configuration
        self.target = self.general_config.get("target", "target")
        self.feature_cols = self.general_config.get("feature_cols", ["discharge", "P", "T"])
        self.static_features = self.general_config.get("static_features", [])
        self.rivers_to_exclude = self.general_config.get("rivers_to_exclude", [])
        self.hparam_tuning_years = self.general_config.get("hparam_tuning_years", 3)
        self.early_stopping_val_fraction = self.general_config.get("early_stopping_val_fraction", 0.1)
        self.num_test_years = self.general_config.get("num_test_years", 2)
        
        # Model storage
        self.fitted_models = {}  # Will store Lightning models per period
        self.feature_processing_artifacts = None
        self.data_scalers = {}

    def __preprocess_data__(self):
        """Preprocess data for deep learning models."""
        logger.info(f"Starting data preprocessing for {self.name}")
        logger.info(f"Initial data shape: {self.data.shape}")
        
        try:
            # Add glacier mapper features
            self.data = du.glacier_mapper_features(
                df=self.data,
                static=self.static_data,
                cols_to_keep=self.general_config["glacier_mapper_features_to_keep"],
            )
        except Exception as e:
            logger.error(f"Error in glacier_mapper_features: {e}")

        # Remove log_discharge if it exists
        if "log_discharge" in self.data.columns:
            self.data.drop(columns=["log_discharge"], inplace=True)

        # Sort by date and filter columns
        self.data.sort_values(by="date", inplace=True)
        cols_to_keep = [
            col for col in self.data.columns
            if any([feature in col for feature in self.feature_cols])
        ]
        self.data = self.data[["date", "code"] + cols_to_keep]

        # Filter rivers
        self.data = self.data[~self.data["code"].isin(self.rivers_to_exclude)].copy()
        
        logger.info(f"Preprocessed data shape: {self.data.shape}")

    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict in operational mode using trained deep learning models.
        
        Args:
            today: Date to use as "today" for prediction
            
        Returns:
            DataFrame with columns ['date', 'model', 'code', 'Q_pred']
        """
        if today is None:
            today = datetime.datetime.now()
            
        logger.info(f"Starting operational prediction for {self.name} on {today}")
        
        # Load models if not already loaded
        if not self.fitted_models:
            self.load_model()
            
        # TODO: Implement operational prediction logic
        # This will involve:
        # 1. Preparing input data in the required tensor format
        # 2. Running inference with the trained models
        # 3. Post-processing predictions
        
        raise NotImplementedError("Operational prediction not yet implemented")

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate deep learning models using Leave-One-Year-Out cross-validation.
        
        Returns:
            DataFrame containing hindcasted values with columns ['date', 'model', 'code', 'Q_pred']
        """
        logger.info(f"Starting calibration and hindcast for {self.name}")
        
        # Preprocess data
        self.__preprocess_data__()
        
        # Get unique years for cross-validation
        years = sorted(self.data['date'].dt.year.unique())
        hindcast_results = []
        
        for test_year in progress_bar(years, desc="Cross-validation"):
            logger.info(f"Training for test year: {test_year}")
            
            # Split data into train/test
            train_data = self.data[self.data['date'].dt.year != test_year].copy()
            test_data = self.data[self.data['date'].dt.year == test_year].copy()
            
            if len(test_data) == 0:
                logger.warning(f"No test data for year {test_year}, skipping")
                continue
                
            # Create datasets and dataloaders
            train_dataset = self._create_dataset(train_data, is_training=True)
            test_dataset = self._create_dataset(test_data, is_training=False)
            
            # Train model for this fold
            model = self._train_model(train_dataset, test_year)
            
            # Generate predictions
            predictions = self._predict_with_model(model, test_dataset, test_year)
            hindcast_results.append(predictions)
            
            # Store the trained model
            self.fitted_models[test_year] = model

        if hindcast_results:
            hindcast_df = pd.concat(hindcast_results, ignore_index=True)
            logger.info(f"Hindcast completed. Shape: {hindcast_df.shape}")
            return hindcast_df
        else:
            logger.error("No hindcast results generated")
            return pd.DataFrame()

    def _create_dataset(self, data: pd.DataFrame, is_training: bool) -> DeepLearningDataset:
        """Create deep learning dataset from preprocessed data."""
        # TODO: Implement dataset creation logic
        # This will involve converting time series data to the required tensor format:
        # - x_past: (batch, past_time_steps, past_features)
        # - x_nan_mask: (batch, past_time_steps, past_features) 
        # - x_future: (batch, future_time_steps, future_vars)
        # - x_now: (batch, 1, now_vars)
        # - x_static: static basin features
        
        raise NotImplementedError("Dataset creation not yet implemented")
    
    def _train_model(self, train_dataset: DeepLearningDataset, test_year: int) -> pl.LightningModule:
        """Train a deep learning model for the given fold."""
        # TODO: Implement training logic
        # This will involve:
        # 1. Creating the Lightning module with the specified architecture
        # 2. Setting up callbacks (checkpoint, early stopping, etc.)
        # 3. Training the model with the specified hyperparameters
        
        raise NotImplementedError("Model training not yet implemented")
    
    def _predict_with_model(self, model: pl.LightningModule, dataset: DeepLearningDataset, year: int) -> pd.DataFrame:
        """Generate predictions using a trained model."""
        # TODO: Implement prediction logic
        # This will involve:
        # 1. Running model inference on the dataset
        # 2. Converting tensor outputs back to DataFrame format
        # 3. Post-processing predictions (inverse scaling, etc.)
        
        raise NotImplementedError("Model prediction not yet implemented")

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune hyperparameters using Optuna.
        
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Starting hyperparameter tuning for {self.name}")
        
        try:
            import optuna
            
            # TODO: Implement hyperparameter tuning
            # This will involve:
            # 1. Defining the objective function
            # 2. Setting up the search space
            # 3. Running optimization
            # 4. Updating model_config with best parameters
            
            return False, "Hyperparameter tuning not yet implemented"
            
        except ImportError:
            return False, "Optuna not available for hyperparameter tuning"

    def save_model(self, is_fitted: bool = False) -> None:
        """Save trained models and preprocessing artifacts."""
        logger.info(f"Saving {self.name} models")
        
        save_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        os.makedirs(save_path, exist_ok=True)
        
        # Save model configuration
        config_path = os.path.join(save_path, "model_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.model_config, f, indent=2)
            
        # Save feature configuration
        feature_config_path = os.path.join(save_path, "feature_config.json")
        with open(feature_config_path, 'w') as f:
            json.dump(self.feature_config, f, indent=2)
            
        # Save experiment configuration
        experiment_config_path = os.path.join(save_path, "experiment_config.json")
        with open(experiment_config_path, 'w') as f:
            json.dump(self.general_config, f, indent=2)
            
        # Save fitted models if available
        if self.fitted_models:
            models_path = os.path.join(save_path, "models")
            os.makedirs(models_path, exist_ok=True)
            
            for year, model in self.fitted_models.items():
                model_path = os.path.join(models_path, f"model_{year}.ckpt")
                torch.save(model.state_dict(), model_path)
                
        # Save feature processing artifacts if available
        if self.feature_processing_artifacts:
            artifacts_path = os.path.join(save_path, "artifacts")
            self.feature_processing_artifacts.save_to_disk(artifacts_path)
            
        logger.info(f"Model saved to {save_path}")

    def load_model(self) -> None:
        """Load trained models and preprocessing artifacts."""
        logger.info(f"Loading {self.name} models")
        
        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Model directory not found: {load_path}")
            
        # Load configurations
        config_path = os.path.join(load_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
                
        # Load fitted models
        models_path = os.path.join(load_path, "models")
        if os.path.exists(models_path):
            self.fitted_models = {}
            for model_file in os.listdir(models_path):
                if model_file.endswith('.ckpt'):
                    year = int(model_file.split('_')[1].split('.')[0])
                    model_path = os.path.join(models_path, model_file)
                    
                    # TODO: Load model with proper architecture
                    # model = self._create_model_architecture()
                    # model.load_state_dict(torch.load(model_path))
                    # self.fitted_models[year] = model
                    
        # Load feature processing artifacts
        artifacts_path = os.path.join(load_path, "artifacts")
        if os.path.exists(artifacts_path):
            self.feature_processing_artifacts = FeatureProcessingArtifacts.load_from_disk(artifacts_path)
            
        logger.info(f"Model loaded from {load_path}")