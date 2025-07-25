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

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class DeepMetaLearner(BaseForecastModel):
    """
    Deep learning meta-learner for monthly discharge forecasting.
    
    This class focuses on meta-learning approaches where deep learning models
    learn to combine predictions from multiple base models (e.g., SciRegressor outputs)
    and/or learn from historical performance patterns.
    
    Typical use cases:
    - Uncertainty quantification using Asymmetric Laplace models
    - Learning to weight predictions from multiple base models
    - Historical performance-based meta-learning
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
        Initialize the DeepMetaLearner model.

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

        # Meta-learning specific configuration
        self.meta_learning_approach = self.model_config.get("meta_learning_approach", "uncertainty")
        
        # Deep learning specific configuration
        self.architecture = self.model_config.get("architecture", "uncertainty_net")
        self.hidden_size = self.model_config.get("hidden_size", 64)
        self.num_layers = self.model_config.get("num_layers", 2)
        self.dropout = self.model_config.get("dropout", 0.1)
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 32)
        self.max_epochs = self.model_config.get("max_epochs", 100)
        self.patience = self.model_config.get("patience", 10)
        
        # Meta-learning data configuration
        self.lookback_steps = self.model_config.get("lookback_steps", 365)
        self.future_steps = self.model_config.get("future_steps", 30)
        self.use_base_predictions = self.model_config.get("use_base_predictions", True)
        self.use_historical_errors = self.model_config.get("use_historical_errors", True)
        self.use_uncertainty_features = self.model_config.get("use_uncertainty_features", True)
        
        # Training configuration  
        self.target = self.general_config.get("target", "target")
        self.feature_cols = self.general_config.get("feature_cols", ["discharge", "P", "T"])
        self.static_features = self.general_config.get("static_features", [])
        self.rivers_to_exclude = self.general_config.get("rivers_to_exclude", [])
        self.base_model_names = self.general_config.get("base_model_names", ["SciRegressor"])
        
        # Cross-validation configuration
        self.hparam_tuning_years = self.general_config.get("hparam_tuning_years", 3)
        self.early_stopping_val_fraction = self.general_config.get("early_stopping_val_fraction", 0.1)
        self.num_test_years = self.general_config.get("num_test_years", 2)
        
        # Model storage
        self.fitted_models = {}  # Will store Lightning models per period
        self.feature_processing_artifacts = None
        self.data_scalers = {}
        self.base_model_predictions = {}  # Store base model predictions for meta-learning

    def __preprocess_data__(self):
        """Preprocess data for deep meta-learning models."""
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

        # Sort by date and filter
        self.data.sort_values(by="date", inplace=True)
        self.data = self.data[~self.data["code"].isin(self.rivers_to_exclude)].copy()
        
        # Load base model predictions if using meta-learning approach
        if self.use_base_predictions:
            self._load_base_model_predictions()
            
        logger.info(f"Preprocessed data shape: {self.data.shape}")

    def _load_base_model_predictions(self):
        """Load predictions from base models for meta-learning."""
        logger.info("Loading base model predictions for meta-learning")
        
        for model_name in self.base_model_names:
            try:
                # TODO: Implement loading of base model predictions
                # This would typically load hindcast results from base models
                # and align them with the current data for meta-learning
                
                logger.info(f"Loading predictions from {model_name}")
                # predictions_path = os.path.join(self.path_config["predictions_path"], f"{model_name}_hindcast.parquet")
                # base_predictions = pd.read_parquet(predictions_path)
                # self.base_model_predictions[model_name] = base_predictions
                
            except Exception as e:
                logger.warning(f"Could not load predictions from {model_name}: {e}")

    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict in operational mode using trained deep meta-learning models.
        
        Args:
            today: Date to use as "today" for prediction
            
        Returns:
            DataFrame with columns ['date', 'model', 'code', 'Q_pred'] and possibly uncertainty quantiles
        """
        if today is None:
            today = datetime.datetime.now()
            
        logger.info(f"Starting operational prediction for {self.name} on {today}")
        
        # Load models if not already loaded
        if not self.fitted_models:
            self.load_model()
            
        # TODO: Implement operational prediction logic for meta-learning
        # This will involve:
        # 1. Loading current base model predictions
        # 2. Preparing meta-features (historical errors, performance metrics, etc.)
        # 3. Running inference with the trained meta-learning models
        # 4. Generating uncertainty estimates if using uncertainty approaches
        
        raise NotImplementedError("Operational prediction not yet implemented")

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate deep meta-learning models using Leave-One-Year-Out cross-validation.
        
        Returns:
            DataFrame containing hindcasted values with uncertainty estimates
        """
        logger.info(f"Starting calibration and hindcast for {self.name}")
        
        # Preprocess data
        self.__preprocess_data__()
        
        # Get unique years for cross-validation
        years = sorted(self.data['date'].dt.year.unique())
        hindcast_results = []
        
        for test_year in progress_bar(years, desc="Meta-learning cross-validation"):
            logger.info(f"Training meta-learner for test year: {test_year}")
            
            # Split data into train/test
            train_data = self.data[self.data['date'].dt.year != test_year].copy()
            test_data = self.data[self.data['date'].dt.year == test_year].copy()
            
            if len(test_data) == 0:
                logger.warning(f"No test data for year {test_year}, skipping")
                continue
                
            # Prepare meta-learning features
            train_meta_features = self._prepare_meta_features(train_data, test_year, is_training=True)
            test_meta_features = self._prepare_meta_features(test_data, test_year, is_training=False)
            
            # Create datasets and dataloaders
            train_dataset = self._create_meta_dataset(train_meta_features, is_training=True)
            test_dataset = self._create_meta_dataset(test_meta_features, is_training=False)
            
            # Train meta-learning model for this fold
            model = self._train_meta_model(train_dataset, test_year)
            
            # Generate meta-predictions
            predictions = self._predict_with_meta_model(model, test_dataset, test_year)
            hindcast_results.append(predictions)
            
            # Store the trained model
            self.fitted_models[test_year] = model

        if hindcast_results:
            hindcast_df = pd.concat(hindcast_results, ignore_index=True)
            logger.info(f"Meta-learning hindcast completed. Shape: {hindcast_df.shape}")
            return hindcast_df
        else:
            logger.error("No hindcast results generated")
            return pd.DataFrame()

    def _prepare_meta_features(self, data: pd.DataFrame, test_year: int, is_training: bool) -> Dict[str, np.ndarray]:
        """Prepare meta-learning features including base predictions and historical performance."""
        # TODO: Implement meta-feature preparation
        # This will involve:
        # 1. Extracting base model predictions for the relevant time period
        # 2. Computing historical error statistics
        # 3. Preparing performance-based features
        # 4. Creating temporal and spatial context features
        
        logger.info(f"Preparing meta-features for {'training' if is_training else 'testing'}")
        
        meta_features = {
            'base_predictions': np.array([]),  # Base model predictions
            'historical_errors': np.array([]),  # Historical error patterns
            'performance_metrics': np.array([]),  # Historical performance metrics
            'temporal_features': np.array([]),  # Time-based features
            'static_features': np.array([]),  # Basin characteristics
        }
        
        return meta_features
    
    def _create_meta_dataset(self, meta_features: Dict[str, np.ndarray], is_training: bool) -> DeepLearningDataset:
        """Create deep learning dataset from meta-learning features."""
        # TODO: Implement meta-dataset creation
        # This will convert meta-features into the tensor format required by the model
        
        raise NotImplementedError("Meta-dataset creation not yet implemented")
    
    def _train_meta_model(self, train_dataset: DeepLearningDataset, test_year: int) -> pl.LightningModule:
        """Train a deep meta-learning model for the given fold."""
        # TODO: Implement meta-model training
        # This will involve:
        # 1. Creating the appropriate meta-learning architecture (UncertaintyNet, etc.)
        # 2. Setting up meta-learning specific loss functions
        # 3. Training with uncertainty-aware objectives
        
        raise NotImplementedError("Meta-model training not yet implemented")
    
    def _predict_with_meta_model(self, model: pl.LightningModule, dataset: DeepLearningDataset, year: int) -> pd.DataFrame:
        """Generate meta-predictions with uncertainty estimates."""
        # TODO: Implement meta-prediction logic
        # This will involve:
        # 1. Running meta-model inference
        # 2. Generating uncertainty estimates (quantiles, intervals, etc.)
        # 3. Post-processing meta-predictions
        
        raise NotImplementedError("Meta-model prediction not yet implemented")

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune meta-learning hyperparameters using Optuna.
        
        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Starting hyperparameter tuning for {self.name}")
        
        try:
            import optuna
            
            # TODO: Implement meta-learning specific hyperparameter tuning
            # This will include tuning of:
            # 1. Architecture-specific parameters
            # 2. Meta-learning approach parameters
            # 3. Uncertainty quantification parameters
            
            return False, "Meta-learning hyperparameter tuning not yet implemented"
            
        except ImportError:
            return False, "Optuna not available for hyperparameter tuning"

    def save_model(self, is_fitted: bool = False) -> None:
        """Save trained meta-learning models and preprocessing artifacts."""
        logger.info(f"Saving {self.name} meta-learning models")
        
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
            
        # Save fitted meta-models if available
        if self.fitted_models:
            models_path = os.path.join(save_path, "meta_models")
            os.makedirs(models_path, exist_ok=True)
            
            for year, model in self.fitted_models.items():
                model_path = os.path.join(models_path, f"meta_model_{year}.ckpt")
                torch.save(model.state_dict(), model_path)
                
        # Save meta-learning artifacts
        if self.base_model_predictions:
            meta_artifacts_path = os.path.join(save_path, "meta_artifacts")
            os.makedirs(meta_artifacts_path, exist_ok=True)
            
            for model_name, predictions in self.base_model_predictions.items():
                pred_path = os.path.join(meta_artifacts_path, f"{model_name}_predictions.parquet")
                predictions.to_parquet(pred_path)
                
        logger.info(f"Meta-learning model saved to {save_path}")

    def load_model(self) -> None:
        """Load trained meta-learning models and preprocessing artifacts."""
        logger.info(f"Loading {self.name} meta-learning models")
        
        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Meta-learning model directory not found: {load_path}")
            
        # Load configurations
        config_path = os.path.join(load_path, "model_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.model_config = json.load(f)
                
        # Load fitted meta-models
        models_path = os.path.join(load_path, "meta_models")
        if os.path.exists(models_path):
            self.fitted_models = {}
            for model_file in os.listdir(models_path):
                if model_file.endswith('.ckpt'):
                    year = int(model_file.split('_')[2].split('.')[0])
                    model_path = os.path.join(models_path, model_file)
                    
                    # TODO: Load meta-model with proper architecture
                    # model = self._create_meta_model_architecture()
                    # model.load_state_dict(torch.load(model_path))
                    # self.fitted_models[year] = model
                    
        # Load meta-learning artifacts
        meta_artifacts_path = os.path.join(load_path, "meta_artifacts")
        if os.path.exists(meta_artifacts_path):
            self.base_model_predictions = {}
            for pred_file in os.listdir(meta_artifacts_path):
                if pred_file.endswith('.parquet'):
                    model_name = pred_file.replace('_predictions.parquet', '')
                    pred_path = os.path.join(meta_artifacts_path, pred_file)
                    self.base_model_predictions[model_name] = pd.read_parquet(pred_path)
                    
        logger.info(f"Meta-learning model loaded from {load_path}")