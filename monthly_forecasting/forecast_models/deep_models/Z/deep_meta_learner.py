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
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

from monthly_forecasting.forecast_models.meta_learners.base_meta_learner import (
    BaseMetaLearner,
)
from monthly_forecasting.scr import FeatureExtractor as FE
from monthly_forecasting.scr import data_utils as du

# Import deep learning components
from monthly_forecasting.deep_scr.meta_base import LitMetaForecastBase
from monthly_forecasting.deep_scr.data_class import MetaMonthDataModule, META_MONTH_DATA

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class DeepMetaLearner(BaseMetaLearner):
    """
    Deep learning meta-learner for monthly discharge forecasting.

    Inherits from BaseMetaLearner to leverage existing meta-learning infrastructure
    (base predictor loading, historical performance calculation) while using
    deep learning models for meta-learning instead of simple weighted ensembles.

    Key differences from HistoricalMetaLearner:
    - Uses neural networks to learn meta-features and weights
    - Incorporates uncertainty quantification
    - Learns from base model predictions, errors, and performance patterns
    - Supports more complex meta-learning architectures
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
            general_config: General configuration including meta-learning settings
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

        # Deep learning model configuration
        self.architecture = self.model_config.get("architecture", "uncertainty_net")
        self.hidden_size = self.model_config.get("hidden_size", 64)
        self.num_layers = self.model_config.get("num_layers", 2)
        self.dropout = self.model_config.get("dropout", 0.1)
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 32)
        self.max_epochs = self.model_config.get("max_epochs", 100)
        self.patience = self.model_config.get("patience", 10)
        self.weight_decay = self.model_config.get("weight_decay", 0.0)

        # Meta-learning configuration
        self.lookback_steps = self.model_config.get("lookback_steps", 90)
        self.future_steps = self.model_config.get("future_steps", 15)
        self.quantiles = self.model_config.get("quantiles", [0.1, 0.5, 0.9])
        self.loss_fn = self.model_config.get("loss_fn", "QuantileLoss")

        # Meta-features configuration
        self.use_base_predictions = self.model_config.get("use_base_predictions", True)
        self.use_historical_performance = self.model_config.get(
            "use_historical_performance", True
        )
        self.use_temporal_features = self.model_config.get(
            "use_temporal_features", True
        )

        # Model storage
        self.fitted_models = {}  # Lightning models per year for LOOCV
        self.data_scalers = {}
        self.base_model_predictions = {}  # Base model predictions for meta-learning

    def __preprocess_data__(self):
        """
        Preprocess data for deep meta-learning models.
        Leverages BaseMetaLearner infrastructure while adding deep learning specific features.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Preprocessed data and list of base model names
        """
        logger.info(f"Starting data preprocessing for deep meta-learning: {self.name}")

        # 1. Create target variable using FeatureExtractor (same as HistoricalMetaLearner)
        logger.info("Creating target variable with FeatureExtractor")

        extractor = FE.StreamflowFeatureExtractor(
            feature_configs={},  # empty dict to not create any features
            prediction_horizon=self.general_config["prediction_horizon"],
            offset=self.general_config.get(
                "offset", self.general_config["prediction_horizon"]
            ),
        )

        # Get target variable
        target = extractor.create_target(self.data)
        dates = pd.to_datetime(self.data["date"])
        codes = self.data["code"].astype(int)
        target_data = pd.DataFrame({"date": dates, "code": codes, "Q_obs": target})

        # 2. Load base predictors using inherited method
        logger.info("Loading base predictors")
        base_predictors, model_names = self.__load_base_predictors__()

        # 3. Merge the base predictors with target data
        logger.info("Merging base predictors with target data")
        merged_data = target_data.merge(
            base_predictors, on=["date", "code"], how="left"
        )

        # 4. Create periods column for temporal features
        logger.info("Creating period columns")
        preprocessed_data = du.get_periods(merged_data)

        # 5. Add temporal features for deep learning
        logger.info("Adding temporal features for deep learning")
        preprocessed_data = self._add_temporal_features(preprocessed_data)

        # 6. Calculate historical performance if needed for meta-features
        if self.use_historical_performance:
            logger.info("Calculating historical performance for meta-features")
            self.historical_performance = self.__calculate_historical_performance__(
                preprocessed_data, model_names
            )

        # 7. Add deep learning specific features
        logger.info("Adding deep learning specific features")
        preprocessed_data = self._add_deep_learning_features(
            preprocessed_data, model_names
        )

        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        logger.info(f"Model names: {model_names}")
        logger.info(f"Available columns: {preprocessed_data.columns.tolist()}")

        return preprocessed_data, model_names

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for deep learning models."""
        data = data.copy()

        if self.use_temporal_features:
            # Add date components
            data["year"] = pd.to_datetime(data["date"]).dt.year
            data["month"] = pd.to_datetime(data["date"]).dt.month
            data["day"] = pd.to_datetime(data["date"]).dt.day
            data["day_of_year"] = pd.to_datetime(data["date"]).dt.dayofyear

            # Add cyclical features
            data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
            data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
            data["day_sin"] = np.sin(2 * np.pi * data["day_of_year"] / 365.25)
            data["day_cos"] = np.cos(2 * np.pi * data["day_of_year"] / 365.25)

        return data

    def _add_deep_learning_features(
        self, data: pd.DataFrame, model_names: List[str]
    ) -> pd.DataFrame:
        """Add deep learning specific meta-features."""
        data = data.copy()

        # Add historical performance features if available
        if hasattr(self, "historical_performance") and self.use_historical_performance:
            logger.info("Adding historical performance features")
            # Merge historical performance metrics
            performance_features = []
            for model_name in model_names:
                if model_name in self.historical_performance.columns:
                    perf_col = f"{model_name}_hist_perf"
                    performance_features.append(perf_col)

            if performance_features:
                data = data.merge(
                    self.historical_performance,
                    on=["code", "period"],
                    how="left",
                    suffixes=("", "_hist_perf"),
                )

        # Add error features for base models
        for model_name in model_names:
            if model_name in data.columns and "Q_obs" in data.columns:
                error_col = f"{model_name}_error"
                data[error_col] = data["Q_obs"] - data[model_name]

                # Add rolling error statistics
                data[f"{model_name}_error_ma7"] = (
                    data.groupby("code")[error_col]
                    .rolling(window=7, min_periods=1)
                    .mean()
                    .values
                )
                data[f"{model_name}_error_std7"] = (
                    data.groupby("code")[error_col]
                    .rolling(window=7, min_periods=1)
                    .std()
                    .values
                )

        return data

    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict in operational mode using trained deep meta-learning models.

        Args:
            today: Date to use as "today" for prediction

        Returns:
            DataFrame with columns ['date', 'code', 'Q_{self.name}'] and uncertainty quantiles
        """
        if today is None:
            today = datetime.datetime.now()

        logger.info(f"Starting operational prediction for {self.name} on {today}")

        # Load models if not already loaded
        if not self.fitted_models:
            self.load_model()

        if "operational" not in self.fitted_models:
            logger.error(
                "No operational model found. Please run calibrate_model_and_hindcast first."
            )
            return pd.DataFrame()

        # Preprocess current data
        data, model_names = self.__preprocess_data__()

        # Filter data for today's prediction (or most recent data)
        today_data = data[data["date"].dt.date == today.date()]

        if len(today_data) == 0:
            logger.warning(
                f"No data available for {today.date()}, using most recent data"
            )
            # Use most recent data available
            latest_date = data["date"].max()
            today_data = data[data["date"] == latest_date]

        if len(today_data) == 0:
            logger.error("No data available for operational prediction")
            return pd.DataFrame()

        # Scale the data using stored scalers
        if not hasattr(self, "data_scalers") or not self.data_scalers:
            logger.error(
                "No data scalers found. Please run calibrate_model_and_hindcast first."
            )
            return pd.DataFrame()

        scaled_data = self._scale_data(today_data, self.data_scalers, model_names)

        # Prepare features
        past_features, future_features, static_features, base_learner_cols = (
            self._prepare_feature_columns(model_names)
        )

        # Create data module for prediction
        data_module = MetaMonthDataModule(
            df=None,
            static_df=self.static_data,
            static_features=static_features,
            past_features=past_features,
            future_features=future_features,
            base_learner_cols=base_learner_cols,
            base_learner_add_cols=[],
            train_years=[],
            val_years=[],
            test_years=[],
            lookback=self.lookback_steps,
            future_known_steps=self.future_steps,
            batch_size=self.batch_size,
            num_workers=0,
            train_df=scaled_data,  # Dummy
            val_df=scaled_data,  # Dummy
            test_df=scaled_data,  # Use for prediction
        )

        # Generate predictions using operational model
        operational_predictions = self._generate_meta_predictions(
            self.fitted_models["operational"],
            data_module,
            self.data_scalers,
            "operational",
        )

        if len(operational_predictions) == 0:
            logger.error("No operational predictions generated")
            return pd.DataFrame()

        # Keep only the required columns
        result_cols = ["date", "code", f"Q_{self.name}"]
        # Add quantile columns if they exist
        quantile_cols = [
            col
            for col in operational_predictions.columns
            if col.startswith("Q") and col != f"Q_{self.name}"
        ]
        result_cols.extend(quantile_cols)

        # Filter to only include existing columns
        available_cols = [
            col for col in result_cols if col in operational_predictions.columns
        ]

        logger.info(
            f"Operational prediction completed: {len(operational_predictions)} predictions"
        )

        return operational_predictions[available_cols].copy()

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate deep meta-learning models using Leave-One-Year-Out cross-validation.

        Key modification: The LOOCV test year includes input for the first forecasting date,
        following the general calibration workflow in deep_scr/train_eval.py.

        Returns:
            DataFrame containing hindcasted values with uncertainty estimates
        """
        logger.info(f"Starting calibration and hindcast for {self.name}")

        # Preprocess data
        data, model_names = self.__preprocess_data__()

        # Drop rows without target values
        data.dropna(subset=["Q_obs"], inplace=True)

        # Get available years for LOOCV
        available_years = sorted(data["date"].dt.year.unique())
        logger.info(f"Available years for LOOCV: {available_years}")

        # Run LOOCV for meta-learning
        hindcast_predictions = self._loocv_deep_meta(data, model_names, available_years)

        if len(hindcast_predictions) == 0:
            logger.error("No hindcast predictions generated")
            return pd.DataFrame()

        # Save the final model weights for operational prediction
        logger.info("Saving final model for operational prediction")
        final_model = self._train_final_meta_model(data, model_names)
        self.fitted_models["operational"] = final_model

        logger.info(
            f"Calibration completed: {len(hindcast_predictions)} hindcast predictions"
        )
        return hindcast_predictions[["date", "code", "Q_obs", f"Q_{self.name}"]].copy()

    def _loocv_deep_meta(
        self, data: pd.DataFrame, model_names: List[str], loocv_years: List[int]
    ) -> pd.DataFrame:
        """
        Perform Leave-One-Year-Out Cross-Validation for deep meta-learning.

        Following the pattern from deep_scr/train_eval.py with modification:
        The LOOCV test year contains input for the first forecasting date.

        Args:
            data: Preprocessed data with targets and base model predictions
            model_names: List of base model names
            loocv_years: Years to use for LOOCV

        Returns:
            DataFrame containing ensemble predictions for all validation years
        """
        logger.info(f"Starting deep meta-learning LOOCV for years: {loocv_years}")

        all_predictions = []

        for year in progress_bar(loocv_years, desc="Deep Meta LOOCV", leave=True):
            logger.info(f"Processing deep meta LOOCV for year {year}")

            # Split data: training vs test (including input for forecast date)
            train_data = data[data["date"].dt.year != year].copy()
            test_data = data[data["date"].dt.year == year].copy()

            if len(train_data) == 0:
                logger.warning(f"No training data for year {year}")
                continue

            if len(test_data) == 0:
                logger.warning(f"No test data for year {year}")
                continue

            # Create data scalers following deep_scr pattern
            scalers = self._create_data_scalers(train_data, model_names)

            # Scale the data
            train_scaled = self._scale_data(train_data, scalers, model_names)
            test_scaled = self._scale_data(test_data, scalers, model_names)

            # Prepare feature columns for deep learning
            past_features, future_features, static_features, base_learner_cols = (
                self._prepare_feature_columns(model_names)
            )

            # Create data module following MetaMonthDataModule pattern
            data_module = MetaMonthDataModule(
                df=None,  # We provide scaled dataframes directly
                static_df=self.static_data,
                static_features=static_features,
                past_features=past_features,
                future_features=future_features,
                base_learner_cols=base_learner_cols,
                base_learner_add_cols=[],  # Error features already added
                train_years=[],  # Not used since we provide dataframes directly
                val_years=[],
                test_years=[],
                lookback=self.lookback_steps,
                future_known_steps=self.future_steps,
                batch_size=self.batch_size,
                num_workers=0,  # Avoid multiprocessing issues
                train_df=train_scaled,
                val_df=test_scaled,  # Use test as val for early stopping
                test_df=test_scaled,
            )

            # Create and train meta-learning model
            meta_model = self._create_meta_model(
                static_features, past_features, future_features, base_learner_cols
            )

            # Train the model
            trained_model = self._train_deep_meta_model(meta_model, data_module, year)

            # Generate predictions
            predictions = self._generate_meta_predictions(
                trained_model, data_module, scalers, year
            )

            if len(predictions) > 0:
                all_predictions.append(predictions)

            # Store the trained model
            self.fitted_models[year] = trained_model

            logger.info(
                f"Completed deep meta LOOCV for year {year}: {len(predictions)} predictions"
            )

        if len(all_predictions) == 0:
            logger.error("No predictions generated from deep meta LOOCV")
            return pd.DataFrame()

        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)
        logger.info(
            f"Deep meta LOOCV completed: {len(final_predictions)} total predictions"
        )

        return final_predictions

    def _create_data_scalers(
        self, train_data: pd.DataFrame, model_names: List[str]
    ) -> Dict[str, Any]:
        """Create data scalers following the pattern from deep_scr/train_eval.py."""
        logger.info("Creating data scalers for deep meta-learning")

        scalers = {}

        # Per-basin scaling for discharge and target (like normalize_train_data)
        scalers["target_scaler"] = {}
        scalers["discharge_scaler"] = {}

        for code in train_data["code"].unique():
            basin_data = train_data[train_data["code"] == code]

            # Target scaling
            if "Q_obs" in basin_data.columns:
                target_mean = basin_data["Q_obs"].mean()
                target_std = basin_data["Q_obs"].std(ddof=0)
                scalers["target_scaler"][code] = (target_mean, target_std)

            # Discharge scaling (if available)
            if "discharge" in basin_data.columns:
                discharge_mean = basin_data["discharge"].mean()
                discharge_std = basin_data["discharge"].std(ddof=0)
                scalers["discharge_scaler"][code] = (discharge_mean, discharge_std)

        # Global scaling for other features
        scalers["feature_scaler"] = {}
        numeric_cols = [
            col
            for col in train_data.columns
            if col not in ["code", "date", "Q_obs", "discharge"] + model_names
            and pd.api.types.is_numeric_dtype(train_data[col])
        ]

        for col in numeric_cols:
            mean_val = train_data[col].mean()
            std_val = train_data[col].std(ddof=0)
            scalers["feature_scaler"][col] = (mean_val, std_val)

        return scalers

    def _scale_data(
        self, data: pd.DataFrame, scalers: Dict[str, Any], model_names: List[str]
    ) -> pd.DataFrame:
        """Scale data using provided scalers."""
        scaled_data = data.copy()

        # Scale target per basin
        if "target_scaler" in scalers:
            for code in scaled_data["code"].unique():
                if code in scalers["target_scaler"] and "Q_obs" in scaled_data.columns:
                    mean_val, std_val = scalers["target_scaler"][code]
                    mask = scaled_data["code"] == code
                    scaled_data.loc[mask, "Q_obs"] = (
                        scaled_data.loc[mask, "Q_obs"] - mean_val
                    ) / std_val

        # Scale discharge per basin
        if "discharge_scaler" in scalers:
            for code in scaled_data["code"].unique():
                if (
                    code in scalers["discharge_scaler"]
                    and "discharge" in scaled_data.columns
                ):
                    mean_val, std_val = scalers["discharge_scaler"][code]
                    mask = scaled_data["code"] == code
                    scaled_data.loc[mask, "discharge"] = (
                        scaled_data.loc[mask, "discharge"] - mean_val
                    ) / std_val

        # Scale base learner predictions with target scaler
        if "target_scaler" in scalers:
            for code in scaled_data["code"].unique():
                if code in scalers["target_scaler"]:
                    mean_val, std_val = scalers["target_scaler"][code]
                    mask = scaled_data["code"] == code
                    for model_name in model_names:
                        if model_name in scaled_data.columns:
                            scaled_data.loc[mask, model_name] = (
                                scaled_data.loc[mask, model_name] - mean_val
                            ) / std_val

        # Scale other features globally
        if "feature_scaler" in scalers:
            for col, (mean_val, std_val) in scalers["feature_scaler"].items():
                if col in scaled_data.columns:
                    scaled_data[col] = (scaled_data[col] - mean_val) / std_val

        return scaled_data

    def _prepare_feature_columns(
        self, model_names: List[str]
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Prepare feature columns for deep learning models."""

        # Past temporal features (features available until forecast date)
        past_features = ["discharge"] if "discharge" in self.data.columns else []

        # Add temporal features if enabled
        if self.use_temporal_features:
            temporal_features = ["month_sin", "month_cos", "day_sin", "day_cos"]
            past_features.extend(
                [f for f in temporal_features if f in self.data.columns]
            )

        # Future temporal features (features available into the future)
        future_features = []
        # Add weather data or other future-known features here
        # For monthly forecasting, we might have some future weather data

        # Static features (basin characteristics)
        static_features = [col for col in self.static_data.columns if col != "code"]

        # Base learner columns (predictions from other models)
        base_learner_cols = model_names.copy()

        # Add error features for base learners
        error_features = []
        for model_name in model_names:
            error_features.extend(
                [
                    f"{model_name}_error",
                    f"{model_name}_error_ma7",
                    f"{model_name}_error_std7",
                ]
            )

        # Add error features to past features
        past_features.extend([f for f in error_features if f in self.data.columns])

        logger.info(f"Feature columns prepared:")
        logger.info(f"  Past features ({len(past_features)}): {past_features}")
        logger.info(f"  Future features ({len(future_features)}): {future_features}")
        logger.info(f"  Static features ({len(static_features)}): {static_features}")
        logger.info(
            f"  Base learner cols ({len(base_learner_cols)}): {base_learner_cols}"
        )

        return past_features, future_features, static_features, base_learner_cols

    def _create_meta_model(
        self,
        static_features: List[str],
        past_features: List[str],
        future_features: List[str],
        base_learner_cols: List[str],
    ) -> LitMetaForecastBase:
        """Create meta-learning model following deep_scr pattern."""

        # Import the specific model architectures from deep_scr
        try:
            from monthly_forecasting.deep_scr.AL_UncertaintyNet import (
                AL_Uncertainty_Forecast,
            )
            from monthly_forecasting.deep_scr.UncertaintyNet import Uncertainty_Forecast
            from monthly_forecasting.deep_scr.MLP import MLPForecast
        except ImportError as e:
            logger.error(f"Could not import deep learning models: {e}")
            raise

        # Calculate dimensions
        past_dim = len(past_features)
        future_dim = len(future_features)
        static_dim = len(static_features)
        base_learner_dim = len(base_learner_cols)
        base_learner_error_dim = 0  # Error features are included in past_features

        # Model configuration
        model_config = {
            "past_dim": past_dim,
            "future_dim": future_dim,
            "static_dim": static_dim,
            "base_learner_dim": base_learner_dim,
            "base_learner_error_dim": base_learner_error_dim,
            "lookback": self.lookback_steps,
            "future_known_steps": self.future_steps,
            "hidden_dim": self.hidden_size,
            "lr": self.learning_rate,
            "weight_decay": self.weight_decay,
            "output_dim": len(self.quantiles),
            "loss_fn": self.loss_fn,
            "quantiles": self.quantiles,
            "dropout": self.dropout,
        }

        logger.info(f"Creating {self.architecture} meta-learning model")
        logger.info(
            f"Model dimensions: past={past_dim}, future={future_dim}, static={static_dim}, base_learner={base_learner_dim}"
        )

        # Create model based on architecture
        if self.architecture == "AL_UncertaintyNet":
            model = AL_Uncertainty_Forecast(
                adaptive_weighting=True,
                correction_term=True,
                weight_by_metrics=self.use_historical_performance,
                **model_config,
            )
        elif self.architecture == "UncertaintyNet":
            model = Uncertainty_Forecast(center_weight=0.5, **model_config)
        elif self.architecture == "MLP":
            model = MLPForecast(**model_config)
        else:
            raise ValueError(f"Unknown architecture: {self.architecture}")

        return model

    def _train_deep_meta_model(
        self, model: LitMetaForecastBase, data_module: MetaMonthDataModule, year: int
    ) -> LitMetaForecastBase:
        """Train deep meta-learning model following deep_scr/train_eval.py pattern."""
        logger.info(f"Training deep meta-learning model for year {year}")

        # Setup accelerator
        accelerator = "cpu"
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"

        # Setup callbacks
        checkpoint_dir = os.path.join(
            self.path_config.get("model_home_path", ""), f"{self.name}_checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)

        early_stop_cb = EarlyStopping(
            monitor="val_loss", patience=self.patience, mode="min"
        )

        lr_monitor_cb = LearningRateMonitor(logging_interval="epoch")

        checkpoint_cb = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            filename=f"meta-{year}-{{epoch:02d}}-{{val_loss:.4f}}",
        )

        # Create trainer
        trainer = Trainer(
            max_epochs=self.max_epochs,
            accelerator=accelerator,
            log_every_n_steps=50,
            enable_progress_bar=False,  # Reduce verbosity
            gradient_clip_val=1.0,
            callbacks=[early_stop_cb, lr_monitor_cb, checkpoint_cb],
            logger=False,  # Disable default logger for cleaner output
        )

        # Train the model
        trainer.fit(model, datamodule=data_module)

        # Load best model from checkpoint
        best_model_path = checkpoint_cb.best_model_path
        if best_model_path:
            ModelClass = type(model)
            model = ModelClass.load_from_checkpoint(best_model_path)
            logger.info(f"Loaded best model from {best_model_path}")

        return model

    def _generate_meta_predictions(
        self,
        model: LitMetaForecastBase,
        data_module: MetaMonthDataModule,
        scalers: Dict[str, Any],
        year: int,
    ) -> pd.DataFrame:
        """Generate meta-predictions following deep_scr/train_eval.py pattern."""
        logger.info(f"Generating meta-predictions for year {year}")

        # Setup trainer for prediction
        accelerator = "cpu"
        if torch.cuda.is_available():
            accelerator = "gpu"
        elif torch.backends.mps.is_available():
            accelerator = "mps"

        trainer = Trainer(
            accelerator=accelerator, logger=False, enable_progress_bar=False
        )

        # Generate predictions
        predictions = trainer.predict(model, datamodule=data_module)

        if not predictions:
            logger.warning(f"No predictions generated for year {year}")
            return pd.DataFrame()

        # Concatenate all prediction batches
        predictions_df = pd.concat(predictions, axis=0, ignore_index=True)

        # Unscale predictions using target scalers
        predictions_df = self._unscale_predictions(predictions_df, scalers)

        # Rename prediction columns to match expected format
        prediction_cols = [col for col in predictions_df.columns if col.startswith("Q")]
        for col in prediction_cols:
            if col == "Q50":  # Main prediction
                predictions_df.rename(columns={col: f"Q_{self.name}"}, inplace=True)
            # Keep quantile columns as they are for uncertainty

        # Add validation year for tracking
        predictions_df["validation_year"] = year

        logger.info(f"Generated {len(predictions_df)} predictions for year {year}")

        return predictions_df

    def _unscale_predictions(
        self, predictions: pd.DataFrame, scalers: Dict[str, Any]
    ) -> pd.DataFrame:
        """Unscale predictions using target scalers."""
        unscaled = predictions.copy()

        if "target_scaler" in scalers:
            pred_cols = [col for col in unscaled.columns if col.startswith("Q")]

            for code in unscaled["code"].unique():
                if code in scalers["target_scaler"]:
                    mean_val, std_val = scalers["target_scaler"][code]
                    mask = unscaled["code"] == code

                    for col in pred_cols:
                        if col in unscaled.columns:
                            unscaled.loc[mask, col] = (
                                unscaled.loc[mask, col] * std_val + mean_val
                            )

        return unscaled

    def _train_final_meta_model(
        self, data: pd.DataFrame, model_names: List[str]
    ) -> LitMetaForecastBase:
        """Train final meta-learning model on all data for operational prediction."""
        logger.info("Training final meta-learning model for operational prediction")

        # Use all data for training
        scalers = self._create_data_scalers(data, model_names)
        scaled_data = self._scale_data(data, scalers, model_names)

        # Prepare features
        past_features, future_features, static_features, base_learner_cols = (
            self._prepare_feature_columns(model_names)
        )

        # Split data for training (use last 10% for validation)
        split_idx = int(len(scaled_data) * 0.9)
        train_data = scaled_data.iloc[:split_idx].copy()
        val_data = scaled_data.iloc[split_idx:].copy()

        # Create data module
        data_module = MetaMonthDataModule(
            df=None,
            static_df=self.static_data,
            static_features=static_features,
            past_features=past_features,
            future_features=future_features,
            base_learner_cols=base_learner_cols,
            base_learner_add_cols=[],
            train_years=[],
            val_years=[],
            test_years=[],
            lookback=self.lookback_steps,
            future_known_steps=self.future_steps,
            batch_size=self.batch_size,
            num_workers=0,
            train_df=train_data,
            val_df=val_data,
            test_df=val_data,
        )

        # Create and train model
        model = self._create_meta_model(
            static_features, past_features, future_features, base_learner_cols
        )
        trained_model = self._train_deep_meta_model(model, data_module, "final")

        # Store scalers for operational prediction
        self.data_scalers = scalers

        return trained_model

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune meta-learning hyperparameters using Optuna.

        Returns:
            Tuple of (success, message)
        """
        logger.info(f"Starting hyperparameter tuning for {self.name}")

        try:
            import optuna

            # Preprocess data
            data, model_names = self.__preprocess_data__()
            data.dropna(subset=["Q_obs"], inplace=True)

            available_years = sorted(data["date"].dt.year.unique())

            # Use last few years for hyperparameter validation
            hparam_tuning_years = self.hparam_tuning_years
            validation_years = available_years[-hparam_tuning_years:]
            training_years = available_years[:-hparam_tuning_years]

            if len(training_years) < 3:
                return False, "Not enough data for hyperparameter tuning"

            def objective(trial):
                # Suggest hyperparameters
                trial_config = {
                    "hidden_size": trial.suggest_categorical(
                        "hidden_size", [32, 64, 128, 256]
                    ),
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 1e-4, 1e-2, log=True
                    ),
                    "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                    "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
                    "weight_decay": trial.suggest_float(
                        "weight_decay", 1e-6, 1e-3, log=True
                    ),
                }

                # Update model configuration temporarily
                original_config = {}
                for key, value in trial_config.items():
                    original_config[key] = getattr(self, key)
                    setattr(self, key, value)

                try:
                    # Train on training years, validate on validation years
                    train_data = data[data["date"].dt.year.isin(training_years)].copy()
                    val_data = data[data["date"].dt.year.isin(validation_years)].copy()

                    # Create scalers and scale data
                    scalers = self._create_data_scalers(train_data, model_names)
                    train_scaled = self._scale_data(train_data, scalers, model_names)
                    val_scaled = self._scale_data(val_data, scalers, model_names)

                    # Prepare features
                    (
                        past_features,
                        future_features,
                        static_features,
                        base_learner_cols,
                    ) = self._prepare_feature_columns(model_names)

                    # Create data module
                    data_module = MetaMonthDataModule(
                        df=None,
                        static_df=self.static_data,
                        static_features=static_features,
                        past_features=past_features,
                        future_features=future_features,
                        base_learner_cols=base_learner_cols,
                        base_learner_add_cols=[],
                        train_years=[],
                        val_years=[],
                        test_years=[],
                        lookback=self.lookback_steps,
                        future_known_steps=self.future_steps,
                        batch_size=self.batch_size,
                        num_workers=0,
                        train_df=train_scaled,
                        val_df=val_scaled,
                        test_df=val_scaled,
                    )

                    # Create and train model
                    model = self._create_meta_model(
                        static_features,
                        past_features,
                        future_features,
                        base_learner_cols,
                    )
                    trained_model = self._train_deep_meta_model(
                        model, data_module, f"trial_{trial.number}"
                    )

                    # Generate predictions and calculate validation score
                    predictions = self._generate_meta_predictions(
                        trained_model, data_module, scalers, f"trial_{trial.number}"
                    )

                    if len(predictions) == 0:
                        return float("inf")  # Return bad score

                    # Calculate NSE as validation metric
                    if (
                        f"Q_{self.name}" in predictions.columns
                        and "Q_obs" in val_data.columns
                    ):
                        # Merge predictions with validation data
                        merged = val_data.merge(
                            predictions[["date", "code", f"Q_{self.name}"]],
                            on=["date", "code"],
                            how="inner",
                        )

                        if len(merged) > 10:
                            observed = merged["Q_obs"].values
                            predicted = merged[f"Q_{self.name}"].values

                            # Remove NaN values
                            valid_mask = ~(np.isnan(observed) | np.isnan(predicted))

                            if np.sum(valid_mask) >= 10:
                                from monthly_forecasting.scr.metrics import (
                                    calculate_NSE,
                                )

                                nse = calculate_NSE(
                                    observed[valid_mask], predicted[valid_mask]
                                )
                                return (
                                    -nse if not np.isnan(nse) else float("inf")
                                )  # Minimize negative NSE

                    return float("inf")  # Return bad score if evaluation failed

                except Exception as e:
                    logger.warning(f"Trial {trial.number} failed: {str(e)}")
                    return float("inf")

                finally:
                    # Restore original configuration
                    for key, value in original_config.items():
                        setattr(self, key, value)

            # Run optimization
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=20, timeout=3600)  # 1 hour timeout

            if study.best_trial:
                # Update model with best parameters
                best_params = study.best_params
                for key, value in best_params.items():
                    setattr(self, key, value)

                best_score = -study.best_value  # Convert back to NSE
                logger.info(
                    f"Hyperparameter tuning completed. Best params: {best_params}, Best NSE: {best_score:.4f}"
                )
                return True, f"Tuning successful. Best NSE: {best_score:.4f}"
            else:
                return False, "No valid hyperparameter combinations found"

        except ImportError:
            return False, "Optuna not available for hyperparameter tuning"
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return False, f"Tuning failed: {str(e)}"

    def save_model(self, is_fitted: bool = False) -> None:
        """Save trained meta-learning models and preprocessing artifacts."""
        logger.info(f"Saving {self.name} meta-learning models")

        save_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        os.makedirs(save_path, exist_ok=True)

        # Save model metadata including hyperparameters
        metadata = {
            "name": self.name,
            "architecture": self.architecture,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "weight_decay": self.weight_decay,
            "lookback_steps": self.lookback_steps,
            "future_steps": self.future_steps,
            "quantiles": self.quantiles,
            "loss_fn": self.loss_fn,
            "model_type": "DeepMetaLearner",
            "version": "1.0",
        }

        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save configurations
        config_data = {
            "general_config": self.general_config,
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "path_config": self.path_config,
        }

        import pickle

        config_path = os.path.join(save_path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config_data, f)

        # Save data scalers
        if hasattr(self, "data_scalers") and self.data_scalers:
            scalers_path = os.path.join(save_path, "data_scalers.json")
            with open(scalers_path, "w") as f:
                json.dump(self.data_scalers, f, indent=2)

        # Save fitted meta-models if available
        if self.fitted_models:
            models_path = os.path.join(save_path, "meta_models")
            os.makedirs(models_path, exist_ok=True)

            for year, model in self.fitted_models.items():
                model_path = os.path.join(models_path, f"meta_model_{year}.ckpt")
                if hasattr(model, "state_dict"):
                    torch.save(model.state_dict(), model_path)
                else:
                    # Save the entire model if state_dict is not available
                    torch.save(model, model_path)

        # Save historical performance if available
        if (
            hasattr(self, "historical_performance")
            and self.historical_performance is not None
        ):
            perf_path = os.path.join(save_path, "historical_performance.parquet")
            self.historical_performance.to_parquet(perf_path, index=False)

        logger.info(f"Meta-learning model saved to {save_path}")

    def load_model(self) -> None:
        """Load trained meta-learning models and preprocessing artifacts."""
        logger.info(f"Loading {self.name} meta-learning models")

        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")

        if not os.path.exists(load_path):
            raise FileNotFoundError(
                f"Meta-learning model directory not found: {load_path}"
            )

        # Load metadata
        metadata_path = os.path.join(load_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update model parameters
            for key, value in metadata.items():
                if hasattr(self, key) and key not in ["name", "model_type", "version"]:
                    setattr(self, key, value)

            logger.info(f"Loaded metadata: {metadata}")
        else:
            logger.warning(f"No metadata file found at {metadata_path}")

        # Load configurations
        import pickle

        config_path = os.path.join(load_path, "config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config_data = pickle.load(f)

            # Update configurations (be careful not to override critical paths)
            self.general_config.update(config_data.get("general_config", {}))
            self.model_config.update(config_data.get("model_config", {}))
            self.feature_config.update(config_data.get("feature_config", {}))

            logger.info("Configuration loaded successfully")
        else:
            logger.warning(f"No configuration file found at {config_path}")

        # Load data scalers
        scalers_path = os.path.join(load_path, "data_scalers.json")
        if os.path.exists(scalers_path):
            with open(scalers_path, "r") as f:
                self.data_scalers = json.load(f)
            logger.info("Data scalers loaded successfully")
        else:
            logger.warning("No data scalers found")

        # Load historical performance
        perf_path = os.path.join(load_path, "historical_performance.parquet")
        if os.path.exists(perf_path):
            self.historical_performance = pd.read_parquet(perf_path)
            logger.info("Historical performance loaded successfully")

        # Load fitted meta-models (note: loading requires recreating the architecture)
        models_path = os.path.join(load_path, "meta_models")
        if os.path.exists(models_path):
            self.fitted_models = {}

            # Note: For now, we just note that models exist
            # Full loading would require recreating the model architecture
            # which depends on the specific data dimensions at loading time
            model_files = [f for f in os.listdir(models_path) if f.endswith(".ckpt")]
            if model_files:
                logger.info(
                    f"Found {len(model_files)} saved models. Models will be loaded when needed."
                )

            # Store the models path for later loading
            self._models_path = models_path
        else:
            logger.warning("No saved models found")

        logger.info(f"Meta-learning model loaded from {load_path}")

    def _load_specific_model(
        self,
        year: str,
        static_features: List[str],
        past_features: List[str],
        future_features: List[str],
        base_learner_cols: List[str],
    ):
        """Load a specific trained model for a given year."""
        if not hasattr(self, "_models_path"):
            raise ValueError("No models path available. Please run load_model() first.")

        model_path = os.path.join(self._models_path, f"meta_model_{year}.ckpt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Create the model architecture
        model = self._create_meta_model(
            static_features, past_features, future_features, base_learner_cols
        )

        # Load the state dict
        try:
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            logger.info(f"Loaded model for year {year}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model for year {year}: {e}")
            # Try loading the entire model
            model = torch.load(model_path, map_location="cpu")
            return model
