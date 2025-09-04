import os
import pandas as pd
import numpy as np
import datetime
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm as progress_bar

import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

from monthly_forecasting.forecast_models.meta_learners.base_meta_learner import (
    BaseMetaLearner,
)
from monthly_forecasting.scr import FeatureExtractor as FE
from monthly_forecasting.scr import data_utils as du
from monthly_forecasting.scr import mixture

# Import deep learning components
from monthly_forecasting.forecast_models.deep_models.architectures.mlp_uncertainty import (
    MLPUncertaintyModel,
)
from monthly_forecasting.deep_scr.data_class import TabularDataset

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class UncertaintyMixtureModel(BaseMetaLearner):
    """
    A model which helps to quantify the uncertainty of the ensemble predictions.
    The main steps are that a neural network approximates the probability distribution for each ensemble member.
    This is done based on temporal features, past residuals and info about the ensemble prediction.
    Finally, the pdf's are combined to form a mixture model.
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

        self.name = self.general_config.get("model_name", "UncertaintyMixtureModel")
        self.model_home_path = self.path_config.get("model_home_path")
        self.save_dir = os.path.join(self.model_home_path, self.name)

        self.hidden_size = self.model_config.get("hidden_size", 64)
        self.num_residual_blocks = self.model_config.get("num_residual_blocks", 2)
        self.dropout = self.model_config.get("dropout", 0.1)
        self.learning_rate = self.model_config.get("learning_rate", 0.001)
        self.batch_size = self.model_config.get("batch_size", 32)
        self.max_epochs = self.model_config.get("max_epochs", 100)
        self.patience = self.model_config.get("patience", 10)
        self.weight_decay = self.model_config.get("weight_decay", 0.0)
        self.gradient_clip_val = self.model_config.get("gradient_clip_val", 1.0)

        self.quantiles = self.model_config.get(
            "quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        )
        self.target_col = "Q_obs"
        self.loss_fn = self.model_config.get("loss_fn", "ALDLoss")

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

        self.data = self._m3_to_mm(self.data)

        # Get target variable
        target = extractor.create_target(self.data)
        dates = pd.to_datetime(self.data["date"])
        codes = self.data["code"].astype(int)
        target_data = pd.DataFrame(
            {"date": dates, "code": codes, self.target_col: target}
        )

        # 2. Load base predictors using inherited method
        logger.info("Loading base predictors")
        base_predictors, model_names = self.__load_base_predictors__()
        self.model_names = model_names

        # 3. Merge the base predictors with target data
        logger.info("Merging base predictors with target data")
        merged_data = target_data.merge(
            base_predictors, on=["date", "code"], how="left"
        )

        # Add error features for base models
        for model_name in model_names:
            if (
                model_name in merged_data.columns
                and self.target_col in merged_data.columns
            ):
                error_col = f"{model_name}_error"
                merged_data[error_col] = (
                    merged_data[self.target_col] - merged_data[model_name]
                )
                abs_error_col = f"{model_name}_abs_error"
                merged_data[abs_error_col] = merged_data[error_col].abs()

        ensemble_mean = merged_data[model_names].mean(axis=1)
        merged_data["ensemble_mean"] = ensemble_mean
        ensemble_std = merged_data[model_names].std(axis=1)
        merged_data["ensemble_std"] = ensemble_std
        ensemble_min = merged_data[model_names].min(axis=1)
        merged_data["ensemble_min"] = ensemble_min
        ensemble_max = merged_data[model_names].max(axis=1)
        merged_data["ensemble_max"] = ensemble_max
        ensemble_skew = merged_data[model_names].skew(axis=1)
        merged_data["ensemble_skew"] = ensemble_skew
        ensemble_median = merged_data[model_names].median(axis=1)
        merged_data["ensemble_median"] = ensemble_median

        # 4. Reformat to long format with ensemble_member column
        logger.info("Reformatting DataFrame to long format")
        preprocessed_data = self._reformat_df(merged_data)

        # 5. Create periods column for temporal features
        logger.info("Creating period columns")
        preprocessed_data = du.get_periods(preprocessed_data)

        preprocessed_data["dist_to_ens"] = (
            preprocessed_data["ensemble_mean"] - preprocessed_data["Q_pred"]
        )

        # 6. Add temporal features for deep learning
        logger.info("Adding temporal features for deep learning")
        preprocessed_data = self._add_temporal_features(preprocessed_data)

        #  create a feature list
        self.features = ["Q_pred", "day_sin", "day_cos", "month_sin", "month_cos"]

        ensemble_cols = [
            "dist_to_ens",
            "ensemble_mean",
            "ensemble_std",
            "ensemble_min",
            "ensemble_max",
            "ensemble_skew",
            "ensemble_median",
        ]
        self.features.extend(ensemble_cols)

        error_cols = [col for col in preprocessed_data.columns if "error" in col]
        self.features.extend(error_cols)

        features_from_aggregation = [
            "error_mean",
            "error_std",
            "error_max",
            "error_skew",
            "abs_error_mean",
            "abs_error_std",
            "abs_error_max",
            "abs_error_skew",
        ]
        self.features.extend(features_from_aggregation)

        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        logger.info(f"Model names: {model_names}")
        logger.info(
            f"Columns in preprocessed data: {preprocessed_data.columns.tolist()}"
        )
        logger.info(f"Feature names: {self.features}")

        self.data = preprocessed_data
        self.model_names = model_names

    def _add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for deep learning models."""
        data = data.copy()

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

    def _m3_to_mm(self, data: pd.DataFrame) -> pd.DataFrame:
        for code in data.code.unique():
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            # transform from m3/s to mm/day
            data.loc[data["code"] == code, "discharge"] = (
                data.loc[data["code"] == code, "discharge"] * 86.4 / area
            )
        return data

    def _mm_to_m3(self, data: pd.DataFrame, col: Union[List[str], str]) -> pd.DataFrame:
        for code in data.code.unique():
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            # transform from mm/day to m3/s
            data.loc[data["code"] == code, col] = (
                data.loc[data["code"] == code, col] * area / 86.4
            )
        return data

    def _reformat_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reformats the DataFrame to use a long format with an 'ensemble_member' column.

        Instead of having separate columns for each model's predictions, errors, and
        absolute errors (e.g., 'model1_error', 'model1_abs_error'), this function
        creates a standardized set of columns ('Q_pred', 'error', 'abs_error') and
        adds an 'ensemble_member' column to indicate which model the row corresponds to.

        Shared columns like 'ensemble_mean', 'ensemble_std', etc., are repeated for
        each ensemble member. This enables consistent feature engineering and modeling
        across all ensemble members.

        Args:
            df: The input DataFrame with wide-format model columns.

        Returns:
            A reformatted DataFrame in long format with standardized columns.

        Raises:
            ValueError: If required columns are missing or model_names is empty.
        """
        if not self.model_names:
            raise ValueError("model_names is empty; cannot reformat DataFrame.")

        # Identify model prediction columns that exist
        existing_models = [m for m in self.model_names if m in df.columns]
        if not existing_models:
            raise ValueError("No model prediction columns found in DataFrame.")

        missing_models = set(self.model_names) - set(existing_models)
        if missing_models:
            logger.warning(f"Model prediction columns not found: {missing_models}")

        # Identify shared columns (not model-specific)
        model_specific_patterns = [f"{model}_" for model in self.model_names]
        shared_cols = [
            col
            for col in df.columns
            if not any(col.startswith(pattern) for pattern in model_specific_patterns)
            and col not in self.model_names
        ]

        # Melt the prediction columns
        pred_df = pd.melt(
            df,
            id_vars=shared_cols,
            value_vars=existing_models,
            var_name="ensemble_member",
            value_name="Q_pred",
        )

        # Process error columns efficiently using vectorized operations
        error_data = []
        abs_error_data = []

        for model in existing_models:
            error_col = f"{model}_error"
            abs_error_col = f"{model}_abs_error"

            # Create arrays of the same length as original df
            if error_col in df.columns:
                error_data.extend(df[error_col].values)
            else:
                logger.warning(
                    f"Error column '{error_col}' not found for model '{model}'."
                )
                error_data.extend([np.nan] * len(df))

            if abs_error_col in df.columns:
                abs_error_data.extend(df[abs_error_col].values)
            else:
                logger.warning(
                    f"Absolute error column '{abs_error_col}' not found for model '{model}'."
                )
                abs_error_data.extend([np.nan] * len(df))

        # Add error columns to melted dataframe
        pred_df["error"] = error_data
        pred_df["abs_error"] = abs_error_data

        # Reorder columns for consistency
        column_order = shared_cols + ["ensemble_member", "Q_pred", "error", "abs_error"]
        reformatted_df = pred_df[column_order]

        logger.info(
            f"Reformatted DataFrame from {len(df)} rows to {len(reformatted_df)} rows "
            f"with {len(existing_models)} ensemble members per original row."
        )

        return reformatted_df

    def _scale_data(
        self, data: pd.DataFrame, scaler: Optional[Dict] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Scales the data using z-score normalization (mean=0, std=1) for all numeric features.
        If the scaler is not provided it will fit a new one.
        """
        data_scaled = data.copy()

        # Initialize scaler dictionary if not provided
        if scaler is None:
            scaler = {}

        # Scale the target
        target_col = self.target_col
        if target_col in data_scaled.columns:
            target_values = data_scaled[target_col].values
            valid_mask = ~np.isnan(target_values)
            if valid_mask.any():
                target_mean = np.mean(target_values[valid_mask])
                target_std = np.std(target_values[valid_mask])
                # Avoid division by zero
                if target_std == 0:
                    target_std = 1.0
                scaler[target_col] = {"mean": target_mean, "std": target_std}
                data_scaled[target_col] = (target_values - target_mean) / target_std

        # Apply z-score scaling to all feature columns
        for col in self.features:
            col_scaler_key = f"{col}_scaler"
            if col_scaler_key not in scaler:
                col_values = data_scaled[col].values
                valid_mask = ~np.isnan(col_values)
                if valid_mask.any():
                    col_mean = np.mean(col_values[valid_mask])
                    col_std = np.std(col_values[valid_mask])
                    # Avoid division by zero
                    if col_std == 0:
                        col_std = 1.0
                    scaler[col_scaler_key] = {"mean": col_mean, "std": col_std}

            # Apply scaling if scaler exists
            if col_scaler_key in scaler:
                col_scaler = scaler[col_scaler_key]
                col_values = data_scaled[col].values
                # Apply standardization: (x - mean) / std
                data_scaled[col] = (col_values - col_scaler["mean"]) / col_scaler["std"]

        return data_scaled, scaler

    def _calculate_aggregated_statistics(self, df):
        """
        Calculates aggregated statistics of the errors for each code, period, and ensemble_member
        for both abs_error and error
        - mean
        - std
        - skewness
        - max

        returns the aggregated statistics as a DataFrame
        cols : [code, period, ensemble_member, error_mean, error_std, error_skew, error_max,
                abs_error_mean, abs_error_std, abs_error_skew, abs_error_max]
        """
        # Initialize list to store results
        results = []

        # Group by code, period, and ensemble_member
        grouped = df.groupby(["code", "period", "ensemble_member"])

        for (code, period, ensemble_member), group in grouped:
            row = {"code": code, "period": period, "ensemble_member": ensemble_member}

            # Calculate statistics for error column
            if "error" in group.columns:
                row["error_mean"] = group["error"].mean()
                row["error_std"] = group["error"].std()
                row["error_skew"] = group["error"].skew()
                row["error_max"] = group["error"].max()

            # Calculate statistics for abs_error column
            if "abs_error" in group.columns:
                row["abs_error_mean"] = group["abs_error"].mean()
                row["abs_error_std"] = group["abs_error"].std()
                row["abs_error_skew"] = group["abs_error"].skew()
                row["abs_error_max"] = group["abs_error"].max()

            results.append(row)

        # Convert to DataFrame
        aggregated_stats = pd.DataFrame(results)

        return aggregated_stats

    def _rescale_predictions(
        self, df: pd.DataFrame, scaler: Dict[str, Any]
    ) -> pd.DataFrame:
        df = df.copy()
        # use the target scaler to inverse all prediction columns
        prediction_columns = [col for col in df.columns if col.startswith("Q_")]
        mean = scaler[self.target_col]["mean"]
        std = scaler[self.target_col]["std"]
        for col in prediction_columns:
            df[col] = df[col] * std + mean

        df = self._mm_to_m3(df, col=prediction_columns)

        return df

    def _init_model(
        self,
        hidden_size: int,
        num_residual_blocks: int,
        dropout: float,
        learning_rate: float,
        weight_decay: float,
        gradient_clip_val: float,
    ) -> pl.LightningModule:
        model = MLPUncertaintyModel(
            num_features=len(self.features),
            hidden_size=hidden_size,
            num_residual_blocks=num_residual_blocks,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_val=gradient_clip_val,
        )

        return model

    def create_mixture_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        mixture_model = mixture.MixtureModel(distribution_type="ALD")

        # we groupby code and date
        grouped = df.groupby(["code", "date"])
        final_pred = []

        for group in grouped:
            # create a dict with ensemble_member : {loc: , scale: , asymmetry:}
            ensemble_dict = {}
            for _, row in group.iterrows():
                ensemble_dict[row["ensemble_member"]] = {
                    "loc": row["loc"],
                    "scale": row["scale"],
                    "asymmetry": row["asymmetry"],
                }

            stats = mixture_model.get_statistic(
                parameter_dict=ensemble_dict, quantiles=self.quantiles
            )
            code = group["code"].iloc[0]
            date = group["date"].iloc[0]
            this_dict = {"code": code, "date": date}
            this_dict.update(stats)
            final_pred.append(this_dict)

        final_pred = pd.DataFrame(final_pred)
        final_pred.rename(columns={"mean": f"Q_{self.name}"}, inplace=True)

        return final_pred

    def fit(
        self,
        train_data: pd.DataFrame = None,
        val_data: pd.DataFrame = None,
        run_name: str = "final",
    ) -> Tuple[pl.LightningModule, Dict[str, Any], Dict[str, Any], pd.DataFrame]:
        # Prepare the data
        # calculate aggregated statistics
        aggregated_stats = self._calculate_aggregated_statistics(train_data)

        # merge aggregated_stats with train_data
        train_data = train_data.merge(
            aggregated_stats, on=["code", "period", "ensemble_member"], how="left"
        )
        val_data = val_data.merge(
            aggregated_stats, on=["code", "period", "ensemble_member"], how="left"
        )
        # Scale it, remove nans
        train_data, scaler = self._scale_data(train_data)
        val_data, _ = self._scale_data(val_data, scaler)

        # Create the data class
        train_data = train_data.dropna(subset=self.features + [self.target_col])
        val_data = val_data.dropna(subset=self.features + [self.target_col])

        train_dataset = TabularDataset(
            df=train_data, features=self.features, target=self.target_col
        )

        val_dataset = TabularDataset(
            df=val_data, features=self.features, target=self.target_col
        )

        logger.info(f"Training dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,  # Use 0 for debugging/main thread
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Use 0 for debugging/main thread
        )

        # Initialize the Model
        model = self._init_model(
            hidden_size=self.hidden_size,
            num_residual_blocks=self.num_residual_blocks,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip_val=self.gradient_clip_val,
        )

        # fit with early stopping
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints", run_name)

        # Clear old checkpoints to prevent conflicts
        if os.path.exists(checkpoint_dir):
            import shutil

            shutil.rmtree(checkpoint_dir)
            logger.info(f"Cleared old checkpoint directory: {checkpoint_dir}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            monitor="val_loss",
            dirpath=checkpoint_dir,
            filename="best_model",
            save_top_k=1,
            mode="min",
        )

        early_stop_callback = EarlyStopping(
            monitor="val_loss", patience=self.patience, verbose=False, mode="min"
        )

        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            devices=1,
            accelerator="cpu",
            callbacks=[checkpoint_callback, early_stop_callback],
            gradient_clip_val=self.gradient_clip_val,
            default_root_dir=checkpoint_dir,
            enable_progress_bar=False,  # Disable progress bar so exceptions are not obscured
            log_every_n_steps=1,
        )

        self.trainer.fit(model, train_loader, val_loader)

        # Load best model
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            model = MLPUncertaintyModel.load_from_checkpoint(best_model_path)
            logger.info(f"Successfully loaded best model from {best_model_path}")
        else:
            logger.warning("Could not find best model path. Using last model state.")

        history = {}
        log_dir = self.trainer.log_dir
        if log_dir and os.path.exists(os.path.join(log_dir, "metrics.csv")):
            metrics_df = pd.read_csv(os.path.join(log_dir, "metrics.csv"))
            history = metrics_df.to_dict(orient="list")

        return model, history, scaler, aggregated_stats

    def predict(
        self,
        df: pd.DataFrame,
        model: pl.LightningModule,
        scaler: Dict[str, Any] = None,
        aggregated_stats: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Make predictions using the trained uncertainty mixture model.

        Args:
            df: Input DataFrame with features
            model: Trained PyTorch Lightning model
            scaler: Feature scaler (optional)
            aggregated_stats: Pre-computed aggregated statistics (optional)

        Returns:
            DataFrame with predictions including uncertainty parameters
        """
        # Prepare the data
        if aggregated_stats is None:
            aggregated_stats = self._calculate_aggregated_statistics(df)

        # Merge aggregated_stats with df
        df = df.merge(
            aggregated_stats, on=["code", "period", "ensemble_member"], how="left"
        )
        df = df.dropna(subset=self.features)
        # Scale with the same procedure as during training
        df, _ = self._scale_data(df, scaler)

        if self.target_col not in df.columns:
            df[self.target_col] = np.nan

        all_predictions = []

        # Process each ensemble member separately
        for ensemble_model in df.ensemble_member.unique():
            df_ens = df[df.ensemble_member == ensemble_model].copy()

            # Create dataset
            dataset = TabularDataset(
                df=df_ens, features=self.features, target=self.target_col
            )

            # Create DataLoader for batch processing
            dataloader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,  # Use 0 for debugging/main thread
            )

            # Use PyTorch Lightning Trainer for prediction
            trainer = Trainer(logger=False, enable_progress_bar=False)
            batch_predictions = trainer.predict(model, dataloader)

            # Combine batch predictions into a single DataFrame
            if batch_predictions:
                ensemble_predictions = pd.concat(batch_predictions, ignore_index=True)
                ensemble_predictions["ensemble_member"] = ensemble_model
                all_predictions.append(ensemble_predictions)

        if all_predictions:
            prediction_df = pd.concat(all_predictions, axis=0, ignore_index=True)
        else:
            prediction_df = pd.DataFrame()

        return prediction_df

    def predict_operational(self, today=None):
        """
        Make operational predictions using the trained uncertainty mixture model.

        Args:
            today: Date for prediction (optional)

        Returns:
            DataFrame with operational predictions
        """
        logger.info(f"Starting operational prediction for {self.name}")

        # Preprocess data (this will call _reformat_df)
        self.__preprocess_data__()

        # Prepare the data for operational prediction
        operational_data = self.data.copy()

        # Calculate aggregated statistics
        aggregated_stats = self._calculate_aggregated_statistics(operational_data)

        # Merge aggregated_stats with operational_data
        operational_data = operational_data.merge(
            aggregated_stats, on=["code", "period", "ensemble_member"], how="left"
        )

        # Scale with the same procedure as during training
        operational_data, _ = self._scale_data(operational_data)

        if self.target_col not in operational_data.columns:
            operational_data[self.target_col] = np.nan

        # Create dataset
        operational_dataset = TabularDataset(
            df=operational_data,
            features=operational_data.columns.tolist(),
            target=self.target_col,
        )

        # Load the Model
        model = self.load_model()

        # Predict
        predictions = model.predict(operational_dataset)

        # Add predictions to the dataframe
        operational_data["Q_pred_scaled"] = predictions

        # Create prediction df with mixture model
        prediction_df = operational_data[
            ["date", "code", "ensemble_member", "Q_pred_scaled"]
        ].copy()

        return prediction_df

    def calibrate_model_and_hindcast(self):
        """
        Calibrate the ensemble models using Leave-One-Year-Out cross-validation.

        Returns:
            hindcast (pd.DataFrame): DataFrame containing the hindcasted values.
        """

        logger.info(f"Starting calibration and hindcasting for {self.name}")

        original_data = self.data.copy()

        self.__preprocess_data__()

        if "year" not in self.data.columns:
            self.data["year"] = self.data["date"].dt.year

        # Add day column if not present
        if "day" not in self.data.columns:
            self.data["day"] = self.data["date"].dt.day

        # Get configuration parameters
        num_test_years = self.general_config.get("num_test_years", 2)

        self.__filter_forecast_days__()

        all_years = sorted(self.data["year"].unique())

        if num_test_years > 0:
            loocv_years = all_years[:-num_test_years]
            test_years = all_years[-num_test_years:]
        else:
            loocv_years = all_years
            test_years = None

        hindcast_df = None

        for loo in loocv_years:
            logger.info(f"LOO Year: {loo}")
            train_data = self.data[self.data["year"] != loo].copy()
            # randomly take val_fraction of train for validation
            val_fraction = self.general_config.get("val_fraction", 0.1)
            val_data = train_data.sample(frac=val_fraction, random_state=42)
            train_data = train_data.drop(val_data.index)

            test_data = self.data[self.data["year"] == loo].copy()

            # Fit the model
            model, cal_history, scaler, aggregated_stats = self.fit(
                train_data=train_data, val_data=val_data, run_name=str(loo)
            )
            # Predict on the test set
            preds = model.predict(
                df=test_data,
                model=model,
                scaler=scaler,
                aggregated_stats=aggregated_stats,
            )

            if hindcast_df is None:
                hindcast_df = preds
            else:
                hindcast_df = pd.concat([hindcast_df, preds], axis=0)

        # Now we fit on all
        logger.info("Fitting final model on all data")
        train_data = self.data[self.data["year"].isin(loocv_years)].copy()
        val_fraction = self.general_config.get("val_fraction", 0.1)
        val_data = train_data.sample(frac=val_fraction, random_state=42)
        train_data = train_data.drop(val_data.index)

        # Fit the model on all (minus test years) data
        model, cal_history, scaler, aggregated_stats = self.fit(
            train_data=train_data, val_data=val_data
        )
        if test_years is not None:
            test_data = self.data[self.data["year"].isin(test_years)].copy()
            preds = model.predict(
                df=test_data,
                model=model,
                scaler=scaler,
                aggregated_stats=aggregated_stats,
            )
            hindcast_df = pd.concat([hindcast_df, preds], axis=0)

        logger.info("Creating mixture predictions")
        mixture_preds = self.create_mixture_predictions(hindcast_df)

        logger.info("Rescaling predictions to original units")
        hindcast = self._rescale_predictions(mixture_preds, scaler)

        ground_truth = original_data[["date", "code", "Q_obs"]].copy()
        pred_cols = [
            col for col in hindcast.columns if col.startswith("Q_") and col != "Q_obs"
        ]
        hindcast = hindcast[pred_cols + ["date", "code"]]
        hindcast = hindcast.merge(ground_truth, on=["date", "code"], how="left")

        # save model (commented out intentionally)
        logger.info("Saving final model (skipped in current implementation)")
        # self.save_model(model=model, scaler=scaler, history=cal_history)

        return hindcast

    def tune_hyperparameters(self):
        return True, "UncertaintyMixtureMLP: Hyperparameter tuning not implemented yet"

    def save_model(self):
        return super().save_model()

    def load_model(self):
        return super().load_model()
