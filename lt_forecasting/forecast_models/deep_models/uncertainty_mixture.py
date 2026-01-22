import os

os.environ["OMP_NUM_THREADS"] = "1"
import shutil
import pandas as pd
import numpy as np
import datetime
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
from tqdm import tqdm as progress_bar

import torch

torch.set_num_threads(1)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from torch.utils.data import DataLoader

import optuna

from lt_forecasting.forecast_models.meta_learners.base_meta_learner import (
    BaseMetaLearner,
)
from lt_forecasting.scr import FeatureExtractor as FE
from lt_forecasting.scr import data_utils as du
from lt_forecasting.scr import mixture

# Import deep learning components
from lt_forecasting.forecast_models.deep_models.architectures.mlp_uncertainty import (
    MLPUncertaintyModel,
)
from lt_forecasting.deep_scr.data_class import TabularDataset

# Shared logging
import logging
from lt_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)
logging.getLogger("fsspec").setLevel(logging.WARNING)


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
        base_predictors: Optional[pd.DataFrame] = None,
        base_model_names: Optional[List[str]] = None,
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
            base_predictors: Optional[pd.DataFrame] = None,
            base_model_names: Optional[List[str]] = None,
        """
        super().__init__(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        self.base_predictors = base_predictors
        self.base_model_names = base_model_names

        # this will be overwritten in calibration / fit
        self.is_fitted = False

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
        self.use_pred_mean = self.model_config.get("use_pred_mean", True)

        self.test_years = self.general_config.get("test_years", [2021, 2022, 2023])
        self.hparam_tuning_years = self.general_config.get(
            "hparam_tuning_years", [2018, 2019, 2020]
        )

        assert isinstance(self.test_years, list), (
            f"test_years should be a list of years but got {type(self.test_years)}"
        )
        assert isinstance(self.hparam_tuning_years, list), (
            f"hparam_tuning_years should be a list of years but got {type(self.hparam_tuning_years)}"
        )

        # check if there are some years both in test_years and hparam_tuning_years
        overlapping_years = set(self.test_years).intersection(
            set(self.hparam_tuning_years)
        )
        if overlapping_years:
            logger.warning(
                f"Overlapping years found in test_years and hparam_tuning_years: {overlapping_years}. Please ensure these are distinct."
            )

        self.quantiles = self.model_config.get(
            "quantiles", [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        )
        self.target_col = "Q_obs"
        self.loss_fn = self.model_config.get("loss_fn", "ALDLoss")

        self.rivers_to_remove = self.general_config.get("rivers_to_exclude", [])

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

        self.ground_truth = target_data.copy()
        self.ground_truth = du.get_periods(self.ground_truth)
        logger.info("Ground Truth created")

        # 2. Load base predictors using inherited method
        if self.base_predictors is not None and self.base_model_names is not None:
            logger.info("Using provided base predictors and model names")
            base_predictors = self.base_predictors
            model_names = self.base_model_names
        else:
            logger.info("Loading base predictors - NOT DATABASE COMPATIBLE YET")
            base_predictors, model_names = self.__load_base_predictors__(
                use_mean_pred=self.general_config.get("use_ens_mean", False)
            )
            self.model_names = model_names

        # sort by date so the newsest date is at the top
        base_predictors = base_predictors.sort_values(by=["date"], ascending=[False])
        today = pd.to_datetime(datetime.datetime.now().date())
        base_predictors = base_predictors[base_predictors["date"] <= today]
        logger.info(f"Head of Base Predictors data:\n{base_predictors.head(10)}")

        logger.info("converting base predictors to mm/day")
        for model in model_names:
            base_predictors = self._m3_to_mm(base_predictors, col=model)

        # 3. Merge the base predictors with target data
        logger.info("Merging base predictors with target data")
        merged_data = target_data.merge(
            base_predictors, on=["date", "code"], how="left"
        )

        # remove rivers which are not in static data
        if self.rivers_to_remove:
            unique_rivers_to_remove = set(self.rivers_to_remove)
            logger.info(f"Removing rivers: {unique_rivers_to_remove}")
            merged_data = merged_data[
                ~merged_data["code"].isin(unique_rivers_to_remove)
            ]

        ensemble_mean = merged_data[model_names].mean(axis=1)
        merged_data["Q_pred"] = ensemble_mean
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
        num_valid_pred = merged_data[model_names].notna().sum(axis=1)
        merged_data["num_valid_pred"] = num_valid_pred
        max_distance = merged_data[model_names].max(axis=1) - merged_data[
            model_names
        ].min(axis=1)
        merged_data["ensemble_range"] = max_distance

        merged_data = merged_data.drop(columns=model_names)
        merged_data["ensemble_member"] = "ensemble"

        # 4. Reformat to long format with ensemble_member column

        # 5. Create periods column for temporal features
        logger.info("Creating period columns")
        preprocessed_data = du.get_periods(merged_data)

        # calculate error statistics for each code, period and ensemble member , loocv style
        preprocessed_data = self._loo_error_stats(preprocessed_data)

        # 6. Add temporal features for deep learning
        logger.info("Adding temporal features for deep learning")
        preprocessed_data = self._add_temporal_features(preprocessed_data)

        #  create a feature list
        self.features = ["Q_pred", "day_sin", "day_cos", "month_sin", "month_cos"]

        ensemble_cols = [
            "ensemble_std",
            "ensemble_min",
            "ensemble_max",
            "ensemble_skew",
            "ensemble_median",
            "num_valid_pred",
        ]
        if len(model_names) > 1:
            self.features.extend(ensemble_cols)

        features_from_aggregation = [
            "error_mean",
            "error_std",
            "error_max",
            "error_skew",
            "abs_error_mean",
            "abs_error_std",
            "abs_error_max",
            f"{self.target_col}_mean",
            f"{self.target_col}_std",
            f"{self.target_col}_max",
            f"{self.target_col}_skew",
            f"{self.target_col}_min",
        ]

        self.features.extend(features_from_aggregation)

        # Force contiguous copy to minimize fragmentation after all transformations
        preprocessed_data = preprocessed_data.copy()

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

    def _m3_to_mm(self, data: pd.DataFrame, col: str = "discharge") -> pd.DataFrame:
        for code in data.code.unique():
            if code in self.rivers_to_remove:
                continue
            if code not in self.static_data["code"].values:
                logger.debug(
                    f"Code {code} not found in static data. Skipping this code."
                )
                self.rivers_to_remove.append(code)
                continue
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            # transform from m3/s to mm/day
            data.loc[data["code"] == code, col] = (
                data.loc[data["code"] == code, col] * 86.4 / area
            )
        return data

    def _mm_to_m3(self, data: pd.DataFrame, col: Union[List[str], str]) -> pd.DataFrame:
        for code in data.code.unique():
            if code not in self.static_data["code"].values:
                logger.debug(
                    f"Code {code} not found in static data. Skipping this code."
                )
                self.rivers_to_remove.append(code)
                continue
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

        # Reorder columns for consistency
        column_order = shared_cols + ["ensemble_member", "Q_pred"]
        reformatted_df = pred_df[column_order]

        # Force contiguous copy and reset index to minimize fragmentation
        reformatted_df = reformatted_df.copy().reset_index(drop=True)

        logger.info(
            f"Reformatted DataFrame from {len(df)} rows to {len(reformatted_df)} rows "
            f"with {len(existing_models)} ensemble members per original row."
        )

        return reformatted_df

    def _calculate_scaler(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculates the scaler parameters (mean and std) for z-score normalization.
        """
        scaler = {}

        # Scale the target
        target_col = self.target_col
        pred_col = self.features[0]  # Assuming first feature is always Q_pred
        logger.info("-" * 40)
        logger.info(f"Calculating scaler for target column: {target_col}")
        logger.info(f"Calculating scaler for prediction column: {pred_col}")

        if "pred" not in pred_col:
            logger.warning(f"First feature is not 'Q_pred', but {pred_col}")
            logger.warning("Make sure that the first features is the prediction column")

        if target_col in data.columns:
            valid_mask = ~np.isnan(data[target_col].values)
            if valid_mask.any():
                target_mean = np.mean(data[target_col].values[valid_mask])
                target_std = np.std(data[target_col].values[valid_mask])
                # Avoid division by zero
                if target_std == 0:
                    target_std = 1.0
                scaler[f"{target_col}_scaler"] = {
                    "mean": target_mean,
                    "std": target_std,
                }

        # Calculate scalers for all feature columns
        for col in self.features[1:]:
            col_values = data[col].values
            valid_mask = ~np.isnan(col_values)
            if valid_mask.any():
                col_mean = np.mean(col_values[valid_mask])
                col_std = np.std(col_values[valid_mask])
                # Avoid division by zero
                if col_std == 0:
                    col_std = 1.0
                scaler[f"{col}_scaler"] = {"mean": col_mean, "std": col_std}

        return scaler

    def _scale_data(self, data: pd.DataFrame, scaler: Dict[str, Any]) -> pd.DataFrame:
        """
        Scales the data using z-score normalization (mean=0, std=1) for all numeric features.
        Uses the provided scaler parameters.
        """
        data_scaled = data.copy()

        # Scale the target
        target_col = self.target_col
        pred_col = self.features[0]  # Assuming first feature is always Q_pred

        if target_col in data_scaled.columns:
            target_col_scaler_key = f"{target_col}_scaler"
            if target_col_scaler_key in scaler:
                target_scaler = scaler[target_col_scaler_key]
                # Apply standardization: (x - mean) / std
                data_scaled[target_col] = (
                    data_scaled[target_col] - target_scaler["mean"]
                ) / target_scaler["std"]

        if pred_col in data_scaled.columns:
            target_col_scaler_key = f"{target_col}_scaler"
            if target_col_scaler_key in scaler:
                target_scaler = scaler[target_col_scaler_key]
                # Apply standardization: (x - mean) / std
                data_scaled[pred_col] = (
                    data_scaled[pred_col] - target_scaler["mean"]
                ) / target_scaler["std"]

        # Apply z-score scaling to all feature columns
        for col in self.features[1:]:
            col_scaler_key = f"{col}_scaler"
            if col_scaler_key in scaler:
                col_scaler = scaler[col_scaler_key]
                col_values = data_scaled[col].values
                # Apply standardization: (x - mean) / std
                data_scaled[col] = (col_values - col_scaler["mean"]) / col_scaler["std"]

        return data_scaled

    def _calculate_aggregated_statistics(self, df):
        """
        Calculates aggregated statistics of the errors for each code, period
        for both abs_error and error
        - mean
        - std
        - skewness
        - max

        returns the aggregated statistics as a DataFrame
        cols : [code, period, error_mean, error_std, error_skew, error_max,
                abs_error_mean, abs_error_std, abs_error_max]
        """
        df = df.copy()
        df["error"] = df["Q_pred"] - df[self.target_col]
        df["abs_error"] = df["error"].abs()
        # Initialize list to store results
        results = []

        # Group by code and period
        grouped = df.groupby(["code", "period"])

        for (code, period), group in grouped:
            row = {"code": code, "period": period}

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
                row["abs_error_max"] = group["abs_error"].max()

            if self.target_col in group.columns:
                row["n_samples"] = group[self.target_col].notna().sum()
                row[f"{self.target_col}_mean"] = group[self.target_col].mean()
                row[f"{self.target_col}_std"] = group[self.target_col].std()
                row[f"{self.target_col}_max"] = group[self.target_col].max()
                row[f"{self.target_col}_skew"] = group[self.target_col].skew()
                row[f"{self.target_col}_min"] = group[self.target_col].min()

            results.append(row)

        # Convert to DataFrame
        aggregated_stats = pd.DataFrame(results)

        return aggregated_stats

    def _loo_error_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the error statistics for each code, period.
        for all years except one and inserts them as a column for the left out year."""

        df = df.copy()
        df["year"] = pd.to_datetime(df["date"]).dt.year

        all_years = df["year"].unique()
        all_years.sort()

        merged_dfs = []

        for year in all_years:
            df_loo = df[df["year"] != year].copy()
            agg_stats = self._calculate_aggregated_statistics(df_loo)

            df_year = df[df["year"] == year].copy()
            df_year = df_year.merge(agg_stats, on=["code", "period"], how="left")
            merged_dfs.append(df_year)

        df_loo = pd.concat(merged_dfs, ignore_index=True)
        # Force contiguous copy after concat to minimize fragmentation
        df_loo = df_loo.copy()
        logger.info("-" * 40)
        logger.info(f"LOO error stats added, new shape: {df_loo.shape}")
        logger.info(f"Columns after LOO error stats: {df_loo.columns.tolist()}")
        logger.info("-" * 40)
        return df_loo

    def _rescale_predictions(
        self, df: pd.DataFrame, scaler: Dict[str, Any]
    ) -> pd.DataFrame:
        df = df.copy()
        # use the target scaler to inverse all prediction columns
        prediction_columns = [col for col in df.columns if col.startswith("Q")]
        scaler_key = f"{self.target_col}_scaler"
        if scaler_key not in scaler:
            logger.warning(f"Scaler key '{scaler_key}' not found in scaler dictionary.")
            df = self._mm_to_m3(df, col=prediction_columns)
            return df

        mean = scaler[scaler_key]["mean"]
        std = scaler[scaler_key]["std"]
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
            lr_scheduler=self.model_config.get("scheduler", None),
            use_pred_mean=self.use_pred_mean,
        )

        return model

    def fit(
        self,
        train_data: pd.DataFrame = None,
        val_data: pd.DataFrame = None,
        run_name: str = "final",
    ) -> Tuple[pl.LightningModule, Dict[str, Any], Dict[str, Any], pd.DataFrame]:
        # Prepare the data
        # Scale it, remove nans
        scaler = self._calculate_scaler(train_data)
        train_data = self._scale_data(train_data, scaler)
        val_data = self._scale_data(val_data, scaler)

        # Reset index to prevent potential issues with shuffling and improve contiguity
        train_data = train_data.reset_index(drop=True).copy()
        # shuffle train data
        train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
        val_data = val_data.reset_index(drop=True).copy()

        # Create the data class
        train_data = (
            train_data.dropna(subset=self.features + [self.target_col])
            .reset_index(drop=True)
            .copy()
        )
        val_data = (
            val_data.dropna(subset=self.features + [self.target_col])
            .reset_index(drop=True)
            .copy()
        )

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
            num_workers=0,  # Set to 0 for debugging
            pin_memory=False,  # Pin memory can cause issues with some setups
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,  # Set to 0 for debugging
            pin_memory=False,
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
            accelerator="auto",
            callbacks=[checkpoint_callback, early_stop_callback],
            gradient_clip_val=self.gradient_clip_val,
            default_root_dir=checkpoint_dir,
            enable_progress_bar=True,  # Disable progress bar so exceptions are not obscured
        )

        try:
            print("Starting model training...")
            self.trainer.fit(model, train_loader, val_loader)
        except Exception as e:
            print(f"Error during training: {e}")
            logger.error(f"Error during training: {e}", exc_info=True)
            raise e

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

        return model, history, scaler

    def predict(
        self,
        df: pd.DataFrame,
        model: pl.LightningModule,
        scaler: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Make predictions using the trained uncertainty mixture model.

        This revised method processes all data in a single pass for efficiency,
        avoiding the inefficient loop over ensemble members.

        Args:
            df: Input DataFrame with features. Must contain 'code', 'date',
                and 'ensemble_member' for identification.
            model: Trained PyTorch Lightning model.
            scaler: Feature scaler dictionary.
            aggregated_stats: Pre-computed aggregated statistics.

        Returns:
            DataFrame with predictions including uncertainty parameters (loc, scale,
            asymmetry) and identifying columns.
        """
        predict_df = df.copy()

        # Drop rows with missing features, which can't be predicted
        predict_df.dropna(subset=self.features, inplace=True)
        predict_df.reset_index(drop=True, inplace=True)
        # Force contiguous copy after dropna
        predict_df = predict_df.copy()
        identifiers = predict_df[["date", "code", "ensemble_member"]].copy()

        if predict_df.empty:
            logger.warning(
                "DataFrame is empty after dropping NaNs. No predictions can be made."
            )
            return pd.DataFrame()

        # 2. Scale features
        predict_df = self._scale_data(predict_df, scaler)

        # Add dummy target if not present (required by TabularDataset)
        if self.target_col not in predict_df.columns:
            predict_df[self.target_col] = np.nan

        # 3. Create Dataset and DataLoader
        dataset = TabularDataset(
            df=predict_df, features=self.features, target=self.target_col
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # 4. Make predictions
        # Initialize a single trainer for prediction
        trainer = Trainer(
            logger=False, enable_progress_bar=False, accelerator="cpu", devices=1
        )
        # push model to the trainer device
        model = model.to("cpu")
        # trainer.predict returns a list of batch results (DataFrames)
        # batch_predictions = trainer.predict(model, dataloader)
        prediction_results = model.MC_quantile_sampling(
            dataloader=dataloader,
            MC_num_samples=self.model_config.get("MC_num_samples", 100),
            ALD_num_samples=self.model_config.get("ALD_num_samples", 500),
            quantiles=self.quantiles,
        )
        """
        # 5. Combine and format results
        if not batch_predictions:
            logger.warning("Prediction returned no results.")
            return pd.DataFrame()

        # Concatenate all batch prediction DataFrames
        prediction_results = pd.concat(batch_predictions, ignore_index=True)
        prediction_results = prediction_results[["loc", "scale", "asymmetry"]]"""
        final_df = pd.concat([identifiers, prediction_results], axis=1)

        # create the mixed quantile predictions
        # final_df = self.create_mixture_predictions(final_df)
        final_df = self._rescale_predictions(final_df, scaler)

        # rename Q_mean to Q_<model_name>
        if f"Q_mean" in final_df.columns:
            final_df = final_df.rename(columns={f"Q_mean": f"Q_{self.name}"})

        logger.info(f"Generated predictions for {len(final_df)} rows.")
        logger.info(f"Prediction columns: {final_df.columns.tolist()}")

        return final_df

    def predict_operational(self, today=None):
        """
        Predict in operational mode using global trained models.

        Args:
            today (datetime.datetime, optional): Date to use as "today" for prediction.
                If None, uses current datetime.

        Returns:
            forecast (pd.DataFrame): DataFrame containing the forecasted values.
        """
        logger.info(f"Starting operational prediction for {self.name}")

        self.__preprocess_data__()

        if today is None:
            today = datetime.datetime.now()
            today = pd.to_datetime(today.strftime("%Y-%m-%d"))
        else:
            today = pd.to_datetime(today.strftime("%Y-%m-%d"))

        # Step 1: Load models and artifacts
        self.load_model()

        # Step 3: Calculate valid period
        if not self.general_config.get("offset"):
            self.general_config["offset"] = self.general_config["prediction_horizon"]
        shift = (
            self.general_config["offset"] - self.general_config["prediction_horizon"]
        )
        valid_from = today + datetime.timedelta(days=1) + datetime.timedelta(days=shift)
        valid_to = valid_from + datetime.timedelta(
            days=self.general_config["prediction_horizon"]
        )

        valid_from_str = valid_from.strftime("%Y-%m-%d")
        valid_to_str = valid_to.strftime("%Y-%m-%d")

        logger.info(f"Forecast valid from: {valid_from_str} to: {valid_to_str}")

        # Step 4: Filter data for operational prediction
        all_codes = self.data["code"].unique()
        # how many days delayed can the prediction be made
        allowable_delay = self.general_config.get("allowable_delay", 3)
        cutoff_date = today - pd.Timedelta(days=allowable_delay)

        logger.info(
            f"Cutoff date for operational prediction: {cutoff_date.date()}. Considering all data after this date."
        )

        logger.info(f"Total number of unique codes: {len(all_codes)}")
        # logger.info("Data Tail before filtering:")
        # logger.info(self.data.tail())

        ## Cutoff between cutoff day and today
        operational_data = self.data[
            (self.data["date"] >= cutoff_date) & (self.data["date"] <= today)
        ].copy()
        logger.info(f"Operational data shape after cutoff: {operational_data.shape}")

        recent_operational_data = []
        ## get the most recent date where data is available for each code
        for code in all_codes:
            code_data = operational_data[operational_data["code"] == code]
            if code_data.empty:
                logger.warning(
                    f"No data available for code {code} in operational data."
                )
                continue
            # sort by date descending
            code_data = code_data.sort_values(by="date", ascending=False)

            # take the most recent date where all features are available
            features_with_na_values = code_data[self.features].isna().any(axis=1)
            # logg the missing features
            if features_with_na_values.any():
                logger.warning(
                    f"The following features are missing for code {code}: {code_data[self.features].columns[code_data[self.features].isna().any()].tolist()}"
                )
            code_data = code_data.dropna(subset=self.features)
            if code_data.empty:
                logger.warning(f"No complete feature data available for code {code}.")
                continue
            most_recent_date = code_data["date"].iloc[0]
            recent_data = code_data[code_data["date"] == most_recent_date]
            recent_operational_data.append(recent_data)

        operational_data = pd.concat(recent_operational_data, ignore_index=True)

        logger.info(
            f"Operational data shape after dropping missing features: {operational_data.shape}"
        )

        # sort by date and take for each code the most recent date
        operational_data = operational_data.sort_values(
            by=["code", "date"], ascending=[True, False]
        )
        operational_data = operational_data.drop_duplicates(
            subset=["code"], keep="first"
        )

        logger.info(f"Operational data shape after filtering: {operational_data.shape}")
        missing_codes = set(all_codes) - set(operational_data["code"].unique())
        if missing_codes:
            logger.warning(
                f"No recent data available for codes: {missing_codes}. These will be skipped in the forecast."
            )

        if operational_data.empty:
            logger.warning(
                "No data available for operational prediction after filtering. Exiting."
            )
            return pd.DataFrame()

        # Step 5: Make predictions
        forecast = self.predict(
            df=operational_data,
            model=self.model,
            scaler=self.scaler,
        )

        if forecast.empty:
            logger.warning("No predictions were made. Exiting.")
            return pd.DataFrame()

        # Step 7: Rescale the predictions to original scale
        pred_cols = [col for col in forecast.columns if col.startswith("Q")]
        pred_cols.append("date")
        pred_cols.append("code")
        forecast = forecast[pred_cols]

        logger.info("Operational prediction completed")
        logger.info("Head of forecast DataFrame:")
        logger.info(forecast.head())

        forecast["valid_from"] = valid_from_str
        forecast["valid_to"] = valid_to_str

        all_pred_cols = [col for col in forecast.columns if col.startswith("Q")]

        if missing_codes:
            # Handle those codes by adding them with NaN predictions
            for code in missing_codes:
                empty_row = {
                    "code": code,
                }
                for col in all_pred_cols:
                    empty_row[col] = np.nan

                empty_row_df = pd.DataFrame([empty_row], index=[0])

                # Add the empty row to the forecast DataFrame
                forecast = pd.concat([forecast, empty_row_df], ignore_index=True)

        # round numeric columns to 2 decimals
        numeric_cols = forecast.select_dtypes(include=[np.number]).columns
        # exclude code column
        numeric_cols = [col for col in numeric_cols if col != "code"]
        forecast[numeric_cols] = forecast[numeric_cols].round(2)

        # set limit of 0 for predictions
        forecast[numeric_cols] = forecast[numeric_cols].clip(lower=0)

        return forecast

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate the ensemble models using Leave-One-Year-Out cross-validation.

        Returns:
            hindcast (pd.DataFrame): DataFrame containing the hindcasted values.
        """

        logger.info(f"Starting calibration and hindcasting for {self.name}")

        self.__preprocess_data__()

        if "year" not in self.data.columns:
            self.data["year"] = self.data["date"].dt.year

        # Add day column if not present
        if "day" not in self.data.columns:
            self.data["day"] = self.data["date"].dt.day

        # Get configuration parameters
        test_years = self.test_years
        if test_years is None:
            test_years = []

        self.__filter_forecast_days__()

        all_years = sorted(self.data["year"].unique())

        loocv_years = [year for year in all_years if year not in test_years]

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
            model, cal_history, scaler = self.fit(
                train_data=train_data, val_data=val_data, run_name=str(loo)
            )
            # Predict on the test set
            preds = self.predict(
                df=test_data,
                model=model,
                scaler=scaler,
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
        self.model, self.cal_history, self.scaler = self.fit(
            train_data=train_data, val_data=val_data
        )
        self.is_fitted = True
        if test_years is not None:
            test_data = self.data[self.data["year"].isin(test_years)].copy()
            preds = self.predict(
                df=test_data,
                model=self.model,
                scaler=self.scaler,
            )
            hindcast_df = pd.concat([hindcast_df, preds], axis=0)

        hindcast = hindcast_df.reset_index(drop=True).copy()

        logger.info("Creating mixture predictions")

        ground_truth = self._mm_to_m3(self.ground_truth, col=self.target_col)

        pred_cols = [
            col for col in hindcast.columns if col.startswith("Q") and col != "Q_obs"
        ]
        pred_cols.append("date")
        pred_cols.append("code")

        hindcast = hindcast[pred_cols]
        hindcast = hindcast.merge(ground_truth, on=["date", "code"], how="left")

        #  Calculate valid period
        if not self.general_config.get("offset"):
            self.general_config["offset"] = self.general_config["prediction_horizon"]
        shift = (
            self.general_config["offset"] - self.general_config["prediction_horizon"]
        )
        valid_from = (
            hindcast["date"]
            + datetime.timedelta(days=1)
            + datetime.timedelta(days=shift)
        )
        valid_to = valid_from + datetime.timedelta(
            days=self.general_config["prediction_horizon"]
        )

        hindcast["valid_from"] = valid_from
        hindcast["valid_to"] = valid_to

        # save model
        self.save_model()

        logger.info("Hindcasting completed")
        logger.info("Head of hindcast DataFrame:")
        logger.info(hindcast.head())

        return hindcast

    def objective(
        self,
        trial: optuna.Trial,
        train_loader: DataLoader,
        val_df: pd.DataFrame,
        scaler: Dict[str, Any],
    ) -> float:
        # tune lr, hidden_size, num_residual_blocks, dropout, weight_decay
        # number epochs is set to a fixed value of 20 for tuning
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        hidden_size = trial.suggest_categorical("hidden_size", [16, 32])
        num_residual_blocks = trial.suggest_int("num_residual_blocks", 1, 4)
        dropout = trial.suggest_uniform("dropout", 0.0, 0.8)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 5e-3)

        val_org_scale = val_df.copy()
        val_df = val_df.drop(columns=[self.target_col]).copy()

        model = self._init_model(
            hidden_size=hidden_size,
            num_residual_blocks=num_residual_blocks,
            dropout=dropout,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gradient_clip_val=self.gradient_clip_val,
        )

        epochs_hparam_tuning = self.general_config.get(
            "epochs_hparam_tuning", self.max_epochs
        )

        # fit the model
        trainer = pl.Trainer(
            max_epochs=epochs_hparam_tuning,
            devices=1,
            accelerator="cpu",
            enable_progress_bar=True,
            gradient_clip_val=self.gradient_clip_val,
        )

        trainer.fit(model, train_loader)

        # predict on the validation set
        preds = self.predict(
            df=val_df,
            model=model,
            scaler=scaler,
        )

        if preds.empty:
            logger.warning("No predictions were made during hyperparameter tuning.")
            return float("inf")

        val_org_scale = self._mm_to_m3(val_org_scale, col=self.target_col)
        val_preds = preds.merge(val_org_scale, on=["date", "code"], how="left")

        # calculate coverage
        coverage_90 = np.mean(
            (val_preds["Q_obs"] >= val_preds["Q5"])
            & (val_preds["Q_obs"] <= val_preds["Q95"])
        )
        coverage_50 = np.mean(
            (val_preds["Q_obs"] >= val_preds["Q25"])
            & (val_preds["Q_obs"] <= val_preds["Q75"])
        )

        # score is the mean squared error of the coverage
        score = abs(coverage_90 - 0.9) + abs(coverage_50 - 0.5)
        score *= 100
        return score

    def tune_hyperparameters(self):
        """
        Tune the hyperparameters of the uncertainty models with optuna.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success and a message.
        """

        logger.info(f"Starting hyperparameter tuning for {self.name}")

        # Apply the same preprocessing as other methods
        self.__preprocess_data__()

        hparam_tuning_years = self.hparam_tuning_years

        if len(hparam_tuning_years) <= 0:
            return False, "UncertaintyMixtureMLP: num_hparam_tuning_years must be > 0"

        all_years = sorted(self.data["year"].unique())
        years_train = [year for year in all_years if year not in hparam_tuning_years]
        years_val = hparam_tuning_years
        train_data = self.data[self.data["year"].isin(years_train)]
        val_data = self.data[self.data["year"].isin(years_val)]

        # process the train and val data like in the fit method
        scaler = self._calculate_scaler(train_data)
        train_data = self._scale_data(train_data, scaler)

        # Reset index and force contiguous copies to minimize fragmentation
        train_data = train_data.reset_index(drop=True).copy()
        val_data = val_data.reset_index(drop=True).copy()

        # Create the data class
        train_data = (
            train_data.dropna(subset=self.features + [self.target_col])
            .reset_index(drop=True)
            .copy()
        )
        val_data = (
            val_data.dropna(subset=self.features + [self.target_col])
            .reset_index(drop=True)
            .copy()
        )

        # shuffle train data
        train_data = train_data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        train_dataset = TabularDataset(
            df=train_data, features=self.features, target=self.target_col
        )

        logger.info(f"Training dataset size: {len(train_dataset)}")

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )

        # create the study
        study = optuna.create_study(direction="minimize")
        # number of trials
        n_trials = self.general_config.get("n_trials", 20)
        # run the optimization
        study.optimize(
            lambda trial: self.objective(trial, train_loader, val_data, scaler),
            n_trials=n_trials,
        )

        # get the best hyperparameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        # Overwrite the model configuration with the best hyperparameters
        self.model_config["learning_rate"] = best_params["learning_rate"]
        self.model_config["hidden_size"] = best_params["hidden_size"]
        self.model_config["num_residual_blocks"] = best_params["num_residual_blocks"]
        self.model_config["dropout"] = best_params["dropout"]
        self.model_config["weight_decay"] = best_params["weight_decay"]
        # save the model configuration
        self.save_model()

        return True, "UncertaintyMixtureMLP: Hyperparameter tuned successfully"

    def save_model(self):
        logger.info(f"Saving {self.name} models")
        save_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        os.makedirs(save_path, exist_ok=True)

        # Save general model configuration
        model_config_path = os.path.join(save_path, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(self.model_config, f, indent=4)
        # Save general feature configuration
        feature_config_path = os.path.join(save_path, "feature_config.json")
        with open(feature_config_path, "w") as f:
            json.dump(self.feature_config, f, indent=4)
        # Save general experiment configuration
        experiment_config_path = os.path.join(save_path, "experiment_config.json")
        with open(experiment_config_path, "w") as f:
            json.dump(self.general_config, f, indent=4)

        if self.is_fitted:
            # Save the trained model
            model_path = os.path.join(save_path, "final_model.ckpt")
            self.model.to("cpu")
            torch.save(self.model.state_dict(), model_path)
            logger.info(f"Model saved to {model_path}")

            # Save the scaler
            scaler_path = os.path.join(save_path, "scaler.json")
            with open(scaler_path, "w") as f:
                json.dump(self.scaler, f, indent=4)
            logger.info(f"Scaler saved to {scaler_path}")

            # Save the training history
            history_path = os.path.join(save_path, "training_history.json")
            with open(history_path, "w") as f:
                json.dump(self.cal_history, f, indent=4)
            logger.info(f"Training history saved to {history_path}")

        else:
            logger.warning(
                "Model is not fitted; skipping model, scaler, and history saving."
            )

    def load_model(self):
        logger.info(f"Loading {self.name} model")
        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")

        model_path = os.path.join(load_path, "final_model.ckpt")
        scaler_path = os.path.join(load_path, "scaler.json")

        # check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found at {scaler_path}")

        # Load the scaler
        with open(scaler_path, "r") as f:
            self.scaler = json.load(f)
        logger.info(f"Scaler loaded from {scaler_path}")

        # Initialize the model architecture
        model = self._init_model(
            hidden_size=self.hidden_size,
            num_residual_blocks=self.num_residual_blocks,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            gradient_clip_val=self.gradient_clip_val,
        )

        model.load_state_dict(torch.load(model_path))
        self.model = model

        logger.info(f"Model loaded from {model_path}")
