import os
import pandas as pd
import numpy as np
import geopandas as gpd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm as progress_bar
import json
import joblib
import datetime
import warnings
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sktime.transformations.panel.rocket import MiniRocket

warnings.filterwarnings("ignore")

from monthly_forecasting.forecast_models.base_class import BaseForecastModel
from monthly_forecasting.scr import data_utils as du
from monthly_forecasting.scr import sci_utils

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)
logging.getLogger("fiona.ogrext").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)


class MiniRocketRegressor(BaseForecastModel):
    """
    A regressor class using MiniRocket for time series feature extraction.
    Uses GLOBAL fitting approach where a model is trained on all basins and periods simultaneously.
    Models include: Linear Regression, XGBoost, LightGBM, SVR, and CatBoost.
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
        Initialize the MiniRocketRegressor model with a configuration dictionary.
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
        self.models = self.general_config.get(
            "models", ["linear", "xgboost", "lgbm", "svr", "catboost"]
        )
        self.fitted_models = {}
        self.minirocket_transformers = {}  # Store fitted MiniRocket transformers

        # MiniRocket specific parameters
        self.input_length = self.general_config.get(
            "input_length", 30
        )  # lookback window
        self.num_kernels = self.general_config.get(
            "num_kernels", 1000
        )  # MiniRocket kernels

        # Get preprocessing configuration
        self.target = self.general_config.get("target", "target")
        self.feature_cols = self.general_config.get(
            "feature_cols",
            ["discharge", "P", "T"],
        )
        self.static_features = self.general_config.get("static_features", [])
        self.rivers_to_exclude = self.general_config.get("rivers_to_exclude", [])
        self.snow_vars = self.general_config.get("snow_vars", ["SWE"])
        self.hparam_tuning_years = self.general_config.get("hparam_tuning_years", 3)
        self.early_stopping_val_fraction = self.general_config.get(
            "early_stopping_val_fraction", 0.1
        )
        self.num_test_years = self.general_config.get("num_test_years", 2)
        self.allowable_missing_value_operational = self.general_config.get(
            "allowable_missing_value_operational", 5
        )

    def __preprocess_data__(self):
        """
        Preprocess the data by adding derived features and preparing for MiniRocket.
        """
        logger.info(f"-" * 50)
        logger.info(f"Starting data preprocessing for {self.name}")
        logger.info(f"Initial data shape: {self.data.shape}")
        logger.info(f"Initial columns: {self.data.columns.tolist()}")
        logger.info(f"-" * 50)

        # Apply glacier mapper features if configured
        try:
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

        # Sort by date
        self.data.sort_values(by="date", inplace=True)

        # Keep relevant columns
        cols_to_keep = [
            col
            for col in self.data.columns
            if any([feature in col for feature in self.feature_cols])
        ]
        self.data = self.data[["date", "code"] + cols_to_keep]

        # Filter out excluded rivers
        self.data = self.data[~self.data["code"].isin(self.rivers_to_exclude)].copy()

        # Convert discharge to mm/day
        for code in self.data.code.unique():
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            self.data.loc[self.data["code"] == code, "discharge"] = (
                self.data.loc[self.data["code"] == code, "discharge"] * 86.4 / area
            )

        # Process snow data to elevation bands
        if self.path_config["path_to_hru_shp"] is not None:
            elevation_band_shp = gpd.read_file(self.path_config["path_to_hru_shp"])
            if "CODE" in elevation_band_shp.columns:
                elevation_band_shp.rename(columns={"CODE": "code"}, inplace=True)
            elevation_band_shp["code"] = elevation_band_shp["code"].astype(int)

            for snow_var in self.snow_vars:
                self.data = du.calculate_percentile_snow_bands(
                    self.data,
                    elevation_band_shp,
                    num_bands=self.general_config["num_elevation_zones"],
                    col_name=snow_var,
                )
                snow_vars_drop = [
                    col
                    for col in self.data.columns
                    if snow_var in col and "elev" not in col
                ]
                self.data = self.data.drop(columns=snow_vars_drop)

        # Extract extended features for snowmapper
        self.data = du.derive_features_from_snowmapper(self.data)

        # Add temporal features
        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month
        self.data["day"] = self.data["date"].dt.day
        self.data["week"] = self.data["date"].dt.isocalendar().week

        # Add cyclical encoding for temporal features
        self.data["month_sin"] = np.sin(2 * np.pi * self.data["month"] / 12)
        self.data["month_cos"] = np.cos(2 * np.pi * self.data["month"] / 12)
        self.data["week_sin"] = np.sin(2 * np.pi * self.data["week"] / 52)
        self.data["week_cos"] = np.cos(2 * np.pi * self.data["week"] / 52)

        # Prepare time series columns for MiniRocket
        self.ts_feature_cols = [
            col
            for col in self.data.columns
            if any([feature in col for feature in self.feature_cols])
        ]

        logger.info(
            f"Time series feature columns for MiniRocket: {self.ts_feature_cols}"
        )
        logger.debug("Data preprocessing completed. Data shape: %s", self.data.shape)

    def __create_time_series_windows__(
        self,
        data: pd.DataFrame,
        target_dates: List[pd.Timestamp] = None,
        for_prediction: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Create time series windows for MiniRocket transformation.

        Args:
            data: Input dataframe with time series data
            target_dates: Specific dates to create windows for (if None, uses all valid dates)
            for_prediction: If True, doesn't require target values

        Returns:
            X: 3D array of shape (n_samples, n_features, input_length)
            metadata: DataFrame with metadata for each sample (date, code, target, etc.)
        """
        X_list = []
        metadata_list = []

        # Get unique basin codes
        basin_codes = data["code"].unique()

        for code in basin_codes:
            basin_data = data[data["code"] == code].sort_values("date").copy()

            if target_dates is not None:
                # Filter to specific dates for this basin
                basin_target_dates = [
                    d for d in target_dates if d in basin_data["date"].values
                ]
            else:
                # Use all dates with enough history
                basin_target_dates = basin_data["date"].values[self.input_length :]

            for target_date in basin_target_dates:
                target_idx = basin_data[basin_data["date"] == target_date].index[0]
                target_loc = basin_data.index.get_loc(target_idx)

                # Check if we have enough history
                if target_loc < self.input_length:
                    continue

                # Get the window
                window_start_loc = target_loc - self.input_length
                window_end_loc = target_loc

                window_data = basin_data.iloc[window_start_loc:window_end_loc]

                # Check for missing values in the window
                if window_data[self.ts_feature_cols].isnull().any().any():
                    continue

                # Extract time series features
                ts_features = window_data[
                    self.ts_feature_cols
                ].values.T  # Shape: (n_features, input_length)
                X_list.append(ts_features)

                # Get metadata for this sample
                target_row = basin_data.iloc[target_loc]
                metadata = {
                    "date": target_date,
                    "code": code,
                    "year": target_row["year"],
                    "month": target_row["month"],
                    "day": target_row["day"],
                    "week": target_row["week"],
                    "month_sin": target_row["month_sin"],
                    "month_cos": target_row["month_cos"],
                    "week_sin": target_row["week_sin"],
                    "week_cos": target_row["week_cos"],
                }

                if not for_prediction and self.target in target_row:
                    metadata[self.target] = target_row[self.target]

                metadata_list.append(metadata)

        if not X_list:
            logger.warning("No valid time series windows created")
            return np.array([]), pd.DataFrame()

        X = np.array(X_list)  # Shape: (n_samples, n_features, input_length)
        metadata_df = pd.DataFrame(metadata_list)

        logger.info(f"Created {X.shape[0]} time series windows with shape {X.shape}")

        return X, metadata_df

    def __apply_minirocket_transform__(
        self, X: np.ndarray, fit: bool = False, transformer_key: str = "default"
    ) -> np.ndarray:
        """
        Apply MiniRocket transformation to time series windows.

        Args:
            X: 3D array of shape (n_samples, n_features, input_length)
            fit: Whether to fit the transformer
            transformer_key: Key to store/retrieve the transformer

        Returns:
            Transformed features of shape (n_samples, n_minirocket_features)
        """
        if fit:
            logger.info(
                f"Fitting MiniRocket transformer with {self.num_kernels} kernels"
            )
            self.minirocket_transformers[transformer_key] = MiniRocket(
                num_kernels=self.num_kernels, random_state=42
            )
            X_transformed = self.minirocket_transformers[transformer_key].fit_transform(
                X
            )
        else:
            if transformer_key not in self.minirocket_transformers:
                raise ValueError(
                    f"MiniRocket transformer '{transformer_key}' not found. Must fit first."
                )
            X_transformed = self.minirocket_transformers[transformer_key].transform(X)

        logger.info(f"MiniRocket features shape: {X_transformed.shape}")
        return X_transformed

    def __prepare_features_for_model__(
        self, X_minirocket: np.ndarray, metadata_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine MiniRocket features with metadata and static features.

        Args:
            X_minirocket: MiniRocket transformed features
            metadata_df: DataFrame with metadata for each sample

        Returns:
            DataFrame with all features combined
        """
        # Create DataFrame from MiniRocket features
        minirocket_cols = [f"minirocket_{i}" for i in range(X_minirocket.shape[1])]
        features_df = pd.DataFrame(X_minirocket, columns=minirocket_cols)

        # Add metadata
        for col in metadata_df.columns:
            features_df[col] = metadata_df[col].values

        # Add static features
        if self.static_features:
            static_df = self.static_data[["code"] + self.static_features].copy()
            features_df = pd.merge(features_df, static_df, on="code", how="left")

        # Create dummy variables for categorical features
        features_df["code_str"] = features_df["code"].astype(str)
        features_df = pd.get_dummies(
            features_df, columns=["month"], prefix="month", dtype=int
        )

        return features_df

    def __loocv__(
        self,
        years: List[int],
        params_dict: Dict[str, Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Perform Leave-One-Year-Out Cross-Validation with MiniRocket features.
        """
        logger.info(f"Starting LOOCV for {self.name} with years: {years}")

        all_predictions = []

        for test_year in progress_bar(years, desc="Processing years", leave=True):
            # Split data
            train_years = [y for y in years if y != test_year]
            train_data = self.data[self.data["year"].isin(train_years)].copy()
            test_data = self.data[self.data["year"] == test_year].copy()

            # Get test dates that have valid targets
            test_dates = test_data[test_data[self.target].notna()]["date"].unique()

            # Create time series windows for training
            X_train_3d, train_metadata = self.__create_time_series_windows__(
                train_data, target_dates=None, for_prediction=False
            )

            if X_train_3d.size == 0:
                logger.warning(f"No valid training windows for year {test_year}")
                continue

            # Apply MiniRocket transformation
            X_train_minirocket = self.__apply_minirocket_transform__(
                X_train_3d, fit=True, transformer_key=f"year_{test_year}"
            )

            # Prepare training features
            train_features_df = self.__prepare_features_for_model__(
                X_train_minirocket, train_metadata
            )

            # Create time series windows for testing
            X_test_3d, test_metadata = self.__create_time_series_windows__(
                test_data, target_dates=test_dates, for_prediction=False
            )

            if X_test_3d.size == 0:
                logger.warning(f"No valid test windows for year {test_year}")
                continue

            # Apply MiniRocket transformation to test data
            X_test_minirocket = self.__apply_minirocket_transform__(
                X_test_3d, fit=False, transformer_key=f"year_{test_year}"
            )

            # Prepare test features
            test_features_df = self.__prepare_features_for_model__(
                X_test_minirocket, test_metadata
            )

            # Get feature columns (excluding metadata)
            feature_cols = [
                col
                for col in train_features_df.columns
                if col not in ["date", "code", "year", "day", "week", self.target]
            ]

            # Train models and make predictions
            year_predictions = test_features_df[["date", "code", self.target]].copy()

            for model_type in self.models:
                logger.info(f"Training {model_type} for year {test_year}")

                # Get model parameters
                if params_dict and model_type in params_dict:
                    params = params_dict[model_type]
                else:
                    params = self.__get_default_params__(model_type)

                # Prepare data
                X_train = train_features_df[feature_cols]
                y_train = train_features_df[self.target]
                X_test = test_features_df[feature_cols]

                # Handle categorical features for CatBoost
                cat_features = []
                if model_type == "catboost":
                    cat_features = ["code_str"] if "code_str" in feature_cols else []

                # Create and train model
                model = self.__create_model__(model_type, params, cat_features)
                model = sci_utils.fit_model(
                    model=model,
                    X=X_train,
                    y=y_train,
                    model_type=model_type,
                    val_fraction=self.early_stopping_val_fraction,
                )

                # Make predictions
                y_pred = model.predict(X_test)
                year_predictions[f"Q_{model_type}"] = y_pred

            all_predictions.append(year_predictions)

        if not all_predictions:
            logger.error("No predictions generated in LOOCV")
            return pd.DataFrame()

        # Combine all predictions
        df_predictions = pd.concat(all_predictions, ignore_index=True)

        # Add ensemble prediction
        pred_cols = [f"Q_{model}" for model in self.models]
        df_predictions[f"Q_{self.name}"] = df_predictions[pred_cols].mean(axis=1)

        # Rename target column
        df_predictions.rename(columns={self.target: "Q_obs"}, inplace=True)

        return df_predictions

    def __fit_on_all__(
        self,
        test_years: List[int] = None,
        params_dict: Dict[str, Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Fit models on all available data using MiniRocket features.
        """
        logger.info(f"Starting global fitting for {self.name}")

        fitted_models = {}
        df_predictions = pd.DataFrame()

        # Prepare training data
        if test_years is None:
            train_data = self.data.copy()
        else:
            train_data = self.data[~self.data["year"].isin(test_years)].copy()
            test_data = self.data[self.data["year"].isin(test_years)].copy()
            test_dates = test_data[test_data[self.target].notna()]["date"].unique()

        # Create time series windows for training
        X_train_3d, train_metadata = self.__create_time_series_windows__(
            train_data, target_dates=None, for_prediction=False
        )

        if X_train_3d.size == 0:
            logger.error("No valid training windows created")
            return fitted_models

        # Apply MiniRocket transformation
        X_train_minirocket = self.__apply_minirocket_transform__(
            X_train_3d, fit=True, transformer_key="global"
        )

        # Prepare training features
        train_features_df = self.__prepare_features_for_model__(
            X_train_minirocket, train_metadata
        )

        # Get feature columns
        feature_cols = [
            col
            for col in train_features_df.columns
            if col not in ["date", "code", "year", "day", "week", self.target]
        ]

        # Prepare test data if provided
        if test_years is not None:
            X_test_3d, test_metadata = self.__create_time_series_windows__(
                test_data, target_dates=test_dates, for_prediction=False
            )

            X_test_minirocket = self.__apply_minirocket_transform__(
                X_test_3d, fit=False, transformer_key="global"
            )

            test_features_df = self.__prepare_features_for_model__(
                X_test_minirocket, test_metadata
            )

            df_predictions = test_features_df[["date", "code", self.target]].copy()

        # Train each model type
        for model_type in self.models:
            logger.info(f"Training {model_type} on all data")

            # Get model parameters
            if params_dict and model_type in params_dict:
                params = params_dict[model_type]
            else:
                params = self.__get_default_params__(model_type)

            # Prepare data
            X_train = train_features_df[feature_cols]
            y_train = train_features_df[self.target]

            # Handle categorical features for CatBoost
            cat_features = []
            if model_type == "catboost":
                cat_features = ["code_str"] if "code_str" in feature_cols else []

            # Create and train model
            model = self.__create_model__(model_type, params, cat_features)
            model = sci_utils.fit_model(
                model=model,
                X=X_train,
                y=y_train,
                model_type=model_type,
                val_fraction=self.early_stopping_val_fraction,
            )

            # Store fitted model
            fitted_models[model_type] = {
                "model": model,
                "feature_cols": feature_cols,
                "feature_importance": sci_utils.get_feature_importance(model)
                if hasattr(model, "feature_importances_")
                else None,
            }

            # Make predictions on test data if provided
            if test_years is not None:
                X_test = test_features_df[feature_cols]
                y_pred = model.predict(X_test)
                df_predictions[f"Q_{model_type}"] = y_pred

        # Add ensemble prediction if test predictions were made
        if test_years is not None and not df_predictions.empty:
            pred_cols = [f"Q_{model}" for model in self.models]
            df_predictions[f"Q_{self.name}"] = df_predictions[pred_cols].mean(axis=1)
            df_predictions.rename(columns={self.target: "Q_obs"}, inplace=True)

        return fitted_models, df_predictions

    def __create_model__(
        self, model_type: str, params: Dict[str, Any], cat_features: List[str] = None
    ):
        """Create a model instance based on type."""
        if model_type == "linear":
            return LinearRegression(**params)
        elif model_type == "svr":
            return SVR(**params)
        else:
            # For tree-based models, use the existing sci_utils function
            return sci_utils.get_model(model_type, params, cat_features)

    def __get_default_params__(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters for each model type."""
        defaults = {
            "linear": {},
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 6,
                "random_state": 42,
            },
            "lgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "random_state": 42,
            },
            "svr": {"kernel": "rbf", "C": 1.0, "epsilon": 0.1},
            "catboost": {
                "iterations": 100,
                "learning_rate": 0.1,
                "depth": 6,
                "random_state": 42,
                "verbose": False,
            },
        }
        return defaults.get(model_type, {})

    def __make_data_continuous__(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make sure that the date record is continuous for each basin.
        """
        if df.empty:
            logger.warning("Input DataFrame is empty")
            return df.copy()

        df = df.copy()

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df["date"]):
            df["date"] = pd.to_datetime(df["date"])

        # Sort by code and date
        df = df.sort_values(["code", "date"]).reset_index(drop=True)

        # Get overall date range
        min_date = df["date"].min()
        max_date = df["date"].max()

        continuous_data_list = []
        basin_codes = df["code"].unique()

        for code in basin_codes:
            basin_data = df[df["code"] == code].copy()

            # Handle duplicate dates
            basin_data = basin_data.drop_duplicates(subset=["date"], keep="last")

            # Create complete date range
            date_range = pd.date_range(start=min_date, end=max_date, freq="D")

            # Reindex to fill missing dates
            basin_indexed = basin_data.set_index("date")
            basin_continuous = basin_indexed.reindex(date_range)
            basin_continuous = basin_continuous.reset_index()
            basin_continuous.rename(columns={"index": "date"}, inplace=True)

            # Ensure code is filled for all rows
            basin_continuous["code"] = code

            continuous_data_list.append(basin_continuous)

        # Combine all basin data
        result_df = pd.concat(continuous_data_list, ignore_index=True)
        result_df = result_df.sort_values(["code", "date"]).reset_index(drop=True)

        return result_df

    def __post_process_data__(
        self, df: pd.DataFrame, pred_cols: List[str], obs_col: str = None
    ) -> pd.DataFrame:
        """
        Post-process the data after model predictions (mm/d to m3/s conversion).
        """
        df = df.copy()
        for code in df["code"].unique():
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            df.loc[df["code"] == code, pred_cols] = (
                df.loc[df["code"] == code, pred_cols] * area / 86.4
            )

            if obs_col is not None and obs_col in df.columns:
                df.loc[df["code"] == code, obs_col] = (
                    df.loc[df["code"] == code, obs_col] * area / 86.4
                )

        return df

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate the ensemble models using Leave-One-Year-Out cross-validation.
        """
        logger.info(f"Starting calibration and hindcasting for {self.name}")

        self.__preprocess_data__()

        # Filter forecast days if configured
        forecast_days = self.general_config.get("forecast_days", None)
        if forecast_days:
            self.__filter_forecast_days__()

        # Split years for LOOCV and testing
        all_years = sorted(self.data["year"].unique())
        num_test_years = self.general_config.get("num_test_years", 2)
        loocv_years = all_years[:-num_test_years]
        test_years = all_years[-num_test_years:]

        # Perform LOOCV
        hindcast_df = self.__loocv__(
            years=loocv_years,
            params_dict=self.model_config,
        )

        # Fit on all data and predict on test years
        logger.info(f"Fitting models on all data for {self.name}")
        self.fitted_models, test_predictions = self.__fit_on_all__(
            test_years=test_years,
            params_dict=self.model_config,
        )

        # Combine LOOCV and test predictions
        if not test_predictions.empty:
            hindcast_df = pd.concat([hindcast_df, test_predictions], ignore_index=True)

        # Post-process predictions
        pred_cols = [f"Q_{model}" for model in self.models] + [f"Q_{self.name}"]
        hindcast_df = self.__post_process_data__(
            df=hindcast_df, pred_cols=pred_cols, obs_col="Q_obs"
        )

        # Calculate valid period
        offset = self.general_config.get(
            "offset", self.general_config["prediction_horizon"]
        )
        shift = offset - self.general_config["prediction_horizon"]
        valid_from = (
            hindcast_df["date"]
            + datetime.timedelta(days=1)
            + datetime.timedelta(days=shift)
        )
        valid_to = valid_from + datetime.timedelta(
            days=self.general_config["prediction_horizon"]
        )

        hindcast_df["valid_from"] = valid_from
        hindcast_df["valid_to"] = valid_to

        # Save fitted models
        self.save_model(is_fitted=True)

        logger.info(f"Finished calibration and hindcasting for {self.name}")
        return hindcast_df

    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict in operational mode using fitted models and MiniRocket features.
        """
        logger.info(f"Starting operational prediction for {self.name}")

        if today is None:
            today = datetime.datetime.now()
            today = pd.to_datetime(today.strftime("%Y-%m-%d"))
        else:
            today = pd.to_datetime(today.strftime("%Y-%m-%d"))

        # Load models
        self.load_model()

        if not self.fitted_models:
            logger.error("No fitted models found. Please train models first.")
            return pd.DataFrame()

        # Filter data to recent period for efficiency
        cutoff_date = today - pd.DateOffset(years=2)
        self.data = self.data[self.data["date"] >= cutoff_date].copy()
        self.data = self.__make_data_continuous__(self.data)

        logger.info(
            f"Filtered data from {cutoff_date.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}"
        )

        # Preprocess data
        self.__preprocess_data__()

        # Set target to 0 for operational mode
        self.data[self.target] = 0

        # Calculate valid period
        offset = self.general_config.get(
            "offset", self.general_config["prediction_horizon"]
        )
        shift = offset - self.general_config["prediction_horizon"]
        valid_from = today + datetime.timedelta(days=1) + datetime.timedelta(days=shift)
        valid_to = valid_from + datetime.timedelta(
            days=self.general_config["prediction_horizon"]
        )

        valid_from_str = valid_from.strftime("%Y-%m-%d")
        valid_to_str = valid_to.strftime("%Y-%m-%d")

        logger.info(f"Forecast valid from: {valid_from_str} to: {valid_to_str}")

        # Get basin codes
        basin_codes = self.data["code"].unique()

        # Create time series windows for prediction
        target_dates = [today] * len(basin_codes)  # Predict for today for each basin

        # Need to ensure we have data up to today for each basin
        prediction_data = []
        for code in basin_codes:
            basin_data = self.data[self.data["code"] == code]
            if today in basin_data["date"].values:
                prediction_data.append(basin_data)

        if not prediction_data:
            logger.error("No basins have data for the prediction date")
            return pd.DataFrame()

        combined_data = pd.concat(prediction_data, ignore_index=True)

        X_pred_3d, pred_metadata = self.__create_time_series_windows__(
            combined_data, target_dates=[today], for_prediction=True
        )

        if X_pred_3d.size == 0:
            logger.error("No valid prediction windows created")
            return pd.DataFrame()

        # Apply MiniRocket transformation
        X_pred_minirocket = self.__apply_minirocket_transform__(
            X_pred_3d, fit=False, transformer_key="global"
        )

        # Prepare features
        pred_features_df = self.__prepare_features_for_model__(
            X_pred_minirocket, pred_metadata
        )

        # Make predictions with each model
        all_predictions = pred_features_df[["date", "code"]].copy()
        pred_cols = []

        for model_type, model_info in self.fitted_models.items():
            logger.info(f"Making predictions with {model_type}")

            model = model_info["model"]
            feature_cols = model_info["feature_cols"]

            # Ensure all required features are present
            missing_features = set(feature_cols) - set(pred_features_df.columns)
            if missing_features:
                logger.warning(f"Missing features for {model_type}: {missing_features}")
                continue

            X_pred = pred_features_df[feature_cols]
            y_pred = model.predict(X_pred)

            pred_col = f"Q_{model_type}"
            all_predictions[pred_col] = y_pred
            pred_cols.append(pred_col)

        # Add ensemble prediction
        ensemble_name = f"Q_{self.name}"
        all_predictions[ensemble_name] = all_predictions[pred_cols].mean(axis=1)
        pred_cols.append(ensemble_name)

        # Post-process predictions
        all_predictions = self.__post_process_data__(
            df=all_predictions, pred_cols=pred_cols, obs_col=None
        )

        # Create final forecast DataFrame
        forecast = pd.DataFrame(
            {
                "code": all_predictions["code"],
                "forecast_date": today.strftime("%Y-%m-%d"),
                "valid_from": valid_from_str,
                "valid_to": valid_to_str,
                "prediction_horizon_days": self.general_config["prediction_horizon"],
                "model_name": self.name,
                "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )

        # Add predictions
        for pred_col in pred_cols:
            forecast[pred_col] = all_predictions[pred_col]

        logger.info(f"Operational forecast completed for {len(forecast)} basins")
        return forecast

    def __filter_forecast_days__(self) -> None:
        """Filter data to include only specified forecast days."""
        forecast_days = self.general_config.get(
            "forecast_days", [5, 10, 15, 20, 25, "end"]
        )

        if forecast_days:
            day_conditions = []
            for forecast_day in forecast_days:
                if forecast_day == "end":
                    day_conditions.append(
                        self.data["date"].dt.day == self.data["date"].dt.days_in_month
                    )
                else:
                    day_conditions.append(self.data["date"].dt.day == forecast_day)

            combined_condition = day_conditions[0]
            for condition in day_conditions[1:]:
                combined_condition = combined_condition | condition

            self.data = self.data[combined_condition]

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune hyperparameters for the models using MiniRocket features.
        """
        logger.info(f"Starting hyperparameter tuning for {self.name}")

        self.__preprocess_data__()

        # Filter forecast days if configured
        if self.general_config.get("forecast_days"):
            self.__filter_forecast_days__()

        # Split data for hyperparameter tuning
        all_years = sorted(self.data["year"].unique())
        num_hparam_years = self.hparam_tuning_years
        train_years = all_years[:-num_hparam_years]
        val_years = all_years[-num_hparam_years:]

        logger.info(f"Training years: {train_years}, Validation years: {val_years}")

        # Create time series windows
        train_data = self.data[self.data["year"].isin(train_years)]
        val_data = self.data[self.data["year"].isin(val_years)]

        X_train_3d, train_metadata = self.__create_time_series_windows__(
            train_data, target_dates=None, for_prediction=False
        )

        X_val_3d, val_metadata = self.__create_time_series_windows__(
            val_data, target_dates=None, for_prediction=False
        )

        if X_train_3d.size == 0 or X_val_3d.size == 0:
            return False, "Not enough data for hyperparameter tuning"

        # Apply MiniRocket
        X_train_minirocket = self.__apply_minirocket_transform__(
            X_train_3d, fit=True, transformer_key="hparam_tuning"
        )

        X_val_minirocket = self.__apply_minirocket_transform__(
            X_val_3d, fit=False, transformer_key="hparam_tuning"
        )

        # Prepare features
        train_features_df = self.__prepare_features_for_model__(
            X_train_minirocket, train_metadata
        )

        val_features_df = self.__prepare_features_for_model__(
            X_val_minirocket, val_metadata
        )

        # Get feature columns
        feature_cols = [
            col
            for col in train_features_df.columns
            if col not in ["date", "code", "year", "day", "week", self.target]
        ]

        X_train = train_features_df[feature_cols]
        y_train = train_features_df[self.target]
        X_val = val_features_df[feature_cols]
        y_val = val_features_df[self.target]

        # Tune hyperparameters for each model
        best_params = {}

        for model_type in self.models:
            logger.info(f"Tuning hyperparameters for {model_type}")

            cat_features = (
                ["code_str"]
                if model_type == "catboost" and "code_str" in feature_cols
                else []
            )

            # For linear regression and SVR, we'll use simpler parameter search
            if model_type == "linear":
                best_params[model_type] = {}  # No hyperparameters to tune
            elif model_type == "svr":
                # Simple grid search for SVR
                from sklearn.model_selection import GridSearchCV

                param_grid = {
                    "C": [0.1, 1, 10, 100],
                    "epsilon": [0.01, 0.1, 0.5],
                    "kernel": ["rbf", "linear"],
                }
                svr = SVR()
                grid_search = GridSearchCV(
                    svr, param_grid, cv=3, scoring="neg_mean_squared_error"
                )
                grid_search.fit(X_train, y_train)
                best_params[model_type] = grid_search.best_params_
            else:
                # Use sci_utils for tree-based models
                tuned_params = sci_utils.optimize_hyperparams(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    model_type=model_type,
                    cat_features=cat_features,
                    n_trials=self.general_config.get("n_trials", 50),
                    artifacts=None,  # MiniRocket doesn't use the same artifacts
                    experiment_config=self.general_config,
                    target=self.target,
                    basin_codes=val_features_df["code"],
                    val_dates=val_features_df["date"],
                )

                if tuned_params is None:
                    logger.error(f"Hyperparameter tuning failed for {model_type}")
                    continue

                best_params[model_type] = tuned_params

            logger.info(f"Best parameters for {model_type}: {best_params[model_type]}")

        # Update model configuration
        self.model_config = best_params

        # Save updated configuration
        self.save_model(is_fitted=False)

        return True, "Hyperparameter tuning completed successfully"

    def save_model(self, is_fitted: bool = False) -> None:
        """Save fitted models and MiniRocket transformers."""
        logger.info(f"Saving {self.name} models")
        save_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        os.makedirs(save_path, exist_ok=True)

        if is_fitted:
            # Save MiniRocket transformer
            transformer_path = os.path.join(save_path, "minirocket_transformer.joblib")
            joblib.dump(self.minirocket_transformers.get("global"), transformer_path)

            # Save each fitted model
            for model_type, model_info in self.fitted_models.items():
                model_path = os.path.join(save_path, f"{model_type}_model.joblib")
                joblib.dump(model_info["model"], model_path)

                # Save feature columns
                features_path = os.path.join(save_path, f"{model_type}_features.json")
                with open(features_path, "w") as f:
                    json.dump(model_info["feature_cols"], f)

                # Save feature importance if available
                if model_info.get("feature_importance") is not None:
                    importance_path = os.path.join(
                        save_path, f"{model_type}_feature_importance.csv"
                    )
                    model_info["feature_importance"].to_csv(
                        importance_path, index=False
                    )

        # Save configurations
        configs = {
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "general_config": self.general_config,
            "minirocket_params": {
                "input_length": self.input_length,
                "num_kernels": self.num_kernels,
                "ts_feature_cols": self.ts_feature_cols
                if hasattr(self, "ts_feature_cols")
                else [],
            },
        }

        for config_name, config_data in configs.items():
            config_path = os.path.join(save_path, f"{config_name}.json")
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=4)

        logger.info(f"Models saved to {save_path}")

    def load_model(self) -> None:
        """Load fitted models and MiniRocket transformers."""
        logger.info(f"Loading {self.name} models")
        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")

        if not os.path.exists(load_path):
            logger.error(f"Model path {load_path} does not exist")
            return

        # Load MiniRocket transformer
        transformer_path = os.path.join(load_path, "minirocket_transformer.joblib")
        if os.path.exists(transformer_path):
            self.minirocket_transformers["global"] = joblib.load(transformer_path)

        # Load minirocket params
        minirocket_params_path = os.path.join(load_path, "minirocket_params.json")
        if os.path.exists(minirocket_params_path):
            with open(minirocket_params_path, "r") as f:
                minirocket_params = json.load(f)
                self.input_length = minirocket_params.get(
                    "input_length", self.input_length
                )
                self.num_kernels = minirocket_params.get(
                    "num_kernels", self.num_kernels
                )
                self.ts_feature_cols = minirocket_params.get("ts_feature_cols", [])

        # Load fitted models
        self.fitted_models = {}
        for model_type in self.models:
            model_path = os.path.join(load_path, f"{model_type}_model.joblib")
            if not os.path.exists(model_path):
                logger.warning(f"Model file for {model_type} not found")
                continue

            model = joblib.load(model_path)

            # Load feature columns
            features_path = os.path.join(load_path, f"{model_type}_features.json")
            with open(features_path, "r") as f:
                feature_cols = json.load(f)

            self.fitted_models[model_type] = {
                "model": model,
                "feature_cols": feature_cols,
            }

        logger.info(f"Models loaded from {load_path}")
