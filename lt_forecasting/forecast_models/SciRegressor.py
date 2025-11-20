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

warnings.filterwarnings("ignore")

from lt_forecasting.forecast_models.base_class import BaseForecastModel
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

from lt_forecasting.scr import FeatureExtractor as FE
from lt_forecasting.scr.FeatureProcessingArtifacts import (
    process_training_data,
    process_test_data,
    post_process_predictions,
)
from lt_forecasting.scr.FeatureProcessingArtifacts import (
    FeatureProcessingArtifacts,
)
from lt_forecasting.scr import data_utils as du
from lt_forecasting.scr import sci_utils

# Shared logging
import logging
from lt_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger
logging.getLogger("fiona.ogrext").setLevel(logging.WARNING)
logging.getLogger("fiona").setLevel(logging.WARNING)


class SciRegressor(BaseForecastModel):
    """
    A regressor class for ensemble models which can be fitted using "Sci-Kit Learn style" (fit/predict) methods with tabular data.,
    XGBoost, LightGBM, CatBoost, etc.
    Uses GLOBAL fitting approach where a model is trained on all basins and periods simultaneously.
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
        self.models = self.general_config.get(
            "models", ["xgboost"]
        )  # List of model types
        self.fitted_models = {}  # Will store fitted model objects per period

        # Get preprocessing configuration
        self.target = self.general_config.get("target", "target")
        self.cat_features = self.general_config.get("cat_features", ["code_str"])
        self.feature_cols = self.general_config.get(
            "feature_cols",
            [
                "discharge",
                "P",
                "T",
            ],
        )
        self.static_features = self.general_config.get("static_features", [])
        self.rivers_to_exclude = self.general_config.get("rivers_to_exclude", [])
        self.snow_vars = self.general_config.get("snow_vars", ["SWE"])
        self.early_stopping_val_fraction = self.general_config.get(
            "early_stopping_val_fraction", 0.1
        )

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

        # New parameters for enhanced long-term mean scaling
        self.use_relative_target = self.general_config.get("use_relative_target", False)
        self.relative_scaling_vars = self.general_config.get(
            "relative_scaling_vars", []
        )

        self.allowbale_missing_value_operational = self.general_config.get(
            "allowbale_missing_value_operational", 0
        )

    def __preprocess_data__(self):
        """
        Preprocess the data by adding position and other derived features.
        """
        logger.info(f"-" * 50)
        logger.info(f"Starting data preprocessing for {self.name}")
        logger.info(f"Initial data shape: {self.data.shape}")
        logger.info(f"Initial columns: {self.data.columns.tolist()}")
        logger.info(f"-" * 50)
        try:
            self.data = du.glacier_mapper_features(
                df=self.data,
                static=self.static_data,
                cols_to_keep=self.general_config["glacier_mapper_features_to_keep"],
            )
        except Exception as e:
            logger.error(f"Error in glacier_mapper_features: {e}")

        # remove log_discharge if it exists
        if "log_discharge" in self.data.columns:
            self.data.drop(columns=["log_discharge"], inplace=True)

        # Sort by date
        self.data.sort_values(by="date", inplace=True)

        cols_to_keep = [
            col
            for col in self.data.columns
            if any([feature in col for feature in self.feature_cols])
        ]
        self.data = self.data[["date", "code"] + cols_to_keep]

        # -------------- 2. Preprocess Discharge ------------------------------
        for code in self.data.code.unique():
            if code not in self.static_data["code"].values:
                logger.warning(
                    f"Code {code} not found in static data. Skipping this code."
                )
                self.rivers_to_exclude.append(code)
                continue
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            # transform from m3/s to mm/day
            self.data.loc[self.data["code"] == code, "discharge"] = (
                self.data.loc[self.data["code"] == code, "discharge"] * 86.4 / area
            )

        # Filter out rivers to exclude
        self.data = self.data[~self.data["code"].isin(self.rivers_to_exclude)].copy()

        # -------------- 3. Snow Data to equal percentage area ------------------------------
        if self.path_config["path_to_hru_shp"] is not None:
            elevation_band_shp = gpd.read_file(self.path_config["path_to_hru_shp"])
            # rename CODE to code
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

        # Extract extended features for snowmapper:
        self.data = du.derive_features_from_snowmapper(self.data)

        logger.debug("Data preprocessing completed. Data shape: %s", self.data.shape)

        # -------------- 4. Feature Extraction ------------------------------
        self.__extract_features__()

        # -------------- 5. Load LR predictors if configured ------------------------------
        if self.general_config["use_lr_predictors"]:
            lr_predictors, lr_pred_cols = self.__load_lr_predictors__()
            self.data = pd.merge(
                self.data, lr_predictors, on=["date", "code"], how="inner"
            )
            if len(lr_pred_cols) > 1:
                self.data["ensemble_pred"] = self.data[lr_pred_cols].mean(axis=1)
                lr_pred_cols.append("ensemble_pred")
        else:
            lr_pred_cols = []

        # -------------- 6. Dummy encoding for categorical features ------------------------------
        self.data["basin"] = self.data["code"]
        self.data["code_str"] = self.data["code"].astype(str)
        self.data = pd.get_dummies(self.data, columns=["month", "basin"], dtype=int)

        # -------------- 7. Merge with static features ------------------------------
        static_df_feat = self.static_data[["code"] + self.static_features].copy()
        self.data = pd.merge(self.data, static_df_feat, on="code", how="inner")

        # -------------- 8. Prepare feature sets ------------------------------
        self.feature_set = (
            [col + "_" for col in self.feature_cols]
            + ["week_sin", "week_cos", "month_sin", "month_cos"]
            + self.static_features
            + lr_pred_cols
        )

        self.dynamic_features = (
            [col + "_" for col in self.feature_cols]
            + ["week_sin", "week_cos", "month_sin", "month_cos"]
            + lr_pred_cols
        )

        self.feature_set = [
            col
            for col in self.data.columns
            if any([f in col for f in self.feature_set])
        ]

        logger.info(f"Feature set for {self.name}: {self.feature_set}")

        # check if the cat_features are in the columns
        for cat_feature in self.cat_features:
            if cat_feature not in self.data.columns:
                logger.warning(
                    f"Categorical feature '{cat_feature}' not found in data columns. Removing from cat_features."
                )
                self.cat_features.remove(cat_feature)

    def __extract_features__(self):
        """
        Extract features from the data using FeatureExtractor and prepare for global fitting.
        """

        keys_to_remove = [
            key for key in self.feature_config.keys() if key not in self.feature_cols
        ]
        for key in keys_to_remove:
            self.feature_config.pop(key)

        logger.debug("Extracting features using FeatureExtractor")
        # Use FeatureExtractor for time series features
        extractor = FE.StreamflowFeatureExtractor(
            feature_configs=self.feature_config,
            prediction_horizon=self.general_config["prediction_horizon"],
            offset=self.general_config.get(
                "offset", self.general_config["prediction_horizon"]
            ),
        )

        self.data = extractor.create_all_features(self.data)

        # Add temporal features
        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month
        self.data["day"] = self.data["date"].dt.day

        # Add basin identity features for global training
        self.data["basin"] = self.data["code"]
        self.data["code_str"] = self.data["code"].astype(str)

        # Add cyclical encoding for temporal features
        self.data["month_sin"] = np.sin(2 * np.pi * self.data["month"] / 12)
        self.data["month_cos"] = np.cos(2 * np.pi * self.data["month"] / 12)

        # Add week features if available
        self.data["week"] = self.data["date"].dt.isocalendar().week
        self.data["week_sin"] = np.sin(2 * np.pi * self.data["week"] / 52)
        self.data["week_cos"] = np.cos(2 * np.pi * self.data["week"] / 52)

        logger.debug("Feature extraction completed. Data shape: %s", self.data.shape)

    def __load_lr_predictors__(self) -> pd.DataFrame:
        """
        Loads all the prediction df's from the path_config['path_to_lr_predictors'] directory.
        Where the model name is the name of the folder the prediction.csv is located in.
        """
        path_list = self.path_config["path_to_lr_predictors"]
        models = []

        all_predictions = None
        pred_cols = []
        for path in path_list:
            model_name = os.path.basename(os.path.dirname(path))
            # check if path ends with .csv
            if not path.endswith(".csv"):
                # add the prediction.csv to the path
                path = os.path.join(path, "predictions.csv")

            df = pd.read_csv(path)

            df["date"] = pd.to_datetime(df["date"])
            df["code"] = df["code"].astype(int)

            pred_col = f"Q_{model_name}"

            # check if pred_col exists in df
            if pred_col not in df.columns:
                logger.warning(
                    f"Prediction column '{pred_col}' not found in {model_name}. Skipping this model."
                )
                continue

            pred_cols.append(pred_col)

            if all_predictions is None:
                all_predictions = df[["date", "code", pred_col]].copy()

            else:
                # Merge predictions on date and code
                all_predictions = pd.merge(
                    all_predictions,
                    df[["date", "code", pred_col]],
                    on=["date", "code"],
                    how="inner",
                )

        for code in all_predictions["code"].unique():
            if code not in self.static_data["code"].values:
                logger.warning(
                    f"Code {code} not found in static data. Skipping this code."
                )
                continue
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            all_predictions.loc[all_predictions["code"] == code, pred_cols] = (
                all_predictions.loc[all_predictions["code"] == code, pred_cols]
                * area
                / 86.4
            )

        return all_predictions, pred_cols

    def __filter_forecast_days__(
        self,
    ) -> None:
        forecast_days = self.general_config.get(
            "forecast_days", [5, 10, 15, 20, 25, "end"]
        )

        # Filter data to include only the specified forecast days
        if forecast_days:
            day_conditions = []
            for forecast_day in forecast_days:
                if forecast_day == "end":
                    day_conditions.append(
                        self.data["date"].dt.day == self.data["date"].dt.days_in_month
                    )
                else:
                    day_conditions.append(self.data["date"].dt.day == forecast_day)

            # Combine all conditions with OR logic
            combined_condition = day_conditions[0]
            for condition in day_conditions[1:]:
                combined_condition = combined_condition | condition

            self.data = self.data[combined_condition]

    def __post_process_data__(
        self, df: pd.DataFrame, pred_cols: List[str], obs_col: str = None
    ) -> pd.DataFrame:
        """
        Post-process the data after model predictions.
        this is the re-transformation from mm/d to m3/s
        """
        # Convert predictions from mm/d to m3/s
        df = df.copy()
        for code in df["code"].unique():
            area = self.static_data[self.static_data["code"] == code][
                "area_km2"
            ].values[0]
            df.loc[df["code"] == code, pred_cols] = (
                df.loc[df["code"] == code, pred_cols] * area / 86.4
            )

            # Only convert observations if obs_col is provided and exists in the dataframe
            if obs_col is not None and obs_col in df.columns:
                df.loc[df["code"] == code, obs_col] = (
                    df.loc[df["code"] == code, obs_col] * area / 86.4
                )

        return df

    def __loocv__(
        self,
        years: List[int],
        features: List[str],
        params: Dict[str, Any] = None,
        model_type: str = "xgb",
    ) -> pd.DataFrame:
        """
        Perform Leave-One-Year-Out Cross-Validation.

        Args:
            years (List[int]): List of years to use for cross-validation.
            features (List[str]): List of features to use for training.
            params (Dict[str, Any], optional): Hyperparameters for the model.
            model_type (str, optional): Type of model to use (default is "xgb").

        Returns:
            pd.DataFrame: DataFrame containing the cross-validated predictions.
        """
        logger.info(f"Starting LOOCV for {self.name} with years: {years}")

        df_predictions = pd.DataFrame()

        pred_col = f"Q_{model_type}"

        if "catboost" in model_type:
            if len(self.cat_features) > 0:
                features = features + self.cat_features
                logger.info(
                    f"Using categorical features for {model_type}: {self.cat_features}"
                )

        # Iterate over each year for LOOCV
        for year in progress_bar(years, desc="Processing years", leave=True):
            df_train = self.data[self.data["year"] != year].dropna(subset=[self.target])
            df_test = self.data[self.data["year"] == year].dropna(subset=[self.target])

            if df_test.empty:
                logger.warning(
                    f"No data available for year {year}. Skipping this year in LOOCV."
                )
                continue
            # Original columns so we don't mess up anything
            df_predictions_year = df_test[["date", "code", self.target]].copy()

            # Create artifacts
            # This handles nan imputation, scaling and variable selection
            train_processed, artifacts = process_training_data(
                df_train=df_train,
                features=features,
                target=self.target,
                experiment_config=self.general_config,
                static_features=self.static_features,
            )

            final_features = artifacts.final_features

            # Process test data
            test_processed = process_test_data(
                df_test=df_test,
                artifacts=artifacts,
                experiment_config=self.general_config,
            )

            # Prepare data
            X_train = train_processed[final_features]
            y_train = train_processed[self.target]
            X_test = test_processed[final_features]

            if X_test.empty:
                logger.warning(
                    f"No valid test data available for year {year} after processing. Skipping this year in LOOCV."
                )
                continue

            # Create model
            if params:
                model = sci_utils.get_model(model_type, params, self.cat_features)
            else:
                params = {}
                model = sci_utils.get_model(model_type, params, self.cat_features)

            # Train and predict
            model = sci_utils.fit_model(
                model=model,
                X=X_train,
                y=y_train,
                model_type=model_type,
                val_fraction=self.early_stopping_val_fraction,
            )

            y_pred = model.predict(X_test)

            # Extra step if we dropped any rows in the test set
            test_processed[pred_col] = y_pred
            test_processed = test_processed[["date", "code", pred_col]].copy()

            df_predictions_year = pd.merge(
                df_predictions_year, test_processed, on=["date", "code"], how="inner"
            )

            df_predictions_year = post_process_predictions(
                df_predictions=df_predictions_year,
                artifacts=artifacts,
                experiment_config=self.general_config,
                prediction_column=pred_col,
                target=self.target,
            )

            df_predictions = pd.concat([df_predictions, df_predictions_year])

        # Rename columns
        df_predictions.rename(columns={self.target: "Q_obs"}, inplace=True)

        return df_predictions

    def __fit_on_all__(
        self,
        features: List[str],
        test_years: List[int] = None,
        params: Dict[str, Any] = None,
        model_type: str = "xgb",
    ) -> Tuple[Any, Any, List[str]]:
        """
        Fit the model on all available data.
        Args:
            features (List[str]): List of features to use for training.
            params (Dict[str, Any], optional): Hyperparameters for the model.
            model_type (str, optional): Type of model to use (default is "xgb").
        Returns:
            pd.DataFrame: DataFrame containing the fitted model predictions.
        """
        logger.info(
            f"Starting global fitting for {self.name} with model type: {model_type}"
        )
        df_predictions = pd.DataFrame()
        pred_col = f"Q_{model_type}"

        if "catboost" in model_type:
            if len(self.cat_features) > 0:
                features = features + self.cat_features
                logger.info(
                    f"Using categorical features for {model_type}: {self.cat_features}"
                )

        if test_years is None:
            train_data = self.data.copy()
            train_data = train_data.dropna(subset=[self.target]).copy()

        else:
            train_data = self.data[self.data["year"].isin(test_years) == False].copy()
            train_data = train_data.dropna(subset=[self.target]).copy()

            test_data = self.data[self.data["year"].isin(test_years)].copy()
            test_data = test_data.dropna(subset=[self.target]).copy()

            df_predictions = test_data.copy()
            df_predictions = df_predictions[["date", "code", self.target]].copy()

        # Create artifacts
        # This handles nan imputation, scaling and variable selection
        train_processed, artifacts = process_training_data(
            df_train=train_data,
            features=features,
            target=self.target,
            experiment_config=self.general_config,
            static_features=self.static_features,
        )

        final_features = artifacts.final_features

        # Prepare data
        X_train = train_processed[final_features]
        y_train = train_processed[self.target]

        # Create model
        if params:
            model = sci_utils.get_model(model_type, params, self.cat_features)
        else:
            params = {}
            model = sci_utils.get_model(model_type, params, self.cat_features)

        # Train the model
        model = sci_utils.fit_model(
            model=model,
            X=X_train,
            y=y_train,
            model_type=model_type,
            val_fraction=self.early_stopping_val_fraction,
        )

        # Predict on test data
        if test_years is not None:
            test_data_processed = process_test_data(
                df_test=test_data,
                artifacts=artifacts,
                experiment_config=self.general_config,
            )

            X_test = test_data_processed[final_features]
            y_pred = model.predict(X_test)

            test_data_processed[pred_col] = y_pred
            test_data_processed = test_data_processed[["date", "code", pred_col]].copy()

            df_predictions = pd.merge(
                df_predictions, test_data_processed, on=["date", "code"], how="inner"
            )
            df_predictions = post_process_predictions(
                df_predictions=df_predictions,
                artifacts=artifacts,
                experiment_config=self.general_config,
                prediction_column=pred_col,
                target=self.target,
            )

            # Rename columns
            df_predictions.rename(columns={self.target: "Q_obs"}, inplace=True)
        else:
            df_predictions = pd.DataFrame()

        return [model, df_predictions, artifacts, final_features]

    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict in operational mode using global trained models.

        Args:
            today (datetime.datetime, optional): Date to use as "today" for prediction.
                If None, uses current datetime.

        Returns:
            forecast (pd.DataFrame): DataFrame containing the forecasted values.
        """
        logger.info(f"Starting operational prediction for {self.name}")

        if today is None:
            today = datetime.datetime.now()
            today = pd.to_datetime(today.strftime("%Y-%m-%d"))
        else:
            today = pd.to_datetime(today.strftime("%Y-%m-%d"))

        # Step 1: Load models and artifacts
        self.load_model()

        if not self.fitted_models:
            logger.error(
                "No fitted models found. Please train models first using calibrate_model_and_hindcast()."
            )
            return pd.DataFrame()

        # Step 2: Filter data to only include last 2 years (for fast processing)
        cutoff_date = today - pd.DateOffset(years=2)
        self.data = self.data[self.data["date"] >= cutoff_date].copy()

        logger.info(
            f"Filtered data from {cutoff_date.strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}"
        )

        # Step 3: Data processing
        self.__preprocess_data__()

        # Set target to 0 for operational mode (no observations available)
        # this ensures that nothing gets filtered out in the next steps
        self.data.drop(columns=[self.target], inplace=True, errors="ignore")
        # self.data[self.target] = 0

        # Step 4: Calculate valid period
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

        # Step 5: Make predictions with ensemble of models
        forecast_predictions = {}
        all_pred_cols = []

        # Get unique basin codes for prediction
        basin_codes = self.data["code"].unique()

        # Prepare base forecast dataframe
        forecast_base = pd.DataFrame(
            {
                "code": basin_codes,
                "date": today.strftime("%Y-%m-%d"),
                "valid_from": valid_from_str,
                "valid_to": valid_to_str,
                "prediction_horizon_days": self.general_config["prediction_horizon"],
            }
        )

        for model_type in self.models:
            if model_type not in self.fitted_models:
                logger.warning(
                    f"Model {model_type} not found in fitted models. Skipping."
                )
                continue

            logger.info(f"Making predictions with {model_type}")

            # Get model components
            model = self.fitted_models[model_type]["model"]
            artifacts = self.fitted_models[model_type]["artifacts"]
            final_features = self.fitted_models[model_type]["final_features"]

            pred_col = f"Q_{model_type}"
            all_pred_cols.append(pred_col)

            # Get the most recent complete data for each basin for prediction
            prediction_data = []
            successful_basins = []
            failed_basins = []

            for code in basin_codes:
                basin_data = self.data[self.data["code"] == code].copy()

                if basin_data.empty:
                    logger.warning(f"No data available for basin {code}. Skipping.")
                    failed_basins.append(code)
                    continue

                # Take the most recent complete observation
                today_row = basin_data[basin_data["date"] == today]
                if today_row[final_features].isnull().any(axis=1).any():
                    logger.warning(
                        f"Missing features for basin {code} on {today.strftime('%Y-%m-%d')}."
                    )

                    logger.warning(f"Head of today_row:\n{today_row.head()}")

                    # Get columns that have NaN values
                    cols_with_nan = (
                        today_row[final_features]
                        .columns[today_row[final_features].isnull().any(axis=0)]
                        .tolist()
                    )
                    logger.warning(f"Features which are Nan : {cols_with_nan}")

                # TODO: Clever way to handle after how many missing features we need to skip the basin
                # Maybe load feature importance and if feature in top 10 skip the basin
                number_of_nan_columns = today_row[final_features].isnull().sum().sum()
                if number_of_nan_columns > self.allowbale_missing_value_operational:
                    logger.warning(
                        f"Too many missing features ({number_of_nan_columns}) for basin {code} on {today.strftime('%Y-%m-%d')}. Skipping."
                    )
                    failed_basins.append(code)
                    continue

                if today_row.empty:
                    logger.warning(
                        f"No data available for basin {code} on {today.strftime('%Y-%m-%d')}. Skipping."
                    )
                    failed_basins.append(code)
                    continue

                prediction_data.append(today_row)
                successful_basins.append(code)

            if not prediction_data:
                logger.error("No prediction data available for any basin.")
                continue

            logger.info(f"Successful dataloading for basins: {len(successful_basins)}")
            logger.info(f"Failed dataloading for basins: {len(failed_basins)}")

            # Combine all basin prediction data
            prediction_df = pd.concat(prediction_data, ignore_index=True)

            # Apply the same preprocessing artifacts as during training
            prediction_processed = process_test_data(
                df_test=prediction_df,
                artifacts=artifacts,
                experiment_config=self.general_config,
            )

            # Make predictions
            X_pred = prediction_processed[final_features]
            y_pred = model.predict(X_pred)

            # Store predictions
            prediction_processed[pred_col] = y_pred

            # Extract relevant columns for forecast
            forecast_model = prediction_processed[["date", "code", pred_col]].copy()

            # check if there are any nan vlaues
            if forecast_model[pred_col].isnull().any():
                logger.info(f"Found NaN values in predictions for {model_type}.")
                # log the codes with NaN predictions
                nan_codes = forecast_model[forecast_model[pred_col].isnull()]["code"]
                logger.info(f"NaN predictions for codes: {nan_codes.unique().tolist()}")

            # Post process predictions
            forecast_model = post_process_predictions(
                df_predictions=forecast_model,
                artifacts=artifacts,
                experiment_config=self.general_config,
                prediction_column=pred_col,
                target=self.target,
            )

            if forecast_model[pred_col].isnull().any():
                logger.info(
                    f"Found NaN values in post-processed predictions for {model_type}."
                )
                # log the codes with NaN predictions
                nan_codes = forecast_model[forecast_model[pred_col].isnull()]["code"]
                logger.info(
                    f"NaN post-processed predictions for codes: {nan_codes.unique().tolist()}"
                )

            forecast_predictions[model_type] = forecast_model

        # Merge all model predictions
        forecast = forecast_base.copy()

        for model_type, model_forecast in forecast_predictions.items():
            forecast = pd.merge(forecast, model_forecast, on="code", how="left")

        if not all_pred_cols:
            logger.error("No successful predictions made.")
            return pd.DataFrame()

        # Create ensemble prediction (average of all models)
        ensemble_name = f"Q_{self.name}"
        forecast[ensemble_name] = forecast[all_pred_cols].mean(axis=1, skipna=True)
        all_pred_cols.append(ensemble_name)

        # Step 6: Post-process data (convert from mm/day back to m³/s)
        forecast = self.__post_process_data__(
            df=forecast,
            pred_cols=all_pred_cols,
            obs_col=None,  # No observations in operational mode
        )

        # Reorder columns for better readability
        base_cols = [
            "code",
            "date",
            "valid_from",
            "valid_to",
            "prediction_horizon_days",
        ]

        pred_cols = [col for col in forecast.columns if col.startswith("Q_")]

        forecast = forecast[base_cols + pred_cols]

        # check if all basin_codes are in the forecast
        missing_basin_codes = set(basin_codes) - set(forecast["code"].unique())
        if missing_basin_codes:
            logger.warning(
                f"Missing basin codes in forecast: {missing_basin_codes}. "
                "This may indicate that some basins had no data for the prediction date."
            )

            # Handle those codes by adding them with NaN predictions
            for code in missing_basin_codes:
                empty_row = {
                    "code": code,
                    "date": today.strftime("%Y-%m-%d"),
                    "valid_from": valid_from_str,
                    "valid_to": valid_to_str,
                    "prediction_horizon_days": self.general_config[
                        "prediction_horizon"
                    ],
                }
                for col in all_pred_cols:
                    empty_row[col] = np.nan

                empty_row_df = pd.DataFrame([empty_row], index=[0])

                # Add the empty row to the forecast DataFrame
                forecast = pd.concat([forecast, empty_row_df], ignore_index=True)

        logger.info(
            f"Operational forecast completed for {len(forecast)} basins with {len(all_pred_cols)} predictions"
        )

        return forecast

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate the ensemble models using Leave-One-Year-Out cross-validation.

        Returns:
            hindcast (pd.DataFrame): DataFrame containing the hindcasted values.
        """
        logger.info(
            f"Starting calibration and hindcasting for {self.name} with models: {self.models}"
        )

        self.__preprocess_data__()

        if "year" not in self.data.columns:
            self.data["year"] = self.data["date"].dt.year

        # Add day column if not present
        if "day" not in self.data.columns:
            self.data["day"] = self.data["date"].dt.day

        # Get configuration parameters
        test_years = self.test_years

        self.__filter_forecast_days__()

        all_years = sorted(self.data["year"].unique())

        if len(test_years) > 0:
            loocv_years = [year for year in all_years if year not in test_years]
            test_years = [year for year in all_years if year in test_years]
        else:
            loocv_years = all_years
            test_years = None

        hindcast_df = None
        all_pred_cols = []
        for model_type, params in self.model_config.items():
            logger.info(f"Calibrating model {model_type} with parameters: {params}")

            # Perform LOOCV for the model
            df_predictions = self.__loocv__(
                years=loocv_years,
                features=self.feature_set,
                params=params,
                model_type=model_type,
            )

            pred_col = f"Q_{model_type}"
            all_pred_cols.append(pred_col)

            if hindcast_df is None:
                hindcast_df = df_predictions
            else:
                # merge on date and code
                df_predictions = df_predictions[["date", "code", pred_col]]
                hindcast_df = pd.merge(
                    hindcast_df, df_predictions, on=["date", "code"], how="inner"
                )

        ensemble_name = f"Q_{self.name}"
        hindcast_df[ensemble_name] = hindcast_df[all_pred_cols].mean(axis=1)

        hindcast_df = self.__post_process_data__(
            df=hindcast_df, pred_cols=all_pred_cols + [ensemble_name], obs_col="Q_obs"
        )

        logger.info(f"Finished hindcasting for {self.name}")

        # ------------ Fit on All Data --------------
        logger.info(f"Fitting models on all data for {self.name}")
        fitted_models = {}

        fit_on_all_predictions = None
        for model_type, params in self.model_config.items():
            logger.info(f"Fitting model {model_type} with parameters: {params}")

            model, df_predictions, artifacts, final_features = self.__fit_on_all__(
                features=self.feature_set,
                test_years=test_years,
                params=params,
                model_type=model_type,
            )

            fitted_models[model_type] = {
                "model": model,
                "artifacts": artifacts,
                "final_features": final_features,
            }

            pred_col = f"Q_{model_type}"
            if fit_on_all_predictions is None:
                fit_on_all_predictions = df_predictions
            elif fit_on_all_predictions.empty:
                logger.debug(
                    f"No predictions available for model {model_type}. Skipping."
                )
            else:
                # merge on date and code
                df_predictions = df_predictions[["date", "code", pred_col]]
                fit_on_all_predictions = pd.merge(
                    fit_on_all_predictions,
                    df_predictions,
                    on=["date", "code"],
                    how="inner",
                )

            feature_importance = sci_utils.get_feature_importance(model)
            # save the feature importance
            fitted_models[model_type]["feature_importance"] = feature_importance

        # Add ensemble prediction
        if not fit_on_all_predictions.empty:
            fit_on_all_predictions[ensemble_name] = fit_on_all_predictions[
                all_pred_cols
            ].mean(axis=1)
            fit_on_all_predictions = self.__post_process_data__(
                df=fit_on_all_predictions,
                pred_cols=all_pred_cols + [ensemble_name],
                obs_col="Q_obs",
            )

            hindcast_df = pd.concat(
                [hindcast_df, fit_on_all_predictions], ignore_index=True
            )

        # Save fitted models
        self.fitted_models = fitted_models
        self.save_model(is_fitted=True)

        #  Calculate valid period
        if not self.general_config.get("offset"):
            self.general_config["offset"] = self.general_config["prediction_horizon"]
        shift = (
            self.general_config["offset"] - self.general_config["prediction_horizon"]
        )
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

        return hindcast_df

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune the hyperparameters of the ensemble models with optuna.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success and a message.
        """
        logger.info(
            f"Starting hyperparameter tuning for {self.name} with models: {self.models}"
        )

        # Apply the same preprocessing as other methods
        self.__preprocess_data__()

        if "year" not in self.data.columns:
            self.data["year"] = self.data["date"].dt.year

        # Add day column if not present
        if "day" not in self.data.columns:
            self.data["day"] = self.data["date"].dt.day

        # Get configuration parameters
        hparam_tuning_years = self.hparam_tuning_years

        self.__filter_forecast_days__()

        all_years = sorted(self.data["year"].unique())
        train_years = [year for year in all_years if year not in hparam_tuning_years]
        val_years = [year for year in all_years if year in hparam_tuning_years]

        df_train = self.data[self.data["year"].isin(train_years)].copy()
        df_val = self.data[self.data["year"].isin(val_years)].copy()

        logger.info(
            f"Hyperparameter Tuning on: Training years: {train_years}, Validation years: {val_years}"
        )

        # Dropna based on target
        df_train = df_train.dropna(subset=[self.target]).copy()
        df_val = df_val.dropna(subset=[self.target]).copy()

        if df_train.empty or df_val.empty:
            logger.error(
                "Not enough data for hyperparameter tuning. Ensure that the dataset contains sufficient years of data."
            )
            return (
                False,
                "Not enough data for hyperparameter tuning. Ensure that the dataset contains sufficient years of data.",
            )

        for model_type, params in self.model_config.items():
            logger.info(f"Tuning hparams for model {model_type}")

            # Use the same feature set logic as other methods
            if "catboost" in model_type:
                if len(self.cat_features) > 0:
                    this_feature_set = self.feature_set + self.cat_features
                    logger.info(
                        f"Using categorical features for {model_type}: {self.cat_features}"
                    )
                else:
                    this_feature_set = self.feature_set
            else:
                this_feature_set = self.feature_set

            df_train_processed, artifacts = process_training_data(
                df_train=df_train,
                features=this_feature_set,
                target=self.target,
                experiment_config=self.general_config,
                static_features=self.static_features,
            )

            if df_train_processed.isna().any().any():
                logger.debug(
                    "Training data contains NaN values after preprocessing. This may affect model performance."
                )

                # Get only columns that have NaN values and their counts
                nan_counts = df_train_processed.isna().sum()
                columns_with_nans = nan_counts[nan_counts > 0]

                if not columns_with_nans.empty:
                    logger.debug(f"Columns with NaN values:\n{columns_with_nans}")

                    # Optional: Also log which rows have NaNs for debugging
                    rows_with_nans = df_train_processed[
                        df_train_processed.isna().any(axis=1)
                    ]
                    logger.debug(
                        f"Number of rows with NaN values: {len(rows_with_nans)}"
                    )

                    # Optional: Show a sample of problematic rows
                    if len(rows_with_nans) > 0:
                        logger.debug(
                            f"Sample rows with NaN values:\n{rows_with_nans.head()}"
                        )
                else:
                    logger.debug("No NaN values found in training data.")

            df_val_processed = process_test_data(
                df_test=df_val,
                artifacts=artifacts,
                experiment_config=self.general_config,
            )

            if df_val_processed.isna().any().any():
                logger.debug(
                    "Validation data contains NaN values after preprocessing. This may affect model performance."
                )

                # Get only columns that have NaN values and their counts
                nan_counts = df_val_processed.isna().sum()
                columns_with_nans = nan_counts[nan_counts > 0]

                if not columns_with_nans.empty:
                    logger.debug(f"Columns with NaN values:\n{columns_with_nans}")

                    # Optional: Also log which rows have NaNs for debugging
                    rows_with_nans = df_val_processed[
                        df_val_processed.isna().any(axis=1)
                    ]
                    logger.debug(
                        f"Number of rows with NaN values: {len(rows_with_nans)}"
                    )

                    # Optional: Show a sample of problematic rows
                    if len(rows_with_nans) > 0:
                        logger.debug(
                            f"Sample rows with NaN values:\n{rows_with_nans.head()}"
                        )
                else:
                    logger.debug("No NaN values found in validation data.")

            final_features = artifacts.final_features

            X_train = df_train_processed[final_features]
            y_train = df_train_processed[self.target]
            X_val = df_val_processed[final_features]
            y_val = df_val_processed[self.target]

            # Get basin codes and dates for validation data if needed
            basin_codes_val = df_val_processed["code"]
            val_dates = df_val_processed["date"]

            best_params = sci_utils.optimize_hyperparams(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                model_type=model_type,
                cat_features=self.cat_features,
                n_trials=self.general_config.get("n_trials", 50),
                artifacts=artifacts,
                experiment_config=self.general_config,
                target=self.target,
                basin_codes=basin_codes_val,
                val_dates=val_dates,
            )

            if best_params is None:
                logger.error(f"Hyperparameter tuning failed for model {model_type}.")
                return False, f"Hyperparameter tuning failed for model {model_type}."

            logger.info(f"Best parameters for {model_type}: {best_params}")

            # Update the model_config with the best parameters
            self.model_config[model_type] = best_params

        # Save the updated model configuration
        self.save_model(is_fitted=False)

        return (
            True,
            "Hyperparameter tuning completed successfully. Updated model configuration saved.",
        )

    def save_model(self, is_fitted: bool = False) -> None:
        """
        Save all fitted models and preprocessing artifacts.
        """
        logger.info(f"Saving {self.name} models")
        save_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        os.makedirs(save_path, exist_ok=True)

        # Save fitted models
        if is_fitted:
            for model_type, model_info in self.fitted_models.items():
                model = model_info["model"]
                artifacts = model_info["artifacts"]
                final_features = model_info["final_features"]
                feature_importance = model_info.get("feature_importance", None)

                # Save the model
                model_path = os.path.join(save_path, f"{model_type}_model.joblib")
                joblib.dump(model, model_path)

                # Save artifacts
                artifacts_path = os.path.join(save_path, f"{model_type}_artifacts")
                artifacts.save(filepath=artifacts_path, format="hybrid")

                # Save final features
                features_path = os.path.join(save_path, f"{model_type}_features.json")
                with open(features_path, "w") as f:
                    json.dump(final_features, f)

                if feature_importance is not None:
                    # save as csv
                    feature_importance_path = os.path.join(
                        save_path, f"{model_type}_feature_importance.csv"
                    )
                    feature_importance.to_csv(feature_importance_path, index=False)

        def convert_numpy_types(obj):
            """Recursively convert numpy types to native Python types for JSON serialization."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj

        # Save general model configuration
        model_config_path = os.path.join(save_path, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(convert_numpy_types(self.model_config), f, indent=4)
        # Save general feature configuration
        feature_config_path = os.path.join(save_path, "feature_config.json")
        with open(feature_config_path, "w") as f:
            json.dump(convert_numpy_types(self.feature_config), f, indent=4)
        # Save general experiment configuration
        experiment_config_path = os.path.join(save_path, "experiment_config.json")
        with open(experiment_config_path, "w") as f:
            json.dump(convert_numpy_types(self.general_config), f, indent=4)

        logger.info(f"Models and artifacts saved to {save_path}")

    def load_model(self) -> None:
        """
        Load all fitted models and preprocessing artifacts.
        """
        logger.info(f"Loading {self.name}  models")
        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")

        if not os.path.exists(load_path):
            logger.error(f"Model path {load_path} does not exist. Cannot load models.")
            return

        # Load fitted models
        for model_type in self.models:
            model_path = os.path.join(load_path, f"{model_type}_model.joblib")
            if not os.path.exists(model_path):
                logger.error(
                    f"Model file {model_path} does not exist. Cannot load model."
                )
                continue

            model = joblib.load(model_path)

            # Load artifacts
            artifacts_path = os.path.join(load_path, f"{model_type}_artifacts")
            artifacts = FeatureProcessingArtifacts.load(
                filepath=artifacts_path, format="hybrid"
            )

            # Load final features
            features_path = os.path.join(load_path, f"{model_type}_features.json")
            with open(features_path, "r") as f:
                final_features = json.load(f)

            self.fitted_models[model_type] = {
                "model": model,
                "artifacts": artifacts,
                "final_features": final_features,
            }

        logger.info(f"Models loaded from {load_path}")
