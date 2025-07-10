import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt

# Ensure the logs directory exists
import datetime

# Shared logging
import logging
from log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


from forecast_models.base_class import BaseForecastModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
from tqdm import tqdm

from scr import FeatureExtractor as FE


class LinearRegressionModel(BaseForecastModel):
    """
    Linear Regression model for forecasting.
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
        Initialize the Linear Regression model with a configuration dictionary.

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

        logger.debug("Extracting features for Linear Regression model")
        self.__extract_features__()
        logger.debug("Features extracted successfully")

    def __extract_features__(self):
        """
        Extract features from the data.

        Args:
            data (pd.DataFrame): DataFrame containing the data.
        """
        # Implement feature extraction logic here
        extractor = FE.StreamflowFeatureExtractor(
            feature_configs=self.feature_config,
            prediction_horizon=self.general_config["prediction_horizon"],
            offset=self.general_config["offset"],
        )

        self.data = extractor.create_all_features(self.data)
        self.data["year"] = self.data["date"].dt.year
        self.data["month"] = self.data["date"].dt.month

    def __get_periods__(self) -> None:
        """
        Get unique periods from the data.
        The period name is <month>-<day>.
        if it is the last day of the month, the period name is <month>-end. (accounts for february)

        Returns:
            pd.DataFrame: DataFrame containing unique periods.
        """
        self.data["day"] = self.data["date"].dt.day
        self.data["month"] = self.data["date"].dt.month
        self.data["period_suffix"] = np.where(
            self.data["date"].dt.day == self.data["date"].dt.days_in_month,
            "end",
            self.data["date"].dt.day.astype(str),
        )
        self.data["period"] = (
            self.data["month"].astype(str) + "-" + self.data["period_suffix"]
        )
        self.data.drop(columns=["period_suffix"], inplace=True)

    def get_highest_corr_features(
        self, df_period: pd.DataFrame, target: str
    ) -> List[str]:
        num_features = self.general_config["num_features"]
        base_features = self.general_config["base_features"]
        snow_vars = self.general_config["snow_vars"]

        possible_features = []
        this_period = df_period["period"].unique()[0]

        for feat in base_features:
            possible_features += [col for col in df_period.columns if feat in col]
        for feat in snow_vars:
            possible_features += [col for col in df_period.columns if feat in col]

        corr = df_period[possible_features + [target]].corr()
        abs_corr_series = corr[target].abs()

        # Convert to a regular Python dictionary to avoid Series comparison issues
        abs_corr = {k: float(v) for k, v in abs_corr_series.items() if not pd.isna(v)}

        nan_test = possible_features.copy()
        nan_test.append(target)
        test_df = df_period.dropna(subset=nan_test, how="all").copy()
        len_data_points = len(test_df)

        feature_types = {}
        for feature in possible_features:
            if feature not in abs_corr:  # Skip features with no correlation
                continue
            # Split by underscore to get variable type (like SWE, SCA, etc.)
            var_type = feature.split("_")[0]
            # If this is a new variable type or has higher correlation than previous best
            if (
                var_type not in feature_types
                or abs_corr[feature] > abs_corr[feature_types[var_type]]
            ):
                feature_types[var_type] = feature

        # Get the best features across all types (sorted by correlation)
        best_features = sorted(
            feature_types.values(), key=lambda x: abs_corr[x], reverse=True
        )
        # Select top features up to num_features
        features_week = best_features[:num_features]

        if len(features_week) < num_features:
            logger.warning(
                f"Warning: Only {len(features_week)} features found for {this_period}."
            )

        # Convert to Index to maintain compatibility with the rest of the code
        features_week = pd.Index(features_week)

        return features_week.to_list()

    def predict_on_sample(
        self,
        calibration_data: pd.DataFrame,
        prediction_data: pd.DataFrame,
        features: List[str],
        target: str,
    ):
        """
        Predict on a sample of data.

        Args:
            calibration_data (pd.DataFrame): DataFrame containing the calibration data.
            prediction_data (pd.DataFrame): DataFrame containing the prediction data.
            features (List[str]): List of feature names.
            target (str): Target variable name.

        Returns:
            y_pred (np.ndarray): Predicted values.
            y_pred_calibration (np.ndarray): Predicted values for calibration data.
            r2_calibration (float): R-squared value for calibration data.
            model (sklearn.linear_model): Fitted model.
        """

        calibration_data = calibration_data.dropna(
            subset=features + [target], how="any"
        ).copy()

        prediction_data = prediction_data.dropna(subset=features, how="any").copy()
        if len(prediction_data) == 0:
            return [np.nan], [np.nan], np.nan, None
        if len(calibration_data) < self.general_config["num_features"] * 2:
            return [np.nan], [np.nan], np.nan, None

        X_calibration = calibration_data[features]
        y_calibration = calibration_data[target]
        X_prediction = prediction_data[features]

        # Fit the model
        lr_type = self.model_config["lr_type"]
        if lr_type == "linear":
            model = LinearRegression()
        elif lr_type == "ridge":
            model = Ridge(alpha=self.model_config["alpha"])
        elif lr_type == "lasso":
            model = Lasso(alpha=self.model_config["alpha"])
        else:
            raise ValueError(f"Unknown model type: {lr_type}")

        model.fit(X_calibration, y_calibration)

        y_pred = model.predict(X_prediction)

        y_pred_calibration = model.predict(X_calibration)
        r2_calibration = r2_score(y_calibration, y_pred_calibration)
        return y_pred, y_pred_calibration, r2_calibration, model

    def predict_operational(self) -> pd.DataFrame:
        """
        Predict in operational mode.

        Args:
            data (pd.DataFrame): DataFrame containing the operational data.

        Returns:
            forecast (pd.DataFrame): DataFrame containing the forecasted values.
                columns: ['forecast_date', 'model_name', 'code','valid_from', 'valid_to', 'Q' (Optional: Q_05, Q_10, Q_50 ...)]
        """

        today = datetime.datetime.now()
        today_day = today.day
        today_month = today.month

        is_last_day_of_month = today.day == pd.Timestamp(today).days_in_month

        period_suffix = "_end" if is_last_day_of_month else str(today_day)
        period_name = f"{today_month}-{period_suffix}"

        logger.debug(f"Predicting operational data for period: {period_name}")
        logger.debug(f"Today's date: {today.strftime('%Y-%m-%d')}")
        logger.debug(
            f"Latest date of data: {self.data['date'].max().strftime('%Y-%m-%d')}"
        )

        self.__get_periods__()

        # Filter the data for the current period
        operational_data = self.data[self.data["period"] == period_name].copy()
        logger.debug(
            f"Operational data for period {period_name} contains {len(operational_data)} samples."
        )

        if len(operational_data) == 0:
            logger.warning(f"No operational data available for period {period_name}.")
            return pd.DataFrame()

        forecast = pd.DataFrame()
        if not self.general_config["offset"]:
            self.general_config["offset"] = 0

        shift = (
            self.general_config["offset"] - self.general_config["prediction_horizon"]
        )
        shift = max(shift, 0)  # Ensure shift is non-negative
        valid_from = today + datetime.timedelta(days=1) + datetime.timedelta(days=shift)
        valid_to = valid_from + datetime.timedelta(
            days=self.general_config["prediction_horizon"]
        )

        valid_from = valid_from.strftime("%Y-%m-%d")
        valid_to = valid_to.strftime("%Y-%m-%d")

        logger.debug(f"Valid from: {valid_from}")
        logger.debug(f"Valid to: {valid_to}")

        codes_to_predict = operational_data["code"].unique()

        # Convert today to pandas Timestamp for consistent comparison
        today_pd = pd.Timestamp(today.date())

        for code in codes_to_predict:
            # Filter the data for the current code
            code_data = operational_data[operational_data["code"] == code].copy()

            # Get the features for the current code
            features = self.get_highest_corr_features(code_data, "target")
            logger.debug(f"Features for {code}: {features}")

            logger.debug(
                f"filtering data for code {code} with {len(code_data)} samples."
            )
            logger.debug(
                f"Latest date of data for code {code}: {code_data['date'].max().strftime('%Y-%m-%d')}"
            )
            logger.debug(f"Today is {today_pd.strftime('%Y-%m-%d')}")
            logger.debug(
                f"Type of date and today: {type(code_data['date'].max())}, {type(today_pd)}"
            )
            calibration_data = code_data[code_data["date"] < today_pd].copy()
            prediction_data = code_data[code_data["date"] == today_pd].copy()

            logger.debug(f"Length Prediction data for {code}: {len(prediction_data)}")
            logger.debug(f"Length Calibration data for {code}: {len(calibration_data)}")

            calibration_target_mean = calibration_data["target"].mean()
            if pd.isna(calibration_target_mean):
                calibration_target_mean = 0

            # Predict on the sample
            predictions, _, r2_cali, model = self.predict_on_sample(
                calibration_data=calibration_data,
                prediction_data=prediction_data,
                features=features,
                target="target",
            )

            if predictions[0] is not np.nan:
                predictions = np.round(predictions, 2)  # Round to 2 decimal places
                predictions = np.maximum(
                    predictions, 0
                )  # Ensure non-negative predictions

                logger.debug(f"Predictions for {code}: {predictions[0]} m3/s")
                logger.debug(
                    f"Prediction for {code} relative to long-term mean: {predictions[0] / calibration_target_mean}"
                )
                logger.debug(f"R2 calibration for {code}: {r2_cali}")
                # construct the model and debug the equation
                model_equation = f"y = {model.intercept_:.3f} + " + " + ".join(
                    [f"{coef:.3f}*{feat}" for coef, feat in zip(model.coef_, features)]
                )
                logger.debug(f"Model equation for {code}: {model_equation}")

            # Create a DataFrame for the predictions
            pred_col = f"Q_{self.name}"
            pred_df = pd.DataFrame(
                {
                    "forecast_date": [today],
                    "code": [code],
                    "valid_from": [valid_from],
                    "valid_to": [valid_to],
                    pred_col: predictions,
                }
            )

            forecast = pd.concat([forecast, pred_df], ignore_index=True)

        return forecast

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate the model using the provided data.
        The highest correlated features are selected for each period and code on each loocv year seperately.

        Returns:
            hindcast (pd.DataFrame): DataFrame containing the hindcasted values.
                columns: ['date', 'model', 'code', 'Q_pred' (Optional: Q_05, Q_10, Q_50 ...)]
        """
        logger.info(f"Starting calibration and hindcasting for {self.name}")

        # Add day column if not present
        if "day" not in self.data.columns:
            self.data["day"] = self.data["date"].dt.day

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

        # get period names
        self.__get_periods__()

        # Get unique years and codes
        years = sorted(self.data["year"].unique())
        codes = self.data["code"].unique()

        filter_years = self.general_config.get("filter_years", None)
        if filter_years is not None:
            loocv_years = [year for year in years if year not in filter_years]
        else:
            loocv_years = years

        # Initialize results container
        all_predictions = []

        Q_pred_col = f"Q_{self.name}"

        logger.info(
            f"Performing Leave-One-Year-Out CV for {len(loocv_years)} years: {loocv_years}"
        )
        logger.info(
            "Unique periods in data: " + ", ".join(self.data["period"].unique())
        )

        # Process each period (day-month combination)
        failed_lr = []
        for period_name in tqdm(
            self.data["period"].unique(), desc="Processing periods"
        ):
            period_df = self.data[self.data["period"] == period_name]

            # Process each code separately
            for code in codes:
                code_period_df = period_df[period_df["code"] == code].copy()

                if len(code_period_df) < 2:  # Need at least 2 samples
                    failed_lr.append(
                        {
                            "code": code,
                            "period": period_name,
                            "reason": "Not enough data points",
                        }
                    )
                    continue

                # Get features for this code and period on the training data

                features = self.get_highest_corr_features(code_period_df, "target")

                # Perform Leave-One-Year-Out CV on non-test years
                for test_year in loocv_years:
                    # Training data: all years except current test year
                    train_data = code_period_df[
                        (code_period_df["year"].isin(loocv_years))
                        & (code_period_df["year"] != test_year)
                    ].copy()

                    # Test data: current test year
                    test_data = code_period_df[
                        code_period_df["year"] == test_year
                    ].copy()

                    if len(train_data) < self.general_config["num_features"] * 2:
                        failed_lr.append(
                            {
                                "code": code,
                                "period": period_name,
                                "reason": "Not enough data points > 2 * num_features",
                            }
                        )
                        continue

                    if not features:
                        failed_lr.append(
                            {
                                "code": code,
                                "period": period_name,
                                "reason": "No features selected",
                            }
                        )
                        continue

                    # Make predictions
                    predictions, _, r2, model = self.predict_on_sample(
                        calibration_data=train_data,
                        prediction_data=test_data,
                        features=features,
                        target="target",
                    )

                    # round predictions to 2 decimal places
                    predictions = np.round(predictions, 2)

                    Q_obs = test_data["target"].values

                    # Store predictions
                    for i, (idx, row) in enumerate(test_data.iterrows()):
                        if not np.isnan(predictions[i]):
                            all_predictions.append(
                                {
                                    "date": row["date"],
                                    "code": code,
                                    Q_pred_col: max(
                                        0, predictions[i]
                                    ),  # Ensure non-negative
                                    "period": period_name,
                                    "Q_obs": Q_obs[i],
                                }
                            )

        logger.debug(
            f"Failed linear regression for {len(failed_lr)} cases: {failed_lr}"
        )

        # Convert to DataFrame
        if all_predictions:
            hindcast_df = pd.DataFrame(all_predictions)
            # Remove period and cv_type columns as they're not in the expected output format
            hindcast_df = hindcast_df[["date", "code", Q_pred_col, "period", "Q_obs"]]
            # Sort by date and code
            hindcast_df = hindcast_df.sort_values(["date", "code"]).reset_index(
                drop=True
            )
            logger.info(
                f"Calibration complete. Generated {len(hindcast_df)} predictions for {len(hindcast_df['code'].unique())} codes"
            )
        else:
            logger.warning("No predictions generated during calibration")
            hindcast_df = pd.DataFrame(
                columns=["date", "code", Q_pred_col, "period", "Q_obs"]
            )

        return hindcast_df

    def tune_hyperparameters(
        self,
    ) -> Tuple[bool, str]:
        """
        Tune the hyperparameters of the model using the provided data.
        For linear regression, this optimizes alpha for Ridge/Lasso using time series CV.

        Returns:
            bool: True if hyperparameters were tuned successfully, False otherwise.
            str: Message indicating the result of the tuning process.
        """
        return (
            False,
            "Hyperparameter tuning is not applicable for Linear Regression models. Please use the model as is or implement a custom tuning method.",
        )

    def save_model(self) -> None:
        """
        Save the model to a file.

        Returns:
            None
        """
        logger.info(
            f"Linear Regression model is being fitted each time seperately - no model to save."
        )

    def load_model(self) -> None:
        """
        Load the model from a file.

        Returns:
            None
        """
        logger.info(
            f"Linear Regression model is being fitted each time seperately - no model to load."
        )
