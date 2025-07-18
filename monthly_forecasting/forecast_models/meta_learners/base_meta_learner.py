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
from monthly_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


from monthly_forecasting.forecast_models.base_class import BaseForecastModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
from tqdm import tqdm

from monthly_forecasting.scr import FeatureExtractor as FE


class BaseMetaLearner(BaseForecastModel):
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
        Initialize the BaseMetaLearner model with a configuration dictionary.

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

        self.num_samples_val = self.model_config.get("num_samples_val", 10)
        self.metric = self.model_config.get("metric", "mse")

        available_metrics = ["nmse", "r2", "nmae", "nrmse"]

        if self.metric not in available_metrics:
            raise ValueError(
                f"Metric '{self.metric}' is not supported. Available metrics: {available_metrics}"
            )

        invert_metrics = ["nmse", "nmae", "nrmse"]

        if self.metric in invert_metrics:
            self.invert_metric = True
        else:
            self.invert_metric = False

    def __load_base_predictors__(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load base predictors from the specified path.

        Returns:
            Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame of base predictors and a list of their names (column names).
        """
        path_to_base_predictors = self.path_config["path_to_base_predictors"]

        base_models_cols = []
        df_to_merge_on = self.data[["date", "code"]].copy()

        for path in path_to_base_predictors:
            # model name is the folder in which the csv file is stored
            model_name = os.path.basename(os.path.dirname(path))

            logger.info(f"Loading base predictors from {path} for model {model_name}")

            pred_df = pd.read_csv(path)
            pred_df["date"] = pd.to_datetime(pred_df["date"])
            pred_df["code"] = pred_df["code"].astype(int)

            pred_cols = [
                col for col in pred_df.columns if "Q_" in col and col != "Q_obs"
            ]

            for col in pred_cols:
                sub_model = col.split("_")[1]
                member_name = sub_model

                if sub_model == model_name:
                    base_models_cols.append(member_name)

                else:
                    member_name = f"{model_name}_{sub_model}"
                    base_models_cols.append(member_name)

                sub_df = pred_df[["date", "code", col]].copy()
                sub_df.rename(columns={col: member_name}, inplace=True)

                df_to_merge_on = df_to_merge_on.merge(
                    sub_df, on=["date", "code"], how="left"
                )

        if not base_models_cols:
            raise ValueError("No base predictors found in the specified paths.")

        logger.info(
            f"Loaded {len(base_models_cols)} base predictors: {base_models_cols}"
        )

        return df_to_merge_on, base_models_cols

    def __calculate_historical_performance__(
        self, base_predictors: pd.DataFrame, model_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate the historical performance of the base models for each code and period.
        And the global performance for each model for each period.
        If there are less than 'num_samples_val' we use the global performance.

        Args:
            base_predictors (pd.DataFrame): The DataFrame containing base predictors.
            model_names (List[str]): The list of model names to evaluate.

        Returns:
            pd.DataFrame: A DataFrame containing the historical performance metrics.
                        Columns: ['code', 'period', 'model_xy', 'model_zy', ...]
                        where 'model_xy' and 'model_zy' are the model names and the values in the columns
                        are the performance metrics (e.g., MSE, R2).
        """
        from monthly_forecasting.scr.metrics import get_metric_function

        logger.info(f"Calculating historical performance using metric: {self.metric}")

        # Get the metric function
        metric_func = get_metric_function(self.metric)

        # Initialize results list
        results = []

        # Get target column name (assuming it's 'Q_obs')
        target_col = "Q_obs"
        if target_col not in base_predictors.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in base_predictors"
            )

        # Calculate performance for each combination of code and period
        for code in base_predictors["code"].unique():
            for period in base_predictors["period"].unique():
                # Filter data for this code and period
                mask = (base_predictors["code"] == code) & (
                    base_predictors["period"] == period
                )
                subset_data = base_predictors[mask]

                if len(subset_data) < self.num_samples_val:
                    logger.debug(
                        f"Insufficient data for code {code}, period {period}: {len(subset_data)} < {self.num_samples_val}"
                    )
                    # Use global performance for this period if insufficient local data
                    global_mask = base_predictors["period"] == period
                    subset_data = base_predictors[global_mask]

                    if len(subset_data) < self.num_samples_val:
                        logger.debug(
                            f"Insufficient global data for period {period}: {len(subset_data)} < {self.num_samples_val}"
                        )
                        continue

                # Calculate performance for each model
                performance_row = {"code": code, "period": period}

                for model_name in model_names:
                    if model_name not in subset_data.columns:
                        logger.debug(f"Model {model_name} not found in data")
                        performance_row[model_name] = np.nan
                        continue

                    # Get observed and predicted values
                    observed = subset_data[target_col].values
                    predicted = subset_data[model_name].values

                    # Remove NaN pairs
                    valid_mask = ~(np.isnan(observed) | np.isnan(predicted))

                    if np.sum(valid_mask) < 2:
                        logger.debug(
                            f"Insufficient valid data for model {model_name}, code {code}, period {period}"
                        )
                        performance_row[model_name] = np.nan
                        continue

                    observed_clean = observed[valid_mask]
                    predicted_clean = predicted[valid_mask]

                    # Calculate performance metric
                    try:
                        performance_value = metric_func(observed_clean, predicted_clean)
                        performance_row[model_name] = performance_value
                    except Exception as e:
                        logger.debug(
                            f"Error calculating {self.metric} for model {model_name}: {str(e)}"
                        )
                        performance_row[model_name] = np.nan

                results.append(performance_row)

        # Convert results to DataFrame
        performance_df = pd.DataFrame(results)

        # Fill remaining NaN values with global performance
        for model_name in model_names:
            if model_name not in performance_df.columns:
                continue

            # Calculate global performance by period for models with NaN values
            for period in performance_df["period"].unique():
                period_mask = performance_df["period"] == period
                nan_mask = performance_df[model_name].isna()

                if np.any(period_mask & nan_mask):
                    # Calculate global performance for this period
                    global_data = base_predictors[base_predictors["period"] == period]

                    if (
                        len(global_data) >= self.num_samples_val
                        and model_name in global_data.columns
                    ):
                        observed = global_data[target_col].values
                        predicted = global_data[model_name].values

                        valid_mask = ~(np.isnan(observed) | np.isnan(predicted))

                        if np.sum(valid_mask) >= 2:
                            try:
                                global_performance = metric_func(
                                    observed[valid_mask], predicted[valid_mask]
                                )

                                # Fill NaN values with global performance
                                performance_df.loc[
                                    period_mask & nan_mask, model_name
                                ] = global_performance

                            except Exception as e:
                                logger.debug(
                                    f"Error calculating global {self.metric} for model {model_name}: {str(e)}"
                                )

        logger.info(
            f"Historical performance calculated for {len(performance_df)} code-period combinations"
        )
        return performance_df
