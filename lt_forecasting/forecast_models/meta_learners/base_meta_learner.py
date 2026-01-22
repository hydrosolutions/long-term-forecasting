import os
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
import matplotlib.pyplot as plt
import warnings

# Ensure the logs directory exists
import datetime

# Shared logging
import logging
from lt_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


from lt_forecasting.forecast_models.base_class import BaseForecastModel
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import json
from tqdm import tqdm

from lt_forecasting.scr import FeatureExtractor as FE


class BaseMetaLearner(BaseForecastModel):
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
        Initialize the BaseMetaLearner model with a configuration dictionary.

        Args:
            general_config (Dict[str, Any]): General configuration for the model.
            model_config (Dict[str, Any]): Model-specific configuration.
            path_config (Dict[str, Any]): Path configuration for saving/loading data.
            base_predictors (Optional[pd.DataFrame]): Pre-loaded base model predictions.
                If provided, will be used instead of loading from filesystem.
            base_model_names (Optional[List[str]]): List of base model column names.
                Required if base_predictors is provided.
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
        self.metric = self.model_config.get("metric", "nmse")

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

        self.target = self.general_config.get("target_column", "Q_obs")

        # Store external predictions if provided
        self._external_base_predictors = base_predictors
        self._external_base_model_names = base_model_names

    def __load_base_predictors__(
        self, use_mean_pred: bool = False
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Load base predictors from the specified path.

        .. deprecated::
            Loading predictions internally is deprecated. Use prediction_loader module
            and pass base_predictors parameter to constructor instead.

        Returns:
            Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame of base predictors and a list of their names (column names).
        """
        # Check if external predictions were provided
        if self._external_base_predictors is not None and self._external_base_model_names is not None:
            logger.info("Using externally provided base predictors")
            # Need to merge with self.data to maintain left join behavior
            df_to_merge_on = self.data[["date", "code"]].copy()
            for col_name in self._external_base_model_names:
                # Handle Q_ prefix - external loader uses Q_ prefix, but BaseMetaLearner expects without prefix
                if col_name.startswith("Q_"):
                    model_name = col_name.replace("Q_", "")
                else:
                    model_name = col_name

                if col_name in self._external_base_predictors.columns:
                    sub_df = self._external_base_predictors[["date", "code", col_name]].copy()
                    sub_df.rename(columns={col_name: model_name}, inplace=True)
                    df_to_merge_on = df_to_merge_on.merge(
                        sub_df, on=["date", "code"], how="left"
                    )

            # Extract model names (without Q_ prefix)
            model_names = [name.replace("Q_", "") if name.startswith("Q_") else name
                          for name in self._external_base_model_names]
            return df_to_merge_on, model_names

        # Fall back to file-based loading with deprecation warning
        warnings.warn(
            "Loading predictions internally is deprecated. "
            "Use lt_forecasting.scr.prediction_loader module and pass "
            "base_predictors parameter to constructor instead.",
            DeprecationWarning,
            stacklevel=2
        )

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

            # Check for and handle duplicate (date, code) pairs
            n_duplicates = pred_df[["date", "code"]].duplicated().sum()
            if n_duplicates > 0:
                logger.warning(
                    f"Found {n_duplicates} duplicate (date, code) pairs in {model_name}. "
                    f"Averaging prediction values for duplicates."
                )
                # Get all Q_ columns for averaging
                q_cols = [
                    col for col in pred_df.columns if "Q_" in col and col != "Q_obs"
                ]

                # Create aggregation dictionary
                agg_dict = {col: "mean" for col in q_cols}

                # Average predictions for duplicate (date, code) combinations
                pred_df = pred_df.groupby(["date", "code"], as_index=False).agg(
                    agg_dict
                )

                logger.debug(
                    f"After deduplication: {len(pred_df)} unique (date, code) pairs"
                )

            pred_cols = [
                col for col in pred_df.columns if "Q_" in col and col != "Q_obs"
            ]

            if len(pred_cols) == 1:
                include_ensemble = True
            else:
                include_ensemble = use_mean_pred

            for col in pred_cols:
                sub_model = col.replace("Q_", "")
                member_name = sub_model

                if sub_model == model_name:
                    if not include_ensemble:
                        continue
                    base_models_cols.append(member_name)

                else:
                    sub_sub_model = sub_model.split("_")[-1]
                    member_name = f"{model_name}_{sub_sub_model}"
                    base_models_cols.append(member_name)

                sub_df = pred_df[["date", "code", col]].copy()
                sub_df.rename(columns={col: member_name}, inplace=True)

                logger.debug(
                    f"Merging base predictor '{member_name}' into main DataFrame"
                )

                # check if member_name already exists in df_to_merge_on
                if member_name in df_to_merge_on.columns:
                    logger.warning(
                        f"Column '{member_name}' already exists in the DataFrame. It will be overwritten."
                    )

                df_to_merge_on = df_to_merge_on.merge(
                    sub_df, on=["date", "code"], how="left"
                )

        if not base_models_cols:
            raise ValueError("No base predictors found in the specified paths.")

        logger.info(
            f"Loaded {len(base_models_cols)} base predictors: {base_models_cols}"
        )

        return df_to_merge_on, base_models_cols

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

    def __calculate_historical_performance__(
        self, base_predictors: pd.DataFrame, model_names: List[str]
    ) -> pd.DataFrame:
        """
        Calculate historical performance metrics for the given base predictors.
        """
        from lt_forecasting.scr.metrics import get_metric_function

        logger.info(f"Calculating historical performance using metric: {self.metric}")
        metric_func = get_metric_function(self.metric)
        target_col = self.target

        if target_col not in base_predictors.columns:
            raise ValueError(
                f"Target column '{target_col}' not found in base_predictors: {list(base_predictors.columns)}"
            )

        # Ensure only needed columns
        cols_needed = ["code", "period", target_col] + [
            m for m in model_names if m in base_predictors.columns
        ]
        df = base_predictors[cols_needed].copy()

        def calc_metric(group, model):
            obs = group[target_col].values
            pred = group[model].values
            mask = ~(np.isnan(obs) | np.isnan(pred))
            if mask.sum() < 2:
                return np.nan
            return metric_func(obs[mask], pred[mask])

        # --- Step 1: local performance ---
        local_perf = (
            df.groupby(["code", "period"])
            .apply(
                lambda g: pd.Series(
                    {m: calc_metric(g, m) for m in model_names if m in g.columns}
                )
            )
            .reset_index()
        )

        # --- Step 2: determine which local groups have enough samples ---
        group_sizes = (
            df.dropna(subset=[target_col])
            .groupby(["code", "period"])
            .size()
            .reset_index(name="n_samples")
        )
        local_perf = local_perf.merge(group_sizes, on=["code", "period"], how="left")

        # --- Step 3: global performance per period ---
        global_perf = (
            df.groupby("period")
            .apply(
                lambda g: pd.Series(
                    {m: calc_metric(g, m) for m in model_names if m in g.columns}
                )
            )
            .reset_index()
        )
        global_sizes = (
            df.dropna(subset=[target_col])
            .groupby("period")
            .size()
            .reset_index(name="n_samples_global")
        )
        global_perf = global_perf.merge(global_sizes, on="period", how="left")

        # --- Step 4: fill NaNs or insufficient samples with global ---
        merged = local_perf.merge(global_perf, on="period", suffixes=("", "_global"))
        for m in model_names:
            if m not in merged.columns:
                continue
            cond_replace = (merged["n_samples"] < self.num_samples_val) | merged[
                m
            ].isna()
            enough_global = merged["n_samples_global"] >= self.num_samples_val
            merged.loc[cond_replace & enough_global, m] = merged.loc[
                cond_replace & enough_global, f"{m}_global"
            ]

        # --- Step 5: final cleanup ---
        perf_df = merged[["code", "period"] + model_names]
        logger.info(
            f"Historical performance calculated for {len(perf_df)} code-period combinations"
        )

        logger.debug(f"Performance DataFrame columns: {perf_df.columns.tolist()}")
        logger.debug(f"Performance DataFrame head:\n{perf_df.head()}")

        return perf_df
