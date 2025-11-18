import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import datetime
from tqdm import tqdm as progress_bar

# Hyperparameter optimization
import optuna

import os
import pickle
import json

# Shared logging
import logging
from lt_forecasting import __version__
from lt_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger

from lt_forecasting.forecast_models.meta_learners.base_meta_learner import (
    BaseMetaLearner,
)

from lt_forecasting.scr import FeatureExtractor as FE
from lt_forecasting.scr import data_utils as du
from lt_forecasting.scr.metrics import (
    calculate_NRMSE,
    calculate_NMSE,
    calculate_NMAE,
    calculate_R2,
)

from lt_forecasting.scr.meta_utils import (
    calculate_weights_softmax,
    weights_hybrid,
    top_n_uniform_weights,
)
from lt_forecasting.scr.meta_utils import create_weighted_ensemble


class HistoricalMetaLearner(BaseMetaLearner):
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
        Initialize the HistoricalMetaLearner model with a configuration dictionary.

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

        self.num_samples_val = self.general_config.get("num_samples_val", 10)
        self.metric = self.general_config.get("metric", "nmse")

        available_metrics = ["nmse", "r2", "nmae", "nrmse"]

        if self.metric not in available_metrics:
            raise ValueError(
                f"Metric '{self.metric}' is not supported. Available metrics: {available_metrics}"
            )

        invert_metrics = ["nmse", "nmae", "nrmse"]

        if self.metric in invert_metrics:
            self.invert_metric = True
            self.is_error_metric = True
        else:
            self.invert_metric = False
            self.is_error_metric = False

        self.weighting_method = self.model_config.get("weighting_method", "hybrid")

        if self.weighting_method not in ["softmax", "hybrid", "top_n_uniform"]:
            raise ValueError(
                f"Weighting method '{self.weighting_method}' is not supported. Available methods: ['softmax', 'hybrid']"
            )

        self.temperature = self.general_config.get(
            "temperature", 1.0
        )  # Temperature for softmax scaling
        self.top_n_models = self.model_config.get(
            "top_n_models", 5
        )  # Number of top models to consider for weighting hybrid
        self.delta_performance = self.model_config.get(
            "delta_performance", 0.1
        )  # Threshold for "small difference" (10%)

        self.target = self.general_config.get("target_column", "Q_obs")

    def __preprocess_data__(self):
        """
        Preprocess the data for the meta learner.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Preprocessed data and list of model names
        """
        # 1. Create the target with feature extractor
        logger.info("Creating target variable with FeatureExtractor")

        extractor = FE.StreamflowFeatureExtractor(
            feature_configs={},  # empty dict to not create any features
            prediction_horizon=self.general_config["prediction_horizon"],
            offset=self.general_config.get(
                "offset", self.general_config["prediction_horizon"]
            ),
        )

        # Get target variable
        self.data = extractor.create_all_features(self.data)
        # rename 'target' to self.target
        self.data.rename(columns={"target": self.target}, inplace=True)

        # 2. Load the base predictors
        logger.info("Loading base predictors")
        base_predictors, model_names = self.__load_base_predictors__()
        self.base_models = model_names
        self.model_names = model_names

        # 3. Merge the base predictors with the data
        logger.info("Merging base predictors with target data")
        self.data = self.data.merge(base_predictors, on=["date", "code"], how="left")

        # 4. Create the periods columns
        logger.info("Creating period columns")

        self.data = du.get_periods(self.data)

        logger.info(f"Preprocessed data shape: {self.data.shape}")
        logger.info(f"Model names: {self.model_names}")

    def __get_weights__(self, performance: pd.DataFrame) -> pd.DataFrame:
        """
        Method for getting the weights [0,1] for each model based on the historical performance.
        Either by a weighted average or by a softmax function.

        Args:
            performance (pd.DataFrame): DataFrame with performance metrics per model, code, and period.

        Returns:
            pd.DataFrame: DataFrame with model names and their corresponding weights.
            Columns: ['period', 'code', 'model_xy', 'model_zx']
            Where 'model_xy' and 'model_zx' are the names of the models and the values are the weights.
        """

        def weighting_function(perf_values: np.ndarray) -> np.ndarray:
            """
            Function to calculate weights based on the specified method.
            """
            if self.weighting_method == "softmax":
                return calculate_weights_softmax(
                    perf_values,
                    temperature=self.temperature,
                    invert=self.invert_metric,
                )
            elif self.weighting_method == "hybrid":
                return weights_hybrid(
                    perf_values,
                    top_n=self.top_n_models,
                    delta=self.delta_performance,
                    temperature=self.temperature,
                    is_error_metric=self.is_error_metric,
                )
            elif self.weighting_method == "top_n_uniform":
                return top_n_uniform_weights(
                    perf_values,
                    top_n=self.top_n_models,
                    invert=self.invert_metric,
                )
            else:
                raise ValueError(
                    f"Unsupported weighting method: {self.weighting_method}"
                )

        logger.info("Calculating weights based on historical performance")

        # Ensure top n is smaller or equal to the number of base models
        number_base_models = len(self.base_models)
        self.top_n_models = min(self.top_n_models, number_base_models)

        # Get model columns (excluding 'code' and 'period')
        model_columns = [
            col for col in performance.columns if col not in ["code", "period"]
        ]

        if len(model_columns) == 0:
            raise ValueError("No model columns found in performance DataFrame")

        # Initialize weights DataFrame
        weights_list = []

        # Calculate weights for each code-period combination
        for _, row in performance.iterrows():
            code = row["code"]
            period = row["period"]

            # Get performance values for this combination
            performance_values = np.array([row[model] for model in model_columns])

            # Calculate weights using softmax
            weights = weighting_function(performance_values)

            # check if weights sum up to 1
            sum_weights = np.nansum(weights)
            if np.abs(sum_weights - 1) > 1e-5:
                raise ValueError(
                    f"Weights for code {code}, period {period} do not sum to 1: {weights}, sum: {sum_weights}"
                )

            # Create weights row
            weights_row = {"code": code, "period": period}
            if len(weights) != len(model_columns):
                raise ValueError(
                    f"Number of weights {len(weights)} does not match number of model columns {len(model_columns)}"
                )
            for i, model in enumerate(model_columns):
                weights_row[model] = weights[i]

            weights_list.append(weights_row)

        weights_df = pd.DataFrame(weights_list)

        logger.info(
            f"Weights calculated for {len(weights_df)} code-period combinations"
        )
        logger.debug(f"Weight columns: {model_columns}")

        return weights_df

    def __create_ensemble__(
        self, data: pd.DataFrame, base_models: List[str], weights: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create an ensemble of base models using the specified weights.

        Args:
            data (pd.DataFrame): DataFrame containing base model predictions.
            base_models (List[str]): List of base model names.
            weights (pd.DataFrame): DataFrame containing model weights.

        Returns:
            pd.DataFrame: DataFrame containing the ensemble predictions.
        """

        logger.info("Creating weighted ensemble predictions")

        # Create ensemble using utility function
        ensemble_data = create_weighted_ensemble(
            predictions=data,
            weights=weights,
            model_columns=base_models,
            group_columns=["code", "period"],
        )

        logger.info(f"Ensemble created with {len(ensemble_data)} predictions")

        return ensemble_data

    def __loocv__(self, loocv_years: List[int]) -> pd.DataFrame:
        """
        Perform Leave-One-Out Cross-Validation for the meta learner.

        Args:
            loocv_years (List[int]): List of years to use for LOOCV.

        Returns:
            pd.DataFrame: DataFrame containing ensemble predictions for all validation years.
        """
        logger.info(f"Starting LOOCV for years: {loocv_years}")

        all_predictions = []

        for year in progress_bar(loocv_years, desc="Processing years", leave=True):
            logger.info(f"Processing LOOCV for year {year}")

            # Split data into train and validation sets
            train_data = self.data[self.data["date"].dt.year != year].copy()
            val_data = self.data[self.data["date"].dt.year == year].copy()

            if len(train_data) == 0:
                logger.warning(f"No training data for year {year}")
                continue

            if len(val_data) == 0:
                logger.warning(f"No validation data for year {year}")
                continue

            # Calculate historical performance on training data
            historical_performance = self.__calculate_historical_performance__(
                train_data, self.model_names
            )

            if len(historical_performance) == 0:
                logger.warning(f"No historical performance data for year {year}")
                continue

            # Get weights based on historical performance
            weights = self.__get_weights__(historical_performance)

            # Create ensemble predictions for validation data
            ensemble_predictions = self.__create_ensemble__(
                val_data, self.base_models, weights
            )

            # rename 'ensemble' to Q_self.name
            ensemble_predictions.rename(
                columns={"ensemble": f"Q_{self.name}"}, inplace=True
            )

            # Add year column for tracking
            ensemble_predictions["validation_year"] = year

            all_predictions.append(ensemble_predictions)

            logger.info(
                f"Completed LOOCV for year {year}: {len(ensemble_predictions)} predictions"
            )

        if len(all_predictions) == 0:
            logger.error("No predictions generated from LOOCV")
            return pd.DataFrame()

        # Combine all predictions
        final_predictions = pd.concat(all_predictions, ignore_index=True)

        logger.info(f"LOOCV completed: {len(final_predictions)} total predictions")

        return final_predictions

    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict the operational forecast using the meta learner.

        Args:
            today (datetime.datetime, optional): The date for which to make the prediction. Defaults to None.

        Returns:
            pd.DataFrame: DataFrame containing the operational predictions.
        """
        import os

        logger.info("Starting operational prediction")

        if today is None:
            today = datetime.datetime.now()

        # 1. Preprocess the data
        data, model_names = self.__preprocess_data__()

        # 2. Check if historical performance weights exist
        weights_path = os.path.join(
            self.path_config.get("model_home_path", self.name),
            f"{self.name}_weights.parquet",
        )

        if os.path.exists(weights_path):
            # 3. Load the pre-calculated weights
            logger.info(f"Loading pre-calculated weights from {weights_path}")
            weights = pd.read_parquet(weights_path)
        else:
            # Calculate historical performance on all available data
            logger.info("Calculating historical performance for operational prediction")
            historical_performance = self.__calculate_historical_performance__(
                data, model_names
            )

            if len(historical_performance) == 0:
                logger.error(
                    "No historical performance data available for operational prediction"
                )
                return pd.DataFrame()

            # 4. Get weights based on historical performance
            weights = self.__get_weights__(historical_performance)

            # Save weights for future use
            logger.info(f"Saving weights to {weights_path}")
            weights.to_parquet(weights_path, index=False)

        # Filter data for today's predictions (or most recent data)
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

        # 5. Create ensemble predictions for today
        operational_predictions = self.__create_ensemble__(
            today_data, model_names, weights
        )

        # Rename 'ensemble' to Q_self.name
        operational_predictions.rename(
            columns={"ensemble": f"Q_{self.name}"}, inplace=True
        )

        logger.info(
            f"Operational prediction completed: {len(operational_predictions)} predictions"
        )

        return operational_predictions

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate the model and perform hindcasting.

        Returns:
            pd.DataFrame: DataFrame containing the hindcast predictions.
        """
        logger.info("Starting model calibration and hindcasting")

        # 1. Preprocess the data
        self.__preprocess_data__()

        self.__filter_forecast_days__()

        self.data.dropna(subset=[self.target], inplace=True)

        # Get all available years for LOOCV
        available_years = sorted(self.data["date"].dt.year.unique())
        logger.info(f"Available years for LOOCV: {available_years}")

        # 2. Run LOOCV for the meta learner
        hindcast_predictions = self.__loocv__(available_years)

        if len(hindcast_predictions) == 0:
            logger.error("No hindcast predictions generated")
            return pd.DataFrame()

        # 3. Calculate historical performance on full dataset
        logger.info("Calculating historical performance on full dataset")
        historical_performance = self.__calculate_historical_performance__(
            self.data, self.base_models
        )

        if len(historical_performance) == 0:
            logger.error("No historical performance data calculated")
            return hindcast_predictions

        self.historical_performance = historical_performance

        # 4. Get weights based on historical performance
        weights = self.__get_weights__(historical_performance)
        self.weights = weights

        # 5. Save the model
        self.save_model()

        # 6. Return the hindcast predictions
        logger.info(
            f"Calibration completed: {len(hindcast_predictions)} hindcast predictions"
        )

        return hindcast_predictions[
            ["date", "code", self.target, f"Q_{self.name}"]
        ].copy()

    def objective(
        self,
        trial: optuna.Trial,
        val_df: pd.DataFrame,
        historical_performance: pd.DataFrame,
    ) -> float:
        """
        Objective function for hyperparameter optimization.

        Args:
            trial (optuna.Trial): The trial object containing hyperparameters.

        Returns:
            float: The objective value to minimize (e.g., validation loss).
        """
        # Set hyperparameters from the trial
        self.temperature = trial.suggest_float("temperature", 0.05, 2.0, log=True)
        self.top_n_models = trial.suggest_int("top_n_models", 1, 10)
        self.delta_performance = trial.suggest_float("delta_performance", 0.0, 1.0)
        self.weighting_method = trial.suggest_categorical(
            "weighting_method", ["softmax", "hybrid", "top_n_uniform"]
        )

        weights = self.__get_weights__(historical_performance)

        ensemble_predictions = self.__create_ensemble__(
            val_df, self.base_models, weights
        )

        # rename 'ensemble' to Q_self.name
        ensemble_predictions.rename(
            columns={"ensemble": f"Q_{self.name}"}, inplace=True
        )

        # merge back with validation data to get Q_obs
        val_df = val_df.merge(
            ensemble_predictions[["date", "code", f"Q_{self.name}"]],
            on=["date", "code"],
            how="left",
        )

        # calcualte the normalized mean squared error (NMSE) as the objective
        nrmse = calculate_NRMSE(
            observed=val_df[self.target],
            simulated=val_df[f"Q_{self.name}"],
        )

        r2 = calculate_R2(
            observed=val_df[self.target],
            simulated=val_df[f"Q_{self.name}"],
        )

        nmae = calculate_NMAE(
            observed=val_df[self.target],
            simulated=val_df[f"Q_{self.name}"],
        )

        nmse = calculate_NMSE(
            observed=val_df[self.target],
            simulated=val_df[f"Q_{self.name}"],
        )

        logger.info(
            f"Trial {trial.number}: NRMSE = {nrmse:.4f}, R2 = {r2:.4f}, NMAE = {nmae:.4f}, NMSE = {nmse:.4f}"
        )

        if self.metric == "nmse":
            return nmse
        elif self.metric == "nmae":
            return nmae
        elif self.metric == "r2":
            return -r2  # Minimize negative R2
        elif self.metric == "nrmse":
            return nrmse
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune the hyperparameters of the meta learner.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success and a message.
        """
        logger.info("Starting hyperparameter tuning")

        # Apply the same preprocessing as other methods
        self.__preprocess_data__()

        if "year" not in self.data.columns:
            self.data["year"] = self.data["date"].dt.year

        # Add day column if not present
        if "day" not in self.data.columns:
            self.data["day"] = self.data["date"].dt.day

        # Get configuration parameters
        self.hparam_tuning_years = self.general_config.get("num_hparam_tuning_years", 3)
        num_hparam_tuning_years = self.hparam_tuning_years

        self.__filter_forecast_days__()

        all_years = sorted(self.data["year"].unique())
        train_years = all_years[:-num_hparam_tuning_years]
        val_years = all_years[-num_hparam_tuning_years:]

        df_train = self.data[self.data["year"].isin(train_years)].copy()
        df_val = self.data[self.data["year"].isin(val_years)].copy()

        logger.info(f"Training years: {train_years}, Validation years: {val_years}")

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

        # Calculate historical performance on training data
        historical_performance = self.__calculate_historical_performance__(
            df_train, self.base_models
        )

        if historical_performance.empty:
            logger.error(
                "No historical performance data available for hyperparameter tuning"
            )
            return (
                False,
                "No historical performance data available for hyperparameter tuning",
            )

        # Create an Optuna study
        study = optuna.create_study(
            direction="minimize",
        )

        n_trials = self.general_config.get("n_trials", 50)
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")

        study.optimize(
            lambda trial: self.objective(trial, df_val, historical_performance),
            n_trials=n_trials,
            show_progress_bar=True,
        )

        logger.info("Hyperparameter tuning completed")

        # Get the best hyperparameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")

        # Update model configuration with best hyperparameters
        self.temperature = best_params.get("temperature", self.temperature)
        self.top_n_models = best_params.get("top_n_models", self.top_n_models)
        self.delta_performance = best_params.get(
            "delta_performance", self.delta_performance
        )
        self.weighting_method = best_params.get(
            "weighting_method", self.weighting_method
        )

        logger.info(
            f"Updated model configuration: temperature={self.temperature}, "
            f"top_n_models={self.top_n_models}, delta_performance={self.delta_performance}, "
            f"weighting_method={self.weighting_method}"
        )

        self.save_model()

        return True, "Hyperparameter tuning completed successfully"

    def save_model(self) -> None:
        """
        Save the model to the specified path.
        """

        logger.info(f"Saving {self.name} model")

        # Create save path
        save_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")
        os.makedirs(save_path, exist_ok=True)

        # Save model metadata
        metadata = {
            "name": self.name,
            "metric": self.metric,
            "num_samples_val": self.num_samples_val,
            "invert_metric": self.invert_metric,
            "weighting_method": self.weighting_method,
            "temperature": self.temperature,
            "top_n_models": self.top_n_models,
            "delta_performance": self.delta_performance,
            "model_type": "historical_meta_learner",
            "version": __version__,
        }

        # save the historical performance as parquet file if it exists
        if hasattr(self, "historical_performance"):
            performance_path = os.path.join(
                save_path, f"{self.name}_performance.parquet"
            )
            self.historical_performance.to_parquet(performance_path, index=False)
            metadata["historical_performance_path"] = performance_path

        # save the weights as parquet file if they exist
        if hasattr(self, "weights"):
            weights_path = os.path.join(save_path, f"{self.name}_weights.parquet")
            self.weights.to_parquet(weights_path, index=False)
            metadata["weights_path"] = weights_path

        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save configuration
        config_data = {
            "general_config": self.general_config,
            "model_config": self.model_config,
            "feature_config": self.feature_config,
            "path_config": self.path_config,
        }

        config_path = os.path.join(save_path, "config.pkl")
        with open(config_path, "wb") as f:
            pickle.dump(config_data, f)

        logger.info(f"Model saved successfully to {save_path}")

    def load_model(self) -> None:
        """
        Load the model from the specified path.
        """
        import os
        import pickle
        import json

        logger.info(f"Loading {self.name} model")
        load_path = os.path.join(self.path_config["model_home_path"], f"{self.name}")

        if not os.path.exists(load_path):
            logger.error(f"Model path {load_path} does not exist. Cannot load models.")
            return

        # Load metadata
        metadata_path = os.path.join(load_path, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Update model parameters
            self.metric = metadata.get("metric", self.metric)
            self.num_samples_val = metadata.get("num_samples_val", self.num_samples_val)
            self.invert_metric = metadata.get("invert_metric", self.invert_metric)

            logger.info(f"Loaded metadata: {metadata}")
        else:
            logger.warning(f"No metadata file found at {metadata_path}")

        # Load configuration
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

        logger.info(f"Model loaded successfully from {load_path}")
