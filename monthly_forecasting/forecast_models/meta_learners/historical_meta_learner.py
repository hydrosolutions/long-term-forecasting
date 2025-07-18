import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import datetime

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger

from monthly_forecasting.forecast_models.meta_learners.base_meta_learner import (
    BaseMetaLearner,
)

from monthly_forecasting.scr import FeatureExtractor as FE


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

    def __preprocess_data__(self):
        """
        Preprocess the data for the meta learner.

        Returns:
            Tuple[pd.DataFrame, List[str]]: Preprocessed data and list of model names
        """
        # 1. Create the target with feature extractor
        logger.info("Creating target variable with FeatureExtractor")
        fe = FE(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            feature_config=self.feature_config,
        )

        # Get target variable
        target_data = fe.get_target()

        # 2. Load the base predictors
        logger.info("Loading base predictors")
        base_predictors, model_names = self.__load_base_predictors__()

        # 3. Merge the base predictors with the data
        logger.info("Merging base predictors with target data")
        merged_data = target_data.merge(
            base_predictors, on=["date", "code"], how="left"
        )

        # 4. Create the periods columns
        logger.info("Creating period columns")
        from monthly_forecasting.scr.meta_utils import get_periods

        preprocessed_data = get_periods(merged_data, period_type="monthly")

        logger.info(f"Preprocessed data shape: {preprocessed_data.shape}")
        logger.info(f"Model names: {model_names}")

        return preprocessed_data, model_names

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
        from monthly_forecasting.scr.meta_utils import calculate_weights_softmax

        logger.info("Calculating weights based on historical performance")

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
            weights = calculate_weights_softmax(
                performance_values, temperature=1.0, invert=self.invert_metric
            )

            # Create weights row
            weights_row = {"code": code, "period": period}
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
        from monthly_forecasting.scr.meta_utils import create_weighted_ensemble

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

        # Preprocess data first
        data, model_names = self.__preprocess_data__()

        all_predictions = []

        for year in loocv_years:
            logger.info(f"Processing LOOCV for year {year}")

            # Split data into train and validation sets
            train_data = data[data["date"].dt.year != year].copy()
            val_data = data[data["date"].dt.year == year].copy()

            if len(train_data) == 0:
                logger.warning(f"No training data for year {year}")
                continue

            if len(val_data) == 0:
                logger.warning(f"No validation data for year {year}")
                continue

            # Calculate historical performance on training data
            historical_performance = self.__calculate_historical_performance__(
                train_data, model_names
            )

            if len(historical_performance) == 0:
                logger.warning(f"No historical performance data for year {year}")
                continue

            # Get weights based on historical performance
            weights = self.__get_weights__(historical_performance)

            # Create ensemble predictions for validation data
            ensemble_predictions = self.__create_ensemble__(
                val_data, model_names, weights
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
            self.path_config.get("model_home_path", ""), f"{self.name}_weights.parquet"
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
        import os

        logger.info("Starting model calibration and hindcasting")

        # 1. Preprocess the data
        data, model_names = self.__preprocess_data__()

        # Get all available years for LOOCV
        available_years = sorted(data["date"].dt.year.unique())
        logger.info(f"Available years for LOOCV: {available_years}")

        # 2. Run LOOCV for the meta learner
        hindcast_predictions = self.__loocv__(available_years)

        if len(hindcast_predictions) == 0:
            logger.error("No hindcast predictions generated")
            return pd.DataFrame()

        # 3. Calculate historical performance on full dataset
        logger.info("Calculating historical performance on full dataset")
        historical_performance = self.__calculate_historical_performance__(
            data, model_names
        )

        if len(historical_performance) == 0:
            logger.error("No historical performance data calculated")
            return hindcast_predictions

        # 4. Get weights based on historical performance
        weights = self.__get_weights__(historical_performance)

        # 5. Save the weights as parquet file
        weights_path = os.path.join(
            self.path_config.get("model_home_path", ""), f"{self.name}_weights.parquet"
        )
        logger.info(f"Saving weights to {weights_path}")

        # Ensure directory exists
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        weights.to_parquet(weights_path, index=False)

        # Also save historical performance for reference
        performance_path = os.path.join(
            self.path_config.get("model_home_path", ""),
            f"{self.name}_performance.parquet",
        )
        logger.info(f"Saving historical performance to {performance_path}")
        historical_performance.to_parquet(performance_path, index=False)

        # 6. Return the hindcast predictions
        logger.info(
            f"Calibration completed: {len(hindcast_predictions)} hindcast predictions"
        )

        return hindcast_predictions

    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune the hyperparameters of the meta learner.

        Returns:
            Tuple[bool, str]: A tuple containing a boolean indicating success and a message.
        """
        logger.info("Starting hyperparameter tuning")

        try:
            # Preprocess data
            data, model_names = self.__preprocess_data__()

            # Get available years for validation
            available_years = sorted(data["date"].dt.year.unique())

            if len(available_years) < 3:
                return (
                    False,
                    "Insufficient data for hyperparameter tuning (need at least 3 years)",
                )

            # Parameters to tune
            num_samples_vals = [5, 10, 15, 20]
            metrics_to_test = ["nmse", "r2", "nrmse"]

            best_score = float("-inf")
            best_params = {}

            # Split years for validation
            validation_years = available_years[-2:]  # Use last 2 years for validation
            training_years = available_years[:-2]  # Use remaining years for training

            for num_samples in num_samples_vals:
                for metric in metrics_to_test:
                    logger.info(
                        f"Testing num_samples_val={num_samples}, metric={metric}"
                    )

                    # Update model configuration
                    original_num_samples = self.num_samples_val
                    original_metric = self.metric

                    self.num_samples_val = num_samples
                    self.metric = metric

                    # Update invert_metric based on new metric
                    invert_metrics = ["nmse", "nmae", "nrmse"]
                    self.invert_metric = metric in invert_metrics

                    try:
                        # Train on training years
                        train_data = data[data["date"].dt.year.isin(training_years)]
                        val_data = data[data["date"].dt.year.isin(validation_years)]

                        # Calculate performance on training data
                        historical_performance = (
                            self.__calculate_historical_performance__(
                                train_data, model_names
                            )
                        )

                        if len(historical_performance) == 0:
                            logger.warning(
                                f"No performance data for params: num_samples={num_samples}, metric={metric}"
                            )
                            continue

                        # Get weights
                        weights = self.__get_weights__(historical_performance)

                        # Create ensemble predictions on validation data
                        val_predictions = self.__create_ensemble__(
                            val_data, model_names, weights
                        )

                        # Calculate validation score (using NSE as evaluation metric)
                        from monthly_forecasting.scr.metrics import calculate_NSE

                        if (
                            "Q_obs" in val_predictions.columns
                            and "ensemble" in val_predictions.columns
                        ):
                            observed = val_predictions["Q_obs"].values
                            predicted = val_predictions["ensemble"].values

                            # Remove NaN values
                            valid_mask = ~(np.isnan(observed) | np.isnan(predicted))

                            if np.sum(valid_mask) >= 10:
                                score = calculate_NSE(
                                    observed[valid_mask], predicted[valid_mask]
                                )

                                if not np.isnan(score) and score > best_score:
                                    best_score = score
                                    best_params = {
                                        "num_samples_val": num_samples,
                                        "metric": metric,
                                    }
                                    logger.info(
                                        f"New best score: {score:.4f} with params: {best_params}"
                                    )

                    except Exception as e:
                        logger.warning(
                            f"Error testing params num_samples={num_samples}, metric={metric}: {str(e)}"
                        )

                    # Restore original parameters
                    self.num_samples_val = original_num_samples
                    self.metric = original_metric
                    self.invert_metric = original_metric in invert_metrics

            if best_params:
                # Update model with best parameters
                self.num_samples_val = best_params["num_samples_val"]
                self.metric = best_params["metric"]
                self.invert_metric = best_params["metric"] in invert_metrics

                logger.info(
                    f"Hyperparameter tuning completed. Best params: {best_params}, Score: {best_score:.4f}"
                )
                return True, f"Tuning successful. Best NSE score: {best_score:.4f}"
            else:
                logger.warning("No valid hyperparameter combinations found")
                return False, "No valid hyperparameter combinations found"

        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            return False, f"Tuning failed: {str(e)}"

    def save_model(self) -> None:
        """
        Save the model to the specified path.
        """
        import os
        import pickle
        import json

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
            "model_type": "HistoricalMetaLearner",
            "version": "1.0",
        }

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
