"""
Base Meta-Learner class for monthly discharge forecasting.

This module provides the abstract base class for all meta-learning models,
defining the interface and common functionality for meta-learning approaches.
"""

import os
import pandas as pd
import numpy as np
import datetime
from abc import ABC, abstractmethod
from typing import Dict, Any, List
import logging

from monthly_forecasting.forecast_models.base_class import BaseForecastModel
from monthly_forecasting.scr.evaluation_utils import calculate_all_metrics

logger = logging.getLogger(__name__)


class BaseMetaLearner(BaseForecastModel):
    """
    Abstract base class for meta-learning models.

    Meta-learners combine predictions from multiple base models using
    intelligent weighting strategies or advanced meta-modeling techniques.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
        base_model_predictions: Dict[str, pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the meta-learner.

        Args:
            data: Historical data for training
            static_data: Static basin characteristics
            general_config: General configuration
            model_config: Meta-learning specific configuration
            feature_config: Feature engineering configuration
            path_config: Path configuration
            base_model_predictions: Dictionary of base model predictions
        """
        super().__init__(
            data, static_data, general_config, model_config, feature_config, path_config
        )

        # Meta-learning specific attributes
        self.base_model_predictions = base_model_predictions or {}
        self.meta_model = None
        self.performance_weights = {}
        self.feature_importance = {}

        # Configuration
        self.meta_learning_config = model_config.get("meta_learning", {})
        self.ensemble_method = self.meta_learning_config.get(
            "ensemble_method", "weighted_mean"
        )
        self.weighting_strategy = self.meta_learning_config.get(
            "weighting_strategy", "performance_based"
        )
        self.performance_metric = self.meta_learning_config.get(
            "performance_metric", "rmse"
        )

        # Validation
        self._validate_configuration()

        # Load base model predictions from paths if specified
        if self.path_config.get("path_to_base_models"):
            self._load_base_model_predictions()

        logger.info(
            f"Initialized {self.__class__.__name__} with {len(self.base_model_predictions)} base models"
        )

    def _validate_configuration(self) -> None:
        """Validate meta-learning configuration."""
        valid_methods = ["mean", "weighted_mean", "meta_model"]
        if self.ensemble_method not in valid_methods:
            raise ValueError(
                f"Invalid ensemble_method: {self.ensemble_method}. Must be one of {valid_methods}"
            )

        valid_strategies = [
            "performance_based",
            "uniform",
            "basin_specific",
            "temporal",
        ]
        if self.weighting_strategy not in valid_strategies:
            raise ValueError(
                f"Invalid weighting_strategy: {self.weighting_strategy}. Must be one of {valid_strategies}"
            )

        valid_metrics = ["rmse", "r2", "nse", "mae", "kge"]
        if self.performance_metric not in valid_metrics:
            raise ValueError(
                f"Invalid performance_metric: {self.performance_metric}. Must be one of {valid_metrics}"
            )

    def add_base_model_predictions(
        self, model_id: str, predictions: pd.DataFrame
    ) -> None:
        """
        Add predictions from a base model.

        Args:
            model_id: Unique identifier for the base model
            predictions: DataFrame with predictions (columns: date, code, Q_obs, Q_pred)
        """
        required_columns = ["date", "code", "Q_obs", "Q_pred"]
        missing_columns = [
            col for col in required_columns if col not in predictions.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in predictions: {missing_columns}"
            )

        self.base_model_predictions[model_id] = predictions.copy()
        logger.info(f"Added predictions for base model: {model_id}")

    def _load_base_model_predictions(self) -> None:
        """
        Load base model predictions from paths specified in path_config.

        Similar to SciRegressor's __load_lr_predictors__ method.
        Loads prediction files from path_config['path_to_base_models'] and extracts
        columns matching pattern 'Q_*' (excluding 'Q_obs').
        """
        path_list = self.path_config.get("path_to_base_models", [])

        if not path_list:
            logger.warning(
                "No base model paths specified in path_config['path_to_base_models']"
            )
            return

        logger.info(f"Loading base model predictions from {len(path_list)} paths")

        # Dictionary to store predictions per model
        loaded_predictions = {}

        for path in path_list:
            try:
                # Extract model name from path
                if path.endswith(".csv"):
                    #model name is the folder in which the CSV is located
                    model_name = os.path.basename(os.path.dirname(path))
                    logger.info(f"Loading predictions for model: {model_name}")
                    csv_path = path
                else:
                    # is the last bit of the path
                    if not path.endswith("/"):
                        path += "/"
                    if not os.path.exists(path):
                        logger.warning(
                            f"Path does not exist: {path}. Skipping {model_name}"
                        )
                        continue
                    # model name is the last part of the path
                    # (e.g., /path/to/models/model_name/)
                    model_name = os.path.basename(path.rstrip("/"))
                    logger.info(f"Loading predictions for model: {model_name}")
                    csv_path = os.path.join(path, "predictions.csv")

                # Load predictions
                if not os.path.exists(csv_path):
                    logger.warning(
                        f"Prediction file not found: {csv_path}. Skipping {model_name}"
                    )
                    continue

                df = pd.read_csv(csv_path)

                # Standardize column types
                df["date"] = pd.to_datetime(df["date"])
                df["code"] = df["code"].astype(int)

                # Find prediction columns (Q_* pattern, excluding Q_obs)
                pred_columns = [
                    col for col in df.columns if col.startswith("Q_") and col != "Q_obs"
                ]

                if not pred_columns:
                    logger.warning(
                        f"No prediction columns (Q_*) found in {model_name}. Skipping."
                    )
                    continue

                # Check if Q_obs column exists for observations
                if "Q_obs" in df.columns:
                    obs_col = "Q_obs"
                elif "Q" in df.columns:
                    obs_col = "Q"
                else:
                    logger.warning(
                        f"No observation column found in {model_name}. Using NaN for observations."
                    )
                    obs_col = None

                # Handle multiple prediction columns - create separate models for each
                for pred_col in pred_columns:
                    # Extract suffix from column name (e.g., Q_xgb -> xgb)
                    col_suffix = pred_col.replace("Q_", "")
                    
                    # Create model name: if suffix is same as model_name, use model_name
                    # Otherwise use model_name_suffix format
                    if col_suffix.lower() == model_name.lower():
                        sub_model_name = model_name
                    else:
                        sub_model_name = f"{model_name}_{col_suffix}"

                    # Create standardized prediction DataFrame for this column
                    pred_df = df[["date", "code"]].copy()
                    pred_df["Q_pred"] = df[pred_col]

                    if obs_col:
                        pred_df["Q_obs"] = df[obs_col]
                    else:
                        pred_df["Q_obs"] = np.nan

                    pred_df["model"] = sub_model_name

                    # Store predictions
                    loaded_predictions[sub_model_name] = pred_df

                    logger.info(
                        f"Loaded {len(pred_df)} predictions for model {sub_model_name} (column: {pred_col}) from {csv_path}"
                    )

            except Exception as e:
                logger.error(f"Error loading predictions from {path}: {e}")
                continue

        # Update base_model_predictions
        self.base_model_predictions.update(loaded_predictions)

        logger.info(
            f"Successfully loaded predictions for {len(loaded_predictions)} base models"
        )

    def get_base_model_ids(self) -> List[str]:
        """Get list of base model IDs."""
        return list(self.base_model_predictions.keys())

    def get_common_prediction_index(self) -> pd.MultiIndex:
        """
        Get common prediction index across all base models.

        Returns:
            MultiIndex with (date, code) combinations present in all base models
        """
        if not self.base_model_predictions:
            return pd.MultiIndex.from_tuples([], names=["date", "code"])

        # Get common index across all base models
        common_index = None
        for model_id, predictions in self.base_model_predictions.items():
            model_index = pd.MultiIndex.from_frame(predictions[["date", "code"]])

            if common_index is None:
                common_index = model_index
            else:
                common_index = common_index.intersection(model_index)

        return common_index

    def calculate_base_model_performance(
        self, model_id: str, metric: str = None, group_by: List[str] = None
    ) -> Dict[str, float]:
        """
        Calculate performance metrics for a base model.

        Args:
            model_id: Base model identifier
            metric: Specific metric to calculate (if None, calculate all)
            group_by: List of columns to group by (e.g., ['code', 'month'])

        Returns:
            Dictionary of performance metrics
        """
        if model_id not in self.base_model_predictions:
            raise ValueError(f"Base model {model_id} not found")

        predictions = self.base_model_predictions[model_id]

        if group_by:
            # Calculate metrics for each group
            def calculate_group_metrics(group_df):
                return pd.Series(
                    calculate_all_metrics(group_df["Q_obs"], group_df["Q_pred"])
                )

            grouped_metrics = predictions.groupby(group_by).apply(
                calculate_group_metrics, include_groups=False
            )
            return grouped_metrics.unstack().to_dict()
        else:
            # Calculate overall metrics
            metrics = calculate_all_metrics(predictions["Q_obs"], predictions["Q_pred"])
            return metrics if metric is None else {metric: metrics[metric]}

    def calculate_all_base_model_performance(
        self, metric: str = None, group_by: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance metrics for all base models.

        Args:
            metric: Specific metric to calculate (if None, calculate all)
            group_by: List of columns to group by

        Returns:
            Nested dictionary {model_id: {metric: value}}
        """
        all_performance = {}

        for model_id in self.get_base_model_ids():
            all_performance[model_id] = self.calculate_base_model_performance(
                model_id, metric, group_by
            )

        return all_performance

    @abstractmethod
    def compute_weights(self, **kwargs) -> Dict[str, float]:
        """
        Compute weights for ensemble combination.

        Returns:
            Dictionary mapping model IDs to weights
        """
        pass

    @abstractmethod
    def train_meta_model(self, **kwargs) -> None:
        """
        Train the meta-model (if using meta-modeling approach).
        """
        pass

    def create_ensemble_predictions(
        self, weights: Dict[str, float] = None, common_index_only: bool = True
    ) -> pd.DataFrame:
        """
        Create ensemble predictions from base models.

        Args:
            weights: Model weights (if None, compute using configured method)
            common_index_only: Whether to use only common prediction index

        Returns:
            DataFrame with ensemble predictions
        """
        if not self.base_model_predictions:
            raise ValueError("No base model predictions available")

        if weights is None:
            weights = self.compute_weights()

        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight == 0:
            raise ValueError("Total weight is zero")

        normalized_weights = {k: v / total_weight for k, v in weights.items()}

        # Get prediction index
        if common_index_only:
            pred_index = self.get_common_prediction_index()
        else:
            # Use union of all indices
            all_indices = []
            for predictions in self.base_model_predictions.values():
                all_indices.append(
                    pd.MultiIndex.from_frame(predictions[["date", "code"]])
                )
            pred_index = (
                all_indices[0].union_many(all_indices[1:])
                if len(all_indices) > 1
                else all_indices[0]
            )

        if pred_index.empty:
            raise ValueError("No common prediction index found")

        # Create ensemble predictions
        ensemble_predictions = []
        observations = []

        for date, code in pred_index:
            weighted_pred = 0.0
            obs_value = None
            weight_sum = 0.0

            for model_id, weight in normalized_weights.items():
                if model_id in self.base_model_predictions:
                    model_preds = self.base_model_predictions[model_id]

                    # Find matching prediction
                    mask = (model_preds["date"] == date) & (model_preds["code"] == code)
                    matching_preds = model_preds[mask]

                    if not matching_preds.empty:
                        pred_value = matching_preds["Q_pred"].iloc[0]
                        if not np.isnan(pred_value):
                            weighted_pred += weight * pred_value
                            weight_sum += weight

                        # Get observation (should be same across models)
                        if obs_value is None:
                            obs_value = matching_preds["Q_obs"].iloc[0]

            # Renormalize if some models didn't have predictions
            if weight_sum > 0:
                ensemble_predictions.append(weighted_pred / weight_sum)
            else:
                ensemble_predictions.append(np.nan)

            observations.append(obs_value)

        # Create result DataFrame
        result_df = pd.DataFrame(
            {
                "date": [idx[0] for idx in pred_index],
                "code": [idx[1] for idx in pred_index],
                "Q_obs": observations,
                "Q_pred": ensemble_predictions,
                "model": [self.name] * len(pred_index),
            }
        )

        return result_df

    def evaluate_ensemble_performance(
        self, ensemble_predictions: pd.DataFrame = None, metric: str = None
    ) -> Dict[str, float]:
        """
        Evaluate ensemble performance.

        Args:
            ensemble_predictions: Ensemble predictions DataFrame
            metric: Specific metric to calculate

        Returns:
            Dictionary of performance metrics
        """
        if ensemble_predictions is None:
            ensemble_predictions = self.create_ensemble_predictions()

        metrics = calculate_all_metrics(
            ensemble_predictions["Q_obs"], ensemble_predictions["Q_pred"]
        )

        return metrics if metric is None else {metric: metrics[metric]}

    def get_model_save_path(self) -> str:
        """Get the save path for the meta-learning model."""
        model_dir = self.path_config.get("model_home_path", "/tmp/meta_learning_models")
        return os.path.join(model_dir, f"{self.name}")

    def save_model(self) -> None:
        """Save meta-learning model to file using JSON/joblib pattern like SciRegressor."""
        import json
        import joblib

        save_path = self.get_model_save_path()

        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)

        # Save general configuration files (following SciRegressor pattern)
        model_config_path = os.path.join(save_path, "model_config.json")
        with open(model_config_path, "w") as f:
            json.dump(self.model_config, f, indent=4)

        feature_config_path = os.path.join(save_path, "feature_config.json")
        with open(feature_config_path, "w") as f:
            json.dump(self.feature_config, f, indent=4)

        experiment_config_path = os.path.join(save_path, "experiment_config.json")
        with open(experiment_config_path, "w") as f:
            json.dump(self.general_config, f, indent=4)

        # Save meta-learning specific configuration
        meta_config_path = os.path.join(save_path, "meta_learning_config.json")
        meta_config = {
            "class_name": self.__class__.__name__,
            "model_name": self.name,
            "ensemble_method": self.ensemble_method,
            "weighting_strategy": self.weighting_strategy,
            "performance_metric": self.performance_metric,
            "meta_learning_config": self.meta_learning_config,
            "save_timestamp": datetime.datetime.now().isoformat(),
            "version": "1.0.0",
        }
        with open(meta_config_path, "w") as f:
            json.dump(meta_config, f, indent=4)

        # Save performance weights as JSON
        weights_path = os.path.join(save_path, "performance_weights.json")
        with open(weights_path, "w") as f:
            json.dump(self.performance_weights, f, indent=4)

        # Save feature importance as CSV if available
        if self.feature_importance:
            feature_importance_path = os.path.join(save_path, "feature_importance.json")
            with open(feature_importance_path, "w") as f:
                json.dump(self.feature_importance, f, indent=4)

        # Save model-specific state using joblib (to be overridden by subclasses)
        model_specific_state = self._get_model_specific_state()
        if model_specific_state:
            for key, value in model_specific_state.items():
                model_file_path = os.path.join(save_path, f"{key}.joblib")
                joblib.dump(value, model_file_path)

        logger.info(f"Saved {self.__class__.__name__} model to {save_path}")

    def load_model(self) -> None:
        """Load meta-learning model from file using JSON/joblib pattern like SciRegressor."""
        import json
        import joblib

        save_path = self.get_model_save_path()

        if not os.path.exists(save_path):
            raise FileNotFoundError(f"Model directory not found: {save_path}")

        # Load meta-learning configuration
        meta_config_path = os.path.join(save_path, "meta_learning_config.json")
        if not os.path.exists(meta_config_path):
            raise FileNotFoundError(
                f"Meta-learning config file not found: {meta_config_path}"
            )

        with open(meta_config_path, "r") as f:
            meta_config = json.load(f)

        # Validate model compatibility
        if meta_config.get("class_name") != self.__class__.__name__:
            raise ValueError(
                f"Model class mismatch: expected {self.__class__.__name__}, got {meta_config.get('class_name')}"
            )

        # Load performance weights
        weights_path = os.path.join(save_path, "performance_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, "r") as f:
                self.performance_weights = json.load(f)
        else:
            self.performance_weights = {}

        # Load feature importance
        feature_importance_path = os.path.join(save_path, "feature_importance.json")
        if os.path.exists(feature_importance_path):
            with open(feature_importance_path, "r") as f:
                self.feature_importance = json.load(f)
        else:
            self.feature_importance = {}

        # Load model-specific state (to be overridden by subclasses)
        self._load_model_specific_state(save_path)

        logger.info(f"Loaded {self.__class__.__name__} model from {save_path}")
        logger.info(f"Model saved at: {meta_config.get('save_timestamp', 'unknown')}")

    def _get_model_specific_state(self) -> Dict[str, Any]:
        """
        Get model-specific state for saving.
        To be overridden by subclasses.

        Returns:
            Dictionary with model-specific state where keys are filenames (without extension)
            and values are objects to be saved with joblib
        """
        return {}

    def _load_model_specific_state(self, save_path: str) -> None:
        """
        Load model-specific state from files.
        To be overridden by subclasses.

        Args:
            save_path: Directory path where model files are stored
        """
        pass
