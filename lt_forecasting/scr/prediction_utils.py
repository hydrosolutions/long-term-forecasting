"""
Prediction Utilities Module

Helper functions for loading base model predictions for different model types.
Provides a centralized interface to handle prediction loading and conversion logic.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from lt_forecasting.scr.prediction_loader import (
    apply_area_conversion,
    load_predictions_from_filesystem,
)

logger = logging.getLogger(__name__)


def load_base_predictions_for_model(
    model_type: str,
    path_config: Dict[str, Any],
    static_data: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Load base model predictions based on model type and configuration.

    Handles different model types with their specific prediction loading requirements:
    - sciregressor: Uses path_to_lr_predictors and applies area conversion
    - historical_meta_learner: Uses path_to_base_predictors, no conversion
    - UncertaintyMixtureModel: Uses path_to_base_predictors, no conversion

    Args:
        model_type: Type of model ("sciregressor", "historical_meta_learner", "UncertaintyMixtureModel")
        path_config: Path configuration dictionary containing prediction file paths
        static_data: DataFrame with basin metadata (required for area conversion)

    Returns:
        Tuple of (predictions_df, model_column_names) or (None, None) if paths not configured
        - predictions_df: DataFrame with columns [date, code, Q_{model1}, Q_{model2}, ...]
        - model_column_names: List of prediction column names
            * For SciRegressor: WITHOUT Q_ prefix (e.g., ['model1', 'model2'])
            * For BaseMetaLearner: WITH Q_ prefix (e.g., ['Q_model1', 'Q_model2'])

    Example:
        >>> preds, cols = load_base_predictions_for_model(
        ...     model_type="sciregressor",
        ...     path_config={"path_to_lr_predictors": ["/data/model1/", "/data/model2/"]},
        ...     static_data=static_df
        ... )
        >>> print(cols)
        ['model1', 'model2']  # Q_ prefix stripped for SciRegressor
    """
    # Determine path key and conversion flag based on model type
    if model_type == "sciregressor":
        path_key = "path_to_lr_predictors"
        apply_conversion = True
    elif model_type in ["historical_meta_learner", "UncertaintyMixtureModel"]:
        path_key = "path_to_base_predictors"
        apply_conversion = False
    else:
        logger.info(
            f"Model type '{model_type}' does not use base predictions. Returning None."
        )
        return None, None

    # Check if path is configured
    prediction_paths = path_config.get(path_key)
    if not prediction_paths:
        logger.info(
            f"No prediction paths configured for '{path_key}'. "
            f"Model will not use external base predictions."
        )
        return None, None

    # Load predictions from filesystem
    logger.info(f"Loading base predictions for {model_type} from {path_key}")
    try:
        predictions, model_names = load_predictions_from_filesystem(
            paths=prediction_paths,
            join_type="inner",
        )
        logger.info(
            f"Successfully loaded {len(model_names)} base models: {model_names}"
        )
    except Exception as e:
        logger.warning(
            f"Failed to load predictions from {path_key}: {e}. "
            f"Model will proceed without external base predictions."
        )
        return None, None

    # Apply area conversion if needed (for sciregressor)
    # Note: Conversion uses Q_-prefixed column names from predictions DataFrame
    if apply_conversion:
        logger.info("Applying area conversion to predictions")
        try:
            predictions = apply_area_conversion(
                predictions=predictions,
                static_data=static_data,
                pred_cols=model_names,  # model_names still has Q_ prefix at this point
            )
            logger.info("Area conversion applied successfully")
        except Exception as e:
            logger.error(f"Failed to apply area conversion: {e}")
            raise

    # For sciregressor, strip Q_ prefix from model names (SciRegressor adds it internally)
    # Note: This happens AFTER conversion since DataFrame columns have Q_ prefix
    if model_type == "sciregressor":
        model_names = [name.replace("Q_", "") for name in model_names]
        logger.debug(f"Stripped Q_ prefix for SciRegressor: {model_names}")

    return predictions, model_names
