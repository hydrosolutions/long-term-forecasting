"""
Ensemble builder module for monthly discharge forecasting.

This module creates ensemble predictions at family and global levels by combining
predictions from multiple models using various ensemble methods.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_simple_ensemble(
    prediction_data: List[pd.DataFrame],
    ensemble_method: str = "mean",
    weights: Optional[List[float]] = None,
) -> pd.DataFrame:
    """
    Create a simple ensemble from multiple prediction DataFrames.

    Parameters:
    -----------
    prediction_data : List[pd.DataFrame]
        List of prediction DataFrames to ensemble
    ensemble_method : str
        Method for ensemble creation ('mean', 'median', 'weighted_mean')
    weights : Optional[List[float]]
        Weights for weighted_mean method

    Returns:
    --------
    pd.DataFrame
        DataFrame with ensemble predictions
    """
    if not prediction_data:
        logger.warning("No prediction data provided for ensemble creation")
        return pd.DataFrame()

    # Validate inputs
    if ensemble_method == "weighted_mean" and weights is None:
        logger.warning("Weights not provided for weighted_mean, using equal weights")
        weights = [1.0 / len(prediction_data)] * len(prediction_data)

    if weights is not None and len(weights) != len(prediction_data):
        logger.warning(
            f"Length mismatch: {len(weights)} weights for {len(prediction_data)} models"
        )
        weights = None

    # Get common structure from first DataFrame
    base_df = prediction_data[0].copy()

    # Extract prediction columns from all DataFrames
    prediction_values = []
    for df in prediction_data:
        if "Q_pred" in df.columns:
            pred_values = df.set_index(["date", "code"])["Q_pred"]
            prediction_values.append(pred_values)
        else:
            logger.warning("Q_pred column not found in prediction data")
            continue

    if not prediction_values:
        logger.error("No valid prediction columns found")
        return pd.DataFrame()

    # Combine predictions into a single DataFrame using outer join
    combined_predictions = pd.concat(prediction_values, axis=1, join="outer")

    # Calculate ensemble
    if ensemble_method == "mean":
        ensemble_pred = combined_predictions.mean(axis=1)
    elif ensemble_method == "median":
        ensemble_pred = combined_predictions.median(axis=1)
    elif ensemble_method == "weighted_mean":
        ensemble_pred = (combined_predictions * weights).sum(axis=1)
    else:
        logger.warning(f"Unknown ensemble method: {ensemble_method}, using mean")
        ensemble_pred = combined_predictions.mean(axis=1)

    # Create ensemble DataFrame from the combined index
    ensemble_df = combined_predictions.reset_index()
    ensemble_df = ensemble_df[["date", "code"]].copy()

    # Add observations from the base DataFrame
    base_obs = base_df.set_index(["date", "code"])["Q_obs"]
    ensemble_df = ensemble_df.set_index(["date", "code"])
    ensemble_df["Q_obs"] = base_obs
    ensemble_df["Q_pred"] = ensemble_pred

    # Add ensemble metadata
    ensemble_df["n_models"] = (~combined_predictions.isna()).sum(axis=1)

    # Reset index
    ensemble_df = ensemble_df.reset_index()

    return ensemble_df


def create_family_ensemble(
    loaded_predictions: Dict[str, pd.DataFrame],
    family_name: str,
    ensemble_method: str = "mean",
    weights: Optional[List[float]] = None,
) -> Optional[pd.DataFrame]:
    """
    Create ensemble predictions for a specific model family.

    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of loaded prediction DataFrames
    family_name : str
        Name of the family to create ensemble for
    ensemble_method : str
        Method for ensemble creation
    weights : Optional[List[float]]
        Weights for weighted ensemble

    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame with family ensemble predictions
    """
    # Filter predictions for this family
    family_predictions = []
    family_model_ids = []

    for model_id, df in loaded_predictions.items():
        if "family" in df.columns and df["family"].iloc[0] == family_name:
            family_predictions.append(df)
            family_model_ids.append(model_id)

    if not family_predictions:
        logger.warning(f"No models found for family: {family_name}")
        return None

    logger.info(
        f"Creating ensemble for {family_name} with {len(family_predictions)} models: {family_model_ids}"
    )

    # Create ensemble
    ensemble_df = create_simple_ensemble(family_predictions, ensemble_method, weights)

    if ensemble_df.empty:
        logger.warning(f"Failed to create ensemble for family: {family_name}")
        return None

    # Add ensemble metadata
    ensemble_df["model_id"] = f"{family_name}_Ensemble"
    ensemble_df["family"] = family_name
    ensemble_df["model_name"] = f"{family_name}_Ensemble"
    ensemble_df["is_ensemble"] = True
    ensemble_df["ensemble_type"] = "family"
    ensemble_df["source_models"] = [family_model_ids] * len(ensemble_df)

    logger.info(f"Created {family_name} ensemble with {len(ensemble_df)} records")
    return ensemble_df


def create_global_ensemble(
    loaded_predictions: Dict[str, pd.DataFrame],
    family_ensembles: Dict[str, pd.DataFrame],
    ensemble_method: str = "mean",
    use_families: bool = True,
    weights: Optional[List[float]] = None,
) -> Optional[pd.DataFrame]:
    """
    Create global ensemble predictions using either family ensembles or individual models.

    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of loaded prediction DataFrames
    family_ensembles : Dict[str, pd.DataFrame]
        Dictionary of family ensemble DataFrames
    ensemble_method : str
        Method for ensemble creation
    use_families : bool
        Whether to use family ensembles or individual models
    weights : Optional[List[float]]
        Weights for weighted ensemble

    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame with global ensemble predictions
    """
    if use_families:
        # Use family ensembles
        ensemble_data = list(family_ensembles.values())
        source_models = list(family_ensembles.keys())
        ensemble_type = "global_family"
    else:
        # Use individual models
        ensemble_data = list(loaded_predictions.values())
        source_models = list(loaded_predictions.keys())
        ensemble_type = "global_individual"

    if not ensemble_data:
        logger.warning("No data available for global ensemble creation")
        return None

    logger.info(
        f"Creating global ensemble with {len(ensemble_data)} components: {source_models}"
    )

    # Create ensemble
    ensemble_df = create_simple_ensemble(ensemble_data, ensemble_method, weights)

    if ensemble_df.empty:
        logger.warning("Failed to create global ensemble")
        return None

    # Add ensemble metadata
    ensemble_df["model_id"] = f"Global_Ensemble_{ensemble_type}"
    ensemble_df["family"] = "Global"
    ensemble_df["model_name"] = f"Global_Ensemble_{ensemble_type}"
    ensemble_df["is_ensemble"] = True
    ensemble_df["ensemble_type"] = ensemble_type
    ensemble_df["source_models"] = [source_models] * len(ensemble_df)

    logger.info(f"Created global ensemble with {len(ensemble_df)} records")
    return ensemble_df


def create_all_ensembles(
    loaded_predictions: Dict[str, pd.DataFrame],
    ensemble_method: str = "mean",
    create_global: bool = True,
    weights: Optional[Dict[str, List[float]]] = None,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """
    Create all ensemble predictions (family and global).

    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of loaded prediction DataFrames
    ensemble_method : str
        Method for ensemble creation
    create_global : bool
        Whether to create global ensembles
    weights : Optional[Dict[str, List[float]]]
        Weights for ensemble creation by family

    Returns:
    --------
    Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]
        Tuple of (family_ensembles, global_ensembles)
    """
    # Get unique families
    families = set()
    for df in loaded_predictions.values():
        if "family" in df.columns:
            families.add(df["family"].iloc[0])

    families = sorted(list(families))
    logger.info(f"Found {len(families)} families: {families}")

    # Create family ensembles
    family_ensembles = {}
    for family in families:
        family_weights = weights.get(family) if weights else None
        family_ensemble = create_family_ensemble(
            loaded_predictions, family, ensemble_method, family_weights
        )

        if family_ensemble is not None:
            family_ensembles[family] = family_ensemble

    logger.info(f"Created {len(family_ensembles)} family ensembles")

    # Create global ensembles
    global_ensembles = {}
    if create_global:
        # Global ensemble using family ensembles
        global_family_ensemble = create_global_ensemble(
            loaded_predictions, family_ensembles, ensemble_method, use_families=True
        )

        if global_family_ensemble is not None:
            global_ensembles["Global_Family"] = global_family_ensemble

        # Global ensemble using individual models
        global_individual_ensemble = create_global_ensemble(
            loaded_predictions, family_ensembles, ensemble_method, use_families=False
        )

        if global_individual_ensemble is not None:
            global_ensembles["Global_Individual"] = global_individual_ensemble

    logger.info(f"Created {len(global_ensembles)} global ensembles")

    return family_ensembles, global_ensembles


def save_ensemble_predictions(
    family_ensembles: Dict[str, pd.DataFrame],
    global_ensembles: Dict[str, pd.DataFrame],
    output_dir: str = "../monthly_forecasting_results/evaluation",
) -> Dict[str, str]:
    """
    Save ensemble predictions to CSV files.

    Parameters:
    -----------
    family_ensembles : Dict[str, pd.DataFrame]
        Dictionary of family ensemble DataFrames
    global_ensembles : Dict[str, pd.DataFrame]
        Dictionary of global ensemble DataFrames
    output_dir : str
        Directory to save ensemble predictions

    Returns:
    --------
    Dict[str, str]
        Dictionary mapping ensemble names to file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved_files = {}

    # Save family ensembles
    for family_name, ensemble_df in family_ensembles.items():
        filename = f"{family_name}_ensemble_predictions.csv"
        filepath = output_path / filename

        # Select relevant columns for saving
        save_columns = ["date", "code", "Q_obs", "Q_pred", "n_models"]
        save_df = ensemble_df[save_columns].copy()

        save_df.to_csv(filepath, index=False)
        saved_files[family_name] = str(filepath)
        logger.info(f"Saved {family_name} ensemble predictions to {filepath}")

    # Save global ensembles
    for ensemble_name, ensemble_df in global_ensembles.items():
        filename = f"{ensemble_name}_predictions.csv"
        filepath = output_path / filename

        # Select relevant columns for saving
        save_columns = ["date", "code", "Q_obs", "Q_pred", "n_models"]
        save_df = ensemble_df[save_columns].copy()

        save_df.to_csv(filepath, index=False)
        saved_files[ensemble_name] = str(filepath)
        logger.info(f"Saved {ensemble_name} predictions to {filepath}")

    # Save ensemble metadata
    metadata = {
        "creation_date": datetime.now().isoformat(),
        "family_ensembles": list(family_ensembles.keys()),
        "global_ensembles": list(global_ensembles.keys()),
        "ensemble_method": "mean",  # TODO: Pass this as parameter
        "total_ensembles": len(family_ensembles) + len(global_ensembles),
    }

    metadata_file = output_path / "ensemble_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved ensemble metadata to {metadata_file}")

    return saved_files


def combine_predictions_and_ensembles(
    loaded_predictions: Dict[str, pd.DataFrame],
    family_ensembles: Dict[str, pd.DataFrame],
    global_ensembles: Dict[str, pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """
    Combine individual predictions with ensemble predictions.

    Parameters:
    -----------
    loaded_predictions : Dict[str, pd.DataFrame]
        Dictionary of individual model predictions
    family_ensembles : Dict[str, pd.DataFrame]
        Dictionary of family ensemble predictions
    global_ensembles : Dict[str, pd.DataFrame]
        Dictionary of global ensemble predictions

    Returns:
    --------
    Dict[str, pd.DataFrame]
        Combined dictionary of all predictions
    """
    combined_predictions = {}

    # Add individual predictions
    combined_predictions.update(loaded_predictions)

    # Add family ensembles
    for family_name, ensemble_df in family_ensembles.items():
        ensemble_id = f"{family_name}_Ensemble"
        combined_predictions[ensemble_id] = ensemble_df

    # Add global ensembles
    for ensemble_name, ensemble_df in global_ensembles.items():
        combined_predictions[ensemble_name] = ensemble_df

    logger.info(
        f"Combined {len(loaded_predictions)} individual models, "
        f"{len(family_ensembles)} family ensembles, "
        f"and {len(global_ensembles)} global ensembles"
    )

    return combined_predictions


def create_ensemble_summary(
    family_ensembles: Dict[str, pd.DataFrame], global_ensembles: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Create a summary of ensemble predictions.

    Parameters:
    -----------
    family_ensembles : Dict[str, pd.DataFrame]
        Dictionary of family ensemble DataFrames
    global_ensembles : Dict[str, pd.DataFrame]
        Dictionary of global ensemble DataFrames

    Returns:
    --------
    pd.DataFrame
        Summary DataFrame with ensemble information
    """
    summary_data = []

    # Add family ensemble summaries
    for family_name, ensemble_df in family_ensembles.items():
        summary_data.append(
            {
                "ensemble_name": family_name,
                "ensemble_type": "family",
                "n_records": len(ensemble_df),
                "n_codes": ensemble_df["code"].nunique(),
                "date_range_start": ensemble_df["date"].min(),
                "date_range_end": ensemble_df["date"].max(),
                "mean_n_models": ensemble_df["n_models"].mean(),
                "min_n_models": ensemble_df["n_models"].min(),
                "max_n_models": ensemble_df["n_models"].max(),
            }
        )

    # Add global ensemble summaries
    for ensemble_name, ensemble_df in global_ensembles.items():
        summary_data.append(
            {
                "ensemble_name": ensemble_name,
                "ensemble_type": "global",
                "n_records": len(ensemble_df),
                "n_codes": ensemble_df["code"].nunique(),
                "date_range_start": ensemble_df["date"].min(),
                "date_range_end": ensemble_df["date"].max(),
                "mean_n_models": ensemble_df["n_models"].mean(),
                "min_n_models": ensemble_df["n_models"].min(),
                "max_n_models": ensemble_df["n_models"].max(),
            }
        )

    return pd.DataFrame(summary_data)


if __name__ == "__main__":
    # Example usage
    try:
        from .prediction_loader import load_all_predictions
    except ImportError:
        from prediction_loader import load_all_predictions

    # Load predictions
    predictions, validation = load_all_predictions()

    # Create all ensembles
    family_ensembles, global_ensembles = create_all_ensembles(predictions)

    # Save ensemble predictions
    saved_files = save_ensemble_predictions(family_ensembles, global_ensembles)

    # Create summary
    summary = create_ensemble_summary(family_ensembles, global_ensembles)

    print("\n=== ENSEMBLE CREATION SUMMARY ===")
    print(f"Created {len(family_ensembles)} family ensembles")
    print(f"Created {len(global_ensembles)} global ensembles")
    print(f"Saved {len(saved_files)} ensemble files")

    print("\n=== ENSEMBLE DETAILS ===")
    print(summary.to_string(index=False))
