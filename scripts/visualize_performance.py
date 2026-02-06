#!/usr/bin/env python3
"""
Visualize Performance Script

Generates standardized performance visualization plots organized by lead time (horizon).
Imports data loading and metrics functions from examine_operational_fc.py.

Output structure:
    {output_dir}/{region}/lead_time_{horizon}/
        ├── r2_distribution.png
        ├── accuracy_distribution.png
        ├── efficiency_distribution.png
        ├── bias_distribution.png
        └── uncertainty_calibration.png  (MC_ALD only)
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Import from examine_operational_fc.py (in same directory)
from examine_operational_fc import (
    load_predictions,
    load_observations,
    calculate_target,
    create_ensemble,
    aggregate,
    create_horizon_metrics_dataframe,
    create_quantile_exceedance_dataframe,
    kgz_path_config,
    taj_path_config,
    day_of_forecast,
    output_dir as default_output_dir,
)

# Import style configuration
from dev_tools.visualization.style_config import set_global_plot_style

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Default models to plot (exclude submodel variants)
DEFAULT_MODELS = [
    "LR_Base",
    # "LR_SM",
    "Ensemble",
    "MC_ALD",
    "Climatology",  # Baseline model
    # "SM_GBT",
    # "SM_GBT_Norm",
    # "SM_GBT_LR",
]

# Suffixes to exclude from model list
EXCLUDE_SUFFIXES = ["_xgb", "_lgbm", "_catboost", "_rf", "_loc"]

# Quantile patterns to exclude (raw quantile columns sometimes appear as "models")
QUANTILE_PATTERN = ["Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]

# Abbreviated month names for x-axis
MONTH_ABBREV = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "May",
    6: "Jun",
    7: "Jul",
    8: "Aug",
    9: "Sep",
    10: "Oct",
    11: "Nov",
    12: "Dec",
}

# Agricultural period months (April - September)
AGRICULTURAL_MONTHS = [4, 5, 6, 7, 8, 9]


def add_climatology_baseline(
    aggregated_df: pd.DataFrame,
    monthly_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add climatology baseline as a pseudo-model to the aggregated DataFrame.

    Uses yearly Leave-One-Out Cross-Validation (LOOCV) for each (code, year, month):
    For each prediction, the climatology mean and std are calculated from all OTHER
    years for that specific (code, month) combination. This ensures no data leakage
    and provides a fair baseline that respects temporal validation principles.

    Quantile predictions are computed assuming a normal distribution.

    Args:
        aggregated_df: DataFrame with model predictions and observations
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly)

    Returns:
        DataFrame with "Climatology" model added (including quantile predictions)
    """
    from scipy import stats

    # Get unique combinations from aggregated_df to create climatology predictions
    # Use only one model to get the structure (avoid duplicates)
    sample_model = aggregated_df["model"].iloc[0]
    template_df = aggregated_df[aggregated_df["model"] == sample_model].copy()

    if template_df.empty:
        logger.warning("No template data for climatology computation")
        return aggregated_df

    # Extract year from valid_from for LOOCV
    if "valid_from" in template_df.columns:
        template_df["pred_year"] = template_df["valid_from"].dt.year
    else:
        logger.warning("valid_from column not found - cannot compute LOOCV climatology")
        return aggregated_df

    # Initialize columns for climatology statistics
    template_df["clim_mean"] = np.nan
    template_df["clim_std"] = np.nan
    template_df["clim_count"] = 0

    # Get all unique codes
    all_codes = template_df["code"].unique()
    all_years = monthly_obs["year"].unique()

    logger.info(
        f"Computing LOOCV climatology for {len(all_codes)} basins, "
        f"{len(all_years)} years ({min(all_years)}-{max(all_years)})"
    )

    # Pre-compute climatology statistics for each (code, month, leave_out_year)
    # This is more efficient than computing per-row
    loocv_stats = {}

    for code in all_codes:
        code_obs = monthly_obs[monthly_obs["code"] == code]
        if code_obs.empty:
            continue

        for month in range(1, 13):
            month_obs = code_obs[code_obs["month"] == month]
            if month_obs.empty:
                continue

            for leave_out_year in all_years:
                # LOOCV: use all years EXCEPT leave_out_year
                train_data = month_obs[month_obs["year"] != leave_out_year][
                    "Q_obs_monthly"
                ]

                if len(train_data) < 2:
                    # Need at least 2 years for meaningful std
                    continue

                mean_val = train_data.mean()
                std_val = train_data.std()

                # Handle edge cases
                if pd.isna(std_val) or std_val == 0:
                    std_val = mean_val * 0.3  # Fallback: 30% CV

                # Ensure std is not too small
                std_val = max(std_val, mean_val * 0.01)

                loocv_stats[(code, month, leave_out_year)] = {
                    "mean": mean_val,
                    "std": std_val,
                    "count": len(train_data),
                }

    # Apply LOOCV statistics to each row
    for idx, row in template_df.iterrows():
        key = (row["code"], row["target_month"], row["pred_year"])
        if key in loocv_stats:
            template_df.loc[idx, "clim_mean"] = loocv_stats[key]["mean"]
            template_df.loc[idx, "clim_std"] = loocv_stats[key]["std"]
            template_df.loc[idx, "clim_count"] = loocv_stats[key]["count"]

    # Remove rows where LOOCV could not be computed
    valid_mask = ~template_df["clim_mean"].isna()
    clim_df = template_df[valid_mask].copy()

    if clim_df.empty:
        logger.warning("No valid LOOCV climatology could be computed")
        return aggregated_df

    # Set up climatology as a "model"
    clim_df["model"] = "Climatology"
    clim_df["Q_pred"] = clim_df["clim_mean"]

    # Compute quantile predictions assuming normal distribution
    # Quantiles: Q5, Q10, Q25, Q50, Q75, Q90, Q95
    quantile_levels = {
        "Q5": 0.05,
        "Q10": 0.10,
        "Q25": 0.25,
        "Q50": 0.50,
        "Q75": 0.75,
        "Q90": 0.90,
        "Q95": 0.95,
    }

    for q_name, q_level in quantile_levels.items():
        # Normal distribution quantile: mean + z_score * std
        z_score = stats.norm.ppf(q_level)
        clim_df[q_name] = clim_df["clim_mean"] + z_score * clim_df["clim_std"]
        # Ensure non-negative discharge predictions
        clim_df[q_name] = clim_df[q_name].clip(lower=0)

    # Drop the extra climatology columns
    cols_to_drop = ["clim_mean", "clim_std", "clim_count", "pred_year"]
    clim_df = clim_df.drop(columns=[c for c in cols_to_drop if c in clim_df.columns])

    # Append to original DataFrame
    combined_df = pd.concat([aggregated_df, clim_df], ignore_index=True)

    logger.info(
        f"Added {len(clim_df)} climatology predictions using yearly LOOCV "
        f"(per code/month, leaving out prediction year)"
    )

    return combined_df


def get_common_codes(df: pd.DataFrame, models: list[str]) -> set:
    """
    Find codes that are present in ALL specified models.

    Args:
        df: DataFrame with 'code' and 'model' columns
        models: List of model names to check

    Returns:
        Set of codes present in all models
    """
    if df.empty or not models:
        return set()

    # Get codes for each model
    codes_per_model = []
    for model in models:
        model_df = df[df["model"] == model]
        if not model_df.empty:
            codes_per_model.append(set(model_df["code"].unique()))

    if not codes_per_model:
        return set()

    # Find intersection of all code sets
    common_codes = codes_per_model[0]
    for codes in codes_per_model[1:]:
        common_codes = common_codes.intersection(codes)

    return common_codes


def filter_to_common_samples_across_horizons(
    df: pd.DataFrame,
    horizons: list[int],
    sample_columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Filter DataFrame to only include samples that exist across ALL horizons.

    This ensures fair comparison across lead times by using the same
    (code, year, target_month) combinations for all horizons.

    Args:
        df: DataFrame with predictions (must have 'horizon' column)
        horizons: List of horizons to consider
        sample_columns: Columns that define a unique sample.
                       Default: ["code", "target_month", "pred_year"]

    Returns:
        Filtered DataFrame with only common samples across all horizons
    """
    if sample_columns is None:
        sample_columns = ["code", "target_month", "pred_year"]

    # Extract year from valid_from if pred_year doesn't exist
    if "pred_year" not in df.columns and "valid_from" in df.columns:
        df = df.copy()
        df["pred_year"] = df["valid_from"].dt.year

    # Check required columns exist
    missing_cols = [c for c in sample_columns if c not in df.columns]
    if missing_cols:
        logger.warning(
            f"Cannot filter to common samples - missing columns: {missing_cols}"
        )
        return df

    # Get unique sample keys for each horizon
    sample_sets = []
    for horizon in horizons:
        horizon_df = df[df["horizon"] == horizon]
        if horizon_df.empty:
            logger.warning(f"No data for horizon {horizon} - skipping from common set")
            continue

        # Create tuple keys for this horizon
        keys = set(
            horizon_df[sample_columns].dropna().itertuples(index=False, name=None)
        )
        sample_sets.append(keys)
        logger.debug(f"Horizon {horizon}: {len(keys)} unique samples")

    if not sample_sets:
        logger.warning("No valid horizons found for filtering")
        return df

    # Find intersection of all sample sets
    common_samples = sample_sets[0]
    for sample_set in sample_sets[1:]:
        common_samples = common_samples.intersection(sample_set)

    if not common_samples:
        logger.warning("No common samples found across all horizons")
        return df

    # Log statistics
    total_original = len(df)
    n_common = len(common_samples)
    logger.info(
        f"Found {n_common} common (code, target_month, year) samples across "
        f"{len(sample_sets)} horizons"
    )

    # Filter DataFrame to only include common samples
    # Create a set of tuples for fast lookup
    df_filtered = df[
        df[sample_columns].apply(tuple, axis=1).isin(common_samples)
    ].copy()

    logger.info(
        f"Filtered from {total_original} to {len(df_filtered)} rows "
        f"({len(df_filtered) / total_original * 100:.1f}% retained)"
    )

    return df_filtered


def filter_models(
    available_models: list[str],
    include_models: list[str] | None = None,
    exclude_suffixes: list[str] | None = None,
) -> list[str]:
    """
    Filter out submodel variants and location models.

    Args:
        available_models: List of all available model names
        include_models: Optional explicit list of models to include.
                       If None, uses DEFAULT_MODELS.
        exclude_suffixes: Suffixes to exclude (e.g., ['_xgb', '_lgbm'])

    Returns:
        List of filtered model names
    """
    if exclude_suffixes is None:
        exclude_suffixes = EXCLUDE_SUFFIXES

    # Start with explicit models or default
    if include_models is not None:
        target_models = include_models
    else:
        target_models = DEFAULT_MODELS

    # Filter to only models that exist in available_models
    filtered = []
    for model in target_models:
        if model in available_models:
            # Check if it should be excluded by suffix
            should_exclude = any(model.endswith(suffix) for suffix in exclude_suffixes)
            # Check if it's a quantile pattern
            is_quantile = model in QUANTILE_PATTERN
            if not should_exclude and not is_quantile:
                filtered.append(model)

    logger.info(f"Filtered models: {filtered} (from {len(available_models)} available)")
    return filtered


def create_color_palette(
    models: list[str],
    palette_name: str = "husl",
) -> dict[str, tuple]:
    """
    Generate consistent color mapping for models.

    Climatology gets a distinct gray color as the baseline reference.

    Args:
        models: List of model names
        palette_name: Seaborn palette name (default: 'husl')

    Returns:
        Dictionary mapping model name to RGB color tuple
    """
    # Separate Climatology from other models
    other_models = [m for m in models if m != "Climatology"]
    n_colors = len(other_models)

    # Generate colors for non-Climatology models
    colors = sns.color_palette(palette_name, n_colors)
    color_map = {model: colors[i] for i, model in enumerate(other_models)}

    # Add Climatology with a distinct gray/black color
    if "Climatology" in models:
        color_map["Climatology"] = (0.3, 0.3, 0.3)  # Dark gray

    return color_map


def plot_metric_distribution(
    metrics_df: pd.DataFrame,
    metric: str,
    models: list[str],
    color_map: dict[str, tuple],
    output_path: Path,
    horizon: int,
    agricultural_period: bool = False,
) -> plt.Figure:
    """
    Create grouped boxplot for a single metric (R2, Accuracy, Efficiency, PBIAS).

    Only includes codes that are present in ALL models for fair comparison.

    Args:
        metrics_df: DataFrame with columns: code, model, month, R2, Accuracy, Efficiency, PBIAS, etc.
        metric: Metric column name ('R2', 'Accuracy', 'Efficiency', 'PBIAS')
        models: List of model names to include
        color_map: Dictionary mapping model name to color
        output_path: Path to save the figure
        horizon: Forecast horizon (for title)
        agricultural_period: If True, only plot months April-September

    Returns:
        matplotlib Figure object
    """
    # Filter for specified models
    df = metrics_df[metrics_df["model"].isin(models)].copy()

    if df.empty:
        logger.warning(f"No data for metric {metric} with specified models")
        return plt.figure()

    # Filter for codes present in ALL models
    common_codes = get_common_codes(df, models)
    if not common_codes:
        logger.warning(f"No common codes found across all models for {metric}")
        return plt.figure()

    df = df[df["code"].isin(common_codes)]
    logger.debug(f"Using {len(common_codes)} common codes for {metric} comparison")

    # Ensure month is integer
    df["month"] = df["month"].astype(int)

    # Filter for agricultural period if requested
    if agricultural_period:
        df = df[df["month"].isin(AGRICULTURAL_MONTHS)]
        if df.empty:
            logger.warning("No data for agricultural period (Apr-Sep)")
            return plt.figure()
        month_order = AGRICULTURAL_MONTHS
        title_suffix = " (Apr-Sep)"
    else:
        month_order = list(range(1, 13))
        title_suffix = ""

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Order models according to input order
    model_order = [m for m in models if m in df["model"].unique()]

    # Create custom palette for boxplot
    palette = [color_map[m] for m in model_order]

    sns.barplot(
        data=df,
        x="month",
        y=metric,
        hue="model",
        order=month_order,
        hue_order=model_order,
        palette=palette,
        ax=ax,
        errorbar=("pi", 50),
        capsize=0.1,
        err_kws={"linewidth": 1.5},
        estimator="median",
    )

    # Set x-tick labels to abbreviated month names
    ax.set_xticks(range(len(month_order)))
    ax.set_xticklabels([MONTH_ABBREV[m] for m in month_order])

    # Add reference line based on metric
    if metric in ["R2", "PBIAS"]:
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    elif metric == "Accuracy":
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
    elif metric == "Efficiency":
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)

    # Set appropriate y-axis limits based on metric
    if metric == "R2":
        ax.set_ylim(-0.2, 1.0)
    elif metric == "Accuracy":
        ax.set_ylim(0.0, 1.0)
    elif metric == "Efficiency":
        ax.set_ylim(0.0, 2.5)
    elif metric == "PBIAS":
        # Let y-axis auto-scale but ensure 0 is visible
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(min(y_min, -30), max(y_max, 30))

    # Labels and title
    ax.set_xlabel("Target Month", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric, fontsize=12, fontweight="bold")
    ax.set_title(
        f"{metric} Distribution by Target Month (Horizon {horizon}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.grid(axis="y", alpha=0.3)

    # Legend - include number of common codes
    ax.legend(
        title=f"Model (n={len(common_codes)} codes)",
        loc="best",
        fontsize=9,
        title_fontsize=10,
        framealpha=0.9,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved {metric} distribution plot to {output_path}")

    return fig


def plot_uncertainty_calibration(
    quantile_exceedance_df: pd.DataFrame,
    horizon: int,
    output_path: Path,
    agricultural_period: bool = False,
) -> plt.Figure:
    """
    Create calibration plot with 1:1 line showing quantile calibration.

    Shows a line plot with confidence bands representing the distribution
    of empirical exceedance rates across codes/months.

    Args:
        quantile_exceedance_df: DataFrame with quantile exceedance rates
                               (columns: code, month, horizon, Q5_exc, Q10_exc, etc.)
        horizon: Forecast horizon to filter by
        output_path: Path to save the figure
        agricultural_period: If True, only use months April-September

    Returns:
        matplotlib Figure object
    """
    # Filter by horizon
    df = quantile_exceedance_df[quantile_exceedance_df["horizon"] == horizon].copy()

    if df.empty:
        logger.warning(f"No quantile exceedance data for horizon {horizon}")
        return plt.figure()

    # Filter for agricultural period if requested
    if agricultural_period:
        df = df[df["month"].isin(AGRICULTURAL_MONTHS)]
        if df.empty:
            logger.warning("No data for agricultural period (Apr-Sep)")
            return plt.figure()
        title_suffix = " (Apr-Sep)"
    else:
        title_suffix = ""

    # Expected exceedance rates
    expected_rates = {
        "Q5": 0.05,
        "Q10": 0.10,
        "Q25": 0.25,
        "Q50": 0.50,
        "Q75": 0.75,
        "Q90": 0.90,
        "Q95": 0.95,
    }

    # Find available quantile columns
    exc_cols = [col for col in df.columns if col.endswith("_exc")]
    quantiles = [
        col.replace("_exc", "")
        for col in exc_cols
        if col.replace("_exc", "") in expected_rates
    ]

    if not quantiles:
        logger.warning("No valid quantile columns found")
        return plt.figure()

    # Order quantiles properly
    quantile_order = ["Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]
    quantile_order = [q for q in quantile_order if q in quantiles]

    # Calculate statistics for each quantile
    x_theoretical = []
    y_mean = []
    y_std = []
    y_lower = []  # 10th percentile
    y_upper = []  # 90th percentile

    for q in quantile_order:
        exc_col = f"{q}_exc"
        if exc_col in df.columns:
            values = df[exc_col].dropna()
            if len(values) > 0:
                x_theoretical.append(expected_rates[q])
                y_mean.append(values.mean())
                y_std.append(values.std())
                y_lower.append(values.quantile(0.10))
                y_upper.append(values.quantile(0.90))

    if not x_theoretical:
        logger.warning("No valid data for calibration plot")
        return plt.figure()

    x_theoretical = np.array(x_theoretical)
    y_mean = np.array(y_mean)
    y_std = np.array(y_std)
    y_lower = np.array(y_lower)
    y_upper = np.array(y_upper)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Add 1:1 perfect calibration line
    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        linewidth=2,
        label="Perfect calibration",
        zorder=1,
    )

    # Plot confidence band (10th-90th percentile)
    ax.fill_between(
        x_theoretical,
        y_lower,
        y_upper,
        color="steelblue",
        alpha=0.3,
        label="10th-90th percentile",
        zorder=2,
    )

    # Plot mean line
    ax.plot(
        x_theoretical,
        y_mean,
        "o-",
        color="steelblue",
        linewidth=2.5,
        markersize=8,
        label="Mean empirical rate",
        zorder=3,
    )

    # Add quantile labels at each point
    for i, q in enumerate(quantile_order):
        if i < len(x_theoretical):
            ax.annotate(
                q,
                (x_theoretical[i], y_mean[i]),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=9,
                fontweight="bold",
            )

    # Labels and title
    ax.set_xlabel("Theoretical Quantile", fontsize=12, fontweight="bold")
    ax.set_ylabel("Empirical Exceedance Rate", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Uncertainty Calibration - MC_ALD (Horizon {horizon}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Set axis limits
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # Grid
    ax.grid(alpha=0.3)

    # Legend
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved uncertainty calibration plot to {output_path}")

    return fig


def plot_calibration_comparison(
    quantile_exceedance_dict: dict[str, pd.DataFrame],
    horizon: int,
    output_path: Path,
    agricultural_period: bool = False,
) -> plt.Figure:
    """
    Create side-by-side calibration plots comparing multiple models.

    Shows calibration curves for MC_ALD and Climatology (or other models) in
    subplots for easy comparison.

    Args:
        quantile_exceedance_dict: Dictionary mapping model name to exceedance DataFrame
        horizon: Forecast horizon to filter by
        output_path: Path to save the figure
        agricultural_period: If True, only use months April-September

    Returns:
        matplotlib Figure object
    """
    models = list(quantile_exceedance_dict.keys())
    n_models = len(models)

    if n_models == 0:
        logger.warning("No models provided for calibration comparison")
        return plt.figure()

    # Create figure with subplots
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    title_suffix = " (Apr-Sep)" if agricultural_period else ""

    # Expected exceedance rates
    expected_rates = {
        "Q5": 0.05,
        "Q10": 0.10,
        "Q25": 0.25,
        "Q50": 0.50,
        "Q75": 0.75,
        "Q90": 0.90,
        "Q95": 0.95,
    }

    # Model colors
    model_colors = {
        "MC_ALD": "steelblue",
        "Climatology": (0.3, 0.3, 0.3),
    }

    for ax, model in zip(axes, models):
        df = quantile_exceedance_dict[model]

        # Filter by horizon
        df = df[df["horizon"] == horizon].copy()

        if df.empty:
            ax.text(
                0.5,
                0.5,
                f"No data\n{model}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # Filter for agricultural period if requested
        if agricultural_period:
            df = df[df["month"].isin(AGRICULTURAL_MONTHS)]
            if df.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No data\n{model}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                continue

        # Find available quantile columns
        exc_cols = [col for col in df.columns if col.endswith("_exc")]
        quantiles = [
            col.replace("_exc", "")
            for col in exc_cols
            if col.replace("_exc", "") in expected_rates
        ]

        if not quantiles:
            ax.text(
                0.5,
                0.5,
                f"No quantiles\n{model}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        # Order quantiles properly
        quantile_order = ["Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]
        quantile_order = [q for q in quantile_order if q in quantiles]

        # Calculate statistics for each quantile
        x_theoretical = []
        y_mean = []
        y_lower = []
        y_upper = []

        for q in quantile_order:
            exc_col = f"{q}_exc"
            if exc_col in df.columns:
                values = df[exc_col].dropna()
                if len(values) > 0:
                    x_theoretical.append(expected_rates[q])
                    y_mean.append(values.mean())
                    y_lower.append(values.quantile(0.10))
                    y_upper.append(values.quantile(0.90))

        if not x_theoretical:
            ax.text(
                0.5,
                0.5,
                f"No valid data\n{model}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            continue

        x_theoretical = np.array(x_theoretical)
        y_mean = np.array(y_mean)
        y_lower = np.array(y_lower)
        y_upper = np.array(y_upper)

        color = model_colors.get(model, "steelblue")

        # Add 1:1 perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, alpha=0.7)

        # Plot confidence band
        ax.fill_between(
            x_theoretical,
            y_lower,
            y_upper,
            color=color,
            alpha=0.3,
        )

        # Plot mean line
        ax.plot(
            x_theoretical,
            y_mean,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
        )

        # Set axis properties
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.set_xlabel("Theoretical Quantile", fontsize=11)
        ax.set_ylabel("Empirical Exceedance Rate", fontsize=11)
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)

    # Overall title
    fig.suptitle(
        f"Uncertainty Calibration Comparison (Horizon {horizon}){title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved calibration comparison plot to {output_path}")

    return fig


def compute_crps_normal(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute CRPS for normal distribution predictions.

    Closed-form solution for CRPS of N(μ, σ²):
    CRPS = σ * (z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π)
    where z = (y - μ) / σ

    Args:
        y_true: Observed values
        mu: Predicted means
        sigma: Predicted standard deviations

    Returns:
        Mean CRPS across all predictions
    """
    from scipy import stats

    # Standardize
    z = (y_true - mu) / sigma

    # Compute CRPS using closed-form for normal distribution
    crps_values = sigma * (
        z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi)
    )

    return np.mean(crps_values)


def compute_crps_quantiles(
    y_true: np.ndarray, quantiles: dict[float, np.ndarray]
) -> float:
    """
    Compute CRPS approximation using quantile predictions.

    Uses the quantile-based CRPS formula:
    CRPS ≈ 2 * Σ_τ (1(y < q_τ) - τ) * (q_τ - y)

    Args:
        y_true: Observed values
        quantiles: Dict mapping quantile level (0-1) to predicted quantile values

    Returns:
        Mean CRPS across all predictions
    """
    crps_sum = np.zeros(len(y_true))

    # Sort quantiles by level
    sorted_levels = sorted(quantiles.keys())

    for tau in sorted_levels:
        q_tau = quantiles[tau]
        # Indicator: 1 if y < q_tau, 0 otherwise
        indicator = (y_true < q_tau).astype(float)
        # CRPS contribution for this quantile
        crps_sum += 2 * np.abs(indicator - tau) * np.abs(q_tau - y_true)

    # Average over quantiles and observations
    return np.mean(crps_sum) / len(sorted_levels)


def compute_crps_dataframe(
    aggregated_df: pd.DataFrame,
    models: list[str],
    horizons: list[int],
    month_column: str = "target_month",
) -> pd.DataFrame:
    """
    Compute CRPS for probabilistic models (MC_ALD and Climatology) per (code, month, horizon).

    Returns individual CRPS values for each (code, month, horizon) combination,
    enabling boxplot visualization of the CRPS distribution.

    Args:
        aggregated_df: DataFrame with predictions and quantiles
        models: List of models to compute CRPS for (should include MC_ALD, Climatology)
        horizons: List of horizons to compute
        month_column: Column to use for month grouping

    Returns:
        DataFrame with columns: code, month, horizon, model, crps, ncrps, n_samples
    """
    results = []

    # Quantile column mapping
    quantile_cols = {
        0.05: "Q5",
        0.10: "Q10",
        0.25: "Q25",
        0.50: "Q50",
        0.75: "Q75",
        0.90: "Q90",
        0.95: "Q95",
    }

    for model in models:
        if model not in ["MC_ALD", "Climatology"]:
            continue

        model_df = aggregated_df[aggregated_df["model"] == model].copy()
        if model_df.empty:
            continue

        # Check for required columns
        has_quantiles = all(col in model_df.columns for col in quantile_cols.values())
        if not has_quantiles:
            logger.warning(f"Model {model} missing quantile columns for CRPS")
            continue

        # Compute basin mean Q_obs for normalization
        basin_means = model_df.groupby("code")["Q_obs"].transform("mean")
        model_df["basin_mean_Q_obs"] = basin_means

        for horizon in horizons:
            h_df = model_df[model_df["horizon"] == horizon].copy()

            # Filter valid data
            mask = ~h_df["Q_obs"].isna() & (h_df["Q_obs"] > 0)
            for q_col in quantile_cols.values():
                mask = mask & ~h_df[q_col].isna()
            h_df = h_df[mask]

            if h_df.empty:
                continue

            # Compute CRPS per (code, month) group
            for (code, month), group in h_df.groupby(["code", month_column]):
                if len(group) < 2:
                    continue

                y_true = group["Q_obs"].values
                basin_mean = group["basin_mean_Q_obs"].values[0]

                # Build quantiles dict for this group
                quantiles = {
                    level: group[col].values for level, col in quantile_cols.items()
                }

                # Compute CRPS for this group
                crps = compute_crps_quantiles(y_true, quantiles)

                # Compute normalized CRPS
                y_true_norm = y_true / basin_mean
                quantiles_norm = {
                    level: group[col].values / basin_mean
                    for level, col in quantile_cols.items()
                }
                ncrps = compute_crps_quantiles(y_true_norm, quantiles_norm)

                results.append(
                    {
                        "code": code,
                        "month": month,
                        "horizon": horizon,
                        "model": model,
                        "crps": crps,
                        "ncrps": ncrps,
                        "n_samples": len(group),
                    }
                )

    return pd.DataFrame(results)


def plot_crps_vs_horizon(
    crps_df: pd.DataFrame,
    output_path: Path,
    normalized: bool = True,
) -> plt.Figure:
    """
    Plot CRPS vs forecast horizon for MC_ALD and Climatology using boxplots.

    Shows the distribution of CRPS values across (code, month) groups for each
    horizon, with separate boxes for each model.

    Args:
        crps_df: DataFrame with columns: code, month, horizon, model, crps, ncrps
        output_path: Path to save the figure
        normalized: If True, plot normalized CRPS (ncrps), else raw CRPS

    Returns:
        matplotlib Figure object
    """
    import seaborn as sns

    if crps_df.empty:
        logger.warning("No CRPS data to plot")
        return plt.figure()

    # Select metric
    metric_col = "ncrps" if normalized else "crps"
    y_label = "nCRPS (normalized by basin mean)" if normalized else "CRPS"

    # Model colors
    palette = {
        "MC_ALD": "steelblue",
        "Climatology": (0.3, 0.3, 0.3),
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create boxplot
    sns.boxplot(
        data=crps_df,
        x="horizon",
        y=metric_col,
        hue="model",
        hue_order=["MC_ALD", "Climatology"],
        palette=palette,
        ax=ax,
        width=0.6,
        linewidth=1.5,
        fliersize=3,
        showfliers=False,
    )

    ax.set_xlabel("Lead Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(
        "Probabilistic Skill: CRPS Distribution vs Forecast Horizon",
        fontsize=14,
        fontweight="bold",
    )

    # Grid
    ax.grid(alpha=0.3, axis="y")

    # Legend
    ax.legend(title="Model", loc="upper left", fontsize=10)

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved CRPS vs horizon plot to {output_path}")

    return fig


def create_uncertainty_error_dataframe(
    aggregated_df: pd.DataFrame,
    model: str = "MC_ALD",
    horizons: list[int] | None = None,
    month_column: str = "target_month",
) -> pd.DataFrame:
    """
    Create DataFrame with individual data points for uncertainty vs error analysis.

    For each individual prediction, normalized by the MEAN Q_obs per basin (code):
    - nSTD = (Q75 - Q25) / mean_Q_obs_basin (predicted uncertainty)
    - nRMSE_point = |Q_pred - Q_obs| / mean_Q_obs_basin (normalized absolute error)

    Normalizing by basin mean Q_obs provides a stable reference across all months.

    Args:
        aggregated_df: DataFrame with predictions, observations, and quantiles
        model: Model name to filter for (default: "MC_ALD")
        horizons: Optional list of horizons to include
        month_column: Column to use for month grouping

    Returns:
        DataFrame with columns: code, month, horizon, nRMSE_point, nSTD, Q_obs, Q_pred
    """
    # Filter for specified model
    df = aggregated_df[aggregated_df["model"] == model].copy()

    if df.empty:
        logger.warning(f"No data found for model {model}")
        return pd.DataFrame()

    # Filter by horizons if specified
    if horizons is not None:
        df = df[df["horizon"].isin(horizons)]
        if df.empty:
            logger.warning(f"No data found for model {model} with specified horizons")
            return pd.DataFrame()

    # Check for required quantile columns
    if "Q75" not in df.columns or "Q25" not in df.columns:
        logger.warning("Q75 and Q25 columns required for nSTD calculation")
        return pd.DataFrame()

    # Drop rows with NaN values in required columns
    mask = (
        ~df["Q_obs"].isna()
        & ~df["Q_pred"].isna()
        & ~df["Q75"].isna()
        & ~df["Q25"].isna()
        & (df["Q_obs"] > 0)  # Avoid issues with zero observations
    )
    df = df[mask].copy()

    if df.empty:
        logger.warning(f"No valid data after filtering NaN values for model {model}")
        return pd.DataFrame()

    # Compute mean Q_obs per basin (code only) for normalization
    mean_q_obs_basin = df.groupby("code")["Q_obs"].transform("mean")

    # Ensure we don't divide by zero
    mean_q_obs_basin = mean_q_obs_basin.clip(lower=1e-6)

    # Calculate individual metrics normalized by basin mean Q_obs
    df["nSTD"] = (df["Q75"] - df["Q25"]) / mean_q_obs_basin
    df["nRMSE_point"] = np.abs(df["Q_pred"] - df["Q_obs"]) / mean_q_obs_basin
    df["mean_Q_obs_basin"] = mean_q_obs_basin

    # Select and rename columns for output
    result_df = df[
        [
            "code",
            month_column,
            "horizon",
            "nRMSE_point",
            "nSTD",
            "Q_obs",
            "Q_pred",
            "mean_Q_obs_basin",
        ]
    ].copy()
    result_df = result_df.rename(columns={month_column: "month"})

    result_df = result_df.sort_values(["code", "month", "horizon"]).reset_index(
        drop=True
    )

    logger.info(
        f"Created uncertainty-error DataFrame for {model} with {len(result_df)} "
        f"individual data points (normalized by basin mean Q_obs)"
    )

    return result_df


def plot_uncertainty_vs_error(
    uncertainty_error_dict: dict[str, pd.DataFrame],
    horizon: int,
    output_path: Path,
    agricultural_period: bool = False,
    n_bins: int = 20,
    percentile_bound: float = 97.5,
) -> plt.Figure:
    """
    Create plot showing nRMSE vs nSTD (predicted uncertainty) for multiple models.

    Shows binned means with linear regressions for each model.
    Reports R² for bins and MAE to the 1:1 line.

    Args:
        uncertainty_error_dict: Dict mapping model name to DataFrame with nRMSE_point and nSTD
        horizon: Forecast horizon to filter by
        output_path: Path to save the figure
        agricultural_period: If True, only use months April-September
        n_bins: Number of bins for aggregating nSTD
        percentile_bound: Percentile for axis and binning bounds (default: 97.5)

    Returns:
        matplotlib Figure object
    """
    from scipy import stats

    # Model styling
    model_styles = {
        "MC_ALD": {"color": "steelblue", "marker": "o", "linestyle": "-"},
        "Climatology": {"color": (0.3, 0.3, 0.3), "marker": "s", "linestyle": "--"},
    }

    # Collect all data to determine common axis bounds
    all_nstd = []
    all_nrmse = []
    filtered_data = {}

    for model_name, df in uncertainty_error_dict.items():
        if df.empty:
            continue

        # Filter by horizon
        model_df = df[df["horizon"] == horizon].copy()

        if model_df.empty:
            continue

        # Filter for agricultural period if requested
        if agricultural_period:
            model_df = model_df[model_df["month"].isin(AGRICULTURAL_MONTHS)]
            if model_df.empty:
                continue

        filtered_data[model_name] = model_df
        all_nstd.extend(model_df["nSTD"].values)
        all_nrmse.extend(model_df["nRMSE_point"].values)

    if not filtered_data:
        logger.warning(f"No uncertainty-error data for horizon {horizon}")
        return plt.figure()

    title_suffix = " (Apr-Sep)" if agricultural_period else ""

    # Compute percentile bounds across all models
    x_bound = np.percentile(all_nstd, percentile_bound)
    y_bound = np.percentile(all_nrmse, percentile_bound)
    plot_max = max(x_bound, y_bound)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Store regression results for legend
    legend_entries = []

    for model_name, model_df in filtered_data.items():
        style = model_styles.get(
            model_name, {"color": "gray", "marker": "^", "linestyle": ":"}
        )

        # Filter data to within bounds
        df_plot = model_df[
            (model_df["nSTD"] <= x_bound) & (model_df["nRMSE_point"] <= y_bound)
        ].copy()

        if df_plot.empty:
            continue

        # Create bins for nSTD within the bounded range
        bin_edges = np.linspace(0, x_bound, n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # Calculate bin statistics
        bin_x = []
        bin_means = []

        for i in range(n_bins):
            if i == n_bins - 1:
                mask = (df_plot["nSTD"] >= bin_edges[i]) & (
                    df_plot["nSTD"] <= bin_edges[i + 1]
                )
            else:
                mask = (df_plot["nSTD"] >= bin_edges[i]) & (
                    df_plot["nSTD"] < bin_edges[i + 1]
                )

            bin_data = df_plot.loc[mask, "nRMSE_point"]

            if len(bin_data) >= 3:
                bin_x.append(bin_centers[i])
                bin_means.append(bin_data.mean())

        if not bin_x:
            continue

        bin_x = np.array(bin_x)
        bin_means = np.array(bin_means)

        # Plot binned means
        ax.scatter(
            bin_x,
            bin_means,
            color=style["color"],
            marker=style["marker"],
            s=100,
            zorder=3,
            edgecolors="white",
            linewidths=1,
        )

        # Linear regression on binned data
        slope, intercept, r_value, *_ = stats.linregress(bin_x, bin_means)
        r2_bins = r_value**2

        # Calculate MAE to 1:1 line (perfect calibration)
        mae_to_11 = np.mean(np.abs(bin_means - bin_x))

        # Plot regression line
        x_line = np.array([0, plot_max])
        y_line = slope * x_line + intercept
        ax.plot(
            x_line,
            y_line,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2,
            zorder=2,
        )

        # Build legend label
        label = (
            f"{model_name}: y={slope:.2f}x+{intercept:.2f} "
            f"(R²={r2_bins:.3f}, MAE₁:₁={mae_to_11:.3f})"
        )
        legend_entries.append(
            plt.Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2,
                markersize=8,
                label=label,
            )
        )

        logger.info(
            f"  {model_name} horizon {horizon}: slope={slope:.3f}, "
            f"R²={r2_bins:.3f}, MAE_to_1:1={mae_to_11:.3f}"
        )

    # Add 1:1 line (perfect uncertainty-error correspondence)
    ax.plot(
        [0, plot_max],
        [0, plot_max],
        "k:",
        linewidth=2,
        alpha=0.7,
        zorder=1,
    )
    legend_entries.append(
        plt.Line2D(
            [0], [0], color="black", linestyle=":", linewidth=2, label="1:1 line"
        )
    )

    # Labels and title
    ax.set_xlabel("nSTD (IQR / basin mean Q_obs)", fontsize=12, fontweight="bold")
    ax.set_ylabel("nRMSE (|error| / basin mean Q_obs)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Uncertainty vs Error (Horizon {horizon}){title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Set axis limits
    ax.set_xlim(0, plot_max * 1.02)
    ax.set_ylim(0, plot_max * 1.02)

    # Grid
    ax.grid(alpha=0.3)

    # Legend
    ax.legend(
        handles=legend_entries,
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved uncertainty vs error plot to {output_path}")

    return fig


def calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R² (coefficient of determination) from arrays.

    Args:
        y_true: Array of observed values
        y_pred: Array of predicted values

    Returns:
        R² value (can be negative for poor predictions)
    """
    if len(y_true) < 2:
        return np.nan
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - (ss_res / ss_tot)


def plot_r2_vs_horizon_monthly_avg(
    horizon_metrics: dict[int, pd.DataFrame],
    models: list[str],
    color_map: dict[str, tuple],
    output_path: Path,
    agricultural_period: bool = False,
) -> plt.Figure:
    """
    Create plot of mean R² vs forecast horizon using monthly R² values.

    Takes pre-calculated monthly R² values and computes mean across months/basins.
    Uses seaborn.lineplot for automatic mean and confidence interval computation.

    For Climatology (which doesn't depend on horizon), R² values are taken from
    a single horizon and replicated across all horizons for consistent display.

    Args:
        horizon_metrics: Dictionary mapping horizon -> metrics DataFrame
        models: List of model names to include
        color_map: Dictionary mapping model name to color
        output_path: Path to save the figure
        agricultural_period: If True, only use months April-September

    Returns:
        matplotlib Figure object
    """
    import seaborn as sns

    horizons_list = sorted(horizon_metrics.keys())

    # Pre-compute climatology R² values once (from first available horizon)
    # This ensures identical climatology display across all horizons
    climatology_r2_values = []
    if "Climatology" in models and horizons_list:
        ref_horizon = horizons_list[0]
        if ref_horizon in horizon_metrics:
            ref_df = horizon_metrics[ref_horizon]
            clim_df = ref_df[ref_df["model"] == "Climatology"].copy()

            if agricultural_period:
                clim_df = clim_df[clim_df["month"].isin(AGRICULTURAL_MONTHS)]

            if not clim_df.empty and "R2" in clim_df.columns:
                climatology_r2_values = clim_df["R2"].dropna().tolist()
                logger.info(
                    f"Pre-computed {len(climatology_r2_values)} climatology R² values "
                    f"(will be replicated across all horizons)"
                )

    # Build long-format DataFrame for seaborn
    plot_data = []
    for horizon, df in horizon_metrics.items():
        for model in models:
            # For Climatology, use pre-computed values (same for all horizons)
            if model == "Climatology":
                for r2_val in climatology_r2_values:
                    plot_data.append({"horizon": horizon, "model": model, "R2": r2_val})
                continue

            model_df = df[df["model"] == model].copy()

            if model_df.empty:
                continue

            # Filter for agricultural period if requested
            if agricultural_period:
                model_df = model_df[model_df["month"].isin(AGRICULTURAL_MONTHS)]

            if model_df.empty or "R2" not in model_df.columns:
                continue

            # Add each R² value as a row
            for r2_val in model_df["R2"].dropna():
                plot_data.append({"horizon": horizon, "model": model, "R2": r2_val})

    if not plot_data:
        logger.warning("No data for R² vs horizon plot")
        fig, ax = plt.subplots(figsize=(12, 8))
        return fig

    plot_df = pd.DataFrame(plot_data)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Build palette from color_map
    palette = {model: color_map.get(model, (0.5, 0.5, 0.5)) for model in models}

    # Define line styles: dashed for Climatology, solid for others
    style_order = models
    dashes = {model: (2, 2) if model == "Climatology" else "" for model in models}

    # Use seaborn lineplot
    sns.lineplot(
        data=plot_df,
        x="horizon",
        y="R2",
        hue="model",
        hue_order=models,
        style="model",
        style_order=style_order,
        dashes=dashes,
        palette=palette,
        markers=True,
        markersize=8,
        linewidth=3,
        errorbar=("sd", 0.675),  # ~ 0.675 * sd
        err_style="band",
        ax=ax,
    )

    # Add reference line at R² = 0
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    # Labels and title
    ax.set_xlabel("Lead Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² (per code/month)", fontsize=12, fontweight="bold")
    title_suffix = " (Apr-Sep)" if agricultural_period else ""
    ax.set_title(
        f"R² vs Forecast Horizon{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Set axis properties
    horizons = sorted(horizon_metrics.keys())
    ax.set_xticks(horizons)
    ax.set_xlim(-0.5, max(horizons) + 0.5)
    ax.set_ylim(-0.3, 1.0)

    # Grid
    ax.grid(alpha=0.3)

    # Update legend
    ax.legend(
        title="Model (mean ~ 0.675 * SD)",
        loc="lower left",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.9,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved R² vs horizon (monthly avg) plot to {output_path}")

    return fig


def plot_r2_vs_horizon_seasonal(
    aggregated_df: pd.DataFrame,
    models: list[str],
    color_map: dict[str, tuple],
    output_path: Path,
    horizons: list[int],
    agricultural_period: bool = False,
) -> plt.Figure:
    """
    Create plot of mean R² vs forecast horizon using seasonal R² per basin.

    Calculates R² per basin using all predictions across months (seasonal R²),
    then shows mean R² across basins with standard deviation band.

    For Climatology (which doesn't depend on horizon), R² values are computed
    once and replicated across all horizons for consistent display.

    Args:
        aggregated_df: DataFrame with Q_pred, Q_obs, code, model, horizon, target_month
        models: List of model names to include
        color_map: Dictionary mapping model name to color
        output_path: Path to save the figure
        horizons: List of horizons to plot
        agricultural_period: If True, only use months April-September

    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Pre-compute climatology R² values once (from first horizon)
    # This ensures identical climatology display across all horizons
    climatology_basin_r2 = None
    climatology_mean = None
    climatology_std = None

    if "Climatology" in models and horizons:
        ref_horizon = horizons[0]
        clim_df = aggregated_df[
            (aggregated_df["model"] == "Climatology")
            & (aggregated_df["horizon"] == ref_horizon)
        ].copy()

        if agricultural_period:
            clim_df = clim_df[clim_df["target_month"].isin(AGRICULTURAL_MONTHS)]

        if not clim_df.empty:
            basin_r2_values = []
            for code in clim_df["code"].unique():
                basin_df = clim_df[clim_df["code"] == code]
                valid_mask = ~basin_df["Q_obs"].isna() & ~basin_df["Q_pred"].isna()
                basin_df = basin_df[valid_mask]

                if len(basin_df) < 2:
                    continue

                r2 = calculate_r2(basin_df["Q_obs"].values, basin_df["Q_pred"].values)
                if not np.isnan(r2):
                    basin_r2_values.append(r2)

            if basin_r2_values:
                climatology_basin_r2 = np.array(basin_r2_values)
                climatology_mean = np.mean(climatology_basin_r2)
                climatology_std = np.std(climatology_basin_r2)
                logger.info(
                    f"Pre-computed climatology seasonal R²: mean={climatology_mean:.3f}, "
                    f"std={climatology_std:.3f} (n={len(climatology_basin_r2)} basins)"
                )

    for model in models:
        # For Climatology, use pre-computed values (same for all horizons)
        if model == "Climatology" and climatology_mean is not None:
            color = color_map.get(model, (0.5, 0.5, 0.5))

            # Plot horizontal std band (same across all horizons)
            ax.fill_between(
                horizons,
                [climatology_mean - climatology_std] * len(horizons),
                [climatology_mean + climatology_std] * len(horizons),
                color=color,
                alpha=0.2,
            )

            # Plot horizontal mean line with markers
            ax.plot(
                horizons,
                [climatology_mean] * len(horizons),
                "o--",
                color=color,
                linewidth=2,
                markersize=8,
                label=model,
            )
            continue

        mean_r2 = []
        std_r2 = []
        valid_horizons = []

        for horizon in horizons:
            # Filter for model and horizon
            df = aggregated_df[
                (aggregated_df["model"] == model)
                & (aggregated_df["horizon"] == horizon)
            ].copy()

            if df.empty:
                continue

            # Filter for agricultural period if requested
            if agricultural_period:
                df = df[df["target_month"].isin(AGRICULTURAL_MONTHS)]

            if df.empty:
                continue

            # Calculate R² per basin (using all months combined)
            basin_r2_values = []
            for code in df["code"].unique():
                basin_df = df[df["code"] == code]
                # Drop NaN values
                valid_mask = ~basin_df["Q_obs"].isna() & ~basin_df["Q_pred"].isna()
                basin_df = basin_df[valid_mask]

                if len(basin_df) < 2:
                    continue

                r2 = calculate_r2(basin_df["Q_obs"].values, basin_df["Q_pred"].values)
                if not np.isnan(r2):
                    basin_r2_values.append(r2)

            if not basin_r2_values:
                continue

            basin_r2_values = np.array(basin_r2_values)
            valid_horizons.append(horizon)
            mean_r2.append(np.mean(basin_r2_values))
            std_r2.append(np.std(basin_r2_values))

        if not valid_horizons:
            continue

        valid_horizons = np.array(valid_horizons)
        mean_r2 = np.array(mean_r2)
        std_r2 = np.array(std_r2)

        color = color_map.get(model, (0.5, 0.5, 0.5))

        # Plot std band
        ax.fill_between(
            valid_horizons,
            mean_r2 - std_r2,
            mean_r2 + std_r2,
            color=color,
            alpha=0.2,
        )

        # Plot mean line with markers
        ax.plot(
            valid_horizons,
            mean_r2,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=model,
        )

    # Add reference line at R² = 0
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    # Labels and title
    ax.set_xlabel("Lead Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("R² (per basin, seasonal)", fontsize=12, fontweight="bold")
    title_suffix = " (Apr-Sep)" if agricultural_period else ""
    ax.set_title(
        f"R² vs Forecast Horizon - Seasonal{title_suffix}",
        fontsize=14,
        fontweight="bold",
    )

    # Set axis properties
    ax.set_xticks(horizons)
    ax.set_xlim(-0.5, max(horizons) + 0.5)
    ax.set_ylim(-0.3, 1.0)

    # Grid
    ax.grid(alpha=0.3)

    # Legend
    ax.legend(
        title="Model (mean ± std)",
        loc="lower left",
        fontsize=10,
        title_fontsize=11,
        framealpha=0.9,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved R² vs horizon (seasonal) plot to {output_path}")

    return fig


def plot_r2_boxplots_by_horizon(
    aggregated_df: pd.DataFrame,
    models: list[str],
    color_map: dict[str, tuple],
    output_dir: Path,
    horizons: list[int],
    agricultural_period: bool = False,
) -> list[plt.Figure]:
    """
    Create boxplot comparison of R² distributions across basins for each lead time.

    For each horizon, calculates R² per basin (using all months combined),
    then shows the distribution across basins as boxplots for each model.

    For Climatology (which doesn't depend on horizon), R² is computed once
    and reused for all horizons to ensure consistent display.

    Args:
        aggregated_df: DataFrame with Q_pred, Q_obs, code, model, horizon, target_month
        models: List of model names to include
        color_map: Dictionary mapping model name to color
        output_dir: Directory to save the figures
        horizons: List of horizons to plot
        agricultural_period: If True, only use months April-September

    Returns:
        List of matplotlib Figure objects (one per horizon)
    """
    figures = []

    # Pre-compute climatology R² values once (horizon-agnostic)
    # This ensures identical climatology display across all horizons
    climatology_r2_per_basin = {}
    if "Climatology" in models and "Climatology" in aggregated_df["model"].unique():
        # Use data from any single horizon (they should all be the same after filtering)
        # Using horizon 0 as reference
        clim_df = aggregated_df[
            (aggregated_df["model"] == "Climatology")
            & (aggregated_df["horizon"] == horizons[0])
        ].copy()

        if agricultural_period:
            clim_df = clim_df[clim_df["target_month"].isin(AGRICULTURAL_MONTHS)]

        for code in clim_df["code"].unique():
            basin_df = clim_df[clim_df["code"] == code]
            valid_mask = ~basin_df["Q_obs"].isna() & ~basin_df["Q_pred"].isna()
            basin_df = basin_df[valid_mask]

            if len(basin_df) < 2:
                continue

            r2 = calculate_r2(basin_df["Q_obs"].values, basin_df["Q_pred"].values)
            if not np.isnan(r2):
                climatology_r2_per_basin[code] = r2

        logger.info(
            f"Pre-computed climatology R² for {len(climatology_r2_per_basin)} basins "
            f"(will be reused across all horizons)"
        )

    for horizon in horizons:
        # Collect R² values per basin for each model
        plot_data = []

        for model in models:
            # For Climatology, use pre-computed values
            if model == "Climatology":
                for code, r2 in climatology_r2_per_basin.items():
                    plot_data.append({"model": model, "R2": r2, "code": code})
                continue

            # Filter for model and horizon
            df = aggregated_df[
                (aggregated_df["model"] == model)
                & (aggregated_df["horizon"] == horizon)
            ].copy()

            if df.empty:
                continue

            # Filter for agricultural period if requested
            if agricultural_period:
                df = df[df["target_month"].isin(AGRICULTURAL_MONTHS)]

            if df.empty:
                continue

            # Calculate R² per basin (using all months combined)
            for code in df["code"].unique():
                basin_df = df[df["code"] == code]
                # Drop NaN values
                valid_mask = ~basin_df["Q_obs"].isna() & ~basin_df["Q_pred"].isna()
                basin_df = basin_df[valid_mask]

                if len(basin_df) < 2:
                    continue

                r2 = calculate_r2(basin_df["Q_obs"].values, basin_df["Q_pred"].values)
                if not np.isnan(r2):
                    plot_data.append({"model": model, "R2": r2, "code": code})

        if not plot_data:
            logger.warning(f"No data for R² boxplot at horizon {horizon}")
            continue

        plot_df = pd.DataFrame(plot_data)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create boxplot with seaborn
        box_colors = [
            color_map.get(m, (0.5, 0.5, 0.5))
            for m in models
            if m in plot_df["model"].unique()
        ]
        models_in_data = [m for m in models if m in plot_df["model"].unique()]

        sns.boxplot(
            data=plot_df,
            x="model",
            y="R2",
            hue="model",
            order=models_in_data,
            hue_order=models_in_data,
            palette=box_colors,
            ax=ax,
            width=0.6,
            linewidth=1.5,
            legend=False,
        )

        # Add individual points (strip plot)
        sns.stripplot(
            data=plot_df,
            x="model",
            y="R2",
            order=models_in_data,
            color="black",
            alpha=0.4,
            size=4,
            ax=ax,
        )

        # Add reference line at R² = 0
        ax.axhline(y=0, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

        # Labels and title
        ax.set_xlabel("Model", fontsize=12, fontweight="bold")
        ax.set_ylabel("R² (per basin)", fontsize=12, fontweight="bold")
        title_suffix = " (Apr-Sep)" if agricultural_period else ""
        ax.set_title(
            f"R² Distribution by Model - Lead Time {horizon}{title_suffix}",
            fontsize=14,
            fontweight="bold",
        )

        # Set y-axis limits
        ax.set_ylim(-0.5, 1.0)

        # Grid
        ax.grid(axis="y", alpha=0.3)

        # Rotate x-axis labels if needed
        plt.xticks(rotation=0)

        plt.tight_layout()

        # Save figure
        output_path = output_dir / f"r2_boxplot_horizon_{horizon}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved R² boxplot for horizon {horizon} to {output_path}")

        figures.append(fig)

    return figures


def plot_calibration_grid(
    quantile_exceedance_dict: dict[str, pd.DataFrame],
    horizons: list[int],
    output_path: Path,
    agricultural_period: bool = False,
) -> plt.Figure:
    """
    Create 3x3 grid of calibration plots for all horizons, comparing models.

    Each subplot shows the calibration curves for all models at one horizon.
    Shows MC_ALD and Climatology (normal distribution baseline) side by side.

    Args:
        quantile_exceedance_dict: Dictionary mapping model name to exceedance DataFrame
        horizons: List of horizons to include (should be 9 for 3x3 grid)
        output_path: Path to save the figure
        agricultural_period: If True, only use months April-September

    Returns:
        matplotlib Figure object
    """
    if not quantile_exceedance_dict:
        logger.warning("No quantile exceedance data for calibration grid")
        return plt.figure()

    # Expected exceedance rates
    expected_rates = {
        "Q5": 0.05,
        "Q10": 0.10,
        "Q25": 0.25,
        "Q50": 0.50,
        "Q75": 0.75,
        "Q90": 0.90,
        "Q95": 0.95,
    }

    # Model colors and styles
    model_styles = {
        "MC_ALD": {"color": "steelblue", "linestyle": "-", "marker": "o"},
        "Climatology": {"color": (0.3, 0.3, 0.3), "linestyle": "--", "marker": "s"},
    }

    # Create 3x3 figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    title_suffix = " (Apr-Sep)" if agricultural_period else ""

    for idx, horizon in enumerate(horizons[:9]):  # Limit to 9 for 3x3 grid
        ax = axes[idx]

        # Add 1:1 perfect calibration line
        ax.plot([0, 1], [0, 1], "k:", linewidth=1.5, alpha=0.5)

        has_data = False

        # Plot each model
        for model_name, exc_df in quantile_exceedance_dict.items():
            if exc_df.empty:
                continue

            # Filter by horizon
            df = exc_df[exc_df["horizon"] == horizon].copy()

            if df.empty:
                continue

            # Filter for agricultural period if requested
            if agricultural_period:
                df = df[df["month"].isin(AGRICULTURAL_MONTHS)]
                if df.empty:
                    continue

            # Find available quantile columns
            exc_cols = [col for col in df.columns if col.endswith("_exc")]
            quantiles = [
                col.replace("_exc", "")
                for col in exc_cols
                if col.replace("_exc", "") in expected_rates
            ]

            if not quantiles:
                continue

            # Order quantiles properly
            quantile_order = ["Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]
            quantile_order = [q for q in quantile_order if q in quantiles]

            # Calculate statistics for each quantile
            x_theoretical = []
            y_mean = []

            for q in quantile_order:
                exc_col = f"{q}_exc"
                if exc_col in df.columns:
                    values = df[exc_col].dropna()
                    if len(values) > 0:
                        x_theoretical.append(expected_rates[q])
                        y_mean.append(values.mean())

            if not x_theoretical:
                continue

            x_theoretical = np.array(x_theoretical)
            y_mean = np.array(y_mean)

            # Get style for this model
            style = model_styles.get(
                model_name,
                {"color": "gray", "linestyle": "-", "marker": "o"},
            )

            # Plot mean line
            ax.plot(
                x_theoretical,
                y_mean,
                marker=style["marker"],
                linestyle=style["linestyle"],
                color=style["color"],
                linewidth=2,
                markersize=5,
                label=model_name,
            )

            has_data = True

        if not has_data:
            ax.text(
                0.5,
                0.5,
                f"No data\nHorizon {horizon}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )

        # Set axis properties
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")

        # Add horizon label
        ax.set_title(f"Horizon {horizon}", fontsize=11, fontweight="bold")

        # Grid
        ax.grid(alpha=0.3)

        # Only add axis labels for edge plots
        if idx >= 6:  # Bottom row
            ax.set_xlabel("Theoretical", fontsize=10)
        if idx % 3 == 0:  # Left column
            ax.set_ylabel("Empirical", fontsize=10)

        # Add legend only to first subplot
        if idx == 0 and has_data:
            ax.legend(fontsize=8, loc="lower right")

    # Hide unused subplots if less than 9 horizons
    for idx in range(len(horizons), 9):
        axes[idx].set_visible(False)

    # Overall title
    fig.suptitle(
        f"Uncertainty Calibration Comparison{title_suffix}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved calibration grid plot to {output_path}")

    return fig


def generate_all_plots(
    horizon_metrics: dict[int, pd.DataFrame],
    quantile_exceedance_dict: dict[str, pd.DataFrame],
    uncertainty_error_dict: dict[str, pd.DataFrame],
    crps_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    models: list[str],
    output_dir: Path,
    region: str,
    agricultural_period: bool = False,
) -> None:
    """
    Orchestrate plot generation for all horizons.

    Args:
        horizon_metrics: Dictionary mapping horizon -> metrics DataFrame
        quantile_exceedance_dict: Dictionary mapping model name to exceedance DataFrame
        uncertainty_error_dict: Dictionary mapping model name to uncertainty-error DataFrame
        crps_df: DataFrame with CRPS by horizon and model
        aggregated_df: DataFrame with raw predictions (Q_pred, Q_obs, etc.)
        models: List of models to plot
        output_dir: Base output directory
        region: Region name (for subdirectory)
        agricultural_period: If True, only plot months April-September
    """
    # Create color palette for models
    color_map = create_color_palette(models)

    # Metrics to plot
    metrics = ["R2", "Accuracy", "Efficiency", "PBIAS"]

    # Generate plots for each horizon
    for horizon, metrics_df in horizon_metrics.items():
        horizon_dir = output_dir / region.lower() / f"lead_time_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating plots for horizon {horizon}...")

        # Plot each metric distribution
        for metric in metrics:
            if metric in metrics_df.columns:
                output_path = horizon_dir / f"{metric.lower()}_distribution.png"
                fig = plot_metric_distribution(
                    metrics_df=metrics_df,
                    metric=metric,
                    models=models,
                    color_map=color_map,
                    output_path=output_path,
                    horizon=horizon,
                    agricultural_period=agricultural_period,
                )
                plt.close(fig)
            else:
                logger.warning(f"Metric {metric} not found in metrics DataFrame")

        # Plot uncertainty calibration for MC_ALD (only if we have quantile data)
        mc_ald_exceedance = quantile_exceedance_dict.get("MC_ALD", pd.DataFrame())
        if not mc_ald_exceedance.empty:
            horizon_data = mc_ald_exceedance[mc_ald_exceedance["horizon"] == horizon]
            if not horizon_data.empty:
                output_path = horizon_dir / "uncertainty_calibration.png"
                fig = plot_uncertainty_calibration(
                    quantile_exceedance_df=mc_ald_exceedance,
                    horizon=horizon,
                    output_path=output_path,
                    agricultural_period=agricultural_period,
                )
                plt.close(fig)

        # Plot uncertainty vs error for all models with uncertainty data
        if uncertainty_error_dict:
            output_path = horizon_dir / "uncertainty_vs_error.png"
            fig = plot_uncertainty_vs_error(
                uncertainty_error_dict=uncertainty_error_dict,
                horizon=horizon,
                output_path=output_path,
                agricultural_period=agricultural_period,
            )
            plt.close(fig)

    # Generate summary plots (all horizons in one figure)
    summary_dir = output_dir / region.lower() / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    horizons_list = sorted(horizon_metrics.keys())

    # R² vs horizon plot - monthly average (averages pre-calculated monthly R² values)
    logger.info("Generating R² vs horizon (monthly average) summary plot...")
    fig = plot_r2_vs_horizon_monthly_avg(
        horizon_metrics=horizon_metrics,
        models=models,
        color_map=color_map,
        output_path=summary_dir / "r2_vs_horizon_monthly_avg.png",
        agricultural_period=agricultural_period,
    )
    plt.close(fig)

    # R² vs horizon plot - seasonal (R² per basin across all months, then averaged)
    logger.info("Generating R² vs horizon (seasonal) summary plot...")
    fig = plot_r2_vs_horizon_seasonal(
        aggregated_df=aggregated_df,
        models=models,
        color_map=color_map,
        output_path=summary_dir / "r2_vs_horizon_seasonal.png",
        horizons=horizons_list,
        agricultural_period=agricultural_period,
    )
    plt.close(fig)

    # R² boxplots per horizon (one figure per lead time)
    logger.info("Generating R² boxplots by horizon...")
    figs = plot_r2_boxplots_by_horizon(
        aggregated_df=aggregated_df,
        models=models,
        color_map=color_map,
        output_dir=summary_dir,
        horizons=horizons_list,
        agricultural_period=agricultural_period,
    )
    for fig in figs:
        plt.close(fig)

    # Calibration grid plot (only if we have quantile data)
    if quantile_exceedance_dict:
        logger.info("Generating calibration grid summary plot...")
        horizons_list = sorted(horizon_metrics.keys())
        fig = plot_calibration_grid(
            quantile_exceedance_dict=quantile_exceedance_dict,
            horizons=horizons_list,
            output_path=summary_dir / "calibration_grid.png",
            agricultural_period=agricultural_period,
        )
        plt.close(fig)

    # CRPS vs horizon plot (only if we have CRPS data)
    if not crps_df.empty:
        logger.info("Generating CRPS vs horizon summary plot...")
        fig = plot_crps_vs_horizon(
            crps_df=crps_df,
            output_path=summary_dir / "crps_vs_horizon.png",
            normalized=True,
        )
        plt.close(fig)

    logger.info(f"All plots generated in {output_dir / region.lower()}")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Generate standardized performance visualization plots."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["Kyrgyzstan", "Tajikistan"],
        default="Kyrgyzstan",
        help="Region to process (default: Kyrgyzstan)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(range(9)),
        help="List of horizons to plot (default: 0-8)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of models to plot (default: use DEFAULT_MODELS)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--use-q-obs",
        action="store_true",
        help="Use Q_obs from predictions instead of loading from observations file",
    )
    parser.add_argument(
        "--agricultural-period",
        action="store_true",
        help="Only plot months April-September (agricultural growing season)",
    )

    args = parser.parse_args()

    # Set global plot style
    set_global_plot_style()

    # Determine paths based on region
    if args.region == "Tajikistan":
        pred_config = taj_path_config
    else:
        pred_config = kgz_path_config

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(default_output_dir)

    # Convert horizons to horizon strings for loading
    horizon_strings = [
        f"month_{h}" for h in args.horizons if f"month_{h}" in day_of_forecast
    ]

    logger.info(f"Loading predictions for {args.region}...")
    logger.info(f"Horizons: {args.horizons}")
    if args.agricultural_period:
        logger.info("Agricultural period filter enabled (April-September only)")

    # Load predictions
    predictions_df = load_predictions(
        base_path=pred_config["pred_dir"],
        horizons=horizon_strings,
    )

    if predictions_df.empty:
        logger.error("No predictions loaded. Exiting.")
        return

    logger.info(f"Loaded {len(predictions_df)} prediction records")
    logger.info(f"Available models: {predictions_df['model'].unique().tolist()}")

    # Create ensemble
    logger.info("Creating ensemble...")
    predictions_df = create_ensemble(predictions_df)

    # Handle observations
    monthly_obs = None
    if args.use_q_obs:
        logger.info("Using Q_obs from prediction files...")
        if predictions_df["Q_obs"].isna().all():
            logger.error(
                "Q_obs not available in prediction files. Run without --use-q-obs."
            )
            return
        predictions_df["target_month"] = predictions_df["valid_from"].dt.month
        predictions_df["issue_month"] = predictions_df["issue_date"].dt.month
        aggregated_df = predictions_df
    else:
        logger.info(f"Loading observations from {pred_config['obs_file']}...")
        obs_df = load_observations(pred_config["obs_file"])
        monthly_obs = calculate_target(obs_df)
        logger.info("Aggregating predictions with observations...")
        aggregated_df = aggregate(predictions_df, monthly_obs)

    # Add climatology baseline
    if monthly_obs is not None:
        logger.info("Adding climatology baseline...")
        aggregated_df = add_climatology_baseline(aggregated_df, monthly_obs)
    else:
        logger.warning(
            "Cannot add climatology baseline without monthly_obs. "
            "Run without --use-q-obs to include climatology."
        )

    # Filter to common samples across all horizons for fair comparison
    # This ensures climatology (and all models) are evaluated on the same data
    logger.info("Filtering to common samples across all horizons...")
    aggregated_df = filter_to_common_samples_across_horizons(
        aggregated_df,
        horizons=args.horizons,
        sample_columns=["code", "target_month", "pred_year"],
    )

    # Filter models
    available_models = aggregated_df["model"].unique().tolist()
    models_to_plot = filter_models(
        available_models=available_models,
        include_models=args.models,
    )

    if not models_to_plot:
        logger.error("No models to plot after filtering. Check --models argument.")
        return

    logger.info(f"Models to plot: {models_to_plot}")

    # Create horizon metrics DataFrames
    logger.info("Creating horizon-specific metrics DataFrames...")
    horizon_metrics: dict[int, pd.DataFrame] = {}
    for horizon in args.horizons:
        horizon_metrics[horizon] = create_horizon_metrics_dataframe(
            aggregated_df,
            horizon=horizon,
            models=models_to_plot,
            month_column="target_month",
        )
        logger.info(f"  Horizon {horizon}: {len(horizon_metrics[horizon])} rows")

    # Create quantile exceedance DataFrames for probabilistic models
    quantile_exceedance_dict: dict[str, pd.DataFrame] = {}
    uncertainty_error_dict: dict[str, pd.DataFrame] = {}

    # MC_ALD exceedance and uncertainty-error
    if "MC_ALD" in aggregated_df["model"].unique():
        logger.info("Creating quantile exceedance DataFrame for MC_ALD...")
        mc_ald_exceedance = create_quantile_exceedance_dataframe(
            aggregated_df,
            model="MC_ALD",
            horizons=args.horizons,
            month_column="target_month",
        )
        quantile_exceedance_dict["MC_ALD"] = mc_ald_exceedance
        logger.info(f"  MC_ALD exceedance rows: {len(mc_ald_exceedance)}")

        # Create uncertainty vs error DataFrame for MC_ALD
        logger.info("Creating uncertainty-error DataFrame for MC_ALD...")
        mc_ald_ue = create_uncertainty_error_dataframe(
            aggregated_df,
            model="MC_ALD",
            horizons=args.horizons,
            month_column="target_month",
        )
        uncertainty_error_dict["MC_ALD"] = mc_ald_ue
        logger.info(f"  MC_ALD uncertainty-error rows: {len(mc_ald_ue)}")
    else:
        logger.warning(
            "MC_ALD model not found - skipping MC_ALD uncertainty calibration"
        )

    # Climatology exceedance and uncertainty-error (if available)
    if "Climatology" in aggregated_df["model"].unique():
        logger.info("Creating quantile exceedance DataFrame for Climatology...")
        clim_exceedance = create_quantile_exceedance_dataframe(
            aggregated_df,
            model="Climatology",
            horizons=args.horizons,
            month_column="target_month",
        )
        quantile_exceedance_dict["Climatology"] = clim_exceedance
        logger.info(f"  Climatology exceedance rows: {len(clim_exceedance)}")

        # Create uncertainty vs error DataFrame for Climatology
        logger.info("Creating uncertainty-error DataFrame for Climatology...")
        clim_ue = create_uncertainty_error_dataframe(
            aggregated_df,
            model="Climatology",
            horizons=args.horizons,
            month_column="target_month",
        )
        uncertainty_error_dict["Climatology"] = clim_ue
        logger.info(f"  Climatology uncertainty-error rows: {len(clim_ue)}")
    else:
        logger.warning("Climatology model not found - skipping Climatology calibration")

    # Compute CRPS for probabilistic models
    logger.info("Computing CRPS for probabilistic models...")
    crps_df = compute_crps_dataframe(
        aggregated_df=aggregated_df,
        models=["MC_ALD", "Climatology"],
        horizons=args.horizons,
        month_column="target_month",
    )
    if not crps_df.empty:
        logger.info(
            f"  CRPS computed for {len(crps_df)} (code, month, horizon, model) groups"
        )
        # Log summary CRPS by model and horizon
        for model in ["MC_ALD", "Climatology"]:
            model_data = crps_df[crps_df["model"] == model]
            if model_data.empty:
                continue
            for horizon in sorted(model_data["horizon"].unique()):
                h_data = model_data[model_data["horizon"] == horizon]
                logger.info(
                    f"    {model} h={horizon}: "
                    f"nCRPS median={h_data['ncrps'].median():.4f}, "
                    f"mean={h_data['ncrps'].mean():.4f} "
                    f"(n={len(h_data)} groups)"
                )
    else:
        logger.warning("No CRPS data computed")

    # Generate all plots
    logger.info("Generating plots...")
    generate_all_plots(
        horizon_metrics=horizon_metrics,
        quantile_exceedance_dict=quantile_exceedance_dict,
        uncertainty_error_dict=uncertainty_error_dict,
        crps_df=crps_df,
        aggregated_df=aggregated_df,
        models=models_to_plot,
        output_dir=output_dir,
        region=args.region,
        agricultural_period=args.agricultural_period,
    )

    logger.info("Visualization complete!")


if __name__ == "__main__":
    main()
