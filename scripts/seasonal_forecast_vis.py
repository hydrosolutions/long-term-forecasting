#!/usr/bin/env python3
"""
Seasonal Forecast Visualization Script

Evaluates seasonal forecasts (6-month and 3-month) by comparing ML model predictions.

Forecast Configurations:
- 6-Month Forecasts (Apr-Sep average): Issue dates Jan 25, Feb 25, Mar 25, Apr 25
- 3-Month Forecasts (rolling): Issue dates Mar 25, Apr 25, May 25

Output structure:
    {output_dir}/{region}/seasonal/
        ├── 6_month/
        │   ├── scatter_Jan25.png
        │   ├── r2_vs_issue_date.png
        │   └── metrics_summary.csv
        └── 3_month/
            ├── scatter_Mar25.png
            ├── r2_vs_issue_date.png
            └── metrics_summary.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load environment variables
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

# Import from examine_operational_fc.py (in same directory)
from examine_operational_fc import (
    load_predictions,
    load_observations,
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

# =============================================================================
# CONSTANTS
# =============================================================================

# 6-month forecast issue dates (month, day) -> target months (Apr-Sep)
ISSUE_DATES_6M = {
    (1, 25): {"target_months": [4, 5, 6, 7, 8, 9], "label": "Jan 25"},
    (2, 25): {"target_months": [4, 5, 6, 7, 8, 9], "label": "Feb 25"},
    (3, 25): {"target_months": [4, 5, 6, 7, 8, 9], "label": "Mar 25"},
    (4, 25): {"target_months": [4, 5, 6, 7, 8, 9], "label": "Apr 25"},
}

# 3-month forecast issue dates (month, day) -> target months (rolling)
ISSUE_DATES_3M = {
    (3, 25): {"target_months": [4, 5, 6], "label": "Mar 25"},
    (4, 25): {"target_months": [5, 6, 7], "label": "Apr 25"},
    (5, 25): {"target_months": [6, 7, 8], "label": "May 25"},
}

# Default models to include
DEFAULT_MODELS = ["LR_Base", "LR_SM", "MC_ALD", "Ensemble"]

# Specific code for scatter plots (None = plot all codes)
SCATTER_PLOT_CODE = 16936

# Month abbreviations
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

# Seasonal model directory names by issue month
SEASONAL_DIR_NAMES = {
    1: "seasonal_january",
    2: "seasonal_february",
    3: "seasonal_march",
    4: "seasonal_april",
}

# Suffix for seasonal model names to distinguish from monthly-reconstructed
SEASONAL_MODEL_SUFFIX = "_seasonal"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_horizon(issue_month: int, target_month: int) -> int:
    """
    Calculate forecast horizon (months between issue and target).

    Args:
        issue_month: Month when forecast is issued (1-12)
        target_month: Month being predicted (1-12)

    Returns:
        Horizon in months (0 = same month, 1 = next month, etc.)
    """
    if target_month >= issue_month:
        return target_month - issue_month
    return (12 - issue_month) + target_month


def calculate_target_with_threshold(
    obs: pd.DataFrame,
    min_coverage: float = 0.9,
) -> pd.DataFrame:
    """
    Aggregates daily observations to monthly means, only if >=90% data is present.

    Args:
        obs: DataFrame with columns: date, code, discharge
        min_coverage: Minimum fraction of non-missing days required (default 0.9)

    Returns:
        DataFrame with monthly means where data coverage meets threshold
    """
    obs = obs.copy()
    obs["year"] = obs["date"].dt.year
    obs["month"] = obs["date"].dt.month
    obs["days_in_month"] = obs["date"].dt.daysinmonth

    grouped = (
        obs.groupby(["code", "year", "month"])
        .agg(
            discharge_mean=("discharge", "mean"),
            discharge_count=("discharge", "count"),
            days_in_month=("days_in_month", "first"),
        )
        .reset_index()
    )

    grouped["coverage"] = grouped["discharge_count"] / grouped["days_in_month"]
    filtered = grouped[grouped["coverage"] >= min_coverage].copy()
    filtered = filtered.rename(columns={"discharge_mean": "Q_obs_monthly"})

    return filtered[["code", "year", "month", "Q_obs_monthly"]]


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate performance metrics.

    Args:
        y_true: Array of observed values
        y_pred: Array of predicted values

    Returns:
        Dictionary with r2, rmse, mae, pbias, nse, n_samples
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "pbias": np.nan,
            "nse": np.nan,
            "n_samples": len(y_true_clean),
        }

    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)

    sum_obs = y_true_clean.sum()
    pbias = (
        100 * (y_pred_clean - y_true_clean).sum() / sum_obs if sum_obs != 0 else np.nan
    )

    ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
    ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
    nse = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan

    return {
        "r2": r2,
        "rmse": rmse,
        "mae": mae,
        "pbias": pbias,
        "nse": nse,
        "n_samples": len(y_true_clean),
    }


def calculate_r2_per_code(
    df: pd.DataFrame,
    obs_col: str = "Q_obs_seasonal",
    pred_col: str = "Q_pred_seasonal",
    min_samples: int = 3,
) -> pd.DataFrame:
    """
    Calculate R² for each code (basin) separately using vectorized groupby.

    Args:
        df: DataFrame with observations and predictions
        obs_col: Name of observation column
        pred_col: Name of prediction column
        min_samples: Minimum samples required per code

    Returns:
        DataFrame with columns: code, r2, n_samples
    """

    def compute_r2(group: pd.DataFrame) -> pd.Series:
        clean = group.dropna(subset=[obs_col, pred_col])
        n = len(clean)
        if n < min_samples:
            return pd.Series({"r2": np.nan, "n_samples": n})
        try:
            r2 = r2_score(clean[obs_col], clean[pred_col])
        except Exception:
            r2 = np.nan
        return pd.Series({"r2": r2, "n_samples": n})

    result = df.groupby("code").apply(compute_r2, include_groups=False).reset_index()
    return result[result["n_samples"] >= min_samples]


def get_shared_codes(seasonal_df: pd.DataFrame) -> set[int]:
    """Get codes that are present in ALL models."""
    codes_per_model = seasonal_df.groupby("model")["code"].apply(set)
    if codes_per_model.empty:
        return set()
    return set.intersection(*codes_per_model)


# =============================================================================
# SEASONAL FORECAST RECONSTRUCTION
# =============================================================================


def reconstruct_seasonal_forecasts(
    predictions_df: pd.DataFrame,
    monthly_obs: pd.DataFrame,
    issue_dates: dict,
) -> pd.DataFrame:
    """
    Reconstruct seasonal forecasts by averaging monthly predictions.

    Uses vectorized operations for performance.

    Args:
        predictions_df: DataFrame with monthly predictions (from load_predictions)
        monthly_obs: DataFrame with monthly observations (from calculate_target)
        issue_dates: Dictionary mapping (issue_month, issue_day) to config

    Returns:
        DataFrame with columns: code, issue_year, issue_month, issue_day,
                               issue_label, Q_pred_seasonal, Q_obs_seasonal, model
    """
    results = []

    for (issue_month, issue_day), config in issue_dates.items():
        target_months = config["target_months"]
        label = config["label"]
        horizons_needed = [get_horizon(issue_month, tm) for tm in target_months]

        logger.info(f"Processing issue date {label} -> targets {target_months}")

        # Filter predictions for this issue month and required horizons
        issue_preds = predictions_df[
            (predictions_df["issue_date"].dt.month == issue_month)
            & (predictions_df["horizon"].isin(horizons_needed))
        ].copy()

        if issue_preds.empty:
            logger.warning(f"No predictions found for issue month {issue_month}")
            continue

        issue_preds["issue_year"] = issue_preds["issue_date"].dt.year

        # Vectorized seasonal averaging per (model, code, issue_year)
        seasonal_pred = (
            issue_preds.groupby(["model", "code", "issue_year"])["Q_pred"]
            .mean()
            .reset_index()
            .rename(columns={"Q_pred": "Q_pred_seasonal"})
        )

        # Get seasonal observations - filter for target months and average
        obs_target = monthly_obs[monthly_obs["month"].isin(target_months)].copy()
        seasonal_obs = (
            obs_target.groupby(["code", "year"])["Q_obs_monthly"]
            .mean()
            .reset_index()
            .rename(columns={"year": "issue_year", "Q_obs_monthly": "Q_obs_seasonal"})
        )

        # Merge predictions with observations
        seasonal = seasonal_pred.merge(
            seasonal_obs, on=["code", "issue_year"], how="left"
        )
        seasonal["issue_month"] = issue_month
        seasonal["issue_day"] = issue_day
        seasonal["issue_label"] = label
        seasonal["n_target_months"] = len(target_months)

        results.append(seasonal)

    if not results:
        logger.warning("No seasonal forecasts reconstructed")
        return pd.DataFrame()

    result_df = pd.concat(results, ignore_index=True)
    logger.info(
        f"Reconstructed {len(result_df)} seasonal forecasts for "
        f"{result_df['model'].nunique()} models"
    )
    return result_df


# =============================================================================
# DIRECT SEASONAL MODEL LOADING
# =============================================================================


def load_seasonal_model_hindcasts(
    base_path: Path,
    issue_months: list[int],
    models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load direct seasonal model hindcast predictions.

    Seasonal models predict Apr-Sep discharge directly (not monthly).
    Hindcast files are stored at:
        {base_path}/seasonal_{month_name}/{model}/{model}_hindcast.csv

    Args:
        base_path: Base directory for predictions (e.g., long_term_predictions/)
        issue_months: List of issue months to load (1=Jan, 2=Feb, etc.)
        models: List of model names to load. If None, loads all available.

    Returns:
        DataFrame with columns: code, issue_year, issue_month, issue_day,
                               issue_label, Q_pred_seasonal, model
    """
    results = []

    for issue_month in issue_months:
        if issue_month not in SEASONAL_DIR_NAMES:
            continue

        dir_name = SEASONAL_DIR_NAMES[issue_month]
        seasonal_dir = Path(base_path) / dir_name

        if not seasonal_dir.exists():
            logger.debug(f"Seasonal directory not found: {seasonal_dir}")
            continue

        model_dirs = [d for d in seasonal_dir.iterdir() if d.is_dir()]

        for model_dir in model_dirs:
            model_name = model_dir.name

            if models is not None and model_name not in models:
                continue

            hindcast_file = model_dir / f"{model_name}_hindcast.csv"
            if not hindcast_file.exists():
                logger.debug(f"Hindcast file not found: {hindcast_file}")
                continue

            try:
                df = pd.read_csv(hindcast_file)
            except Exception as e:
                logger.warning(f"Failed to read {hindcast_file}: {e}")
                continue

            if df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"])
            df["issue_year"] = df["date"].dt.year
            df["issue_month"] = df["date"].dt.month
            df["issue_day"] = df["date"].dt.day

            month_abbrev = MONTH_ABBREV.get(issue_month, str(issue_month))
            df["issue_label"] = f"{month_abbrev} 25"

            pred_col = f"Q_{model_name}"
            if pred_col not in df.columns:
                q_cols = [c for c in df.columns if c.startswith("Q_")]
                if q_cols:
                    pred_col = q_cols[0]
                else:
                    logger.warning(f"No prediction column found in {hindcast_file}")
                    continue

            df["Q_pred_seasonal"] = df[pred_col]
            df["model"] = f"{model_name}{SEASONAL_MODEL_SUFFIX}"

            result = df[
                [
                    "code",
                    "issue_year",
                    "issue_month",
                    "issue_day",
                    "issue_label",
                    "Q_pred_seasonal",
                    "model",
                ]
            ].copy()

            results.append(result)
            logger.info(
                f"Loaded {len(result)} seasonal hindcasts for {model_name} "
                f"({month_abbrev})"
            )

    if not results:
        logger.info("No direct seasonal model hindcasts found")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    logger.info(
        f"Loaded {len(combined)} total seasonal hindcasts for "
        f"{combined['model'].nunique()} models"
    )
    return combined


def get_seasonal_observations(
    monthly_obs: pd.DataFrame,
    target_months: list[int] | None = None,
) -> pd.DataFrame:
    """
    Calculate seasonal observations by averaging monthly values for Apr-Sep.

    Args:
        monthly_obs: DataFrame with monthly observations (from calculate_target)
                    Expected columns: code, year, month, Q_obs_monthly
        target_months: Months to average (default: [4, 5, 6, 7, 8, 9] for Apr-Sep)

    Returns:
        DataFrame with columns: code, year, Q_obs_seasonal
    """
    if target_months is None:
        target_months = [4, 5, 6, 7, 8, 9]

    filtered = monthly_obs[monthly_obs["month"].isin(target_months)].copy()

    if filtered.empty:
        return pd.DataFrame()

    seasonal_obs = (
        filtered.groupby(["code", "year"])["Q_obs_monthly"].mean().reset_index()
    )
    seasonal_obs.columns = ["code", "year", "Q_obs_seasonal"]

    return seasonal_obs


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_seasonal_scatter(
    seasonal_df: pd.DataFrame,
    models: list[str],
    issue_label: str,
    output_path: Path,
    forecast_type: str = "6-month",
    code_filter: int | None = SCATTER_PLOT_CODE,
) -> plt.Figure:
    """
    Create scatter plot of predicted vs observed seasonal discharge.

    Args:
        seasonal_df: DataFrame with Q_pred_seasonal, Q_obs_seasonal, model
        models: List of models to plot
        issue_label: Issue date label (e.g., "Jan 25")
        output_path: Path to save figure
        forecast_type: "6-month" or "3-month"
        code_filter: If specified, only plot this specific code (basin)

    Returns:
        matplotlib Figure
    """
    df = seasonal_df[seasonal_df["issue_label"] == issue_label].copy()

    if df.empty:
        logger.warning(f"No data for issue date {issue_label}")
        return plt.figure()

    if code_filter is not None:
        df = df[df["code"] == code_filter].copy()
        if df.empty:
            logger.warning(
                f"No data for code {code_filter} at issue date {issue_label}"
            )
            return plt.figure()

    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = sns.color_palette("husl", n_models)

    for idx, model in enumerate(models):
        ax = axes[idx]
        plot_data = df[df["model"] == model]
        x_col = "Q_obs_seasonal"
        y_col = "Q_pred_seasonal"
        color = colors[idx]

        if plot_data.empty:
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

        plot_data = plot_data.dropna(subset=[x_col, y_col])

        if plot_data.empty:
            ax.text(
                0.5,
                0.5,
                f"No valid data\n{model}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        x = plot_data[x_col].values
        y = plot_data[y_col].values

        metrics = calculate_metrics(x, y)

        ax.scatter(
            x, y, alpha=0.6, color=color, s=40, edgecolors="white", linewidths=0.5
        )

        max_val = max(x.max(), y.max()) * 1.1
        min_val = min(x.min(), y.min()) * 0.9
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5)

        slope, intercept, *_ = stats.linregress(x, y)
        x_line = np.array([min_val, max_val])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2, alpha=0.8)

        ax.set_xlabel("Observed (m³/s)", fontsize=10)
        ax.set_ylabel("Predicted (m³/s)", fontsize=10)
        ax.set_title(
            f"{model}\nR²={metrics['r2']:.3f}, n={metrics['n_samples']}", fontsize=11
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    code_str = f" - Code {code_filter}" if code_filter is not None else ""
    fig.suptitle(
        f"{forecast_type} Seasonal Forecast - Issue Date: {issue_label}{code_str}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved scatter plot to {output_path}")

    return fig


def plot_r2_vs_issue_date(
    seasonal_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
    forecast_type: str = "6-month",
) -> plt.Figure:
    """
    Plot R² distribution across codes for each model and issue date.

    Shows boxplot of per-code R² values to visualize skill distribution.

    Args:
        seasonal_df: DataFrame with seasonal forecasts
        models: List of models to plot
        output_path: Path to save figure
        forecast_type: "6-month" or "3-month"

    Returns:
        matplotlib Figure
    """
    plot_data = []
    issue_labels = sorted(seasonal_df["issue_label"].unique())

    for model in models:
        model_df = seasonal_df[seasonal_df["model"] == model]

        for label in issue_labels:
            label_df = model_df[model_df["issue_label"] == label]

            if label_df.empty:
                continue

            r2_per_code = calculate_r2_per_code(
                label_df, obs_col="Q_obs_seasonal", pred_col="Q_pred_seasonal"
            )

            for _, row in r2_per_code.iterrows():
                plot_data.append(
                    {
                        "Issue Date": label,
                        "R²": row["r2"],
                        "Model": model,
                        "code": row["code"],
                    }
                )

    if not plot_data:
        logger.warning("No data for R² vs issue date plot")
        return plt.figure()

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("husl", len(models))

    sns.boxplot(
        data=plot_df,
        x="Issue Date",
        y="R²",
        hue="Model",
        hue_order=models,
        palette=colors,
        ax=ax,
        width=0.7,
    )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_ylabel("R² (per basin)", fontsize=12)
    ax.set_xlabel("Forecast Issue Date", fontsize=12)
    ax.set_title(
        f"{forecast_type} Forecast Skill Distribution by Issue Date",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved R² vs issue date plot to {output_path}")

    return fig


def plot_skill_comparison(
    seasonal_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> plt.Figure:
    """
    Create boxplot comparing R² distribution per code for ML models.

    Args:
        seasonal_df: DataFrame with seasonal forecasts
        models: List of ML models
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    plot_data = []

    for model in models:
        model_df = seasonal_df[seasonal_df["model"] == model]

        if model_df.empty:
            continue

        r2_per_code = calculate_r2_per_code(
            model_df, obs_col="Q_obs_seasonal", pred_col="Q_pred_seasonal"
        )

        for _, row in r2_per_code.iterrows():
            plot_data.append(
                {
                    "R²": row["r2"],
                    "Model": model,
                    "code": row["code"],
                }
            )

    if not plot_data:
        logger.warning("No data for skill comparison plot")
        return plt.figure()

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("husl", len(models))

    sns.boxplot(
        data=plot_df,
        x="Model",
        y="R²",
        hue="Model",
        order=models,
        palette=colors,
        ax=ax,
        width=0.6,
        legend=False,
    )

    sns.stripplot(
        data=plot_df,
        x="Model",
        y="R²",
        order=models,
        ax=ax,
        color="black",
        alpha=0.3,
        size=3,
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel("R² (per basin)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "ML Model Comparison - R² Distribution Across Basins",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved skill comparison plot to {output_path}")

    return fig


# =============================================================================
# METRICS SUMMARY
# =============================================================================


def generate_metrics_summary(
    seasonal_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate CSV summary of per-code R² statistics per model and issue date.

    Args:
        seasonal_df: DataFrame with seasonal forecasts
        models: List of ML models
        output_path: Path to save CSV

    Returns:
        Summary DataFrame
    """
    results = []
    issue_labels = seasonal_df["issue_label"].unique()

    for label in issue_labels:
        for model in models:
            model_df = seasonal_df[
                (seasonal_df["model"] == model) & (seasonal_df["issue_label"] == label)
            ]

            if model_df.empty:
                continue

            r2_per_code = calculate_r2_per_code(
                model_df, obs_col="Q_obs_seasonal", pred_col="Q_pred_seasonal"
            )

            if r2_per_code.empty:
                continue

            r2_values = r2_per_code["r2"].dropna()

            results.append(
                {
                    "issue_label": label,
                    "model": model,
                    "n_codes": len(r2_values),
                    "r2_mean": r2_values.mean(),
                    "r2_median": r2_values.median(),
                    "r2_std": r2_values.std(),
                    "r2_min": r2_values.min(),
                    "r2_max": r2_values.max(),
                    "r2_q25": r2_values.quantile(0.25),
                    "r2_q75": r2_values.quantile(0.75),
                }
            )

    if not results:
        return pd.DataFrame()

    summary_df = pd.DataFrame(results)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Saved metrics summary to {output_path}")

    return summary_df


# =============================================================================
# MAIN ORCHESTRATION
# =============================================================================


def process_seasonal_forecasts(
    region: str,
    forecast_type: str,
    models: list[str],
    output_dir: Path,
) -> None:
    """
    Main processing function for seasonal forecast visualization.

    Args:
        region: "Kyrgyzstan" or "Tajikistan"
        forecast_type: "6_month", "3_month", or "both"
        models: List of models to evaluate
        output_dir: Base output directory
    """
    if region == "Tajikistan":
        pred_config = taj_path_config
    else:
        pred_config = kgz_path_config

    horizons = list(day_of_forecast.keys())
    logger.info(f"Loading predictions for {region}...")

    predictions_df = load_predictions(
        base_path=pred_config["pred_dir"],
        horizons=horizons,
    )

    if predictions_df.empty:
        logger.error("No predictions loaded. Exiting.")
        return

    logger.info(f"Loading observations from {pred_config['obs_file']}...")
    obs_df = load_observations(pred_config["obs_file"])
    monthly_obs = calculate_target_with_threshold(obs_df)

    forecast_configs = []
    if forecast_type in ["6_month", "both"]:
        forecast_configs.append(("6_month", ISSUE_DATES_6M))
    if forecast_type in ["3_month", "both"]:
        forecast_configs.append(("3_month", ISSUE_DATES_3M))

    for fc_type, issue_dates in forecast_configs:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {fc_type} forecasts")
        logger.info(f"{'=' * 60}")

        fc_output_dir = output_dir / region.lower() / "seasonal" / fc_type
        fc_output_dir.mkdir(parents=True, exist_ok=True)

        # Reconstruct seasonal forecasts from monthly predictions
        seasonal_df = reconstruct_seasonal_forecasts(
            predictions_df, monthly_obs, issue_dates
        )

        # Load direct seasonal model hindcasts
        issue_months_list = [k[0] for k in issue_dates.keys()]
        seasonal_model_df = load_seasonal_model_hindcasts(
            base_path=pred_config["pred_dir"],
            issue_months=issue_months_list,
            models=models,
        )

        # Merge seasonal model predictions with observations
        if not seasonal_model_df.empty:
            target_months = list(issue_dates.values())[0]["target_months"]
            seasonal_obs_for_models = get_seasonal_observations(
                monthly_obs, target_months=target_months
            )

            if not seasonal_obs_for_models.empty:
                seasonal_model_df = seasonal_model_df.merge(
                    seasonal_obs_for_models,
                    left_on=["code", "issue_year"],
                    right_on=["code", "year"],
                    how="left",
                )
                if "year" in seasonal_model_df.columns:
                    seasonal_model_df = seasonal_model_df.drop(columns=["year"])
            else:
                seasonal_model_df["Q_obs_seasonal"] = np.nan

            if not seasonal_df.empty:
                seasonal_df = pd.concat(
                    [seasonal_df, seasonal_model_df], ignore_index=True
                )
            else:
                seasonal_df = seasonal_model_df

        if seasonal_df.empty:
            logger.warning(f"No seasonal forecasts for {fc_type}")
            continue

        # Filter to requested models (including seasonal variants)
        seasonal_model_names = [f"{m}{SEASONAL_MODEL_SUFFIX}" for m in models]
        all_model_names = models + seasonal_model_names
        available_models = [
            m for m in all_model_names if m in seasonal_df["model"].unique()
        ]
        if not available_models:
            logger.warning(f"None of the requested models found in {fc_type} forecasts")
            continue

        # Filter to shared codes across all models
        filtered_df = seasonal_df[seasonal_df["model"].isin(available_models)]
        shared_codes = get_shared_codes(filtered_df)
        if not shared_codes:
            logger.warning("No shared codes found across all models")
            continue

        seasonal_df = seasonal_df[seasonal_df["code"].isin(shared_codes)]
        logger.info(f"Filtered to {len(shared_codes)} shared codes across all models")
        logger.info(f"Available models: {available_models}")

        # Generate plots
        logger.info("Generating plots...")

        # 1. Scatter plots per issue date
        for config in issue_dates.values():
            label = config["label"]
            label_clean = label.replace(" ", "")

            fig = plot_seasonal_scatter(
                seasonal_df=seasonal_df,
                models=available_models,
                issue_label=label,
                output_path=fc_output_dir / f"scatter_{label_clean}.png",
                forecast_type=fc_type.replace("_", "-"),
            )
            plt.close(fig)

        # 2. R² distribution vs issue date
        fig = plot_r2_vs_issue_date(
            seasonal_df=seasonal_df,
            models=available_models,
            output_path=fc_output_dir / "r2_distribution_vs_issue_date.png",
            forecast_type=fc_type.replace("_", "-"),
        )
        plt.close(fig)

        # 3. Skill comparison plot
        fig = plot_skill_comparison(
            seasonal_df=seasonal_df,
            models=available_models,
            output_path=fc_output_dir / "skill_comparison.png",
        )
        plt.close(fig)

        # 4. Metrics summary CSV
        generate_metrics_summary(
            seasonal_df=seasonal_df,
            models=available_models,
            output_path=fc_output_dir / "metrics_summary.csv",
        )

    logger.info("\nProcessing complete!")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate seasonal forecasts and compare ML models."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["Kyrgyzstan", "Tajikistan"],
        default="Kyrgyzstan",
        help="Region to process (default: Kyrgyzstan)",
    )
    parser.add_argument(
        "--forecast-type",
        type=str,
        choices=["6_month", "3_month", "both"],
        default="both",
        help="Forecast type to evaluate (default: both)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=DEFAULT_MODELS,
        help="List of models to evaluate",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )

    args = parser.parse_args()

    set_global_plot_style()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(default_output_dir)

    logger.info(f"Region: {args.region}")
    logger.info(f"Forecast type: {args.forecast_type}")
    logger.info(f"Models: {args.models}")
    logger.info(f"Output directory: {output_dir}")

    process_seasonal_forecasts(
        region=args.region,
        forecast_type=args.forecast_type,
        models=args.models,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
