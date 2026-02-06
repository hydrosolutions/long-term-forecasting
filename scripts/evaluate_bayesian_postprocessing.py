"""
Evaluation Script for Bayesian Post-Processing of Monthly Discharge Forecasts

This script compares raw hindcast predictions with Bayesian ratio-corrected predictions
to evaluate the effectiveness of the post-processing approach.

The ratio-based correction transforms forecasts to ratio space (R = Q_curr / Q_prev),
applies Bayesian updating with historical ratio statistics, and transforms back.

Usage:
    python evaluate_bayesian_postprocessing.py [--region Kyrgyzstan] [--prior-strength 1.0]

    # Strong prior (more weight to historical ratios)
    python evaluate_bayesian_postprocessing.py --prior-strength 2.0

    # Weak prior (more weight to forecast)
    python evaluate_bayesian_postprocessing.py --prior-strength 0.5

    # Disable soft bounds
    python evaluate_bayesian_postprocessing.py --no-soft-bounds

Output:
    - metrics_comparison.csv: Per-basin, per-horizon metrics for raw and corrected
    - metrics_by_month.csv: Per-basin, per-month metric breakdowns
    - ratio_statistics.csv: Historical ratio statistics
    - plots/: Time series comparison and diagnostic plots
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import r2_score, mean_squared_error

# Setup logging
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

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables
load_dotenv(dotenv_path=project_root / ".env")

from lt_forecasting.scr.bayesian_post_processing import (  # noqa: E402
    RatioPriorConfig,
    infer_q_columns,
    compute_historical_ratios,
    apply_ratio_bayesian_correction,
)

# =============================================================================
# Configuration
# =============================================================================

# Path configurations from environment variables
kgz_path_config = {
    "pred_dir": os.getenv("kgz_path_discharge"),
    "obs_file": os.getenv("kgz_path_base_pred"),
}

taj_path_config = {
    "pred_dir": os.getenv("taj_path_base_pred"),
    "obs_file": os.getenv("taj_path_discharge"),
}

output_dir = os.getenv("out_dir_op_lt")

# Forecast issue day configuration
day_of_forecast = {
    "month_0": 10,
    "month_1": 25,
    "month_2": 25,
    "month_3": 25,
    "month_4": 25,
    "month_5": 25,
    "month_6": 25,
    "month_7": 25,
    "month_8": 25,
    "month_9": 25,
}

# Default configuration
DEFAULT_CONFIG = {
    "region": "Kyrgyzstan",  # "Kyrgyzstan" or "Tajikistan"
    # Models to evaluate (must match model folder names)
    "model_names": ["MC_ALD"],
    # Horizons to test
    "horizons": [
        "month_0",
        "month_1",
        "month_2",
        "month_3",
        "month_4",
        "month_5",
        "month_6",
        "month_7",
        "month_8",
        "month_9",
    ],
    # Minimum observations required for Bayesian correction
    "min_observations": 5,
    # Prior configuration for ratio-based correction
    # prior_precision_scale: 1.0 = natural, >1 = stronger prior, <1 = weaker prior
    "prior_precision_scale": 1.0,
    "apply_soft_bounds": True,
    # Peak capping: prevent forecasts from exceeding seasonal peak during recession
    "apply_peak_capping": True,
    "peak_cap_softness": 0.1,  # 0=hard cap, 1=no cap
}


# =============================================================================
# Data Loading Functions (adapted from examine_operational_fc.py)
# =============================================================================


def load_observations(obs_file: str) -> pd.DataFrame:
    """
    Load observed discharge data from a CSV file.

    Args:
        obs_file: Path to the CSV file containing observed discharge data.

    Returns:
        DataFrame with columns: date, code, discharge (daily observations)
    """
    logger.info(f"Loading observations from: {obs_file}")
    obs_df = pd.read_csv(obs_file)
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    obs_df["code"] = obs_df["code"].astype(int)

    logger.info(f"Loaded {len(obs_df)} daily observations")
    return obs_df


def calculate_monthly_target(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily observations to monthly means for each code.

    Args:
        obs: DataFrame with columns: date, code, discharge (daily observations)

    Returns:
        DataFrame with columns: code, year, month, Q_obs (monthly mean discharge)
    """
    obs = obs.copy()
    obs["year"] = obs["date"].dt.year
    obs["month"] = obs["date"].dt.month

    monthly_obs = (
        obs.groupby(["code", "year", "month"])["discharge"]
        .mean()
        .reset_index()
        .rename(columns={"discharge": "Q_obs"})
    )

    logger.info(f"Calculated {len(monthly_obs)} monthly observations")
    return monthly_obs


def load_hindcast_predictions(
    base_path: str,
    horizons: List[str],
    model_names: List[str],
) -> Dict[int, pd.DataFrame]:
    """
    Load hindcast predictions organized by horizon.

    Args:
        base_path: Base directory containing horizon subdirectories
        horizons: List of horizon identifiers (e.g., ["month_0", "month_1", ...])
        model_names: List of model names to load

    Returns:
        Dictionary mapping horizon number to DataFrame with predictions
    """
    base_path = Path(base_path)
    predictions_by_horizon = {}

    for horizon in horizons:
        horizon_path = base_path / horizon
        if not horizon_path.exists():
            logger.warning(f"Horizon directory not found: {horizon_path}")
            continue

        horizon_num = int(horizon.split("_")[1])
        forecast_day = day_of_forecast.get(horizon, 10)

        all_predictions = []

        for model_name in model_names:
            model_dir = horizon_path / model_name
            hindcast_file = model_dir / f"{model_name}_hindcast.csv"

            if not hindcast_file.exists():
                logger.warning(f"Hindcast file not found: {hindcast_file}")
                continue

            try:
                df = pd.read_csv(hindcast_file)
            except Exception as e:
                logger.error(f"Failed to read {hindcast_file}: {e}")
                continue

            if df.empty:
                continue

            # Convert dates
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df["valid_from"] = pd.to_datetime(df["valid_from"], format="mixed")
            df["valid_to"] = pd.to_datetime(df["valid_to"], format="mixed")
            df["code"] = df["code"].astype(int)

            # Filter to forecast day
            df = df[df["date"].dt.day == forecast_day].copy()

            if df.empty:
                continue

            # Find prediction column and quantile columns
            quantile_cols = [col for col in df.columns if re.fullmatch(r"Q\d+", col)]
            pred_col = f"Q_{model_name}"

            if pred_col not in df.columns:
                # Try to find any Q_ column that's not a quantile
                q_cols = [
                    c
                    for c in df.columns
                    if c.startswith("Q_") and c not in quantile_cols and c != "Q_obs"
                ]
                if q_cols:
                    pred_col = q_cols[0]
                else:
                    logger.warning(f"No prediction column found for {model_name}")
                    continue

            # Build result DataFrame
            result_df = pd.DataFrame(
                {
                    "date": df["date"],
                    "code": df["code"],
                    "valid_from": df["valid_from"],
                    "valid_to": df["valid_to"],
                    "Q50": df[pred_col] if pred_col in df.columns else np.nan,
                    "model": model_name,
                }
            )

            # Add quantile columns
            for q_col in quantile_cols:
                if q_col in df.columns:
                    result_df[q_col] = df[q_col].values

            # Add Q_obs if available
            if "Q_obs" in df.columns:
                result_df["Q_obs_from_file"] = df["Q_obs"].values

            all_predictions.append(result_df)

        if all_predictions:
            combined = pd.concat(all_predictions, ignore_index=True)

            # Add target month/year
            combined["target_month"] = combined["valid_from"].dt.month
            combined["target_year"] = combined["valid_from"].dt.year

            predictions_by_horizon[horizon_num] = combined
            logger.info(f"Horizon {horizon_num}: Loaded {len(combined)} predictions")

    return predictions_by_horizon


def merge_predictions_with_observations(
    predictions: pd.DataFrame,
    monthly_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge predictions with monthly observations.

    Args:
        predictions: DataFrame with predictions
        monthly_obs: DataFrame with monthly observations

    Returns:
        Merged DataFrame with Q_obs column
    """
    merged = predictions.merge(
        monthly_obs,
        left_on=["code", "target_year", "target_month"],
        right_on=["code", "year", "month"],
        how="left",
    )

    # Drop redundant columns
    merged = merged.drop(columns=["year", "month"], errors="ignore")

    logger.info(
        f"Merged: {merged['Q_obs'].notna().sum()} / {len(merged)} have observations"
    )

    return merged


# =============================================================================
# Metric Calculation Functions
# =============================================================================


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate various performance metrics.
    """
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    if len(y_true_clean) < 2:
        return {
            "r2": np.nan,
            "rmse": np.nan,
            "nrmse": np.nan,
            "mae": np.nan,
            "nmae": np.nan,
            "bias": np.nan,
            "n_samples": len(y_true_clean),
        }

    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    bias = np.mean(y_pred_clean - y_true_clean)

    mean_obs = y_true_clean.mean()
    nrmse = rmse / mean_obs if mean_obs != 0 else np.nan
    nmae = mae / mean_obs if mean_obs != 0 else np.nan

    return {
        "r2": r2,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "nmae": nmae,
        "bias": bias,
        "n_samples": len(y_true_clean),
    }


def calculate_coverage(
    observed: np.ndarray, q_low: np.ndarray, q_high: np.ndarray
) -> float:
    """Calculate coverage - fraction of observations within interval."""
    valid = ~(np.isnan(observed) | np.isnan(q_low) | np.isnan(q_high))
    if np.sum(valid) == 0:
        return np.nan
    in_interval = (observed[valid] >= q_low[valid]) & (observed[valid] <= q_high[valid])
    return np.mean(in_interval)


def calculate_comparison_metrics(
    raw_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    pred_col: str = "Q50",
    obs_col: str = "Q_obs",
) -> pd.DataFrame:
    """
    Calculate and compare metrics for raw vs corrected forecasts.
    """
    results = []

    for code in raw_df["code"].unique():
        raw_basin = raw_df[raw_df["code"] == code]
        corr_basin = corrected_df[corrected_df["code"] == code]

        if len(raw_basin) < 5:
            continue

        # Calculate deterministic metrics
        raw_metrics = calculate_metrics(raw_basin[obs_col], raw_basin[pred_col])
        corr_metrics = calculate_metrics(corr_basin[obs_col], corr_basin[pred_col])

        # Calculate coverage if quantiles available
        quantile_cols = infer_q_columns(raw_df)
        if "Q5" in quantile_cols and "Q95" in quantile_cols:
            raw_metrics["coverage_90"] = calculate_coverage(
                raw_basin[obs_col].values,
                raw_basin["Q5"].values,
                raw_basin["Q95"].values,
            )
            corr_metrics["coverage_90"] = calculate_coverage(
                corr_basin[obs_col].values,
                corr_basin["Q5"].values,
                corr_basin["Q95"].values,
            )

        # Build comparison rows
        for metric_name in raw_metrics:
            if metric_name == "n_samples":
                continue

            raw_val = raw_metrics[metric_name]
            corr_val = corr_metrics.get(metric_name, np.nan)

            # Calculate improvement
            if np.isnan(raw_val) or np.isnan(corr_val):
                improvement = np.nan
                pct_improvement = np.nan
            elif metric_name in ["mae", "rmse", "nrmse", "nmae"]:
                # Lower is better
                improvement = raw_val - corr_val
                pct_improvement = (
                    (improvement / abs(raw_val) * 100) if raw_val != 0 else np.nan
                )
            elif metric_name == "bias":
                # Closer to 0 is better
                improvement = abs(raw_val) - abs(corr_val)
                pct_improvement = (
                    (improvement / abs(raw_val) * 100) if raw_val != 0 else np.nan
                )
            elif metric_name == "coverage_90":
                # Closer to 0.90 is better
                raw_dist = abs(raw_val - 0.90)
                corr_dist = abs(corr_val - 0.90)
                improvement = raw_dist - corr_dist
                pct_improvement = np.nan
            else:
                # Higher is better (r2)
                improvement = corr_val - raw_val
                pct_improvement = (
                    (improvement / abs(raw_val) * 100) if raw_val != 0 else np.nan
                )

            results.append(
                {
                    "code": code,
                    "metric": metric_name,
                    "raw_value": raw_val,
                    "corrected_value": corr_val,
                    "improvement": improvement,
                    "pct_improvement": pct_improvement,
                }
            )

    return pd.DataFrame(results)


def calculate_metrics_by_month(
    raw_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    pred_col: str = "Q50",
    obs_col: str = "Q_obs",
) -> pd.DataFrame:
    """
    Calculate metrics broken down by target month and basin.

    This helps identify seasonal patterns in forecast performance.

    Args:
        raw_df: DataFrame with raw forecasts
        corrected_df: DataFrame with corrected forecasts
        pred_col: Column name for predictions
        obs_col: Column name for observations

    Returns:
        DataFrame with metrics per basin and month
    """
    results = []

    for code in raw_df["code"].unique():
        raw_basin = raw_df[raw_df["code"] == code]
        corr_basin = corrected_df[corrected_df["code"] == code]

        for month in range(1, 13):
            raw_month = raw_basin[raw_basin["target_month"] == month]
            corr_month = corr_basin[corr_basin["target_month"] == month]

            if len(raw_month) < 2:
                continue

            raw_metrics = calculate_metrics(raw_month[obs_col], raw_month[pred_col])
            corr_metrics = calculate_metrics(corr_month[obs_col], corr_month[pred_col])

            for metric_name in ["r2", "rmse", "mae", "bias", "nrmse"]:
                raw_val = raw_metrics.get(metric_name, np.nan)
                corr_val = corr_metrics.get(metric_name, np.nan)

                # Calculate improvement
                if np.isnan(raw_val) or np.isnan(corr_val):
                    improvement = np.nan
                elif metric_name in ["mae", "rmse", "nrmse"]:
                    improvement = raw_val - corr_val
                elif metric_name == "bias":
                    improvement = abs(raw_val) - abs(corr_val)
                else:
                    improvement = corr_val - raw_val

                results.append(
                    {
                        "code": code,
                        "target_month": month,
                        "metric": metric_name,
                        "raw_value": raw_val,
                        "corrected_value": corr_val,
                        "improvement": improvement,
                        "n_samples": raw_metrics.get("n_samples", 0),
                    }
                )

    return pd.DataFrame(results)


def log_ratio_statistics(
    ratio_stats: pd.DataFrame,
    save_path: Optional[str] = None,
) -> None:
    """
    Log and optionally save historical ratio statistics.

    Args:
        ratio_stats: DataFrame with ratio statistics from compute_historical_ratios
        save_path: Optional path to save statistics CSV
    """
    logger.info("\n" + "=" * 60)
    logger.info("HISTORICAL RATIO STATISTICS")
    logger.info("=" * 60)

    # Summary by month
    if "target_month" in ratio_stats.columns:
        monthly_summary = (
            ratio_stats.groupby("target_month")
            .agg(
                {
                    "mean_ratio": ["mean", "std"],
                    "median_ratio": "mean",
                    "n_samples": "sum",
                }
            )
            .round(3)
        )
        monthly_summary.columns = [
            "mean_ratio_avg",
            "mean_ratio_std",
            "median_ratio_avg",
            "total_samples",
        ]

        logger.info("\nBy Month:")
        for month in range(1, 13):
            if month in monthly_summary.index:
                row = monthly_summary.loc[month]
                logger.info(
                    f"  Month {month:2d}: mean_ratio={row['mean_ratio_avg']:.3f} "
                    f"(+/- {row['mean_ratio_std']:.3f}), "
                    f"n={int(row['total_samples'])}"
                )

    # Summary by basin
    if "code" in ratio_stats.columns:
        logger.info("\nBy Basin (top 5 by sample size):")
        basin_summary = (
            ratio_stats.groupby("code")
            .agg({"mean_ratio": "mean", "n_samples": "sum"})
            .sort_values("n_samples", ascending=False)
            .head(5)
        )
        for code, row in basin_summary.iterrows():
            logger.info(
                f"  Basin {code}: mean_ratio={row['mean_ratio']:.3f}, "
                f"n={int(row['n_samples'])}"
            )

    if save_path:
        ratio_stats.to_csv(save_path, index=False)
        logger.info(f"\nSaved ratio statistics to: {save_path}")


# =============================================================================
# Visualization Functions
# =============================================================================


def plot_ratio_distributions_by_month(
    ratio_stats: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create boxplots showing historical ratio distributions by month.

    Args:
        ratio_stats: DataFrame with ratio statistics including 'target_month' and ratio values
        output_path: Path to save the plot
    """
    if ratio_stats.empty or "target_month" not in ratio_stats.columns:
        logger.warning("No ratio statistics available for plotting")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Prepare data for boxplot
    months = sorted(ratio_stats["target_month"].unique())
    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    # Left plot: Mean ratio by month (boxplot across basins)
    ax1 = axes[0]
    data_by_month = [
        ratio_stats[ratio_stats["target_month"] == m]["mean_ratio"].dropna().values
        for m in months
    ]
    bp = ax1.boxplot(
        data_by_month,
        labels=[month_labels[m - 1] for m in months],
        patch_artist=True,
    )

    # Color by season
    colors = []
    for m in months:
        if m in [12, 1, 2]:
            colors.append("lightblue")  # Winter
        elif m in [3, 4, 5]:
            colors.append("lightgreen")  # Spring
        elif m in [6, 7, 8]:
            colors.append("lightyellow")  # Summer
        else:
            colors.append("orange")  # Fall

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    ax1.axhline(y=1.0, color="red", linestyle="--", alpha=0.7, label="Perfect ratio")
    ax1.set_xlabel("Month", fontsize=11)
    ax1.set_ylabel("Obs/Pred Ratio", fontsize=11)
    ax1.set_title(
        "Historical Ratio Distribution by Month", fontsize=12, fontweight="bold"
    )
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    # Right plot: Sample size by month
    ax2 = axes[1]
    sample_counts = ratio_stats.groupby("target_month")["n_samples"].sum()
    ax2.bar(
        [month_labels[m - 1] for m in months],
        [sample_counts.get(m, 0) for m in months],
        color="steelblue",
        alpha=0.7,
    )
    ax2.set_xlabel("Month", fontsize=11)
    ax2.set_ylabel("Total Samples", fontsize=11)
    ax2.set_title("Sample Size by Month", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Ratio Correction Diagnostics",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved ratio distributions plot to: {output_path}")


def plot_correction_strength_vs_sample_size(
    ratio_stats: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create scatter plot showing correction strength vs sample size.

    Args:
        ratio_stats: DataFrame with ratio statistics
        output_path: Path to save the plot
    """
    if ratio_stats.empty:
        logger.warning("No ratio statistics available for plotting")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    # Correction strength = abs(ratio - 1)
    ratio_stats = ratio_stats.copy()
    ratio_stats["correction_strength"] = abs(ratio_stats["mean_ratio"] - 1)

    # Scatter plot
    scatter = ax.scatter(
        ratio_stats["n_samples"],
        ratio_stats["correction_strength"],
        c=ratio_stats["target_month"],
        cmap="hsv",
        alpha=0.6,
        s=50,
    )

    # Add colorbar for months
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Month", fontsize=10)

    ax.set_xlabel("Sample Size (n)", fontsize=11)
    ax.set_ylabel("Correction Strength |ratio - 1|", fontsize=11)
    ax.set_title(
        "Correction Strength vs Sample Size",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add trend line
    if len(ratio_stats) > 2:
        z = np.polyfit(ratio_stats["n_samples"], ratio_stats["correction_strength"], 1)
        p = np.poly1d(z)
        x_range = np.linspace(
            ratio_stats["n_samples"].min(), ratio_stats["n_samples"].max(), 100
        )
        ax.plot(x_range, p(x_range), "r--", alpha=0.5, label="Trend")
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved correction strength plot to: {output_path}")


def plot_improvement_by_month(
    metrics_by_month_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create heatmap showing per-month improvement metrics.

    Args:
        metrics_by_month_df: DataFrame from calculate_metrics_by_month
        output_path: Path to save the plot
    """
    if metrics_by_month_df.empty:
        logger.warning("No monthly metrics available for plotting")
        return

    metrics_to_plot = ["mae", "rmse", "r2"]
    available_metrics = [
        m for m in metrics_to_plot if m in metrics_by_month_df["metric"].unique()
    ]

    if not available_metrics:
        return

    fig, axes = plt.subplots(
        1, len(available_metrics), figsize=(5 * len(available_metrics), 5)
    )

    if len(available_metrics) == 1:
        axes = [axes]

    month_labels = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_data = metrics_by_month_df[metrics_by_month_df["metric"] == metric]

        # Aggregate improvement by month
        monthly_improvement = (
            metric_data.groupby("target_month")["improvement"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )

        months = monthly_improvement["target_month"].values
        improvements = monthly_improvement["mean"].values
        stds = monthly_improvement["std"].values

        # Bar plot with error bars
        colors = ["green" if imp > 0 else "red" for imp in improvements]
        ax.bar(
            months,
            improvements,
            color=colors,
            alpha=0.7,
            yerr=stds,
            capsize=3,
        )

        ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(month_labels, rotation=45, ha="right")
        ax.set_xlabel("Target Month", fontsize=10)
        ax.set_ylabel(f"{metric.upper()} Improvement", fontsize=10)
        ax.set_title(f"{metric.upper()}", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Improvement by Target Month (positive = better)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved improvement by month plot to: {output_path}")


def plot_metric_distributions(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create distribution comparison plots for raw vs corrected metrics.

    Shows side-by-side boxplots and histograms comparing metric distributions.
    """
    # Metrics to plot (lower is better for these, higher for r2)
    metrics_to_plot = ["r2", "mae", "rmse", "bias", "nrmse"]
    available_metrics = [
        m for m in metrics_to_plot if m in metrics_df["metric"].unique()
    ]

    if not available_metrics:
        logger.warning("No metrics available for plotting")
        return

    n_metrics = len(available_metrics)
    _, axes = plt.subplots(2, n_metrics, figsize=(4 * n_metrics, 8))

    if n_metrics == 1:
        axes = axes.reshape(2, 1)

    for idx, metric in enumerate(available_metrics):
        metric_data = metrics_df[metrics_df["metric"] == metric]

        raw_values = metric_data["raw_value"].dropna().values
        corr_values = metric_data["corrected_value"].dropna().values

        # Top row: Boxplots
        ax_box = axes[0, idx]
        box_data = [raw_values, corr_values]
        bp = ax_box.boxplot(box_data, labels=["Raw", "Corrected"], patch_artist=True)
        bp["boxes"][0].set_facecolor("lightblue")
        bp["boxes"][1].set_facecolor("lightgreen")

        ax_box.set_title(f"{metric.upper()}", fontsize=12, fontweight="bold")
        ax_box.grid(True, alpha=0.3)

        # Add mean markers
        ax_box.scatter(
            [1, 2],
            [np.mean(raw_values), np.mean(corr_values)],
            color="red",
            marker="D",
            s=50,
            zorder=5,
            label="Mean",
        )

        # Bottom row: Histograms
        ax_hist = axes[1, idx]
        bins = np.linspace(
            min(np.min(raw_values), np.min(corr_values)),
            max(np.max(raw_values), np.max(corr_values)),
            15,
        )
        ax_hist.hist(raw_values, bins=bins, alpha=0.5, label="Raw", color="blue")
        ax_hist.hist(
            corr_values, bins=bins, alpha=0.5, label="Corrected", color="green"
        )
        ax_hist.set_xlabel(metric.upper(), fontsize=10)
        ax_hist.set_ylabel("Count", fontsize=10)
        ax_hist.legend(fontsize=8)
        ax_hist.grid(True, alpha=0.3)

    plt.suptitle(
        "Metric Distributions: Raw vs Bayesian Corrected",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved metric distributions plot to: {output_path}")


def plot_improvement_by_horizon(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create plot showing improvement by horizon for each metric.
    """
    metrics_to_plot = ["r2", "mae", "rmse", "bias"]
    available_metrics = [
        m for m in metrics_to_plot if m in metrics_df["metric"].unique()
    ]

    if not available_metrics:
        return

    horizons = sorted(metrics_df["horizon"].unique())

    _, axes = plt.subplots(
        1, len(available_metrics), figsize=(4 * len(available_metrics), 5)
    )

    if len(available_metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_data = metrics_df[metrics_df["metric"] == metric]

        # Calculate mean raw, corrected, and improvement per horizon
        raw_means = []
        corr_means = []
        raw_stds = []
        corr_stds = []

        for h in horizons:
            h_data = metric_data[metric_data["horizon"] == h]
            raw_means.append(h_data["raw_value"].mean())
            corr_means.append(h_data["corrected_value"].mean())
            raw_stds.append(h_data["raw_value"].std())
            corr_stds.append(h_data["corrected_value"].std())

        x = np.arange(len(horizons))
        width = 0.35

        ax.bar(
            x - width / 2,
            raw_means,
            width,
            label="Raw",
            color="lightblue",
            yerr=raw_stds,
            capsize=3,
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            corr_means,
            width,
            label="Corrected",
            color="lightgreen",
            yerr=corr_stds,
            capsize=3,
            alpha=0.8,
        )

        ax.set_xlabel("Horizon", fontsize=11)
        ax.set_ylabel(metric.upper(), fontsize=11)
        ax.set_title(f"{metric.upper()} by Horizon", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"h{h}" for h in horizons])
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(
        "Raw vs Corrected Metrics by Forecast Horizon", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved improvement by horizon plot to: {output_path}")


def plot_scatter_raw_vs_corrected(
    metrics_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Create scatter plots showing raw vs corrected metric values per basin.
    Points above the diagonal indicate improvement (for metrics where higher is better).
    """
    metrics_to_plot = ["r2", "mae", "rmse"]
    available_metrics = [
        m for m in metrics_to_plot if m in metrics_df["metric"].unique()
    ]

    if not available_metrics:
        return

    _, axes = plt.subplots(
        1, len(available_metrics), figsize=(5 * len(available_metrics), 5)
    )

    if len(available_metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(available_metrics):
        ax = axes[idx]
        metric_data = metrics_df[metrics_df["metric"] == metric]

        raw_vals = metric_data["raw_value"].values
        corr_vals = metric_data["corrected_value"].values
        horizons = metric_data["horizon"].values

        # Color by horizon
        unique_horizons = sorted(set(horizons))
        colors = plt.cm.viridis(np.linspace(0, 0.8, len(unique_horizons)))
        color_map = {h: colors[i] for i, h in enumerate(unique_horizons)}

        for h in unique_horizons:
            mask = horizons == h
            ax.scatter(
                raw_vals[mask],
                corr_vals[mask],
                c=[color_map[h]],
                label=f"h{h}",
                alpha=0.7,
                s=50,
            )

        # Add diagonal line
        lims = [
            min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1]),
        ]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1, label="No change")
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_xlabel(f"Raw {metric.upper()}", fontsize=11)
        ax.set_ylabel(f"Corrected {metric.upper()}", fontsize=11)
        ax.set_title(
            f"{metric.upper()}: Raw vs Corrected", fontsize=12, fontweight="bold"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Add annotation about which side is better
        if metric == "r2":
            ax.annotate(
                "Better →", xy=(0.05, 0.95), xycoords="axes fraction", fontsize=9
            )
        else:
            ax.annotate(
                "← Better",
                xy=(0.95, 0.05),
                xycoords="axes fraction",
                fontsize=9,
                ha="right",
            )

    plt.suptitle("Per-Basin Metric Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved scatter comparison plot to: {output_path}")


def plot_timeseries_comparison(
    raw_df: pd.DataFrame,
    corrected_df: pd.DataFrame,
    code: int,
    start_year: int,
    end_year: int,
    pred_col: str,
    obs_col: str,
    output_path: str,
) -> bool:
    """
    Plot time series comparison for a specific basin and year range.

    Uses valid_from (target month) on x-axis, not issue date.

    Returns True if plot was created, False if no data available.
    """
    # Use valid_from for filtering and plotting (target month, not issue date)
    raw_basin = raw_df[
        (raw_df["code"] == code)
        & (raw_df["valid_from"].dt.year >= start_year)
        & (raw_df["valid_from"].dt.year <= end_year)
    ].copy()

    corr_basin = corrected_df[
        (corrected_df["code"] == code)
        & (corrected_df["valid_from"].dt.year >= start_year)
        & (corrected_df["valid_from"].dt.year <= end_year)
    ].copy()

    if raw_basin.empty:
        logger.warning(f"No data for code {code} in years {start_year}-{end_year}")
        return False

    # Sort by valid_from (target month)
    raw_basin = raw_basin.sort_values("valid_from")
    corr_basin = corr_basin.sort_values("valid_from")

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot observed
    ax.plot(
        raw_basin["valid_from"],
        raw_basin[obs_col],
        "ko-",
        linewidth=2,
        markersize=8,
        label="Observed",
        zorder=3,
    )

    # Plot raw forecast
    ax.plot(
        raw_basin["valid_from"],
        raw_basin[pred_col],
        "b^--",
        linewidth=1.5,
        markersize=7,
        alpha=0.8,
        label="Raw Forecast",
    )

    # Plot corrected forecast
    ax.plot(
        corr_basin["valid_from"],
        corr_basin[pred_col],
        "gs-",
        linewidth=1.5,
        markersize=7,
        alpha=0.8,
        label="Bayesian Corrected",
    )

    # Add uncertainty bands if available
    if "Q5" in raw_basin.columns and "Q95" in raw_basin.columns:
        ax.fill_between(
            raw_basin["valid_from"],
            raw_basin["Q5"],
            raw_basin["Q95"],
            alpha=0.15,
            color="blue",
            label="Raw 90% CI",
        )

    if "Q5" in corr_basin.columns and "Q95" in corr_basin.columns:
        ax.fill_between(
            corr_basin["valid_from"],
            corr_basin["Q5"],
            corr_basin["Q95"],
            alpha=0.15,
            color="green",
            label="Corrected 90% CI",
        )

    ax.set_xlabel("Target Month", fontsize=12)
    ax.set_ylabel("Discharge", fontsize=12)
    ax.set_title(
        f"Basin {code}: Raw vs Bayesian Corrected Forecasts ({start_year}-{end_year})",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis for better date display
    fig.autofmt_xdate()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved time series plot to: {output_path}")
    return True


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================


def run_evaluation(config: Dict) -> None:
    """
    Run the complete Bayesian post-processing evaluation pipeline.

    Uses ratio-based multiplicative correction with configurable prior strength.
    """
    logger.info("=" * 60)
    logger.info("Starting Bayesian Post-Processing Evaluation")
    logger.info("=" * 60)
    logger.info(f"Prior precision scale: {config.get('prior_precision_scale', 1.0)}")

    # Get path configuration based on region
    region = config.get("region", "Kyrgyzstan")
    if region == "Tajikistan":
        path_config = taj_path_config
    else:
        path_config = kgz_path_config

    # Validate paths
    if not path_config["pred_dir"]:
        logger.error(
            f"Prediction directory not configured. "
            f"Set environment variable for {region}."
        )
        return

    if not path_config["obs_file"]:
        logger.error(
            f"Observation file not configured. Set environment variable for {region}."
        )
        return

    # Setup output directory
    save_dir = Path(output_dir) if output_dir else Path("bayesian_evaluation_results")
    precision_str = f"_ps{config.get('prior_precision_scale', 1.0)}"
    save_dir = save_dir / f"{region.lower()}_bayesian_ratio{precision_str}"
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Region: {region}")
    logger.info(f"Predictions: {path_config['pred_dir']}")
    logger.info(f"Observations: {path_config['obs_file']}")
    logger.info(f"Output: {save_dir}")

    # Load observations
    obs_df = load_observations(path_config["obs_file"])
    monthly_obs = calculate_monthly_target(obs_df)

    # Load predictions by horizon
    predictions_by_horizon = load_hindcast_predictions(
        base_path=path_config["pred_dir"],
        horizons=config["horizons"],
        model_names=config["model_names"],
    )

    if not predictions_by_horizon:
        logger.error("No predictions loaded. Check paths and configuration.")
        return

    # Merge with observations
    for horizon, pred_df in predictions_by_horizon.items():
        predictions_by_horizon[horizon] = merge_predictions_with_observations(
            pred_df, monthly_obs
        )

    # Setup prior configuration for ratio-based correction
    prior_config = RatioPriorConfig(
        prior_precision_scale=config.get("prior_precision_scale", 1.0),
        apply_soft_bounds=config.get("apply_soft_bounds", True),
        min_effective_n=config.get("min_observations", 5),
        apply_peak_capping=config.get("apply_peak_capping", True),
        peak_cap_softness=config.get("peak_cap_softness", 0.1),
    )
    logger.info(
        f"Prior config: precision_scale={prior_config.prior_precision_scale}, "
        f"soft_bounds={prior_config.apply_soft_bounds}, "
        f"peak_capping={prior_config.apply_peak_capping} "
        f"(softness={prior_config.peak_cap_softness})"
    )

    # Initialize corrected forecasts storage
    corrected_forecasts = {}
    ratio_stats = None

    # Compute historical ratios first
    logger.info("\nComputing historical ratio statistics...")
    ratio_stats_dict = compute_historical_ratios(
        observations_df=monthly_obs,
        basin_col="code",
        month_col="month",
        year_col="year",
        value_col="Q_obs",
    )

    # Convert dict to DataFrame for logging/plotting
    ratio_stats_records = []
    for (basin, month), stats in ratio_stats_dict.items():
        ratio_stats_records.append(
            {
                "code": basin,
                "target_month": month,
                "mean_ratio": stats.mean_R,
                "std_ratio": stats.std_R,
                "median_ratio": stats.mean_R,  # Use mean as proxy
                "n_samples": stats.n,
                "R_max": stats.R_max,
                "constraint_exists": stats.constraint_exists,
            }
        )
    ratio_stats = pd.DataFrame(ratio_stats_records)

    # Log and save ratio statistics
    log_ratio_statistics(
        ratio_stats,
        save_path=str(save_dir / "ratio_statistics.csv"),
    )

    # Apply ratio-based Bayesian correction
    logger.info("\nApplying ratio-based Bayesian correction...")
    corrected_forecasts = apply_ratio_bayesian_correction(
        forecasts_by_horizon=predictions_by_horizon,
        observations_df=monthly_obs,
        config=prior_config,
        pred_col="Q50",
        basin_col="code",
        date_col="valid_from",
        value_col="Q_obs",
    )

    logger.info("Ratio correction applied successfully.")

    # Calculate metrics for each horizon
    all_metrics = []
    all_metrics_by_month = []

    for horizon in predictions_by_horizon.keys():
        raw_df = predictions_by_horizon[horizon]
        corr_df = corrected_forecasts.get(horizon, raw_df)

        # Calculate comparison metrics (by horizon)
        horizon_metrics = calculate_comparison_metrics(
            raw_df=raw_df,
            corrected_df=corr_df,
            pred_col="Q50",
        )

        if len(horizon_metrics) > 0:
            horizon_metrics["horizon"] = horizon
            all_metrics.append(horizon_metrics)

        # Calculate metrics by month
        month_metrics = calculate_metrics_by_month(
            raw_df=raw_df,
            corrected_df=corr_df,
            pred_col="Q50",
            obs_col="Q_obs",
        )
        if len(month_metrics) > 0:
            month_metrics["horizon"] = horizon
            all_metrics_by_month.append(month_metrics)

    # Combine and save metrics
    if all_metrics:
        metrics_df = pd.concat(all_metrics, ignore_index=True)

        # Save detailed metrics
        metrics_path = save_dir / "metrics_comparison.csv"
        metrics_df.to_csv(metrics_path, index=False)
        logger.info(f"\nSaved detailed metrics to: {metrics_path}")

        # Save metrics by month
        if all_metrics_by_month:
            metrics_by_month_df = pd.concat(all_metrics_by_month, ignore_index=True)
            metrics_by_month_path = save_dir / "metrics_by_month.csv"
            metrics_by_month_df.to_csv(metrics_by_month_path, index=False)
            logger.info(f"Saved monthly metrics to: {metrics_by_month_path}")
        else:
            metrics_by_month_df = pd.DataFrame()

        # Create summary plots
        logger.info("\nCreating summary plots...")

        # 1. Metric distributions (boxplots + histograms)
        try:
            plot_metric_distributions(
                metrics_df,
                str(save_dir / "metric_distributions.png"),
            )
        except Exception as e:
            logger.warning(f"Could not create distribution plot: {e}")

        # 2. Improvement by horizon
        try:
            plot_improvement_by_horizon(
                metrics_df,
                str(save_dir / "improvement_by_horizon.png"),
            )
        except Exception as e:
            logger.warning(f"Could not create horizon plot: {e}")

        # 3. Scatter plot raw vs corrected
        try:
            plot_scatter_raw_vs_corrected(
                metrics_df,
                str(save_dir / "scatter_raw_vs_corrected.png"),
            )
        except Exception as e:
            logger.warning(f"Could not create scatter plot: {e}")

        # 4. Ratio-specific diagnostic plots
        if ratio_stats is not None and len(ratio_stats) > 0:
            try:
                plot_ratio_distributions_by_month(
                    ratio_stats,
                    str(save_dir / "ratio_distributions_by_month.png"),
                )
            except Exception as e:
                logger.warning(f"Could not create ratio distribution plot: {e}")

            try:
                plot_correction_strength_vs_sample_size(
                    ratio_stats,
                    str(save_dir / "correction_strength_vs_samples.png"),
                )
            except Exception as e:
                logger.warning(f"Could not create correction strength plot: {e}")

        # 5. Improvement by month plot
        if len(metrics_by_month_df) > 0:
            try:
                plot_improvement_by_month(
                    metrics_by_month_df,
                    str(save_dir / "improvement_by_month.png"),
                )
            except Exception as e:
                logger.warning(f"Could not create improvement by month plot: {e}")

        # 6. Time series plot for code 16936 (2024-2025) - one plot per horizon
        try:
            for horizon in sorted(predictions_by_horizon.keys()):
                raw_h = predictions_by_horizon[horizon]
                corr_h = corrected_forecasts.get(horizon, raw_h)

                plot_timeseries_comparison(
                    raw_df=raw_h,
                    corrected_df=corr_h,
                    code=16936,
                    start_year=2024,
                    end_year=2025,
                    pred_col="Q50",
                    obs_col="Q_obs",
                    output_path=str(
                        save_dir / f"timeseries_16936_2024_2025_h{horizon}.png"
                    ),
                )
        except Exception as e:
            logger.warning(f"Could not create time series plot: {e}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info(
            f"EVALUATION SUMMARY (RATIO CORRECTION, "
            f"prior_precision_scale={config.get('prior_precision_scale', 1.0)})"
        )
        logger.info("=" * 60)

        for horizon in sorted(predictions_by_horizon.keys()):
            h_metrics = metrics_df[metrics_df["horizon"] == horizon]
            if len(h_metrics) == 0:
                continue

            logger.info(f"\nHorizon {horizon}:")
            for metric in ["mae", "rmse", "r2", "bias"]:
                m_data = h_metrics[h_metrics["metric"] == metric]
                if len(m_data) > 0:
                    raw_mean = m_data["raw_value"].mean()
                    corr_mean = m_data["corrected_value"].mean()
                    impr = m_data["improvement"].mean()

                    # Count improvements
                    n_improved = (m_data["improvement"] > 0).sum()
                    n_total = len(m_data)

                    logger.info(
                        f"  {metric.upper():>6}: Raw={raw_mean:.4f}, "
                        f"Corrected={corr_mean:.4f}, Improvement={impr:+.4f} "
                        f"({n_improved}/{n_total} basins improved)"
                    )

        # Overall summary across all horizons
        logger.info("\n" + "-" * 40)
        logger.info("OVERALL (all horizons):")
        for metric in ["mae", "rmse", "r2", "bias"]:
            m_data = metrics_df[metrics_df["metric"] == metric]
            if len(m_data) > 0:
                raw_mean = m_data["raw_value"].mean()
                corr_mean = m_data["corrected_value"].mean()
                impr = m_data["improvement"].mean()
                n_improved = (m_data["improvement"] > 0).sum()
                n_total = len(m_data)
                logger.info(
                    f"  {metric.upper():>6}: Raw={raw_mean:.4f}, "
                    f"Corrected={corr_mean:.4f}, Improvement={impr:+.4f} "
                    f"({n_improved}/{n_total} improved)"
                )

        # Per-month summary
        if len(metrics_by_month_df) > 0:
            logger.info("\n" + "-" * 40)
            logger.info("PER-MONTH IMPROVEMENT (MAE):")
            mae_by_month = metrics_by_month_df[metrics_by_month_df["metric"] == "mae"]
            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            for month in range(1, 13):
                month_data = mae_by_month[mae_by_month["target_month"] == month]
                if len(month_data) > 0:
                    impr = month_data["improvement"].mean()
                    n_improved = (month_data["improvement"] > 0).sum()
                    n_total = len(month_data)
                    logger.info(
                        f"  {month_names[month - 1]:>3}: Improvement={impr:+.4f} "
                        f"({n_improved}/{n_total} improved)"
                    )

    else:
        logger.warning(
            "No metrics calculated. Check if predictions and observations match."
        )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 60)


# =============================================================================
# CLI Entry Point
# =============================================================================


def main():
    """Main entry point for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate Bayesian post-processing for monthly discharge forecasts",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="Kyrgyzstan",
        choices=["Kyrgyzstan", "Tajikistan"],
        help="Region to evaluate",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        default=None,
        help="Model names to evaluate",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        nargs="+",
        default=None,
        help="Horizons to evaluate (e.g., month_0 month_1 month_2)",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=1.0,
        help="Prior precision scale: 1.0=natural, >1=stronger prior, <1=weaker prior",
    )
    parser.add_argument(
        "--no-soft-bounds",
        action="store_true",
        help="Disable soft bounds based on historical constraints",
    )
    parser.add_argument(
        "--no-peak-capping",
        action="store_true",
        help="Disable peak capping (allow forecasts to exceed seasonal peak)",
    )
    parser.add_argument(
        "--peak-cap-softness",
        type=float,
        default=0.1,
        help="Peak cap softness: 0=hard cap, 1=no cap (default: 0.1)",
    )

    args = parser.parse_args()

    # Build config
    config = DEFAULT_CONFIG.copy()
    config["region"] = args.region
    config["prior_precision_scale"] = args.prior_strength
    config["apply_soft_bounds"] = not args.no_soft_bounds
    config["apply_peak_capping"] = not args.no_peak_capping
    config["peak_cap_softness"] = args.peak_cap_softness

    if args.model_names:
        config["model_names"] = args.model_names
    if args.horizons:
        config["horizons"] = args.horizons

    # Run evaluation
    try:
        run_evaluation(config)
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        import traceback

        tb = traceback.format_exc()
        logger.error(f"Evaluation failed: {e}\n{tb}")
        sys.exit(1)


if __name__ == "__main__":
    main()
