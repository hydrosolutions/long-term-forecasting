#!/usr/bin/env python3
"""
Quarter Forecast Visualization Script

Evaluates 3-month (quarterly) discharge forecasts by comparing ML model predictions
with quarter baseline models (LR_SM, LR_Base).

Forecast Configurations:
- Quarter Forecasts (3-month rolling): Issue dates Mar 25, Apr 25, May 25, Jun 25

Issue Date → Target Period Mapping:
| Issue Date | Target Period | Target Months |
|------------|---------------|---------------|
| March 25   | Apr-Jun       | [4, 5, 6]     |
| April 25   | May-Jul       | [5, 6, 7]     |
| May 25     | Jun-Aug       | [6, 7, 8]     |
| June 25    | Jul-Sep       | [7, 8, 9]     |

Output structure:
    {output_dir}/{region}/quarter/
        ├── scatter_Mar25.png
        ├── scatter_Apr25.png
        ├── scatter_May25.png
        ├── scatter_Jun25.png
        ├── r2_distribution_vs_issue_date.png
        ├── skill_comparison.png
        ├── pi_coverage.png
        ├── pi_coverage_by_issue_date.png
        ├── timeseries_16936_*.png
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

# Quarter forecast issue dates (month, day) -> target months (rolling 3-month)
ISSUE_DATES_QUARTER = {
    (3, 25): {"target_months": [4, 5, 6], "label": "Mar 25"},
    (4, 25): {"target_months": [5, 6, 7], "label": "Apr 25"},
    (5, 25): {"target_months": [6, 7, 8], "label": "May 25"},
    (6, 25): {"target_months": [7, 8, 9], "label": "Jun 25"},
}

# Suffix for quarter model names to distinguish from monthly-reconstructed
QUARTER_MODEL_SUFFIX = "_quarter"

# Quarter directory name
QUARTER_DIR_NAME = "quarter"

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

# Reverse mapping for sorting issue labels by month
MONTH_ORDER = {abbrev: num for num, abbrev in MONTH_ABBREV.items()}

# Quantile column names (must match LINEAR_REGRESSION.py)
QUANTILE_COLS = ["Q5", "Q10", "Q25", "Q75", "Q90", "Q95"]

# PI coverage definitions: (lower_quantile_col, upper_quantile_col, nominal_coverage)
PI_COVERAGE_DEFS = {
    "50%": ("Q25", "Q75", 0.50),
    "90%": ("Q5", "Q95", 0.90),
}


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


def calculate_pi_coverage(
    df: pd.DataFrame,
    obs_col: str = "Q_obs_quarter",
) -> dict[str, float]:
    """
    Calculate prediction interval coverage for models with quantile columns.

    Args:
        df: DataFrame with observations and quantile predictions
        obs_col: Name of observation column

    Returns:
        Dictionary with coverage stats: {
            "coverage_50": actual 50% PI coverage,
            "coverage_90": actual 90% PI coverage,
            "n_samples": number of valid samples,
            "has_quantiles": whether quantile columns exist (with actual values)
        }
    """
    result = {
        "coverage_50": np.nan,
        "coverage_90": np.nan,
        "n_samples": 0,
        "has_quantiles": False,
    }

    # Check if quantile columns exist AND have non-NaN values
    available_quantiles = [
        col for col in QUANTILE_COLS if col in df.columns and df[col].notna().any()
    ]
    if not available_quantiles:
        return result

    result["has_quantiles"] = True

    # Calculate coverage for each PI definition
    for pi_name, (lower_col, upper_col, _) in PI_COVERAGE_DEFS.items():
        if lower_col not in df.columns or upper_col not in df.columns:
            continue

        # Filter to rows with valid obs and quantile values
        valid_mask = df[obs_col].notna() & df[lower_col].notna() & df[upper_col].notna()
        valid_df = df[valid_mask]

        if len(valid_df) == 0:
            continue

        # Calculate coverage: fraction of observations within PI
        in_interval = (valid_df[obs_col] >= valid_df[lower_col]) & (
            valid_df[obs_col] <= valid_df[upper_col]
        )
        coverage = in_interval.mean()

        key = f"coverage_{pi_name.replace('%', '')}"
        result[key] = coverage
        result["n_samples"] = len(valid_df)

    return result


def calculate_r2_per_code(
    df: pd.DataFrame,
    obs_col: str = "Q_obs_quarter",
    pred_col: str = "Q_pred_quarter",
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


def get_shared_codes(quarter_df: pd.DataFrame) -> set[int]:
    """Get codes that are present in ALL models."""
    codes_per_model = quarter_df.groupby("model")["code"].apply(set)
    if codes_per_model.empty:
        return set()
    return set.intersection(*codes_per_model)


# =============================================================================
# QUARTER FORECAST RECONSTRUCTION
# =============================================================================


def calculate_partial_month_obs(
    daily_obs: pd.DataFrame,
    issue_month: int,
    issue_day: int,
) -> pd.DataFrame:
    """
    Calculate observed mean discharge for partial month (day 1 to issue_day).

    For forecasts issued on e.g. April 25, this calculates the mean observed
    discharge from April 1-25 for each code and year.

    Args:
        daily_obs: DataFrame with columns: date, code, discharge
        issue_month: Month of forecast issue (1-12)
        issue_day: Day of forecast issue (1-31)

    Returns:
        DataFrame with columns: code, year, Q_obs_partial
    """
    obs = daily_obs.copy()
    obs["year"] = obs["date"].dt.year
    obs["month"] = obs["date"].dt.month
    obs["day"] = obs["date"].dt.day

    # Filter to issue month and days 1 to issue_day
    partial = obs[(obs["month"] == issue_month) & (obs["day"] <= issue_day)]

    # Calculate mean per code and year
    partial_means = (
        partial.groupby(["code", "year"])["discharge"]
        .mean()
        .reset_index()
        .rename(columns={"discharge": "Q_obs_partial"})
    )

    return partial_means


def reconstruct_quarter_forecasts(
    predictions_df: pd.DataFrame,
    monthly_obs: pd.DataFrame,
    issue_dates: dict,
    daily_obs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Reconstruct quarter forecasts by averaging monthly predictions over 3 target months.

    Uses vectorized operations for performance.

    If daily_obs is provided and the issue month is in the target months,
    the model prediction for the issue month is replaced with observed
    partial month mean (days 1 to issue_day). This uses actual data we
    have at forecast time instead of predicting what we already know.

    Args:
        predictions_df: DataFrame with monthly predictions (from load_predictions)
        monthly_obs: DataFrame with monthly observations (from calculate_target)
        issue_dates: Dictionary mapping (issue_month, issue_day) to config
        daily_obs: Optional DataFrame with daily observations for partial month
                   calculation. Columns: date, code, discharge

    Returns:
        DataFrame with columns: code, issue_year, issue_month, issue_day,
                               issue_label, Q_pred_quarter, Q_obs_quarter, model
    """
    results = []

    for (issue_month, issue_day), config in issue_dates.items():
        target_months = config["target_months"]
        label = config["label"]
        horizons_needed = [get_horizon(issue_month, tm) for tm in target_months]

        logger.info(f"Processing issue date {label} -> targets {target_months}")

        # Check if issue month is in target months (e.g., Apr 25 for Apr-Jun)
        use_partial_obs = daily_obs is not None and issue_month in target_months

        # Filter predictions for this issue month and required horizons
        issue_preds = predictions_df[
            (predictions_df["issue_date"].dt.month == issue_month)
            & (predictions_df["horizon"].isin(horizons_needed))
        ].copy()

        if issue_preds.empty:
            logger.warning(f"No predictions found for issue month {issue_month}")
            continue

        issue_preds["issue_year"] = issue_preds["issue_date"].dt.year

        # Replace predictions for issue month with observed partial month mean
        if use_partial_obs:
            partial_obs = calculate_partial_month_obs(daily_obs, issue_month, issue_day)

            # horizon=0 means predicting the same month as issue
            # Reset index to ensure proper alignment after merge
            issue_preds = issue_preds.reset_index(drop=True)
            issue_month_mask = issue_preds["horizon"] == 0

            if issue_month_mask.any():
                # Merge partial observations
                issue_preds = issue_preds.merge(
                    partial_obs,
                    left_on=["code", "issue_year"],
                    right_on=["code", "year"],
                    how="left",
                )
                if "year" in issue_preds.columns:
                    issue_preds = issue_preds.drop(columns=["year"])

                # Recalculate mask after merge (index may have changed)
                issue_month_mask = issue_preds["horizon"] == 0

                # Replace Q_pred with observed partial mean for issue month
                issue_preds.loc[issue_month_mask, "Q_pred"] = issue_preds.loc[
                    issue_month_mask, "Q_obs_partial"
                ].values

                # Clear quantiles for issue month (we don't have uncertainty for obs)
                for q_col in QUANTILE_COLS:
                    if q_col in issue_preds.columns:
                        issue_preds.loc[issue_month_mask, q_col] = np.nan

                if "Q_obs_partial" in issue_preds.columns:
                    issue_preds = issue_preds.drop(columns=["Q_obs_partial"])

                n_replaced = issue_month_mask.sum()
                logger.info(
                    f"  Replaced {n_replaced} predictions for {MONTH_ABBREV[issue_month]} "
                    f"with observed partial month mean (days 1-{issue_day})"
                )

        # Identify columns to aggregate (Q_pred + any quantile columns)
        agg_cols = ["Q_pred"]
        available_quantiles = [c for c in QUANTILE_COLS if c in issue_preds.columns]
        agg_cols.extend(available_quantiles)

        # Vectorized quarter averaging per (model, code, issue_year)
        quarter_pred = (
            issue_preds.groupby(["model", "code", "issue_year"])[agg_cols]
            .mean()
            .reset_index()
            .rename(columns={"Q_pred": "Q_pred_quarter"})
        )

        if available_quantiles:
            logger.debug(f"Averaged quantile columns: {available_quantiles}")

        # Get quarter observations - filter for target months and average
        obs_target = monthly_obs[monthly_obs["month"].isin(target_months)].copy()
        quarter_obs = (
            obs_target.groupby(["code", "year"])["Q_obs_monthly"]
            .mean()
            .reset_index()
            .rename(columns={"year": "issue_year", "Q_obs_monthly": "Q_obs_quarter"})
        )

        # Merge predictions with observations
        quarter = quarter_pred.merge(quarter_obs, on=["code", "issue_year"], how="left")
        quarter["issue_month"] = issue_month
        quarter["issue_day"] = issue_day
        quarter["issue_label"] = label
        quarter["n_target_months"] = len(target_months)

        results.append(quarter)

    if not results:
        logger.warning("No quarter forecasts reconstructed")
        return pd.DataFrame()

    result_df = pd.concat(results, ignore_index=True)
    logger.info(
        f"Reconstructed {len(result_df)} quarter forecasts for "
        f"{result_df['model'].nunique()} models"
    )
    return result_df


# =============================================================================
# DIRECT QUARTER MODEL LOADING
# =============================================================================


def load_quarter_model_hindcasts(
    base_path: Path,
    issue_months: list[int],
    models: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load direct quarter model hindcast predictions.

    Quarter models predict 3-month discharge directly (not monthly).
    Hindcast files are stored at:
        {base_path}/quarter/{model}/{model}_hindcast.csv

    File format:
        date,code,valid_from,valid_to,flag,Q_{model},Q5,Q10,Q25,Q75,Q90,Q95

    Args:
        base_path: Base directory for predictions (e.g., long_term_predictions/)
        issue_months: List of issue months to filter (3, 4, 5, 6 for quarter)
        models: List of model names to load. If None, loads all available.

    Returns:
        DataFrame with columns: code, issue_year, issue_month, issue_day,
                               issue_label, Q_pred_quarter, model
    """
    results = []

    quarter_dir = Path(base_path) / QUARTER_DIR_NAME

    if not quarter_dir.exists():
        logger.info(f"Quarter directory not found: {quarter_dir}")
        return pd.DataFrame()

    model_dirs = [d for d in quarter_dir.iterdir() if d.is_dir()]

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
        df["valid_from"] = pd.to_datetime(df["valid_from"])
        df["valid_to"] = pd.to_datetime(df["valid_to"])

        df["issue_year"] = df["date"].dt.year
        df["issue_month"] = df["date"].dt.month
        df["issue_day"] = df["date"].dt.day

        # Filter by issue_months
        df = df[df["issue_month"].isin(issue_months)].copy()
        if df.empty:
            logger.debug(f"No data after filtering by issue months in {hindcast_file}")
            continue

        # Determine target months from valid_from/valid_to for label mapping
        df["target_start_month"] = df["valid_from"].dt.month

        # Create issue label based on issue month
        df["issue_label"] = df["issue_month"].map(
            lambda m: f"{MONTH_ABBREV.get(m, str(m))} 25"
        )

        # Find prediction column (Q_{model_name})
        pred_col = f"Q_{model_name}"
        if pred_col not in df.columns:
            q_cols = [
                c for c in df.columns if c.startswith("Q_") and c not in QUANTILE_COLS
            ]
            if q_cols:
                pred_col = q_cols[0]
            else:
                logger.warning(f"No prediction column found in {hindcast_file}")
                continue

        df["Q_pred_quarter"] = df[pred_col]
        df["model"] = f"{model_name}{QUARTER_MODEL_SUFFIX}"

        # Base columns to keep
        base_cols = [
            "code",
            "issue_year",
            "issue_month",
            "issue_day",
            "issue_label",
            "Q_pred_quarter",
            "model",
        ]

        # Add quantile columns if present
        available_quantile_cols = [c for c in QUANTILE_COLS if c in df.columns]
        keep_cols = base_cols + available_quantile_cols

        result = df[keep_cols].copy()

        results.append(result)
        quantile_info = (
            f" (with quantiles: {available_quantile_cols})"
            if available_quantile_cols
            else ""
        )
        logger.info(
            f"Loaded {len(result)} quarter hindcasts for {model_name}{quantile_info}"
        )

    if not results:
        logger.info("No direct quarter model hindcasts found")
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)
    logger.info(
        f"Loaded {len(combined)} total quarter hindcasts for "
        f"{combined['model'].nunique()} models"
    )
    return combined


def get_quarter_observations(
    monthly_obs: pd.DataFrame,
    target_months: list[int],
) -> pd.DataFrame:
    """
    Calculate quarter observations by averaging monthly values for target 3-month period.

    Args:
        monthly_obs: DataFrame with monthly observations (from calculate_target)
                    Expected columns: code, year, month, Q_obs_monthly
        target_months: Months to average (e.g., [4, 5, 6] for Apr-Jun)

    Returns:
        DataFrame with columns: code, year, Q_obs_quarter
    """
    filtered = monthly_obs[monthly_obs["month"].isin(target_months)].copy()

    if filtered.empty:
        return pd.DataFrame()

    quarter_obs = (
        filtered.groupby(["code", "year"])["Q_obs_monthly"].mean().reset_index()
    )
    quarter_obs.columns = ["code", "year", "Q_obs_quarter"]

    return quarter_obs


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_quarter_scatter(
    quarter_df: pd.DataFrame,
    models: list[str],
    issue_label: str,
    output_path: Path,
    code_filter: int | None = SCATTER_PLOT_CODE,
) -> plt.Figure:
    """
    Create scatter plot of predicted vs observed quarter discharge.

    Args:
        quarter_df: DataFrame with Q_pred_quarter, Q_obs_quarter, model
        models: List of models to plot
        issue_label: Issue date label (e.g., "Mar 25")
        output_path: Path to save figure
        code_filter: If specified, only plot this specific code (basin)

    Returns:
        matplotlib Figure
    """
    df = quarter_df[quarter_df["issue_label"] == issue_label].copy()

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
        x_col = "Q_obs_quarter"
        y_col = "Q_pred_quarter"
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

        ax.set_xlabel("Observed (m\u00b3/s)", fontsize=10)
        ax.set_ylabel("Predicted (m\u00b3/s)", fontsize=10)
        ax.set_title(
            f"{model}\nR\u00b2={metrics['r2']:.3f}, n={metrics['n_samples']}",
            fontsize=11,
        )

        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    for idx in range(len(models), len(axes)):
        axes[idx].set_visible(False)

    code_str = f" - Code {code_filter}" if code_filter is not None else ""
    fig.suptitle(
        f"Quarter Forecast - Issue Date: {issue_label}{code_str}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved scatter plot to {output_path}")

    return fig


def _sort_issue_labels(labels: list[str]) -> list[str]:
    """Sort issue labels by month order (Jan, Feb, Mar, ...)."""

    def month_key(label: str) -> int:
        month_abbrev = label.split()[0]
        return MONTH_ORDER.get(month_abbrev, 99)

    return sorted(labels, key=month_key)


def plot_r2_vs_issue_date(
    quarter_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> plt.Figure:
    """
    Plot R² distribution across codes for each model and issue date.

    Shows boxplot of per-code R² values to visualize skill distribution.

    Args:
        quarter_df: DataFrame with quarter forecasts
        models: List of models to plot
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    plot_data = []
    issue_labels = _sort_issue_labels(list(quarter_df["issue_label"].unique()))

    for model in models:
        model_df = quarter_df[quarter_df["model"] == model]

        for label in issue_labels:
            label_df = model_df[model_df["issue_label"] == label]

            if label_df.empty:
                continue

            r2_per_code = calculate_r2_per_code(
                label_df, obs_col="Q_obs_quarter", pred_col="Q_pred_quarter"
            )

            for _, row in r2_per_code.iterrows():
                plot_data.append(
                    {
                        "Issue Date": label,
                        "R\u00b2": row["r2"],
                        "Model": model,
                        "code": row["code"],
                    }
                )

    if not plot_data:
        logger.warning("No data for R\u00b2 vs issue date plot")
        return plt.figure()

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = sns.color_palette("husl", len(models))

    sns.boxplot(
        data=plot_df,
        x="Issue Date",
        y="R\u00b2",
        hue="Model",
        order=issue_labels,
        hue_order=models,
        palette=colors,
        ax=ax,
        width=0.7,
    )

    ax.axhline(y=0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_ylabel("R\u00b2 (per basin)", fontsize=12)
    ax.set_xlabel("Forecast Issue Date", fontsize=12)
    ax.set_title(
        "Quarter Forecast Skill Distribution by Issue Date",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved R\u00b2 vs issue date plot to {output_path}")

    return fig


def plot_skill_comparison(
    quarter_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> plt.Figure:
    """
    Create boxplot comparing R² distribution per code for ML models.

    Args:
        quarter_df: DataFrame with quarter forecasts
        models: List of ML models
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    plot_data = []

    for model in models:
        model_df = quarter_df[quarter_df["model"] == model]

        if model_df.empty:
            continue

        r2_per_code = calculate_r2_per_code(
            model_df, obs_col="Q_obs_quarter", pred_col="Q_pred_quarter"
        )

        for _, row in r2_per_code.iterrows():
            plot_data.append(
                {
                    "R\u00b2": row["r2"],
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
        y="R\u00b2",
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
        y="R\u00b2",
        order=models,
        ax=ax,
        color="black",
        alpha=0.3,
        size=3,
    )

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel("R\u00b2 (per basin)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Model Comparison - R\u00b2 Distribution Across Basins",
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


def plot_pi_coverage(
    quarter_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> plt.Figure:
    """
    Create bar plot comparing prediction interval coverage for probabilistic models.

    Shows actual vs nominal coverage for 50% and 90% prediction intervals.

    Args:
        quarter_df: DataFrame with quarter forecasts (must include quantile columns)
        models: List of ML models to evaluate
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    coverage_data = []

    for model in models:
        model_df = quarter_df[quarter_df["model"] == model]
        if model_df.empty:
            continue

        coverage = calculate_pi_coverage(model_df, obs_col="Q_obs_quarter")

        if not coverage["has_quantiles"]:
            continue

        for pi_name, (_, __, nominal) in PI_COVERAGE_DEFS.items():
            key = f"coverage_{pi_name.replace('%', '')}"
            actual = coverage.get(key, np.nan)
            if not np.isnan(actual):
                coverage_data.append(
                    {
                        "Model": model,
                        "PI": pi_name,
                        "Coverage": actual,
                        "Nominal": nominal,
                        "n_samples": coverage["n_samples"],
                    }
                )

    if not coverage_data:
        logger.info("No models with quantile predictions found for PI coverage plot")
        return plt.figure()

    plot_df = pd.DataFrame(coverage_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Get unique models with quantiles
    models_with_quantiles = plot_df["Model"].unique()
    n_models = len(models_with_quantiles)
    x = np.arange(n_models)
    width = 0.35

    # Plot 50% and 90% coverage side by side
    coverage_50 = plot_df[plot_df["PI"] == "50%"].set_index("Model")["Coverage"]
    coverage_90 = plot_df[plot_df["PI"] == "90%"].set_index("Model")["Coverage"]

    bars_50 = ax.bar(
        x - width / 2,
        [coverage_50.get(m, 0) for m in models_with_quantiles],
        width,
        label="50% PI Coverage",
        color="steelblue",
        alpha=0.8,
    )
    bars_90 = ax.bar(
        x + width / 2,
        [coverage_90.get(m, 0) for m in models_with_quantiles],
        width,
        label="90% PI Coverage",
        color="darkorange",
        alpha=0.8,
    )

    # Add nominal coverage reference lines
    ax.axhline(y=0.50, color="steelblue", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=0.90, color="darkorange", linestyle="--", linewidth=1.5, alpha=0.7)

    # Add text labels for nominal lines
    ax.text(
        n_models - 0.5, 0.52, "Nominal 50%", fontsize=9, color="steelblue", alpha=0.8
    )
    ax.text(
        n_models - 0.5, 0.92, "Nominal 90%", fontsize=9, color="darkorange", alpha=0.8
    )

    # Add value labels on bars
    for bars in [bars_50, bars_90]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(
                    f"{height:.0%}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "Prediction Interval Coverage\n(Dashed lines = nominal coverage)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(models_with_quantiles, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved PI coverage plot to {output_path}")

    return fig


def plot_pi_coverage_by_issue_date(
    quarter_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> plt.Figure:
    """
    Create 2x2 figure showing PI coverage for each issue date.

    Each subplot corresponds to one forecast issue date, with bars showing
    actual vs nominal coverage for 50% and 90% prediction intervals.

    Args:
        quarter_df: DataFrame with quarter forecasts (must include quantile columns)
        models: List of ML models to evaluate
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    issue_labels = _sort_issue_labels(list(quarter_df["issue_label"].unique()))
    n_dates = len(issue_labels)

    if n_dates == 0:
        logger.warning("No issue dates found for PI coverage by issue date plot")
        return plt.figure()

    # Collect coverage data per issue date
    coverage_by_date: dict[str, list[dict]] = {label: [] for label in issue_labels}

    for label in issue_labels:
        label_df = quarter_df[quarter_df["issue_label"] == label]

        for model in models:
            model_df = label_df[label_df["model"] == model]
            if model_df.empty:
                continue

            coverage = calculate_pi_coverage(model_df, obs_col="Q_obs_quarter")

            if not coverage["has_quantiles"]:
                continue

            for pi_name, (_, __, nominal) in PI_COVERAGE_DEFS.items():
                key = f"coverage_{pi_name.replace('%', '')}"
                actual = coverage.get(key, np.nan)
                if not np.isnan(actual):
                    coverage_by_date[label].append(
                        {
                            "Model": model,
                            "PI": pi_name,
                            "Coverage": actual,
                            "Nominal": nominal,
                            "n_samples": coverage["n_samples"],
                        }
                    )

    # Check if any data exists
    has_data = any(len(data) > 0 for data in coverage_by_date.values())
    if not has_data:
        logger.info(
            "No models with quantile predictions found for PI coverage by issue date"
        )
        return plt.figure()

    # Create 2x2 figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, label in enumerate(issue_labels):
        if idx >= 4:
            break
        ax = axes[idx]
        coverage_data = coverage_by_date[label]

        if not coverage_data:
            ax.text(
                0.5,
                0.5,
                f"No quantile data\n{label}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title(f"Issue Date: {label}", fontsize=12, fontweight="bold")
            continue

        plot_df = pd.DataFrame(coverage_data)
        models_with_quantiles = plot_df["Model"].unique()
        n_models = len(models_with_quantiles)
        x = np.arange(n_models)
        width = 0.35

        coverage_50 = plot_df[plot_df["PI"] == "50%"].set_index("Model")["Coverage"]
        coverage_90 = plot_df[plot_df["PI"] == "90%"].set_index("Model")["Coverage"]

        bars_50 = ax.bar(
            x - width / 2,
            [coverage_50.get(m, 0) for m in models_with_quantiles],
            width,
            label="50% PI" if idx == 0 else "",
            color="steelblue",
            alpha=0.8,
        )
        bars_90 = ax.bar(
            x + width / 2,
            [coverage_90.get(m, 0) for m in models_with_quantiles],
            width,
            label="90% PI" if idx == 0 else "",
            color="darkorange",
            alpha=0.8,
        )

        ax.axhline(y=0.50, color="steelblue", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.axhline(y=0.90, color="darkorange", linestyle="--", linewidth=1.5, alpha=0.7)

        # Add value labels on bars
        for bars in [bars_50, bars_90]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.annotate(
                        f"{height:.0%}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        ax.set_ylabel("Coverage", fontsize=11)
        ax.set_xlabel("Model", fontsize=11)
        ax.set_title(f"Issue Date: {label}", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(models_with_quantiles, rotation=45, ha="right")
        ax.set_ylim(0, 1.08)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots if fewer than 4 issue dates
    for idx in range(n_dates, 4):
        axes[idx].set_visible(False)

    # Add legend to first plot
    if coverage_by_date.get(issue_labels[0]):
        axes[0].legend(loc="upper left", fontsize=11)

    fig.suptitle(
        "Quarter Prediction Interval Coverage by Issue Date\n"
        "(Dashed lines = nominal coverage)",
        fontsize=15,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved PI coverage by issue date plot to {output_path}")

    return fig


def plot_timeseries_with_uncertainty(
    quarter_df: pd.DataFrame,
    models: list[str],
    issue_label: str,
    code: int,
    output_path: Path,
) -> plt.Figure:
    """
    Plot time series of observed vs predicted quarter discharge with uncertainty.

    Args:
        quarter_df: DataFrame with quarter forecasts including quantile columns
        models: List of models to plot
        issue_label: Issue date label (e.g., "Mar 25")
        code: Basin code to plot
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    df = quarter_df[
        (quarter_df["issue_label"] == issue_label) & (quarter_df["code"] == code)
    ].copy()

    if df.empty:
        logger.warning(f"No data for code {code} at issue date {issue_label}")
        return plt.figure()

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique years and sort
    years = sorted(df["issue_year"].unique())

    # Plot observed (use data from first model, obs should be same across models)
    obs_df = df[df["model"] == models[0]].sort_values("issue_year")
    obs_years = obs_df["issue_year"].values
    obs_values = obs_df["Q_obs_quarter"].values

    ax.plot(
        obs_years,
        obs_values,
        color="black",
        linewidth=2.5,
        label="Observed",
        zorder=10,
    )

    # Color palette for models
    colors = {
        "LR_SM": "steelblue",
        "MC_ALD": "darkorange",
        "LR_SM_quarter": "steelblue",
        "LR_Base_quarter": "forestgreen",
        "MC_ALD_quarter": "darkorange",
    }
    default_colors = sns.color_palette("husl", len(models))

    # Offset for each model to avoid overlap
    n_models = len(models)
    offsets = np.linspace(-0.15, 0.15, n_models) if n_models > 1 else [0]

    for idx, model in enumerate(models):
        model_df = df[df["model"] == model].sort_values("issue_year")

        if model_df.empty:
            continue

        pred_years = model_df["issue_year"].values + offsets[idx]
        pred_values = model_df["Q_pred_quarter"].values

        color = colors.get(model, default_colors[idx])

        # Check if quantile columns exist
        has_q5 = "Q5" in model_df.columns and model_df["Q5"].notna().any()
        has_q95 = "Q95" in model_df.columns and model_df["Q95"].notna().any()

        if has_q5 and has_q95:
            q5 = model_df["Q5"].values
            q95 = model_df["Q95"].values

            # Calculate error bar lengths (asymmetric)
            yerr_lower = pred_values - q5
            yerr_upper = q95 - pred_values

            ax.errorbar(
                pred_years,
                pred_values,
                yerr=[yerr_lower, yerr_upper],
                fmt="o",
                color=color,
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1,
                capsize=4,
                capthick=1.5,
                elinewidth=1.5,
                linestyle="none",
                label=f"{model} (90% PI)",
                alpha=0.5,
            )
        else:
            # No quantiles - just plot points
            ax.plot(
                pred_years,
                pred_values,
                linestyle="none",
                marker="o",
                markersize=7,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=1,
                color=color,
                label=model,
                alpha=0.8,
            )

    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Quarter Discharge (m\u00b3/s)", fontsize=12)
    ax.set_title(
        f"Quarter Forecast - Code {code} - Issue Date: {issue_label}",
        fontsize=14,
        fontweight="bold",
    )

    ax.set_xticks(years)
    ax.set_xticklabels(years, rotation=45, ha="right")
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved time series plot to {output_path}")

    return fig


# =============================================================================
# METRICS SUMMARY
# =============================================================================


def generate_metrics_summary(
    quarter_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate CSV summary of per-code R² statistics per model and issue date.

    Args:
        quarter_df: DataFrame with quarter forecasts
        models: List of ML models
        output_path: Path to save CSV

    Returns:
        Summary DataFrame
    """
    results = []
    issue_labels = quarter_df["issue_label"].unique()

    for label in issue_labels:
        for model in models:
            model_df = quarter_df[
                (quarter_df["model"] == model) & (quarter_df["issue_label"] == label)
            ]

            if model_df.empty:
                continue

            r2_per_code = calculate_r2_per_code(
                model_df, obs_col="Q_obs_quarter", pred_col="Q_pred_quarter"
            )

            if r2_per_code.empty:
                continue

            r2_values = r2_per_code["r2"].dropna()

            record = {
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

            # Add PI coverage metrics if quantiles available
            coverage = calculate_pi_coverage(model_df, obs_col="Q_obs_quarter")
            if coverage["has_quantiles"]:
                record["coverage_50"] = coverage["coverage_50"]
                record["coverage_90"] = coverage["coverage_90"]
                record["n_coverage_samples"] = coverage["n_samples"]

            results.append(record)

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


def process_quarter_forecasts(
    region: str,
    models: list[str],
    output_dir: Path,
) -> None:
    """
    Main processing function for quarter forecast visualization.

    Args:
        region: "Kyrgyzstan" or "Tajikistan"
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

    logger.info("\n" + "=" * 60)
    logger.info("Processing quarter forecasts")
    logger.info("=" * 60)

    fc_output_dir = output_dir / region.lower() / QUARTER_DIR_NAME
    fc_output_dir.mkdir(parents=True, exist_ok=True)

    # Reconstruct quarter forecasts from monthly predictions
    # Pass daily_obs to use actual observations for issue month when in target period
    quarter_df = reconstruct_quarter_forecasts(
        predictions_df, monthly_obs, ISSUE_DATES_QUARTER, daily_obs=obs_df
    )

    # Load direct quarter model hindcasts
    issue_months_list = [k[0] for k in ISSUE_DATES_QUARTER.keys()]
    quarter_model_df = load_quarter_model_hindcasts(
        base_path=pred_config["pred_dir"],
        issue_months=issue_months_list,
        models=models,
    )

    # Merge quarter model predictions with observations
    if not quarter_model_df.empty:
        # Initialize Q_obs_quarter column
        quarter_model_df["Q_obs_quarter"] = np.nan

        # For each issue date, get observations for its target months
        for (issue_month, _), config in ISSUE_DATES_QUARTER.items():
            target_months = config["target_months"]
            issue_label = config["label"]

            quarter_obs_for_models = get_quarter_observations(
                monthly_obs, target_months=target_months
            )

            if quarter_obs_for_models.empty:
                continue

            # Filter quarter_model_df to this issue date
            mask = quarter_model_df["issue_label"] == issue_label
            if not mask.any():
                continue

            # Merge observations for this specific issue date
            subset = quarter_model_df.loc[mask].copy()
            merged = subset.merge(
                quarter_obs_for_models,
                left_on=["code", "issue_year"],
                right_on=["code", "year"],
                how="left",
                suffixes=("", "_obs"),
            )

            # Update the Q_obs_quarter column with merged values
            if "Q_obs_quarter_obs" in merged.columns:
                quarter_model_df.loc[mask, "Q_obs_quarter"] = merged[
                    "Q_obs_quarter_obs"
                ].values
            elif "Q_obs_quarter" in merged.columns:
                quarter_model_df.loc[mask, "Q_obs_quarter"] = merged[
                    "Q_obs_quarter"
                ].values

        if not quarter_df.empty:
            quarter_df = pd.concat([quarter_df, quarter_model_df], ignore_index=True)
        else:
            quarter_df = quarter_model_df

    if quarter_df.empty:
        logger.warning("No quarter forecasts found")
        return

    # Filter to requested models (including quarter variants)
    quarter_model_names = [f"{m}{QUARTER_MODEL_SUFFIX}" for m in models]
    all_model_names = models + quarter_model_names
    available_models = [m for m in all_model_names if m in quarter_df["model"].unique()]
    if not available_models:
        logger.warning("None of the requested models found in quarter forecasts")
        return

    # Filter to shared codes across all models
    filtered_df = quarter_df[quarter_df["model"].isin(available_models)]
    shared_codes = get_shared_codes(filtered_df)
    if not shared_codes:
        logger.warning("No shared codes found across all models")
        return

    quarter_df = quarter_df[quarter_df["code"].isin(shared_codes)]
    logger.info(f"Filtered to {len(shared_codes)} shared codes across all models")
    logger.info(f"Available models: {available_models}")

    # Generate plots
    logger.info("Generating plots...")

    # 1. Scatter plots per issue date
    for config in ISSUE_DATES_QUARTER.values():
        label = config["label"]
        label_clean = label.replace(" ", "")

        fig = plot_quarter_scatter(
            quarter_df=quarter_df,
            models=available_models,
            issue_label=label,
            output_path=fc_output_dir / f"scatter_{label_clean}.png",
        )
        plt.close(fig)

    # 2. R² distribution vs issue date
    fig = plot_r2_vs_issue_date(
        quarter_df=quarter_df,
        models=available_models,
        output_path=fc_output_dir / "r2_distribution_vs_issue_date.png",
    )
    plt.close(fig)

    # 3. Skill comparison plot
    fig = plot_skill_comparison(
        quarter_df=quarter_df,
        models=available_models,
        output_path=fc_output_dir / "skill_comparison.png",
    )
    plt.close(fig)

    # 4. PI coverage plot (for models with quantile predictions)
    fig = plot_pi_coverage(
        quarter_df=quarter_df,
        models=available_models,
        output_path=fc_output_dir / "pi_coverage.png",
    )
    plt.close(fig)

    # 5. PI coverage by issue date (one row per issue date)
    fig = plot_pi_coverage_by_issue_date(
        quarter_df=quarter_df,
        models=available_models,
        output_path=fc_output_dir / "pi_coverage_by_issue_date.png",
    )
    plt.close(fig)

    # 6. Time series with uncertainty for specific code (16936)
    timeseries_models = ["LR_SM_quarter", "LR_Base_quarter", "MC_ALD"]
    timeseries_models_available = [
        m for m in timeseries_models if m in available_models
    ]
    if timeseries_models_available and SCATTER_PLOT_CODE in shared_codes:
        for issue_lbl in ["Mar 25", "Jun 25"]:
            if issue_lbl in quarter_df["issue_label"].values:
                label_clean = issue_lbl.replace(" ", "")
                fig = plot_timeseries_with_uncertainty(
                    quarter_df=quarter_df,
                    models=timeseries_models_available,
                    issue_label=issue_lbl,
                    code=SCATTER_PLOT_CODE,
                    output_path=fc_output_dir
                    / f"timeseries_{SCATTER_PLOT_CODE}_{label_clean}.png",
                )
                plt.close(fig)

    # 7. Metrics summary CSV
    generate_metrics_summary(
        quarter_df=quarter_df,
        models=available_models,
        output_path=fc_output_dir / "metrics_summary.csv",
    )

    logger.info("\nProcessing complete!")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate quarter (3-month) forecasts and compare ML models."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["Kyrgyzstan", "Tajikistan"],
        default="Kyrgyzstan",
        help="Region to process (default: Kyrgyzstan)",
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
    logger.info(f"Models: {args.models}")
    logger.info(f"Output directory: {output_dir}")

    process_quarter_forecasts(
        region=args.region,
        models=args.models,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
