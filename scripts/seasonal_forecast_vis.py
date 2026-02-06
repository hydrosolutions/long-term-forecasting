#!/usr/bin/env python3
"""
Seasonal Forecast Visualization Script

Evaluates seasonal forecasts (6-month and 3-month) and compares ML model predictions
with SWE-based linear regression predictions using LOOCV.

Forecast Configurations:
- 6-Month Forecasts (Apr-Sep average): Issue dates Jan 25, Feb 25, Mar 25, Apr 25
- 3-Month Forecasts (rolling): Issue dates Mar 25, Apr 25, May 25

Output structure:
    {output_dir}/{region}/seasonal/
        ├── 6_month/
        │   ├── scatter_Jan25.png
        │   ├── r2_vs_issue_date.png
        │   └── metrics_summary.csv
        ├── 3_month/
        │   ├── scatter_Mar25.png
        │   ├── r2_vs_issue_date.png
        │   └── metrics_summary.csv
        └── swe_analysis/
            ├── swe_correlation_boxplot.png
            ├── swe_vs_ml_comparison.png
            └── correlation_summary.csv
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
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
    calculate_target,
    kgz_path_config,
    taj_path_config,
    day_of_forecast,
    output_dir as default_output_dir,
)

# Import data loading
from lt_forecasting.scr.data_loading import load_snow_data

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

# Number of days to average SWE before issue date
SWE_AVERAGING_DAYS = 10

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


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate performance metrics.

    Args:
        y_true: Array of observed values
        y_pred: Array of predicted values

    Returns:
        Dictionary with r2, rmse, mae, pbias, nse, n_samples
    """
    # Drop NaN values
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

    # Calculate metrics
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = mean_absolute_error(y_true_clean, y_pred_clean)

    # PBIAS: Percent Bias
    sum_obs = y_true_clean.sum()
    pbias = (
        100 * (y_pred_clean - y_true_clean).sum() / sum_obs if sum_obs != 0 else np.nan
    )

    # NSE: Nash-Sutcliffe Efficiency
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
    Calculate R² for each code (basin) separately.

    Args:
        df: DataFrame with observations and predictions
        obs_col: Name of observation column
        pred_col: Name of prediction column
        min_samples: Minimum samples required per code

    Returns:
        DataFrame with columns: code, r2, n_samples
    """
    results = []

    for code in df["code"].unique():
        code_data = df[df["code"] == code].dropna(subset=[obs_col, pred_col])

        if len(code_data) < min_samples:
            continue

        y_true = code_data[obs_col].values
        y_pred = code_data[pred_col].values

        try:
            r2 = r2_score(y_true, y_pred)
        except Exception:
            r2 = np.nan

        results.append(
            {
                "code": code,
                "r2": r2,
                "n_samples": len(code_data),
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# DATA LOADING
# =============================================================================


def load_swe_data(swe_path: str | None = None) -> pd.DataFrame | None:
    """
    Load SWE data from environment variable path or specified path.

    Args:
        swe_path: Optional path to SWE file. If None, uses env variable.

    Returns:
        DataFrame with columns: date, code, SWE
    """
    if swe_path is None:
        swe_path = os.getenv("path_SWE_00003")

    if swe_path is None or not Path(swe_path).exists():
        logger.warning(f"SWE data path not found: {swe_path}")
        return None

    logger.info(f"Loading SWE data from {swe_path}")
    swe_df = load_snow_data(swe_path, "SWE")

    if swe_df is None or swe_df.empty:
        logger.warning("SWE data is empty")
        return None

    # Standardize column names
    if "SWE" not in swe_df.columns:
        # Try to find SWE column
        swe_cols = [c for c in swe_df.columns if "swe" in c.lower()]
        if swe_cols:
            swe_df = swe_df.rename(columns={swe_cols[0]: "SWE"})
        else:
            logger.warning("Could not find SWE column")
            return None

    logger.info(
        f"Loaded SWE data: {len(swe_df)} rows, codes: {swe_df['code'].nunique()}"
    )
    return swe_df


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

    For each issue date, loads monthly predictions for target months and
    averages them to get seasonal prediction. Also averages monthly
    observations to get seasonal observation.

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

        logger.info(f"Processing issue date {label} -> targets {target_months}")

        # Calculate required horizons for each target month
        horizons_needed = [get_horizon(issue_month, tm) for tm in target_months]

        # Filter predictions for this issue month
        issue_preds = predictions_df[
            predictions_df["issue_date"].dt.month == issue_month
        ].copy()

        if issue_preds.empty:
            logger.warning(f"No predictions found for issue month {issue_month}")
            continue

        # Get unique issue years
        issue_years = issue_preds["issue_date"].dt.year.unique()

        for model in issue_preds["model"].unique():
            model_preds = issue_preds[issue_preds["model"] == model]

            for issue_year in issue_years:
                # Filter for this issue year
                year_preds = model_preds[
                    model_preds["issue_date"].dt.year == issue_year
                ]

                # Collect monthly predictions for target months
                monthly_preds_list = []
                monthly_obs_list = []

                for target_month, horizon in zip(target_months, horizons_needed):
                    # Determine target year (handle year boundary)
                    target_year = (
                        issue_year if target_month >= issue_month else issue_year + 1
                    )

                    # Find prediction for this horizon
                    horizon_pred = year_preds[year_preds["horizon"] == horizon]

                    if horizon_pred.empty:
                        continue

                    # Get prediction (average if multiple per code)
                    for code in horizon_pred["code"].unique():
                        code_pred = horizon_pred[horizon_pred["code"] == code]

                        pred_val = code_pred["Q_pred"].mean()
                        monthly_preds_list.append(
                            {
                                "code": code,
                                "target_year": target_year,
                                "target_month": target_month,
                                "Q_pred": pred_val,
                            }
                        )

                        # Get corresponding observation
                        obs = monthly_obs[
                            (monthly_obs["code"] == code)
                            & (monthly_obs["year"] == target_year)
                            & (monthly_obs["month"] == target_month)
                        ]

                        if not obs.empty:
                            monthly_obs_list.append(
                                {
                                    "code": code,
                                    "target_year": target_year,
                                    "target_month": target_month,
                                    "Q_obs": obs["Q_obs_monthly"].iloc[0],
                                }
                            )

                if not monthly_preds_list:
                    continue

                # Convert to DataFrames
                preds_df = pd.DataFrame(monthly_preds_list)
                obs_df = pd.DataFrame(monthly_obs_list) if monthly_obs_list else None

                # Calculate seasonal average per code
                seasonal_pred = preds_df.groupby("code")["Q_pred"].mean().reset_index()
                seasonal_pred.columns = ["code", "Q_pred_seasonal"]

                if obs_df is not None and not obs_df.empty:
                    seasonal_obs = obs_df.groupby("code")["Q_obs"].mean().reset_index()
                    seasonal_obs.columns = ["code", "Q_obs_seasonal"]

                    # Merge predictions and observations
                    seasonal = seasonal_pred.merge(seasonal_obs, on="code", how="left")
                else:
                    seasonal = seasonal_pred
                    seasonal["Q_obs_seasonal"] = np.nan

                # Add metadata
                seasonal["issue_year"] = issue_year
                seasonal["issue_month"] = issue_month
                seasonal["issue_day"] = issue_day
                seasonal["issue_label"] = label
                seasonal["model"] = model
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
# SWE-BASED PREDICTION (LOOCV)
# =============================================================================


def calculate_swe_predictor(
    swe_df: pd.DataFrame,
    issue_month: int,
    issue_day: int,
    n_days: int = SWE_AVERAGING_DAYS,
) -> pd.DataFrame:
    """
    Calculate average SWE from n_days before issue date per basin/year.

    Args:
        swe_df: DataFrame with SWE data (date, code, SWE)
        issue_month: Issue date month
        issue_day: Issue date day
        n_days: Number of days to average before issue date

    Returns:
        DataFrame with columns: code, year, SWE_avg
    """
    results = []

    for year in swe_df["date"].dt.year.unique():
        # Create issue date for this year
        try:
            issue_date = pd.Timestamp(year=year, month=issue_month, day=issue_day)
        except ValueError:
            # Handle edge cases (e.g., Feb 29)
            continue

        # Calculate date range (issue_date - n_days + 1 to issue_date)
        start_date = issue_date - pd.Timedelta(days=n_days - 1)
        end_date = issue_date

        # Filter SWE data for this date range
        mask = (swe_df["date"] >= start_date) & (swe_df["date"] <= end_date)
        period_swe = swe_df[mask]

        if period_swe.empty:
            continue

        # Calculate average SWE per code
        avg_swe = period_swe.groupby("code")["SWE"].mean().reset_index()
        avg_swe.columns = ["code", "SWE_avg"]
        avg_swe["year"] = year

        results.append(avg_swe)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def swe_loocv_prediction(
    swe_predictor: pd.DataFrame,
    seasonal_obs: pd.DataFrame,
    min_train_years: int = 3,
) -> pd.DataFrame:
    """
    Perform leave-one-year-out cross-validation for SWE-based predictions.

    For each year, trains a linear regression on all other years to predict
    seasonal discharge from SWE.

    Args:
        swe_predictor: DataFrame with SWE_avg per code/year
        seasonal_obs: DataFrame with Q_obs_seasonal per code/year
        min_train_years: Minimum training years required per basin

    Returns:
        DataFrame with columns: code, year, Q_pred_swe, Q_obs_seasonal, SWE_avg
    """
    # Merge SWE predictor with seasonal observations
    merged = swe_predictor.merge(seasonal_obs, on=["code", "year"], how="inner")

    if merged.empty:
        logger.warning("No matching data between SWE predictor and observations")
        return pd.DataFrame()

    # Filter out rows with NaN values
    merged = merged.dropna(subset=["SWE_avg", "Q_obs_seasonal"])

    if merged.empty:
        logger.warning("No valid data after removing NaN values")
        return pd.DataFrame()

    results = []
    all_years = sorted(merged["year"].unique())

    for code in merged["code"].unique():
        code_data = merged[merged["code"] == code].copy()

        # Skip if insufficient data
        if len(code_data) < min_train_years + 1:
            continue

        for test_year in all_years:
            # Split data
            train_data = code_data[code_data["year"] != test_year]
            test_data = code_data[code_data["year"] == test_year]

            if len(train_data) < min_train_years or test_data.empty:
                continue

            # Ensure no NaN in training data
            train_data = train_data.dropna(subset=["SWE_avg", "Q_obs_seasonal"])
            if len(train_data) < min_train_years:
                continue

            # Train linear regression
            X_train = train_data[["SWE_avg"]].values
            y_train = train_data["Q_obs_seasonal"].values

            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict
            X_test = test_data[["SWE_avg"]].values
            y_pred = model.predict(X_test)[0]

            results.append(
                {
                    "code": code,
                    "year": test_year,
                    "Q_pred_swe": y_pred,
                    "Q_obs_seasonal": test_data["Q_obs_seasonal"].iloc[0],
                    "SWE_avg": test_data["SWE_avg"].iloc[0],
                    "n_train_years": len(train_data),
                }
            )

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================


def calculate_swe_correlations(
    swe_predictor: pd.DataFrame,
    seasonal_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate Pearson correlation between SWE and seasonal discharge per basin.

    Args:
        swe_predictor: DataFrame with SWE_avg per code/year
        seasonal_obs: DataFrame with Q_obs_seasonal per code/year

    Returns:
        DataFrame with columns: code, correlation, p_value, n_samples
    """
    # Merge SWE predictor with seasonal observations
    merged = swe_predictor.merge(seasonal_obs, on=["code", "year"], how="inner")

    if merged.empty:
        return pd.DataFrame()

    # Filter out rows with NaN values
    merged = merged.dropna(subset=["SWE_avg", "Q_obs_seasonal"])

    if merged.empty:
        return pd.DataFrame()

    results = []
    for code in merged["code"].unique():
        code_data = merged[merged["code"] == code].dropna(
            subset=["SWE_avg", "Q_obs_seasonal"]
        )

        if len(code_data) < 3:
            continue

        # Calculate Pearson correlation
        r, p_value = stats.pearsonr(code_data["SWE_avg"], code_data["Q_obs_seasonal"])

        results.append(
            {
                "code": code,
                "correlation": r,
                "p_value": p_value,
                "n_samples": len(code_data),
            }
        )

    return pd.DataFrame(results)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_seasonal_scatter(
    seasonal_df: pd.DataFrame,
    models: list[str],
    issue_label: str,
    output_path: Path,
    swe_predictions: pd.DataFrame | None = None,
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
        swe_predictions: Optional DataFrame with SWE-based predictions
        forecast_type: "6-month" or "3-month"
        code_filter: If specified, only plot this specific code (basin)

    Returns:
        matplotlib Figure
    """
    # Filter data
    df = seasonal_df[seasonal_df["issue_label"] == issue_label].copy()

    if df.empty:
        logger.warning(f"No data for issue date {issue_label}")
        return plt.figure()

    # Filter for specific code if requested
    if code_filter is not None:
        df = df[df["code"] == code_filter].copy()
        if swe_predictions is not None:
            swe_predictions = swe_predictions[
                swe_predictions["code"] == code_filter
            ].copy()

        if df.empty:
            logger.warning(
                f"No data for code {code_filter} at issue date {issue_label}"
            )
            return plt.figure()

    n_models = len(models) + (1 if swe_predictions is not None else 0)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Color palette
    colors = sns.color_palette("husl", n_models)

    all_models = list(models) + (["SWE_LR"] if swe_predictions is not None else [])

    for idx, model in enumerate(all_models):
        ax = axes[idx]

        if model == "SWE_LR" and swe_predictions is not None:
            # Use SWE predictions
            plot_data = swe_predictions
            x_col = "Q_obs_seasonal"
            y_col = "Q_pred_swe"
            color = colors[idx]
        else:
            # Use ML model predictions
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

        # Drop NaN values
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

        # Calculate metrics
        metrics = calculate_metrics(x, y)

        # Scatter plot
        ax.scatter(
            x, y, alpha=0.6, color=color, s=40, edgecolors="white", linewidths=0.5
        )

        # 1:1 line
        max_val = max(x.max(), y.max()) * 1.1
        min_val = min(x.min(), y.min()) * 0.9
        ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5)

        # Linear regression line
        slope, intercept, r_value, *_ = stats.linregress(x, y)
        x_line = np.array([min_val, max_val])
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color=color, linewidth=2, alpha=0.8)

        # Labels
        ax.set_xlabel("Observed (m³/s)", fontsize=10)
        ax.set_ylabel("Predicted (m³/s)", fontsize=10)
        ax.set_title(
            f"{model}\nR²={metrics['r2']:.3f}, n={metrics['n_samples']}", fontsize=11
        )

        # Equal aspect ratio
        ax.set_xlim(min_val, max_val)
        ax.set_ylim(min_val, max_val)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)

    # Hide unused axes
    for idx in range(len(all_models), len(axes)):
        axes[idx].set_visible(False)

    code_str = f" - Code {code_filter}" if code_filter is not None else ""
    fig.suptitle(
        f"{forecast_type} Seasonal Forecast - Issue Date: {issue_label}{code_str}",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved scatter plot to {output_path}")

    return fig


def plot_swe_correlation_boxplot(
    correlations: dict[str, pd.DataFrame],
    output_path: Path,
) -> plt.Figure:
    """
    Create boxplot showing SWE-discharge correlation distribution across basins.

    Args:
        correlations: Dict mapping issue_label to correlation DataFrame
        output_path: Path to save figure

    Returns:
        matplotlib Figure
    """
    # Prepare data for plotting
    plot_data = []
    for label, corr_df in correlations.items():
        for _, row in corr_df.iterrows():
            plot_data.append(
                {
                    "Issue Date": label,
                    "Correlation": row["correlation"],
                    "code": row["code"],
                }
            )

    if not plot_data:
        logger.warning("No correlation data to plot")
        return plt.figure()

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(
        data=plot_df,
        x="Issue Date",
        y="Correlation",
        hue="Issue Date",
        ax=ax,
        palette="Blues",
        width=0.6,
        legend=False,
    )

    # Add individual points
    sns.stripplot(
        data=plot_df,
        x="Issue Date",
        y="Correlation",
        ax=ax,
        color="black",
        alpha=0.4,
        size=4,
    )

    # Reference lines
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.axhline(
        y=0.7,
        color="green",
        linestyle=":",
        linewidth=1,
        alpha=0.5,
        label="Strong (r=0.7)",
    )
    ax.axhline(y=-0.7, color="green", linestyle=":", linewidth=1, alpha=0.5)

    ax.set_ylabel("Pearson Correlation (SWE vs Seasonal Q)", fontsize=12)
    ax.set_xlabel("Forecast Issue Date", fontsize=12)
    ax.set_title(
        "SWE-Discharge Correlation by Issue Date",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-1, 1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved SWE correlation boxplot to {output_path}")

    return fig


def plot_r2_vs_issue_date(
    seasonal_df: pd.DataFrame,
    models: list[str],
    output_path: Path,
    swe_results: dict[str, pd.DataFrame] | None = None,
    forecast_type: str = "6-month",
) -> plt.Figure:
    """
    Plot R² distribution across codes for each model and issue date.

    Shows boxplot of per-code R² values to visualize skill distribution.

    Args:
        seasonal_df: DataFrame with seasonal forecasts
        models: List of models to plot
        output_path: Path to save figure
        swe_results: Optional dict mapping issue_label to SWE predictions DataFrame
        forecast_type: "6-month" or "3-month"

    Returns:
        matplotlib Figure
    """
    # Calculate R² per code, model, and issue date
    plot_data = []

    issue_labels = sorted(seasonal_df["issue_label"].unique())

    for model in models:
        model_df = seasonal_df[seasonal_df["model"] == model]

        for label in issue_labels:
            label_df = model_df[model_df["issue_label"] == label]

            if label_df.empty:
                continue

            # Calculate R² per code
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

    # Add SWE baseline if available
    if swe_results:
        for label, swe_df in swe_results.items():
            if swe_df.empty:
                continue

            r2_per_code = calculate_r2_per_code(
                swe_df, obs_col="Q_obs_seasonal", pred_col="Q_pred_swe"
            )

            for _, row in r2_per_code.iterrows():
                plot_data.append(
                    {
                        "Issue Date": label,
                        "R²": row["r2"],
                        "Model": "SWE_LR",
                        "code": row["code"],
                    }
                )

    if not plot_data:
        logger.warning("No data for R² vs issue date plot")
        return plt.figure()

    plot_df = pd.DataFrame(plot_data)

    # Create figure with boxplots
    fig, ax = plt.subplots(figsize=(12, 7))

    all_models = list(models) + (["SWE_LR"] if swe_results else [])
    colors = sns.color_palette("husl", len(all_models))

    sns.boxplot(
        data=plot_df,
        x="Issue Date",
        y="R²",
        hue="Model",
        hue_order=all_models,
        palette=colors,
        ax=ax,
        width=0.7,
    )

    # Reference line
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
    swe_results: dict[str, pd.DataFrame],
    output_path: Path,
    issue_dates: dict,
) -> plt.Figure:
    """
    Create boxplot comparing R² distribution per code for ML models vs SWE baseline.

    Args:
        seasonal_df: DataFrame with seasonal forecasts
        models: List of ML models
        swe_results: Dict mapping issue_label to SWE prediction DataFrame
        output_path: Path to save figure
        issue_dates: Issue dates configuration

    Returns:
        matplotlib Figure
    """
    # Calculate R² per code for each model (aggregated across all issue dates)
    plot_data = []

    # ML models - aggregate across all issue dates
    for model in models:
        model_df = seasonal_df[seasonal_df["model"] == model]

        if model_df.empty:
            continue

        # Calculate R² per code across all issue dates
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

    # SWE baseline - aggregate across all issue dates
    if swe_results:
        all_swe = []
        for label, swe_df in swe_results.items():
            if not swe_df.empty:
                all_swe.append(swe_df)

        if all_swe:
            combined_swe = pd.concat(all_swe, ignore_index=True)
            r2_per_code = calculate_r2_per_code(
                combined_swe, obs_col="Q_obs_seasonal", pred_col="Q_pred_swe"
            )

            for _, row in r2_per_code.iterrows():
                plot_data.append(
                    {
                        "R²": row["r2"],
                        "Model": "SWE_LR",
                        "code": row["code"],
                    }
                )

    if not plot_data:
        logger.warning("No data for skill comparison plot")
        return plt.figure()

    plot_df = pd.DataFrame(plot_data)

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create grouped boxplot
    all_models = list(models) + (["SWE_LR"] if swe_results else [])
    colors = sns.color_palette("husl", len(all_models))

    sns.boxplot(
        data=plot_df,
        x="Model",
        y="R²",
        hue="Model",
        order=all_models,
        palette=colors,
        ax=ax,
        width=0.6,
        legend=False,
    )

    # Add individual points
    sns.stripplot(
        data=plot_df,
        x="Model",
        y="R²",
        order=all_models,
        ax=ax,
        color="black",
        alpha=0.3,
        size=3,
    )

    # Reference line
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

    ax.set_ylabel("R² (per basin)", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(
        "ML Models vs SWE Baseline - R² Distribution Across Basins",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylim(-0.5, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(title="Model", loc="best", fontsize=9)

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
    swe_results: dict[str, pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    """
    Generate CSV summary of per-code R² statistics per model and issue date.

    Now calculates R² per code and reports distribution statistics.

    Args:
        seasonal_df: DataFrame with seasonal forecasts
        models: List of ML models
        swe_results: Dict mapping issue_label to SWE prediction DataFrame
        output_path: Path to save CSV

    Returns:
        Summary DataFrame
    """
    results = []

    issue_labels = seasonal_df["issue_label"].unique()

    for label in issue_labels:
        # ML models
        for model in models:
            model_df = seasonal_df[
                (seasonal_df["model"] == model) & (seasonal_df["issue_label"] == label)
            ]

            if model_df.empty:
                continue

            # Calculate R² per code
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

        # SWE baseline
        if label in swe_results and not swe_results[label].empty:
            swe_df = swe_results[label]
            r2_per_code = calculate_r2_per_code(
                swe_df, obs_col="Q_obs_seasonal", pred_col="Q_pred_swe"
            )

            if not r2_per_code.empty:
                r2_values = r2_per_code["r2"].dropna()

                results.append(
                    {
                        "issue_label": label,
                        "model": "SWE_LR",
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
    swe_path: str | None = None,
) -> None:
    """
    Main processing function for seasonal forecast visualization.

    Args:
        region: "Kyrgyzstan" or "Tajikistan"
        forecast_type: "6_month", "3_month", or "both"
        models: List of models to evaluate
        output_dir: Base output directory
        swe_path: Optional path to SWE data
    """
    # Determine paths based on region
    if region == "Tajikistan":
        pred_config = taj_path_config
    else:
        pred_config = kgz_path_config

    # Load all horizon predictions
    horizons = list(day_of_forecast.keys())
    logger.info(f"Loading predictions for {region}...")

    predictions_df = load_predictions(
        base_path=pred_config["pred_dir"],
        horizons=horizons,
    )

    if predictions_df.empty:
        logger.error("No predictions loaded. Exiting.")
        return

    # Load observations
    logger.info(f"Loading observations from {pred_config['obs_file']}...")
    obs_df = load_observations(pred_config["obs_file"])
    monthly_obs = calculate_target(obs_df)

    # Load SWE data
    swe_df = load_swe_data(swe_path)

    # Process forecast types
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

        # Reconstruct seasonal forecasts
        seasonal_df = reconstruct_seasonal_forecasts(
            predictions_df, monthly_obs, issue_dates
        )

        if seasonal_df.empty:
            logger.warning(f"No seasonal forecasts for {fc_type}")
            continue

        # Filter to requested models
        available_models = [m for m in models if m in seasonal_df["model"].unique()]
        if not available_models:
            logger.warning(f"None of the requested models found in {fc_type} forecasts")
            continue

        logger.info(f"Available models: {available_models}")

        # Process SWE baseline if data available
        swe_results = {}
        swe_correlations = {}

        if swe_df is not None:
            logger.info("Processing SWE baseline...")

            for (issue_month, issue_day), config in issue_dates.items():
                label = config["label"]

                # Calculate SWE predictor
                swe_predictor = calculate_swe_predictor(swe_df, issue_month, issue_day)

                if swe_predictor.empty:
                    continue

                # Get seasonal observations for matching years
                seasonal_obs = (
                    seasonal_df[seasonal_df["issue_label"] == label]
                    .groupby(["code", "issue_year"])["Q_obs_seasonal"]
                    .first()
                    .reset_index()
                )
                seasonal_obs.columns = ["code", "year", "Q_obs_seasonal"]

                if seasonal_obs.empty:
                    continue

                # LOOCV prediction
                swe_pred = swe_loocv_prediction(swe_predictor, seasonal_obs)
                if not swe_pred.empty:
                    swe_results[label] = swe_pred
                    logger.info(f"  {label}: {len(swe_pred)} SWE predictions")

                # Correlation analysis
                corr_df = calculate_swe_correlations(swe_predictor, seasonal_obs)
                if not corr_df.empty:
                    swe_correlations[label] = corr_df

        # Generate plots
        logger.info("Generating plots...")

        # 1. Scatter plots per issue date
        for (issue_month, issue_day), config in issue_dates.items():
            label = config["label"]
            label_clean = label.replace(" ", "")

            swe_for_plot = swe_results.get(label) if swe_results else None

            fig = plot_seasonal_scatter(
                seasonal_df=seasonal_df,
                models=available_models,
                issue_label=label,
                output_path=fc_output_dir / f"scatter_{label_clean}.png",
                swe_predictions=swe_for_plot,
                forecast_type=fc_type.replace("_", "-"),
            )
            plt.close(fig)

        # 2. R² distribution vs issue date
        fig = plot_r2_vs_issue_date(
            seasonal_df=seasonal_df,
            models=available_models,
            output_path=fc_output_dir / "r2_distribution_vs_issue_date.png",
            swe_results=swe_results if swe_results else None,
            forecast_type=fc_type.replace("_", "-"),
        )
        plt.close(fig)

        # 3. Skill comparison bar chart
        fig = plot_skill_comparison(
            seasonal_df=seasonal_df,
            models=available_models,
            swe_results=swe_results,
            output_path=fc_output_dir / "skill_comparison.png",
            issue_dates=issue_dates,
        )
        plt.close(fig)

        # 4. Metrics summary CSV
        generate_metrics_summary(
            seasonal_df=seasonal_df,
            models=available_models,
            swe_results=swe_results,
            output_path=fc_output_dir / "metrics_summary.csv",
        )

        # 5. SWE correlation boxplot (in swe_analysis subfolder)
        if swe_correlations:
            swe_output_dir = output_dir / region.lower() / "seasonal" / "swe_analysis"
            swe_output_dir.mkdir(parents=True, exist_ok=True)

            fig = plot_swe_correlation_boxplot(
                correlations=swe_correlations,
                output_path=swe_output_dir / f"swe_correlation_boxplot_{fc_type}.png",
            )
            plt.close(fig)

            # Save correlation summary
            all_corr = []
            for label, corr_df in swe_correlations.items():
                corr_df = corr_df.copy()
                corr_df["issue_label"] = label
                all_corr.append(corr_df)

            if all_corr:
                corr_summary = pd.concat(all_corr, ignore_index=True)
                corr_summary.to_csv(
                    swe_output_dir / f"correlation_summary_{fc_type}.csv", index=False
                )

    logger.info("\nProcessing complete!")


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Evaluate seasonal forecasts and compare with SWE baseline."
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
    parser.add_argument(
        "--swe-path",
        type=str,
        default=None,
        help="Path to SWE data file (uses env variable if not specified)",
    )

    args = parser.parse_args()

    # Set global plot style
    set_global_plot_style()

    # Determine output directory
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
        swe_path=args.swe_path,
    )


if __name__ == "__main__":
    main()
