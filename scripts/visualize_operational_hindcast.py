"""
Visualize Operational Hindcast

This script loads operational forecasts (model_name_forecast.csv) for a specific basin
and creates time series comparison plots with multiple models and metrics.

Features:
- Support for multiple models in a single comparison plot
- Calculate MAE and RMSE for all models
- Display metrics in the figure caption/subtitle
- Support probabilistic forecast metrics (CRPS, coverage) for models with quantiles
- One separate plot per lead time (horizon)
- X-axis: year-month (e.g., 2020-Apr, 2020-May, ..., 2024-Sep)
"""

import logging
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from dev_tools.eval_scr.metric_functions import (
    calculate_coverage,
    calculate_mean_CRPS,
    calculate_R2,
    sdivsigma_nse,
)
from lt_forecasting.scr.metrics import calculate_MAE, calculate_RMSE, calculate_NMAE

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a stream handler to output logs to the terminal
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Load environment variables
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# =============================================================================
# CONFIGURATION - Set your target basin code here
# =============================================================================
TARGET_CODE: int = 16936  # Basin code to analyze

# Region configuration: "Kyrgyzstan" or "Tajikistan"
REGION: str = "Kyrgyzstan"  # Set to "Kyrgyzstan" or "Tajikistan"

# Models to analyze (multiple models supported)
MODEL_NAMES: list[str] = ["LR_Base", "LR_SM", "MC_ALD", "SM_GBT_Norm"]

YEARS = [2023, 2024, 2025]

day_of_forecast = {
    "month_0": 10,
    "month_1": 25,
    "month_2": 25,
    "month_3": 25,
    "month_4": 25,
    "month_5": 25,
    "month_6": 25,
    # "month_7": 25,
    # "month_8": 25,
    # "month_9": 25,
}

kgz_path_config = {
    "pred_dir": os.getenv("kgz_path_discharge"),
    "obs_file": os.getenv("kgz_path_base_pred"),
}

taj_path_config = {
    "pred_dir": os.getenv("taj_path_base_pred"),
    "obs_file": os.getenv("taj_path_discharge"),
}

output_dir = os.getenv("out_dir_op_lt")

horizons = list(day_of_forecast.keys())

month_renaming = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

# Short month names for x-axis labels
month_short = {
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

# Color palette for models
MODEL_COLORS = {
    "LR_Base": "#1f77b4",  # blue
    "LR_SM": "#ff7f0e",  # orange
    "MC_ALD": "#2ca02c",  # green
    "SM_GBT_Norm": "#d62728",  # red
}

# Default color cycle for models not in the palette
DEFAULT_COLORS = ["#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

# Color palette for lead times (horizons)
HORIZON_COLORS = {
    0: "#1f77b4",  # blue - month_0
    1: "#ff7f0e",  # orange - month_1
    2: "#2ca02c",  # green - month_2
    3: "#d62728",  # red - month_3
    4: "#9467bd",  # purple - month_4
    5: "#8c564b",  # brown - month_5
}


def load_observations(obs_file: str) -> pd.DataFrame:
    """
    Load observed discharge data from a CSV file.

    Args:
        obs_file: Path to the CSV file containing observed discharge data.

    Returns:
        DataFrame with columns: date, code, discharge (daily observations)
    """
    obs_df = pd.read_csv(obs_file)
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    obs_df["code"] = obs_df["code"].astype(int)

    return obs_df


def calculate_monthly_observations(
    daily_obs: pd.DataFrame,
    target_code: int,
) -> pd.DataFrame:
    """
    Aggregate daily observations to monthly means for a specific basin.

    Args:
        daily_obs: DataFrame with columns: date, code, discharge
        target_code: Basin code to filter for

    Returns:
        DataFrame with columns: year, month, Q_obs_monthly
    """
    df = daily_obs[daily_obs["code"] == target_code].copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_obs = (
        df.groupby(["year", "month"])["discharge"]
        .mean()
        .reset_index()
        .rename(columns={"discharge": "Q_obs_monthly"})
    )

    return monthly_obs


def calculate_monthly_climatology(
    monthly_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate long-term mean and std for each calendar month.

    Args:
        monthly_obs: DataFrame with columns: year, month, Q_obs_monthly

    Returns:
        DataFrame with columns: month, Q_ltm_monthly, Q_std_monthly, Q_min_monthly, Q_max_monthly
    """
    monthly_clim = (
        monthly_obs.groupby("month")["Q_obs_monthly"]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    monthly_clim.columns = [
        "month",
        "Q_ltm_monthly",
        "Q_std_monthly",
        "Q_min_monthly",
        "Q_max_monthly",
        "n_years",
    ]

    logger.info(f"Calculated monthly climatology for {len(monthly_clim)} months")

    return monthly_clim


def load_operational_forecasts(
    base_path: str,
    horizons: list[str],
    target_code: int,
    model_names: list[str],
) -> pd.DataFrame:
    """
    Load operational forecasts for a specific basin and multiple models.

    Operational forecasts have only one row per basin for a given issue date.
    Files are named model_name_hindcast.csv.

    Args:
        base_path: Base directory containing horizon subdirectories
        horizons: List of horizon identifiers (e.g., ["month_0", "month_1", ...])
        target_code: Basin code to filter for
        model_names: List of model names to load

    Returns:
        DataFrame with forecast data for the target basin across all horizons and models,
        with a 'model' column identifying each model.
    """
    base_path = Path(base_path)
    all_forecasts = []

    for model_name in model_names:
        for horizon in horizons:
            horizon_path = base_path / horizon
            if not horizon_path.exists():
                logger.warning(f"Horizon directory not found: {horizon_path}")
                continue

            # Extract horizon number from 'month_X' format
            horizon_num = int(horizon.split("_")[1])

            # Look for the model directory
            model_dir = horizon_path / model_name
            if not model_dir.exists():
                logger.warning(f"Model directory not found: {model_dir}")
                continue

            # Load forecast file (hindcast)
            forecast_file = model_dir / f"{model_name}_hindcast.csv"

            if not forecast_file.exists():
                logger.warning(f"Forecast file not found: {forecast_file}")
                continue

            try:
                df = pd.read_csv(forecast_file)
            except Exception as e:
                logger.error(f"Failed to read {forecast_file}: {e}")
                continue

            if df.empty:
                logger.debug(f"Empty dataframe in {forecast_file}")
                continue

            # Convert dates to datetime (use format='mixed' to handle mixed formats)
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df["valid_from"] = pd.to_datetime(df["valid_from"], format="mixed")
            df["valid_to"] = pd.to_datetime(df["valid_to"], format="mixed")

            # Convert code to int
            df["code"] = df["code"].astype(int)

            # Filter for target code
            df = df[df["code"] == target_code].copy()

            if df.empty:
                logger.debug(f"No data for code {target_code} in {forecast_file}")
                continue

            # Find quantile columns
            quantile_cols = [col for col in df.columns if re.fullmatch(r"Q\d+", col)]

            # Find prediction column (Q_model_name or Q_*)
            excluded_cols = ["Q_obs", "Q_Obs", "Q_OBS"] + quantile_cols
            q_cols = [
                c for c in df.columns if c.startswith("Q_") and c not in excluded_cols
            ]

            if not q_cols:
                logger.warning(f"No prediction column found in {forecast_file}")
                continue

            # Use the first Q_ column as Q_pred
            q_pred_col = q_cols[0]

            # Create result DataFrame
            result_df = pd.DataFrame(
                {
                    "code": df["code"],
                    "issue_date": df["date"],
                    "valid_from": df["valid_from"],
                    "valid_to": df["valid_to"],
                    "Q_pred": df[q_pred_col],
                    "horizon": horizon_num,
                    "model": model_name,
                }
            )

            # Add quantile columns if they exist
            for qcol in quantile_cols:
                if qcol in df.columns:
                    result_df[qcol] = df[qcol].values

            # Add Q_loc column if it exists (for MC_ALD model)
            if "Q_loc" in df.columns:
                result_df["Q_loc"] = df["Q_loc"].values

            all_forecasts.append(result_df)
            logger.info(
                f"Loaded forecast for {model_name} horizon {horizon_num} "
                f"from {forecast_file}"
            )

    if not all_forecasts:
        logger.error(
            f"No forecasts loaded for code {target_code}, models {model_names}"
        )
        return pd.DataFrame()

    combined_df = pd.concat(all_forecasts, ignore_index=True)
    logger.info(
        f"Loaded {len(combined_df)} forecast records for code {target_code} "
        f"across {len(model_names)} models"
    )

    return combined_df


def create_forecast_summary(
    forecasts_df: pd.DataFrame,
    monthly_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary dataframe with predictions and observations for visualization.

    Args:
        forecasts_df: DataFrame with operational forecasts (must have 'model' column)
        monthly_obs: DataFrame with monthly observed discharge (year, month, Q_obs_monthly)

    Returns:
        DataFrame with columns: year, month, year_month, Q_obs, Q_pred, model,
        and quantile columns if available
    """
    if forecasts_df.empty:
        return pd.DataFrame()

    df = forecasts_df.copy()

    # Extract target year and month from valid_from
    df["year"] = df["valid_from"].dt.year
    df["month"] = df["valid_from"].dt.month

    # Create year_month column for time series x-axis (e.g., "2020-04")
    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )

    # Merge with monthly observations
    df = df.merge(
        monthly_obs[["year", "month", "Q_obs_monthly"]],
        on=["year", "month"],
        how="left",
    )

    # Rename Q_obs_monthly to Q_obs for consistency
    df = df.rename(columns={"Q_obs_monthly": "Q_obs"})

    # Sort by year_month and model
    df = df.sort_values(["year", "month", "model"]).reset_index(drop=True)

    # Add month name for display
    df["month_name"] = df["month"].map(month_short)

    return df


def calculate_hindcast_metrics(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate MAE, RMSE, and probabilistic metrics per model.

    Args:
        df: DataFrame with columns Q_obs, Q_pred, model, and optional quantile columns

    Returns:
        DataFrame with columns: model, MAE, RMSE, Coverage_90, CRPS, n_samples
    """
    if df.empty:
        return pd.DataFrame()

    metrics_list = []
    quantile_levels = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])

    for model in df["model"].unique():
        model_df = df[df["model"] == model].copy()

        # Drop rows with NaN in Q_obs or Q_pred
        valid_df = model_df.dropna(subset=["Q_obs", "Q_pred"])

        if valid_df.empty:
            logger.warning(f"No valid data for model {model}")
            continue

        obs = valid_df["Q_obs"].values
        pred = valid_df["Q_pred"].values
        n_samples = len(obs)

        # Calculate MAE and RMSE (non-normalized)
        mae = calculate_MAE(obs, pred, normalize=False)
        rmse = calculate_RMSE(obs, pred, normalize=False)

        metrics = {
            "model": model,
            "MAE": mae,
            "RMSE": rmse,
            "n_samples": n_samples,
        }

        # Check for quantile columns (for probabilistic metrics)
        has_q5 = "Q5" in valid_df.columns and not valid_df["Q5"].isna().all()
        has_q95 = "Q95" in valid_df.columns and not valid_df["Q95"].isna().all()

        if has_q5 and has_q95:
            # Calculate 90% coverage
            coverage_90 = calculate_coverage(
                obs,
                valid_df["Q5"].values,
                valid_df["Q95"].values,
            )
            metrics["Coverage_90"] = coverage_90

        # Check for all quantile columns for CRPS
        quantile_cols = ["Q5", "Q10", "Q25", "Q50", "Q75", "Q90", "Q95"]
        has_all_quantiles = all(
            col in valid_df.columns and not valid_df[col].isna().all()
            for col in quantile_cols
        )

        if has_all_quantiles:
            # Build quantile forecast array
            quantile_forecasts = valid_df[quantile_cols].values
            crps = calculate_mean_CRPS(obs, quantile_forecasts, quantile_levels)
            metrics["CRPS"] = crps

        metrics_list.append(metrics)

    if not metrics_list:
        return pd.DataFrame()

    return pd.DataFrame(metrics_list)


def format_metrics_caption(metrics_df: pd.DataFrame) -> str:
    """
    Format metrics as figure caption.

    Args:
        metrics_df: DataFrame with columns model, MAE, RMSE, Coverage_90, CRPS

    Returns:
        Formatted string for figure caption
    """
    if metrics_df.empty:
        return ""

    lines = []

    # Format MAE
    mae_parts = []
    for _, row in metrics_df.iterrows():
        if not np.isnan(row["MAE"]):
            mae_parts.append(f"{row['model']}={row['MAE']:.1f}")
    if mae_parts:
        lines.append(f"MAE [m³/s]: {', '.join(mae_parts)}")

    # Format RMSE
    rmse_parts = []
    for _, row in metrics_df.iterrows():
        if not np.isnan(row["RMSE"]):
            rmse_parts.append(f"{row['model']}={row['RMSE']:.1f}")
    if rmse_parts:
        lines.append(f"RMSE [m³/s]: {', '.join(rmse_parts)}")

    # Format Coverage (if available)
    if "Coverage_90" in metrics_df.columns:
        cov_parts = []
        for _, row in metrics_df.iterrows():
            if "Coverage_90" in row and not np.isnan(row.get("Coverage_90", np.nan)):
                cov_parts.append(f"{row['model']}={row['Coverage_90']:.2f}")
        if cov_parts:
            lines.append(f"Coverage (90%): {', '.join(cov_parts)}")

    # Format CRPS (if available)
    if "CRPS" in metrics_df.columns:
        crps_parts = []
        for _, row in metrics_df.iterrows():
            if "CRPS" in row and not np.isnan(row.get("CRPS", np.nan)):
                crps_parts.append(f"{row['model']}={row['CRPS']:.3f}")
        if crps_parts:
            lines.append(f"CRPS (norm.): {', '.join(crps_parts)}")

    return "\n".join(lines)


def get_model_color(model_name: str, model_index: int) -> str:
    """Get color for a model, using predefined palette or default colors."""
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]
    return DEFAULT_COLORS[model_index % len(DEFAULT_COLORS)]


def plot_hindcast_timeseries(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    target_code: int,
    horizon: int,
    output_path: str | None = None,
) -> None:
    """
    Plot observed vs predicted discharge as time series.

    - X-axis: year-month (e.g., 2020-Apr, 2020-May, ...)
    - Y-axis: Discharge [m³/s]
    - Black line: Observed monthly discharge
    - Colored lines: Model predictions (one color per model)
    - Shaded bands: Uncertainty intervals for probabilistic models
    - Caption: Metrics table (MAE, RMSE per model)

    Args:
        summary_df: DataFrame from create_forecast_summary with multiple models
        metrics_df: DataFrame from calculate_hindcast_metrics
        target_code: Basin code for title
        horizon: Lead time (horizon number) for title
        output_path: Optional path to save the figure
    """
    if summary_df.empty:
        logger.warning("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique models and year-month values
    models = summary_df["model"].unique()
    year_months = sorted(summary_df["year_month"].unique())

    # Create x-axis positions based on year-month
    x_positions = {ym: i for i, ym in enumerate(year_months)}

    # Create x-axis labels (e.g., "2020-Apr")
    x_labels = []
    for ym in year_months:
        year, month = ym.split("-")
        month_name = month_short[int(month)]
        x_labels.append(f"{year}-{month_name}")

    # Plot observed discharge (black line with markers)
    # Get one row per year_month for observed values
    obs_df = (
        summary_df.drop_duplicates(subset=["year_month"])
        .sort_values("year_month")
        .copy()
    )
    obs_x = [x_positions[ym] for ym in obs_df["year_month"]]
    obs_y = obs_df["Q_obs"].values

    ax.plot(
        obs_x,
        obs_y,
        "ko-",
        linewidth=2.5,
        markersize=8,
        label="Observed",
        zorder=10,
    )

    # Plot each model's predictions
    for i, model in enumerate(models):
        model_df = summary_df[summary_df["model"] == model].sort_values("year_month")
        model_x = [x_positions[ym] for ym in model_df["year_month"]]
        model_y = model_df["Q_pred"].values

        color = get_model_color(model, i)

        # Plot prediction line
        ax.plot(
            model_x,
            model_y,
            "-",
            linewidth=2,
            color=color,
            marker="s",
            markersize=6,
            label=model,
            alpha=0.9,
        )

        # Plot uncertainty bands if quantiles are available
        has_q5 = "Q5" in model_df.columns and not model_df["Q5"].isna().all()
        has_q95 = "Q95" in model_df.columns and not model_df["Q95"].isna().all()
        has_q25 = "Q25" in model_df.columns and not model_df["Q25"].isna().all()
        has_q75 = "Q75" in model_df.columns and not model_df["Q75"].isna().all()

        has_Q_loc = "Q_loc" in model_df.columns and not model_df["Q_loc"].isna().all()

        # Plot Q_loc only for MC_ALD model
        if has_Q_loc and model == "MC_ALD":
            # Plot location parameter as dashed line
            ax.plot(
                model_x,
                model_df["Q_loc"].values,
                "--",
                linewidth=1.5,
                color=color,
                label=f"{model} Loc",
                alpha=0.7,
            )

        if has_q5 and has_q95:
            # 90% CI (Q5-Q95)
            ax.fill_between(
                model_x,
                model_df["Q5"].values,
                model_df["Q95"].values,
                alpha=0.15,
                color=color,
                label=f"{model} 90% CI" if i == 0 else None,
            )

        if has_q25 and has_q75:
            # 50% CI (Q25-Q75)
            ax.fill_between(
                model_x,
                model_df["Q25"].values,
                model_df["Q75"].values,
                alpha=0.25,
                color=color,
                label=f"{model} 50% CI" if i == 0 else None,
            )

    # Format x-axis
    ax.set_xticks(range(len(year_months)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Year-Month", fontsize=12)
    ax.set_ylabel("Discharge [m³/s]", fontsize=12)

    # Create title with horizon info
    ax.set_title(
        f"Hindcast Comparison - Basin: {target_code}, Lead Time: {horizon} month(s)",
        fontsize=14,
        fontweight="bold",
    )

    # Add metrics caption as subtitle
    metrics_caption = format_metrics_caption(metrics_df)
    if metrics_caption:
        fig.text(
            0.5,
            0.01,
            metrics_caption,
            ha="center",
            fontsize=10,
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)
    ax.grid(True, alpha=0.3)

    # Adjust layout to accommodate legend and caption
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, right=0.85)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    plt.show()


def plot_rolling_forecast_by_leadtime(
    all_forecasts_df: pd.DataFrame,
    monthly_obs: pd.DataFrame,
    target_code: int,
    model_name: str,
    output_path: str | None = None,
) -> None:
    """
    Plot rolling forecasts with different lead times in different colors.

    Shows how forecasts for the same target month evolve as lead time decreases.
    Each lead time (month_0, month_1, month_2) is shown in a different color.

    Args:
        all_forecasts_df: DataFrame with forecasts for all horizons (must have 'horizon' column)
        monthly_obs: DataFrame with monthly observed discharge (year, month, Q_obs_monthly)
        target_code: Basin code for title
        model_name: Model name to filter and display
        output_path: Optional path to save the figure
    """
    if all_forecasts_df.empty:
        logger.warning("No data to plot")
        return

    # Filter for the specified model
    df = all_forecasts_df[all_forecasts_df["model"] == model_name].copy()

    if df.empty:
        logger.warning(f"No data for model {model_name}")
        return

    # Extract target year and month from valid_from
    df["year"] = df["valid_from"].dt.year
    df["month"] = df["valid_from"].dt.month
    df["year_month"] = (
        df["year"].astype(str) + "-" + df["month"].astype(str).str.zfill(2)
    )

    # Merge with monthly observations
    df = df.merge(
        monthly_obs[["year", "month", "Q_obs_monthly"]],
        on=["year", "month"],
        how="left",
    )
    df = df.rename(columns={"Q_obs_monthly": "Q_obs"})

    # Sort by year_month
    df = df.sort_values(["year", "month", "horizon"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique year-month values and horizons
    year_months = sorted(df["year_month"].unique())
    horizons = sorted(df["horizon"].unique())

    # Create x-axis positions based on year-month
    x_positions = {ym: i for i, ym in enumerate(year_months)}

    # Create x-axis labels (e.g., "2020-Apr")
    x_labels = []
    for ym in year_months:
        year, month = ym.split("-")
        month_name = month_short[int(month)]
        x_labels.append(f"{year}-{month_name}")

    # Plot observed discharge (black line with markers)
    obs_df = df.drop_duplicates(subset=["year_month"]).sort_values("year_month").copy()
    obs_x = [x_positions[ym] for ym in obs_df["year_month"]]
    obs_y = obs_df["Q_obs"].values

    ax.plot(
        obs_x,
        obs_y,
        "ko-",
        linewidth=2.5,
        markersize=8,
        label="Observed",
        zorder=10,
    )

    # Plot each horizon (lead time) in a different color
    for horizon in horizons:
        horizon_df = df[df["horizon"] == horizon].sort_values("year_month")
        horizon_x = [x_positions[ym] for ym in horizon_df["year_month"]]
        horizon_y = horizon_df["Q_pred"].values

        color = HORIZON_COLORS.get(
            horizon, DEFAULT_COLORS[horizon % len(DEFAULT_COLORS)]
        )

        # Plot prediction line
        ax.plot(
            horizon_x,
            horizon_y,
            "-",
            linewidth=2,
            color=color,
            marker="s",
            markersize=6,
            label=f"Lead time: {horizon} month(s)",
            alpha=0.85,
        )

        # Plot uncertainty bands if quantiles are available
        has_q5 = "Q5" in horizon_df.columns and not horizon_df["Q5"].isna().all()
        has_q95 = "Q95" in horizon_df.columns and not horizon_df["Q95"].isna().all()
        has_q25 = "Q25" in horizon_df.columns and not horizon_df["Q25"].isna().all()
        has_q75 = "Q75" in horizon_df.columns and not horizon_df["Q75"].isna().all()

        if has_q5 and has_q95:
            ax.fill_between(
                horizon_x,
                horizon_df["Q5"].values,
                horizon_df["Q95"].values,
                alpha=0.1,
                color=color,
            )

        if has_q25 and has_q75:
            ax.fill_between(
                horizon_x,
                horizon_df["Q25"].values,
                horizon_df["Q75"].values,
                alpha=0.2,
                color=color,
            )

    # Format x-axis
    ax.set_xticks(range(len(year_months)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Target Month", fontsize=12)
    ax.set_ylabel("Discharge [m³/s]", fontsize=12)

    ax.set_title(
        f"Rolling Forecast by Lead Time - Basin: {target_code}, Model: {model_name}",
        fontsize=14,
        fontweight="bold",
    )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved rolling forecast figure to {output_path}")

    plt.show()


def plot_forecast_vs_climatology(
    summary_df: pd.DataFrame,
    target_code: int,
    model_name: str,
    output_path: str | None = None,
) -> None:
    """
    Create a visualization of normalized forecasts vs monthly climatology.

    This is the legacy function for single-model comparison with climatology.

    Args:
        summary_df: DataFrame from create_forecast_summary
        target_code: Basin code for title
        model_name: Model name for title
        output_path: Optional path to save the figure
    """
    if summary_df.empty:
        logger.warning("No data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique target months
    target_months = sorted(summary_df["month"].unique())

    # Create x-axis positions
    x_positions = []
    x_labels = []
    x_pos = 0

    for month in target_months:
        month_data = summary_df[summary_df["month"] == month].sort_values("horizon")

        for _, row in month_data.iterrows():
            x_positions.append(x_pos)
            x_labels.append(f"{month_renaming[month][:3]}\n(+{int(row['horizon'])}m)")
            x_pos += 1

        x_pos += 0.5  # Add gap between months

    # Reset index for plotting
    plot_data = summary_df.sort_values(["month", "horizon"]).reset_index(drop=True)

    # Plot observations
    ax.plot(
        x_positions,
        plot_data["Q_obs"].values,
        "ko-",
        linewidth=2,
        markersize=6,
        label="Observed",
    )

    # Plot predictions
    ax.plot(
        x_positions,
        plot_data["Q_pred"].values,
        "b-",
        linewidth=2,
        label="Forecast",
        marker="s",
        markersize=8,
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Target Month (+Lead Time)", fontsize=12)
    ax.set_ylabel("Discharge [m³/s]", fontsize=12)
    ax.set_title(
        f"Operational Forecast vs Observed\nBasin: {target_code}, Model: {model_name}",
        fontsize=14,
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    plt.show()


def plot_yearly_trajectories(
    monthly_obs: pd.DataFrame,
    monthly_climatology: pd.DataFrame,
    target_code: int,
    output_path: str | None = None,
) -> None:
    """
    Create a visualization of yearly discharge trajectories with long-term mean.

    Shows each observed year as a separate colored trajectory overlaid on the
    long-term mean with standard deviation bands.

    Args:
        monthly_obs: DataFrame with columns: year, month, Q_obs_monthly
        monthly_climatology: DataFrame with columns: month, Q_ltm_monthly, Q_std_monthly
        target_code: Basin code for title
        output_path: Optional path to save the figure
    """
    if monthly_obs.empty:
        logger.warning("No observation data to plot")
        return

    fig, ax = plt.subplots(figsize=(14, 8))

    # Get unique years and create a color palette
    years = sorted(monthly_obs["year"].unique())
    n_years = len(years)

    # Use a colormap that provides good distinction between years
    cmap = plt.cm.viridis
    colors = [cmap(i / max(n_years - 1, 1)) for i in range(n_years)]

    # X-axis: months 1-12
    months = np.arange(1, 13)
    month_labels = [month_renaming[m][:3] for m in months]

    # Plot long-term mean with ±std band
    clim_data = monthly_climatology.sort_values("month")
    q_ltm = clim_data["Q_ltm_monthly"].values
    q_std = clim_data["Q_std_monthly"].values

    # Plot ±1 std band
    ax.fill_between(
        months,
        q_ltm - q_std,
        q_ltm + q_std,
        alpha=0.3,
        color="gray",
        label="Long-term mean ±1 std",
    )

    # Plot long-term mean line
    ax.plot(months, q_ltm, "k-", linewidth=3, label="Long-term mean", zorder=10)

    # Plot each year as a trajectory
    for year, color in zip(years, colors):
        year_data = monthly_obs[monthly_obs["year"] == year].sort_values("month")

        # Create a full 12-month array with NaN for missing months
        year_values = np.full(12, np.nan)
        for _, row in year_data.iterrows():
            month_idx = int(row["month"]) - 1
            year_values[month_idx] = row["Q_obs_monthly"]

        # Plot the year trajectory
        ax.plot(
            months,
            year_values,
            "-",
            linewidth=1.5,
            alpha=0.7,
            color=color,
            label=str(year),
        )

    ax.set_xticks(months)
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Discharge [m³/s]", fontsize=12)
    ax.set_title(f"Yearly Discharge Trajectories\nBasin: {target_code}", fontsize=14)

    # Create legend with two columns if many years
    if n_years > 15:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=2, fontsize=8)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, fontsize=9)

    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.5, 12.5)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved yearly trajectories figure to {output_path}")

    plt.show()


def load_all_basin_forecasts(
    base_path: str,
    horizons: list[str],
    model_names: list[str],
    daily_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Load operational forecasts for all basins and all models.

    Args:
        base_path: Base directory containing horizon subdirectories
        horizons: List of horizon identifiers (e.g., ["month_0", "month_1", ...])
        model_names: List of model names to load
        daily_obs: DataFrame with daily observations to get available basin codes

    Returns:
        DataFrame with forecast data for all basins across all horizons and models
    """
    base_path = Path(base_path)
    all_forecasts = []

    # Get unique basin codes from observations
    available_codes = daily_obs["code"].unique()
    logger.info(f"Found {len(available_codes)} basins in observations")

    for model_name in model_names:
        for horizon in horizons:
            horizon_path = base_path / horizon
            if not horizon_path.exists():
                continue

            horizon_num = int(horizon.split("_")[1])
            model_dir = horizon_path / model_name
            if not model_dir.exists():
                continue

            forecast_file = model_dir / f"{model_name}_hindcast.csv"
            if not forecast_file.exists():
                continue

            try:
                df = pd.read_csv(forecast_file)
            except Exception as e:
                logger.error(f"Failed to read {forecast_file}: {e}")
                continue

            if df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df["valid_from"] = pd.to_datetime(df["valid_from"], format="mixed")
            df["valid_to"] = pd.to_datetime(df["valid_to"], format="mixed")
            df["code"] = df["code"].astype(int)

            # Find quantile columns
            quantile_cols = [col for col in df.columns if re.fullmatch(r"Q\d+", col)]

            # Find prediction column
            excluded_cols = ["Q_obs", "Q_Obs", "Q_OBS"] + quantile_cols
            q_cols = [
                c for c in df.columns if c.startswith("Q_") and c not in excluded_cols
            ]

            if not q_cols:
                continue

            q_pred_col = q_cols[0]

            result_df = pd.DataFrame(
                {
                    "code": df["code"],
                    "issue_date": df["date"],
                    "valid_from": df["valid_from"],
                    "valid_to": df["valid_to"],
                    "Q_pred": df[q_pred_col],
                    "horizon": horizon_num,
                    "model": model_name,
                }
            )

            for qcol in quantile_cols:
                if qcol in df.columns:
                    result_df[qcol] = df[qcol].values

            all_forecasts.append(result_df)

    if not all_forecasts:
        return pd.DataFrame()

    combined_df = pd.concat(all_forecasts, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} total forecast records across all basins")

    return combined_df


def calculate_all_basin_metrics(
    all_forecasts_df: pd.DataFrame,
    daily_obs: pd.DataFrame,
    model_names: list[str],
    horizons: list[str],
) -> pd.DataFrame:
    """
    Calculate R², nMAE, Accuracy, and Efficiency metrics across all basins.

    Args:
        all_forecasts_df: DataFrame with forecasts for all basins
        daily_obs: DataFrame with daily observations
        model_names: List of model names
        horizons: List of horizon identifiers

    Returns:
        DataFrame with metrics per model and horizon
    """
    if all_forecasts_df.empty:
        return pd.DataFrame()

    metrics_list = []

    for model_name in model_names:
        for horizon in horizons:
            horizon_num = int(horizon.split("_")[1])

            # Filter for this model and horizon
            mask = (all_forecasts_df["model"] == model_name) & (
                all_forecasts_df["horizon"] == horizon_num
            )
            model_df = all_forecasts_df[mask].copy()

            if model_df.empty:
                continue

            # Extract year and month for merging with observations
            model_df["year"] = model_df["valid_from"].dt.year
            model_df["month"] = model_df["valid_from"].dt.month

            # Calculate monthly observations for all basins
            daily_obs_copy = daily_obs.copy()
            daily_obs_copy["year"] = daily_obs_copy["date"].dt.year
            daily_obs_copy["month"] = daily_obs_copy["date"].dt.month

            obs_monthly = (
                daily_obs_copy.groupby(["code", "year", "month"])["discharge"]
                .mean()
                .reset_index()
            )
            obs_monthly = obs_monthly.rename(columns={"discharge": "Q_obs"})

            # Also calculate long-term std per basin-month for accuracy threshold
            obs_std = (
                daily_obs_copy.groupby(["code", "month"])["discharge"]
                .std()
                .reset_index()
            )
            obs_std = obs_std.rename(columns={"discharge": "Q_std"})

            # Merge forecasts with observations
            merged = model_df.merge(
                obs_monthly, on=["code", "year", "month"], how="inner"
            )

            # Merge with std for accuracy threshold
            merged = merged.merge(obs_std, on=["code", "month"], how="left")
            merged["delta"] = 0.674 * merged["Q_std"]

            if merged.empty or merged["Q_obs"].isna().all():
                continue

            # Drop NaN values
            valid = merged.dropna(subset=["Q_obs", "Q_pred"])

            if valid.empty:
                continue

            obs = valid["Q_obs"].values
            pred = valid["Q_pred"].values

            # Calculate R²
            r2 = calculate_R2(obs, pred)

            # Calculate nMAE
            nmae = calculate_NMAE(obs, pred)

            # Calculate Efficiency (s/sigma from sdivsigma_nse)
            eff_df = valid[["Q_obs", "Q_pred"]].copy()
            eff_result = sdivsigma_nse(eff_df, "Q_obs", "Q_pred")
            efficiency = eff_result["sdivsigma"]

            # Calculate Accuracy (using 0.674 * std as threshold)
            if "delta" in valid.columns and not valid["delta"].isna().all():
                residual = np.abs(obs - pred)
                accuracy = np.mean(residual <= valid["delta"].values)
            else:
                accuracy = np.nan

            metrics_list.append(
                {
                    "model": model_name,
                    "horizon": horizon_num,
                    "R2": r2,
                    "nMAE": nmae,
                    "Accuracy": accuracy,
                    "Efficiency": efficiency,
                    "n_samples": len(valid),
                    "n_basins": valid["code"].nunique(),
                }
            )

    if not metrics_list:
        return pd.DataFrame()

    return pd.DataFrame(metrics_list)


def plot_metrics_2x2(
    metrics_df: pd.DataFrame,
    target_code: int,
    output_path: str | None = None,
) -> None:
    """
    Create a 2x2 plot showing R², nMAE, Accuracy, and Efficiency across all basins.

    Args:
        metrics_df: DataFrame with metrics per model and horizon
        target_code: Basin code for reference in title
        output_path: Optional path to save the figure
    """
    if metrics_df.empty:
        logger.warning("No metrics data to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    metrics_to_plot = [
        ("R2", "R² (Coefficient of Determination)", axes[0, 0], True),
        ("nMAE", "nMAE (Normalized Mean Absolute Error)", axes[0, 1], False),
        ("Accuracy", "Accuracy (within 0.674σ)", axes[1, 0], True),
        ("Efficiency", "Efficiency (s/σ)", axes[1, 1], False),
    ]

    models = metrics_df["model"].unique()
    horizons = sorted(metrics_df["horizon"].unique())
    n_models = len(models)
    x = np.arange(len(horizons))
    width = 0.8 / n_models

    for metric_name, metric_label, ax, higher_is_better in metrics_to_plot:
        for i, model in enumerate(models):
            model_data = metrics_df[metrics_df["model"] == model]
            values = []
            for h in horizons:
                h_data = model_data[model_data["horizon"] == h]
                if not h_data.empty and metric_name in h_data.columns:
                    val = h_data[metric_name].values[0]
                    values.append(val if not np.isnan(val) else 0)
                else:
                    values.append(0)

            color = get_model_color(model, i)
            offset = (i - n_models / 2 + 0.5) * width
            bars = ax.bar(
                x + offset, values, width, label=model, color=color, alpha=0.8
            )

            # Add value labels on bars
            for bar, val in zip(bars, values):
                if val != 0:
                    height = bar.get_height()
                    ax.annotate(
                        f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

        ax.set_xlabel("Lead Time (months)", fontsize=10)
        ax.set_ylabel(metric_name, fontsize=10)
        ax.set_title(metric_label, fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}" for h in horizons])
        ax.grid(True, alpha=0.3, axis="y")

        # Add reference lines
        if metric_name == "R2":
            ax.axhline(y=1.0, color="green", linestyle="--", alpha=0.5, label="Perfect")
            ax.set_ylim(0, 1.1)
        elif metric_name == "Accuracy":
            ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.5, label="50%")
            ax.set_ylim(0, 1.1)
        elif metric_name == "Efficiency":
            ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="s=σ")
            ax.set_ylim(
                0,
                max(2, metrics_df[metric_name].max() * 1.1)
                if not metrics_df[metric_name].isna().all()
                else 2,
            )

    # Add legend to first subplot only
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles[:n_models],
        labels[:n_models],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(n_models, 4),
        fontsize=10,
    )

    # Get sample size info
    total_samples = metrics_df["n_samples"].sum()
    n_basins = metrics_df["n_basins"].max() if "n_basins" in metrics_df.columns else "?"

    fig.suptitle(
        f"Cross-Basin Performance Metrics\n({n_basins} basins, {total_samples} total samples)",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, bottom=0.08)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved 2x2 metrics figure to {output_path}")

    plt.show()


def plot_hindcast_timeseries_clean(
    summary_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    target_code: int,
    horizon: int,
    output_path: str | None = None,
) -> None:
    """
    Plot observed vs predicted discharge as time series with clean metrics placement.

    This is an improved version with metrics displayed in a table format on the side.

    Args:
        summary_df: DataFrame from create_forecast_summary with multiple models
        metrics_df: DataFrame from calculate_hindcast_metrics
        target_code: Basin code for title
        horizon: Lead time (horizon number) for title
        output_path: Optional path to save the figure
    """
    if summary_df.empty:
        logger.warning("No data to plot")
        return

    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.05)

    ax_main = fig.add_subplot(gs[0])
    ax_metrics = fig.add_subplot(gs[1])

    # Get unique models and year-month values
    models = summary_df["model"].unique()
    year_months = sorted(summary_df["year_month"].unique())

    # Create x-axis positions based on year-month
    x_positions = {ym: i for i, ym in enumerate(year_months)}

    # Create x-axis labels
    x_labels = []
    for ym in year_months:
        year, month = ym.split("-")
        month_name = month_short[int(month)]
        x_labels.append(f"{year}-{month_name}")

    # Plot observed discharge
    obs_df = (
        summary_df.drop_duplicates(subset=["year_month"])
        .sort_values("year_month")
        .copy()
    )
    obs_x = [x_positions[ym] for ym in obs_df["year_month"]]
    obs_y = obs_df["Q_obs"].values

    ax_main.plot(
        obs_x,
        obs_y,
        "ko-",
        linewidth=2.5,
        markersize=8,
        label="Observed",
        zorder=10,
    )

    # Plot each model's predictions
    for i, model in enumerate(models):
        model_df = summary_df[summary_df["model"] == model].sort_values("year_month")
        model_x = [x_positions[ym] for ym in model_df["year_month"]]
        model_y = model_df["Q_pred"].values

        color = get_model_color(model, i)

        ax_main.plot(
            model_x,
            model_y,
            "-",
            linewidth=2,
            color=color,
            marker="s",
            markersize=6,
            label=model,
            alpha=0.9,
        )

        # Plot uncertainty bands if available
        has_q5 = "Q5" in model_df.columns and not model_df["Q5"].isna().all()
        has_q95 = "Q95" in model_df.columns and not model_df["Q95"].isna().all()
        has_q25 = "Q25" in model_df.columns and not model_df["Q25"].isna().all()
        has_q75 = "Q75" in model_df.columns and not model_df["Q75"].isna().all()

        if has_q5 and has_q95:
            ax_main.fill_between(
                model_x,
                model_df["Q5"].values,
                model_df["Q95"].values,
                alpha=0.15,
                color=color,
            )

        if has_q25 and has_q75:
            ax_main.fill_between(
                model_x,
                model_df["Q25"].values,
                model_df["Q75"].values,
                alpha=0.25,
                color=color,
            )

    # Format main axis
    ax_main.set_xticks(range(len(year_months)))
    ax_main.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax_main.set_xlabel("Year-Month", fontsize=12)
    ax_main.set_ylabel("Discharge [m³/s]", fontsize=12)
    ax_main.set_title(
        f"Hindcast Comparison - Basin: {target_code}, Lead Time: {horizon} month(s)",
        fontsize=14,
        fontweight="bold",
    )
    ax_main.legend(loc="upper left", fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # Create metrics table in the right panel
    ax_metrics.axis("off")

    if not metrics_df.empty:
        # Build table data
        columns = ["Model", "MAE", "RMSE"]
        has_coverage = "Coverage_90" in metrics_df.columns
        has_crps = "CRPS" in metrics_df.columns

        if has_coverage:
            columns.append("Cov90")
        if has_crps:
            columns.append("CRPS")
        columns.append("N")

        table_data = []
        for _, row in metrics_df.iterrows():
            row_data = [
                row["model"],
                f"{row['MAE']:.1f}" if not np.isnan(row["MAE"]) else "-",
                f"{row['RMSE']:.1f}" if not np.isnan(row["RMSE"]) else "-",
            ]
            if has_coverage:
                cov = row.get("Coverage_90", np.nan)
                row_data.append(f"{cov:.2f}" if not np.isnan(cov) else "-")
            if has_crps:
                crps = row.get("CRPS", np.nan)
                row_data.append(f"{crps:.3f}" if not np.isnan(crps) else "-")
            row_data.append(str(int(row["n_samples"])))
            table_data.append(row_data)

        # Create table
        table = ax_metrics.table(
            cellText=table_data,
            colLabels=columns,
            loc="center",
            cellLoc="center",
            colColours=["lightsteelblue"] * len(columns),
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax_metrics.set_title(
            "Error Metrics\n(for selected basin)",
            fontsize=11,
            fontweight="bold",
            pad=20,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved figure to {output_path}")

    plt.show()


def main():
    """Main function to run the operational forecast visualization."""
    logger.info("=" * 60)
    logger.info("Operational Hindcast Visualization")
    logger.info(f"Target Code: {TARGET_CODE}")
    logger.info(f"Region: {REGION}")
    logger.info(f"Models: {MODEL_NAMES}")
    logger.info("=" * 60)

    # Get path configuration based on region
    if REGION == "Kyrgyzstan":
        path_config = kgz_path_config
    elif REGION == "Tajikistan":
        path_config = taj_path_config
    else:
        raise ValueError(f"Unknown region: {REGION}")

    pred_dir = path_config["pred_dir"]
    obs_file = path_config["obs_file"]

    save_dir = Path(output_dir) / f"{REGION.lower()}"

    if not pred_dir or not obs_file:
        logger.error("Path configuration not set. Check your .env file.")
        return

    logger.info(f"Loading observations from: {obs_file}")

    # Load observations
    daily_obs = load_observations(obs_file)

    # Filter by years
    daily_obs = daily_obs[daily_obs["date"].dt.year.isin(YEARS)]

    # Filter observations for target code
    code_obs = daily_obs[daily_obs["code"] == TARGET_CODE]
    if code_obs.empty:
        logger.error(f"No observations found for code {TARGET_CODE}")
        return

    logger.info(f"Loaded {len(code_obs)} daily observations for code {TARGET_CODE}")

    # Calculate monthly observations and climatology
    monthly_obs = calculate_monthly_observations(daily_obs, TARGET_CODE)
    monthly_climatology = calculate_monthly_climatology(monthly_obs)

    # =========================================================================
    # CROSS-BASIN METRICS: Load all basins and calculate aggregate metrics
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Loading forecasts for ALL BASINS to calculate cross-basin metrics")
    logger.info("=" * 60)

    all_basin_forecasts = load_all_basin_forecasts(
        pred_dir, horizons, MODEL_NAMES, daily_obs
    )

    if not all_basin_forecasts.empty:
        # Filter by years
        all_basin_forecasts = all_basin_forecasts[
            all_basin_forecasts["issue_date"].dt.year.isin(YEARS)
        ]

        # Calculate cross-basin metrics
        cross_basin_metrics = calculate_all_basin_metrics(
            all_basin_forecasts, daily_obs, MODEL_NAMES, horizons
        )

        if not cross_basin_metrics.empty:
            logger.info("\nCross-Basin Metrics Summary:")
            print(cross_basin_metrics.to_string(index=False))

            # Plot 2x2 metrics figure
            metrics_output = f"cross_basin_metrics_2x2_{REGION.lower()}.png"
            plot_metrics_2x2(
                cross_basin_metrics,
                TARGET_CODE,
                output_path=Path(save_dir) / metrics_output,
            )
        else:
            logger.warning("No cross-basin metrics calculated")
    else:
        logger.warning("No forecasts loaded for cross-basin analysis")

    # =========================================================================
    # SINGLE BASIN PLOTS: Create hindcast plots for the selected basin
    # =========================================================================
    for horizon in horizons:
        horizon_num = int(horizon.split("_")[1])
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing horizon: {horizon} (lead time: {horizon_num} months)")
        logger.info("=" * 60)

        # Load operational forecasts for all models at this horizon
        logger.info(f"Loading operational forecasts from: {pred_dir}")
        forecasts_df = load_operational_forecasts(
            pred_dir, [horizon], TARGET_CODE, MODEL_NAMES
        )

        if forecasts_df.empty:
            logger.warning(
                f"No forecasts loaded for horizon {horizon}. Skipping this horizon."
            )
            continue

        # Filter by years
        forecasts_df = forecasts_df[forecasts_df["issue_date"].dt.year.isin(YEARS)]

        if forecasts_df.empty:
            logger.warning(
                f"No forecasts loaded for horizon {horizon}. Skipping this horizon."
            )
            continue

        # Create summary dataframe with observed values merged
        summary_df = create_forecast_summary(forecasts_df, monthly_obs)

        logger.info(f"\nForecast Summary for horizon {horizon}:")
        print(
            summary_df[["year_month", "model", "Q_obs", "Q_pred"]]
            .head(20)
            .to_string(index=False)
        )

        # Calculate metrics for this basin
        metrics_df = calculate_hindcast_metrics(summary_df)

        if not metrics_df.empty:
            logger.info(f"\nMetrics for horizon {horizon}:")
            print(metrics_df.to_string(index=False))

        # Plot time series with clean metrics table
        output_filename = f"hindcast_timeseries_{TARGET_CODE}_horizon_{horizon_num}.png"
        plot_hindcast_timeseries_clean(
            summary_df,
            metrics_df,
            TARGET_CODE,
            horizon_num,
            output_path=Path(save_dir) / output_filename,
        )

    # Load all horizons at once for rolling forecast plot
    logger.info("\n" + "=" * 60)
    logger.info("Creating rolling forecast comparison by lead time")
    logger.info("=" * 60)

    all_forecasts_df = load_operational_forecasts(
        pred_dir, horizons, TARGET_CODE, MODEL_NAMES
    )

    if not all_forecasts_df.empty:
        # Filter by years
        all_forecasts_df = all_forecasts_df[
            all_forecasts_df["issue_date"].dt.year.isin(YEARS)
        ]

        # Create one rolling forecast plot per model
        # for model_name in MODEL_NAMES:
        #     model_forecasts = all_forecasts_df[all_forecasts_df["model"] == model_name]
        #     if not model_forecasts.empty:
        #         output_filename = (
        #             f"rolling_forecast_{TARGET_CODE}_{model_name}_by_leadtime.png"
        #         )
        #         plot_rolling_forecast_by_leadtime(
        #             all_forecasts_df,
        #             monthly_obs,
        #             TARGET_CODE,
        #             model_name,
        #             output_path=Path(save_dir) / output_filename,
        #         )

    # Also plot yearly trajectories for context
    plot_yearly_trajectories(
        monthly_obs,
        monthly_climatology,
        TARGET_CODE,
        output_path=Path(save_dir) / f"yearly_trajectories_code_{TARGET_CODE}.png",
    )

    return summary_df


if __name__ == "__main__":
    summary = main()
