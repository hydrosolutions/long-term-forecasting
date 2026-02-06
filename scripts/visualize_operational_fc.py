"""
Visualize Operational Forecasts

This script loads operational forecasts (model_name_forecast.csv) for a specific basin
and creates a summary dataframe with predictions vs long-term mean for visualization.

Key differences from hindcast evaluation:
- Forecasts have only one row per basin (code) for a given issue date
- Focus on a single basin specified at the beginning
- Output: target_month, lead_time, Q5, Q50, Q95, Q_pred, Q_norm (long-term mean)
"""

import os
import re
import sys
import logging
from pathlib import Path

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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load the .env file
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# =============================================================================
# CONFIGURATION - Set your target basin code here
# =============================================================================
TARGET_CODE: int = 17288  # Basin code to analyze

# Region configuration: "Kyrgyzstan" or "Tajikistan"
REGION: str = "Tajikistan"  # Set to "Kyrgyzstan" or "Tajikistan"

# Model to analyze
MODEL_NAME: str = "MC_ALD"

day_of_forecast = {
    # "month_0": 15,
    "month_1": 1,
    "month_2": 1,
    "month_3": 1,
    "month_4": 1,
    "month_5": 1,
    "month_6": 1,
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

    # Use sample std (ddof=1 is default in pandas)
    logger.info(f"Calculated monthly climatology for {len(monthly_clim)} months")

    return monthly_clim


def load_operational_forecasts(
    base_path: str,
    horizons: list[str],
    target_code: int,
    model_name: str,
) -> pd.DataFrame:
    """
    Load operational forecasts for a specific basin and model.

    Operational forecasts have only one row per basin for a given issue date.
    Files are named model_name_forecast.csv (not hindcast).

    Args:
        base_path: Base directory containing horizon subdirectories
        horizons: List of horizon identifiers (e.g., ["month_0", "month_1", ...])
        target_code: Basin code to filter for
        model_name: Name of the model to load

    Returns:
        DataFrame with forecast data for the target basin across all horizons
    """
    base_path = Path(base_path)
    all_forecasts = []

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

        # Load forecast file (not hindcast)
        forecast_file = model_dir / f"{model_name}_forecast.csv"

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

        # Convert dates to datetime
        df["date"] = pd.to_datetime(df["date"])
        df["valid_from"] = pd.to_datetime(df["valid_from"])
        df["valid_to"] = pd.to_datetime(df["valid_to"])

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

        all_forecasts.append(result_df)
        logger.info(f"Loaded forecast for horizon {horizon_num} from {forecast_file}")

    if not all_forecasts:
        logger.error(f"No forecasts loaded for code {target_code}, model {model_name}")
        return pd.DataFrame()

    combined_df = pd.concat(all_forecasts, ignore_index=True)
    logger.info(f"Loaded {len(combined_df)} forecast records for code {target_code}")

    return combined_df


def create_forecast_summary(
    forecasts_df: pd.DataFrame,
    monthly_climatology: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a summary dataframe with predictions for visualization.

    Target month is extracted directly from valid_from (no z-score normalization needed).

    Args:
        forecasts_df: DataFrame with operational forecasts
        monthly_climatology: Pre-computed monthly climatology (Q_ltm_monthly, Q_std_monthly per month)

    Returns:
        DataFrame with columns: target_month, lead_time, Q_pred, Q_pred_normalized, Q_ltm_monthly, etc.
    """
    if forecasts_df.empty:
        return pd.DataFrame()

    df = forecasts_df.copy()

    # Extract target month directly from valid_from (no shift calculation)
    df["target_month"] = df["valid_from"].dt.month
    df["issue_month"] = df["issue_date"].dt.month

    # Calculate lead time in months
    df["lead_time"] = df["horizon"]

    # Merge with monthly climatology based on target_month
    df = df.merge(
        monthly_climatology[
            [
                "month",
                "Q_ltm_monthly",
                "Q_std_monthly",
                "Q_min_monthly",
                "Q_max_monthly",
            ]
        ],
        left_on="target_month",
        right_on="month",
        how="left",
    )
    df = df.drop(columns=["month"], errors="ignore")

    # Direct comparison - no z-score needed
    # Keep Q_pred_normalized for backwards compatibility (equals Q_pred)
    df["Q_pred_normalized"] = df["Q_pred"]

    # Also set normalized quantiles equal to raw quantiles for backwards compatibility
    quantile_cols = ["Q5", "Q25", "Q50", "Q75", "Q95"]
    for qcol in quantile_cols:
        if qcol in df.columns:
            df[f"{qcol}_normalized"] = df[qcol]

    # Select columns for output
    output_cols = [
        "target_month",
        "lead_time",
        "Q_pred",
        "Q_pred_normalized",
        "Q_ltm_monthly",
        "Q_std_monthly",
        "Q_min_monthly",
        "Q_max_monthly",
    ]

    # Add normalized quantile columns if they exist
    for qcol in quantile_cols:
        if f"{qcol}_normalized" in df.columns:
            output_cols.append(f"{qcol}_normalized")

    result_df = df[output_cols].copy()

    # Sort by target_month, then lead_time
    result_df = result_df.sort_values(["target_month", "lead_time"]).reset_index(
        drop=True
    )

    # Add month names for readability
    result_df["target_month_name"] = result_df["target_month"].map(month_renaming)

    return result_df


def plot_forecast_vs_climatology(
    summary_df: pd.DataFrame,
    target_code: int,
    model_name: str,
    output_path: str | None = None,
) -> None:
    """
    Create a visualization of normalized forecasts vs monthly climatology.

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
    target_months = sorted(summary_df["target_month"].unique())

    # Create x-axis positions
    x_positions = []
    x_labels = []
    x_pos = 0

    for month in target_months:
        month_data = summary_df[summary_df["target_month"] == month].sort_values(
            "lead_time"
        )

        for _, row in month_data.iterrows():
            x_positions.append(x_pos)
            x_labels.append(f"{month_renaming[month][:3]}\n(+{int(row['lead_time'])}m)")
            x_pos += 1

        x_pos += 0.5  # Add gap between months

    # Reset index for plotting
    plot_data = summary_df.sort_values(["target_month", "lead_time"]).reset_index(
        drop=True
    )

    # Use monthly climatology values (Q_ltm_monthly) for the norm
    q_norm_values = plot_data["Q_ltm_monthly"].values
    q_norm_std = (
        plot_data["Q_std_monthly"].values
        if "Q_std_monthly" in plot_data.columns
        else np.zeros_like(q_norm_values)
    )
    q_norm_min = (
        plot_data["Q_min_monthly"].values
        if "Q_min_monthly" in plot_data.columns
        else q_norm_values
    )
    q_norm_max = (
        plot_data["Q_max_monthly"].values
        if "Q_max_monthly" in plot_data.columns
        else q_norm_values
    )

    # Calculate error bar lengths for min/max (asymmetric)
    yerr_minmax_lower = q_norm_values - q_norm_min
    yerr_minmax_upper = q_norm_max - q_norm_values

    # Plot min/max as thin error bars with small caps
    """ax.errorbar(
        x_positions, q_norm_values,
        yerr=[yerr_minmax_lower, yerr_minmax_upper],
        fmt='none', capsize=3, capthick=2, ecolor='darkgray', elinewidth=2,
        label="Monthly Climatology (min/max)"
    )"""

    # Plot ±std as thicker error bars with larger caps (slightly offset for visibility)
    ax.errorbar(
        x_positions,
        q_norm_values,
        yerr=q_norm_std,
        fmt="ko-",
        linewidth=2,
        markersize=6,
        capsize=6,
        capthick=2,
        ecolor="black",
        elinewidth=2.5,
        label="Monthly Climatology (±1 std)",
    )

    # Plot normalized predictions
    ax.plot(
        x_positions,
        plot_data["Q_pred_normalized"].values,
        "b-",
        linewidth=2,
        label="Forecast (normalized)",
        marker="s",
        markersize=8,
    )

    # Add uncertainty bands if normalized quantiles exist
    if "Q10_normalized" in plot_data.columns and "Q90_normalized" in plot_data.columns:
        ax.fill_between(
            x_positions,
            plot_data["Q10_normalized"].values,
            plot_data["Q90_normalized"].values,
            alpha=0.2,
            color="blue",
            label="80% CI (Q10-Q90 normalized)",
        )

    if "Q25_normalized" in plot_data.columns and "Q75_normalized" in plot_data.columns:
        ax.fill_between(
            x_positions,
            plot_data["Q25_normalized"].values,
            plot_data["Q75_normalized"].values,
            alpha=0.3,
            color="blue",
            label="50% CI (Q25-Q75 normalized)",
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Target Month (+Lead Time)", fontsize=12)
    ax.set_ylabel("Discharge [m³/s]", fontsize=12)
    ax.set_title(
        f"Operational Forecast vs Monthly Climatology \nBasin: {target_code}, Model: {model_name}",
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


def main():
    """Main function to run the operational forecast visualization."""
    logger.info(f"=" * 60)
    logger.info(f"Operational Forecast Visualization")
    logger.info(f"Target Code: {TARGET_CODE}")
    logger.info(f"Region: {REGION}")
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"=" * 60)

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

    # Filter observations for target code
    code_obs = daily_obs[daily_obs["code"] == TARGET_CODE]
    if code_obs.empty:
        logger.error(f"No observations found for code {TARGET_CODE}")
        return

    logger.info(f"Loaded {len(code_obs)} daily observations for code {TARGET_CODE}")

    # Calculate monthly observations and climatology
    monthly_obs = calculate_monthly_observations(daily_obs, TARGET_CODE)
    monthly_climatology = calculate_monthly_climatology(monthly_obs)

    # Load operational forecasts
    logger.info(f"Loading operational forecasts from: {pred_dir}")
    forecasts_df = load_operational_forecasts(
        pred_dir, horizons, TARGET_CODE, MODEL_NAME
    )

    if forecasts_df.empty:
        logger.error("No forecasts loaded. Check the forecast files exist.")
        return

    # Create summary dataframe
    summary_df = create_forecast_summary(forecasts_df, monthly_climatology)

    logger.info("\n" + "=" * 60)
    logger.info("FORECAST SUMMARY")
    logger.info("=" * 60)
    print(summary_df.to_string(index=False))

    # Plot the results
    plot_forecast_vs_climatology(
        summary_df,
        TARGET_CODE,
        MODEL_NAME,
        output_path=Path(save_dir)
        / f"forecast_vs_climatology_code_{TARGET_CODE}_{MODEL_NAME}.png",
    )

    # Plot yearly trajectories with long-term mean
    plot_yearly_trajectories(
        monthly_obs,
        monthly_climatology,
        TARGET_CODE,
        output_path=Path(save_dir) / f"yearly_trajectories_code_{TARGET_CODE}.png",
    )

    return summary_df


if __name__ == "__main__":
    summary = main()
