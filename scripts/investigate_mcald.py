"""
Script to investigate MC_ALD model predictions and test correction strategies.

This script analyzes the MC_ALD model performance across different regions and
forecast horizons, training linear regression correction models using leave-one-out
cross-validation.
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add a stream handler to output logs to the terminal
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

# load the .env file
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


# =============================================================================
# CONFIGURATION
# =============================================================================

# Region path configurations from environment variables
kgz_path_config = {
    "pred_dir": os.getenv("kgz_path_discharge"),
    "obs_file": os.getenv("kgz_path_base_pred"),
}

taj_path_config = {
    "pred_dir": os.getenv("taj_path_base_pred"),
    "obs_file": os.getenv("taj_path_discharge"),
}

output_dir = os.getenv("out_dir_op_lt")

# Available forecast horizons
ALL_HORIZONS = [
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
]

# Day of forecast for each horizon (when forecasts are issued)
day_of_forecast = {
    "month_0": 15,
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

# =============================================================================
# USER CONFIGURATION - MODIFY THESE
# =============================================================================

# Regions to analyze: ["Kyrgyzstan", "Tajikistan"] or just one
REGIONS_TO_ANALYZE: list[str] = ["Kyrgyzstan"]

# Horizons to analyze: list of horizon names or "all"
# Examples: ["month_1", "month_2", "month_3"] or ["month_0"] or "all"
HORIZONS_TO_ANALYZE: list[str] | str = ["month_0", "month_1", "month_2", "month_3", "month_4", "month_5"]

# Whether to save plots to files (True) or display interactively (False)
SAVE_PLOTS: bool = True

# =============================================================================
# FUNCTIONS
# =============================================================================


def get_path_config(region: str) -> dict[str, str]:
    """Get path configuration for a specific region."""
    if region == "Kyrgyzstan":
        return kgz_path_config
    elif region == "Tajikistan":
        return taj_path_config
    else:
        raise ValueError(f"Unknown region: {region}. Must be 'Kyrgyzstan' or 'Tajikistan'")


def get_mc_ald_path(pred_dir: str, horizon: str) -> Path:
    """
    Construct the path to MC_ALD hindcast file for a given horizon.
    
    Args:
        pred_dir: Base prediction directory
        horizon: Horizon name (e.g., "month_1")
        
    Returns:
        Path to the MC_ALD hindcast CSV file
    """
    return Path(pred_dir) / horizon / "MC_ALD" / "MC_ALD_hindcast.csv"


def load_mc_ald_data(
    pred_dir: str,
    horizon: str,
) -> pd.DataFrame | None:
    """
    Load MC_ALD hindcast data for a specific horizon.
    
    Args:
        pred_dir: Base prediction directory
        horizon: Horizon name (e.g., "month_1")
        
    Returns:
        DataFrame with MC_ALD data or None if file not found
    """
    file_path = get_mc_ald_path(pred_dir, horizon)
    
    if not file_path.exists():
        logger.warning(f"MC_ALD hindcast file not found: {file_path}")
        return None
    
    try:
        data = pd.read_csv(file_path, parse_dates=["date"])
        
        # Filter by day of forecast if specified
        forecast_day = day_of_forecast.get(horizon)
        if forecast_day is not None:
            data = data[data["date"].dt.day == forecast_day].copy()
        
        # Select relevant columns
        required_cols = ["date", "code", "Q_loc", "Q_MC_ALD", "Q_obs"]
        available_cols = [col for col in required_cols if col in data.columns]
        
        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            logger.warning(f"Missing columns in {file_path}: {missing}")
            return None
        
        data = data[available_cols].dropna()
        data["month"] = data["date"].dt.month
        data["horizon"] = horizon
        data["horizon_num"] = int(horizon.split("_")[1])
        
        logger.info(
            f"Loaded {len(data)} records from {horizon} "
            f"({data['code'].nunique()} codes, months: {sorted(data['month'].unique())})"
        )
        
        return data
        
    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def train_correction_model_loocv(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Train a linear regression per code and month to predict the correction
    (Q_obs - Q_loc) based on Q_loc using leave-one-out cross-validation.

    Args:
        data: DataFrame with columns 'code', 'month', 'Q_loc', 'Q_obs', 'Q_MC_ALD'

    Returns:
        DataFrame with additional columns for predictions and corrected values
    """
    # Calculate the target correction
    data = data.copy()
    data["target_correction"] = data["Q_obs"] - data["Q_loc"]

    # Initialize columns for predictions
    data["Q_loc_corrected"] = np.nan

    # Get unique codes
    codes = data["code"].unique()

    for code in codes:
        code_mask = data["code"] == code
        code_data = data[code_mask]

        # Get unique months for this code
        months = code_data["month"].unique()

        for month in months:
            month_mask = (data["code"] == code) & (data["month"] == month)
            month_data = data[month_mask]

            if len(month_data) < 2:
                # Not enough data for LOOCV, use mean correction
                logger.warning(
                    f"Code {code}, month {month}: only {len(month_data)} sample(s), "
                    "using Q_loc as prediction"
                )
                data.loc[month_mask, "Q_loc_corrected"] = month_data["Q_loc"]
                continue

            # Prepare features and target
            X = month_data[["Q_loc"]].values
            y = month_data["target_correction"].values
            indices = month_data.index.tolist()

            # Leave-one-out cross-validation
            loo = LeaveOneOut()
            predictions = np.zeros(len(X))

            for train_idx, test_idx in loo.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train = y[train_idx]

                model = LinearRegression()
                model.fit(X_train, y_train)
                predictions[test_idx] = model.predict(X_test)

            # Calculate corrected Q_loc
            corrected_values = month_data["Q_loc"].values + predictions

            # Update the dataframe
            for i, idx in enumerate(indices):
                data.loc[idx, "Q_loc_corrected"] = corrected_values[i]

    return data


def calculate_r2_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate R² scores per month for different predictions vs Q_obs.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, and Q_loc_corrected

    Returns:
        DataFrame with R² scores per month
    """
    results = []

    for month in sorted(data["month"].unique()):
        month_data = data[data["month"] == month].dropna(
            subset=["Q_obs", "Q_loc", "Q_MC_ALD", "Q_loc_corrected"]
        )

        if len(month_data) < 2:
            continue

        r2_q_loc = r2_score(month_data["Q_obs"], month_data["Q_loc"])
        r2_mc_ald = r2_score(month_data["Q_obs"], month_data["Q_MC_ALD"])
        r2_corrected = r2_score(month_data["Q_obs"], month_data["Q_loc_corrected"])

        results.append(
            {
                "month": month,
                "R² Q_loc (original)": r2_q_loc,
                "R² Q_MC_ALD": r2_mc_ald,
                "R² Q_loc (corrected)": r2_corrected,
                "n_samples": len(month_data),
            }
        )

    return pd.DataFrame(results)


def calculate_r2_by_code_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate R² scores per code and month for different predictions vs Q_obs.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, and Q_loc_corrected

    Returns:
        DataFrame with R² scores per code and month
    """
    results = []

    for code in data["code"].unique():
        for month in sorted(data["month"].unique()):
            mask = (data["code"] == code) & (data["month"] == month)
            subset = data[mask].dropna(
                subset=["Q_obs", "Q_loc", "Q_MC_ALD", "Q_loc_corrected"]
            )

            if len(subset) < 2:
                continue

            r2_q_loc = r2_score(subset["Q_obs"], subset["Q_loc"])
            r2_mc_ald = r2_score(subset["Q_obs"], subset["Q_MC_ALD"])
            r2_corrected = r2_score(subset["Q_obs"], subset["Q_loc_corrected"])

            results.append(
                {
                    "code": code,
                    "month": month,
                    "R² Q_loc (original)": r2_q_loc,
                    "R² Q_MC_ALD": r2_mc_ald,
                    "R² Q_loc (corrected)": r2_corrected,
                    "n_samples": len(subset),
                }
            )

    return pd.DataFrame(results)


def plot_r2_distribution_by_month(
    r2_df: pd.DataFrame,
    region: str,
    horizon: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot the distribution of R² scores per month for all three methods.

    Args:
        r2_df: DataFrame with R² scores per code and month
        region: Region name for the title
        horizon: Horizon name for the title
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Reshape data for plotting
    plot_data = []
    for _, row in r2_df.iterrows():
        plot_data.append(
            {
                "month": row["month"],
                "Method": "Q_loc (original)",
                "R²": row["R² Q_loc (original)"],
            }
        )
        plot_data.append(
            {"month": row["month"], "Method": "Q_MC_ALD", "R²": row["R² Q_MC_ALD"]}
        )
        plot_data.append(
            {
                "month": row["month"],
                "Method": "Q_loc (corrected)",
                "R²": row["R² Q_loc (corrected)"],
            }
        )

    plot_df = pd.DataFrame(plot_data)

    # Create the boxplot and violin plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Define colors for each method
    palette = {
        "Q_loc (original)": "#3498db",  # Blue
        "Q_MC_ALD": "#e74c3c",  # Red
        "Q_loc (corrected)": "#2ecc71",  # Green
    }

    # Boxplot
    sns.boxplot(x="month", y="R²", hue="Method", data=plot_df, palette=palette, ax=axes[0])
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].set_title(
        f"{region} - {horizon}\nR² Distribution per Month (Leave-One-Out CV)",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xlabel("Month", fontsize=11)
    axes[0].set_ylabel("R² Score", fontsize=11)
    axes[0].legend(title="Method", loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Violin plot
    sns.violinplot(
        x="month", y="R²", hue="Method", data=plot_df, palette=palette, inner="box", ax=axes[1]
    )
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].set_title(
        f"{region} - {horizon}\nR² Violin Plot (Leave-One-Out CV)",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].set_xlabel("Month", fontsize=11)
    axes[1].set_ylabel("R² Score", fontsize=11)
    axes[1].legend(title="Method", loc="lower right", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved R² distribution plot to {save_path}")

    return fig


def plot_summary_statistics(
    r2_df: pd.DataFrame,
    region: str,
    horizon: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot summary statistics comparing the three methods.

    Args:
        r2_df: DataFrame with R² scores per code and month
        region: Region name for the title
        horizon: Horizon name for the title
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Calculate median R² per month for each method
    summary = (
        r2_df.groupby("month")
        .agg(
            {
                "R² Q_loc (original)": ["median", "mean", "std", "count"],
                "R² Q_MC_ALD": ["median", "mean", "std"],
                "R² Q_loc (corrected)": ["median", "mean", "std"],
            }
        )
        .round(3)
    )

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{region} - {horizon}: Summary Statistics per Month")
    logger.info("=" * 80)
    print(summary.to_string())

    # Calculate overall statistics
    overall_stats = pd.DataFrame(
        {
            "Method": ["Q_loc (original)", "Q_MC_ALD", "Q_loc (corrected)"],
            "Median R²": [
                r2_df["R² Q_loc (original)"].median(),
                r2_df["R² Q_MC_ALD"].median(),
                r2_df["R² Q_loc (corrected)"].median(),
            ],
            "Mean R²": [
                r2_df["R² Q_loc (original)"].mean(),
                r2_df["R² Q_MC_ALD"].mean(),
                r2_df["R² Q_loc (corrected)"].mean(),
            ],
            "Std R²": [
                r2_df["R² Q_loc (original)"].std(),
                r2_df["R² Q_MC_ALD"].std(),
                r2_df["R² Q_loc (corrected)"].std(),
            ],
        }
    ).round(3)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{region} - {horizon}: Overall Statistics")
    logger.info("=" * 80)
    print(overall_stats.to_string(index=False))

    # Plot bar chart of median R² per month
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    months = sorted(r2_df["month"].unique())
    x = np.arange(len(months))
    width = 0.25

    medians_orig = r2_df.groupby("month")["R² Q_loc (original)"].median()
    medians_mcald = r2_df.groupby("month")["R² Q_MC_ALD"].median()
    medians_corr = r2_df.groupby("month")["R² Q_loc (corrected)"].median()

    # Median R² bar chart
    axes[0].bar(
        x - width, medians_orig, width, label="Q_loc (original)", color="#3498db"
    )
    axes[0].bar(x, medians_mcald, width, label="Q_MC_ALD", color="#e74c3c")
    axes[0].bar(
        x + width, medians_corr, width, label="Q_loc (corrected)", color="#2ecc71"
    )

    axes[0].set_xlabel("Month", fontsize=11)
    axes[0].set_ylabel("Median R²", fontsize=11)
    axes[0].set_title(f"{region} - {horizon}\nMedian R² per Month by Method", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([month_renaming.get(m, m)[:3] for m in months])
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Improvement comparison
    improvement_vs_orig = medians_corr - medians_orig
    improvement_vs_mcald = medians_corr - medians_mcald

    axes[1].bar(
        x - width / 2,
        improvement_vs_orig,
        width,
        label="vs Q_loc (original)",
        color="#3498db",
    )
    axes[1].bar(
        x + width / 2, improvement_vs_mcald, width, label="vs Q_MC_ALD", color="#e74c3c"
    )

    axes[1].set_xlabel("Month", fontsize=11)
    axes[1].set_ylabel("R² Improvement", fontsize=11)
    axes[1].set_title(f"{region} - {horizon}\nR² Improvement of Corrected Q_loc", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([month_renaming.get(m, m)[:3] for m in months])
    axes[1].legend(fontsize=9)
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved summary statistics plot to {save_path}")

    return fig


def plot_r2_by_horizon(
    all_r2_data: pd.DataFrame,
    region: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot R² vs forecast horizon for all methods.

    Args:
        all_r2_data: DataFrame with R² scores including horizon_num column
        region: Region name for the title
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Aggregate by horizon
    agg_data = (
        all_r2_data.groupby("horizon_num")
        .agg(
            {
                "R² Q_loc (original)": ["mean", "std"],
                "R² Q_MC_ALD": ["mean", "std"],
                "R² Q_loc (corrected)": ["mean", "std"],
            }
        )
        .reset_index()
    )
    
    # Flatten column names
    agg_data.columns = [
        "horizon",
        "Q_loc_mean", "Q_loc_std",
        "Q_MC_ALD_mean", "Q_MC_ALD_std",
        "Q_loc_corr_mean", "Q_loc_corr_std",
    ]

    fig, ax = plt.subplots(figsize=(12, 7))

    horizons = agg_data["horizon"].values

    # Plot lines with error bands
    ax.plot(horizons, agg_data["Q_loc_mean"], "o-", color="#3498db", linewidth=2, 
            markersize=8, label="Q_loc (original)")
    ax.fill_between(horizons, 
                    agg_data["Q_loc_mean"] - agg_data["Q_loc_std"],
                    agg_data["Q_loc_mean"] + agg_data["Q_loc_std"],
                    color="#3498db", alpha=0.2)

    ax.plot(horizons, agg_data["Q_MC_ALD_mean"], "s-", color="#e74c3c", linewidth=2,
            markersize=8, label="Q_MC_ALD")
    ax.fill_between(horizons,
                    agg_data["Q_MC_ALD_mean"] - agg_data["Q_MC_ALD_std"],
                    agg_data["Q_MC_ALD_mean"] + agg_data["Q_MC_ALD_std"],
                    color="#e74c3c", alpha=0.2)

    ax.plot(horizons, agg_data["Q_loc_corr_mean"], "^-", color="#2ecc71", linewidth=2,
            markersize=8, label="Q_loc (corrected)")
    ax.fill_between(horizons,
                    agg_data["Q_loc_corr_mean"] - agg_data["Q_loc_corr_std"],
                    agg_data["Q_loc_corr_mean"] + agg_data["Q_loc_corr_std"],
                    color="#2ecc71", alpha=0.2)

    ax.set_xlabel("Forecast Lead Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean R² Score", fontsize=12, fontweight="bold")
    ax.set_title(f"{region}: R² vs Forecast Lead Time\n(Mean ± Std across all codes and months)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(horizons)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved R² by horizon plot to {save_path}")

    return fig


def plot_bias_by_month(
    data: pd.DataFrame,
    region: str,
    horizon: str,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plot bias boxplots by month as percentage of observed, averaged per code:
    mean((Q_obs - Q_loc) / Q_obs * 100) and mean((Q_MC_ALD - Q_loc) / Q_obs * 100) per code.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, code, month columns
        region: Region name for the title
        horizon: Horizon name for the title
        ax: Optional matplotlib Axes to plot on

    Returns:
        matplotlib Axes object
    """
    # Calculate biases as percentage of observed
    data = data.copy()
    
    # Skip rows where Q_obs is zero or very small
    data = data[data["Q_obs"] > 0.01]
    
    # Calculate bias percentages
    data["true_bias_pct"] = (data["Q_obs"] - data["Q_loc"]) / data["Q_obs"] * 100
    data["model_corr_pct"] = (data["Q_MC_ALD"] - data["Q_loc"]) / data["Q_obs"] * 100
    
    # Average per code and month
    avg_bias = (
        data.groupby(["code", "month"])
        .agg({
            "true_bias_pct": "mean",
            "model_corr_pct": "mean",
        })
        .reset_index()
    )
    
    # Reshape for plotting
    plot_data = []
    for _, row in avg_bias.iterrows():
        plot_data.append({
            "month": row["month"],
            "Bias Type": "(Q_obs - Q_loc) / Q_obs (true bias)",
            "Bias (%)": row["true_bias_pct"],
        })
        plot_data.append({
            "month": row["month"],
            "Bias Type": "(Q_MC_ALD - Q_loc) / Q_obs (model correction)",
            "Bias (%)": row["model_corr_pct"],
        })

    plot_df = pd.DataFrame(plot_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Define colors
    palette = {
        "(Q_obs - Q_loc) / Q_obs (true bias)": "#2ecc71",  # Green
        "(Q_MC_ALD - Q_loc) / Q_obs (model correction)": "#e74c3c",  # Red
    }

    sns.boxplot(
        x="month", y="Bias (%)", hue="Bias Type", data=plot_df, 
        palette=palette, ax=ax
    )


    ax.set_ylim(-100, 100)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.set_title(f"{region} - {horizon}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Bias (% of Q_obs, avg per code)", fontsize=10)
    ax.legend(title="", loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    return ax


def plot_bias_all_horizons(
    all_data: list[pd.DataFrame],
    horizons: list[str],
    region: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot bias boxplots for all horizons in a single figure.

    Args:
        all_data: List of DataFrames, one per horizon
        horizons: List of horizon names
        region: Region name for the title
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    n_horizons = len(horizons)
    n_cols = min(3, n_horizons)
    n_rows = (n_horizons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, (data, horizon) in enumerate(zip(all_data, horizons)):
        ax = axes_flat[idx]
        plot_bias_by_month(data, region, horizon, ax=ax)
        
        # Only show legend on first subplot
        if idx > 0:
            ax.get_legend().remove()

    # Hide unused subplots
    for idx in range(n_horizons, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle(
        f"{region}: Bias Comparison by Month (% of Q_obs)\n((Q_obs - Q_loc) / Q_obs vs (Q_MC_ALD - Q_loc) / Q_obs)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved bias comparison plot to {save_path}")

    return fig


def analyze_region_horizon(
    region: str,
    horizon: str,
    save_dir: Path | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Analyze MC_ALD for a specific region and horizon.

    Args:
        region: Region name
        horizon: Horizon name (e.g., "month_1")
        save_dir: Optional directory to save plots

    Returns:
        Tuple of (data_with_corrections, r2_by_code_month) or (None, None) if no data
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Analyzing {region} - {horizon}")
    logger.info("=" * 80)

    # Get path configuration
    path_config = get_path_config(region)
    
    # Load data
    data = load_mc_ald_data(path_config["pred_dir"], horizon)
    
    if data is None or data.empty:
        logger.warning(f"No data available for {region} - {horizon}")
        return None, None

    logger.info(f"Loaded {len(data)} records with {data['code'].nunique()} unique codes")

    # Train correction model using LOOCV
    logger.info("Training linear regression correction model with Leave-One-Out CV...")
    data_with_corrections = train_correction_model_loocv(data)

    # Calculate R² scores per code and month
    logger.info("Calculating R² scores per code and month...")
    r2_by_code_month = calculate_r2_by_code_month(data_with_corrections)

    if r2_by_code_month.empty:
        logger.error("No R² scores could be calculated. Check your data.")
        return data_with_corrections, None

    logger.info(f"Calculated R² for {len(r2_by_code_month)} code-month combinations")

    # Add horizon info to R² data
    r2_by_code_month["horizon"] = horizon
    r2_by_code_month["horizon_num"] = int(horizon.split("_")[1])

    # Create plots
    if save_dir:
        dist_path = save_dir / f"{region.lower()}_{horizon}_r2_distribution.png"
        summary_path = save_dir / f"{region.lower()}_{horizon}_summary_stats.png"
    else:
        dist_path = None
        summary_path = None

    fig1 = plot_r2_distribution_by_month(r2_by_code_month, region, horizon, dist_path)
    fig2 = plot_summary_statistics(r2_by_code_month, region, horizon, summary_path)

    if not SAVE_PLOTS:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    # Show aggregated R² per month
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{region} - {horizon}: Aggregated R² per Month (all samples pooled)")
    logger.info("=" * 80)
    r2_aggregated = calculate_r2_by_month(data_with_corrections)
    print(r2_aggregated.to_string(index=False))

    return data_with_corrections, r2_by_code_month


def main() -> None:
    """Main function to run MC_ALD analysis."""
    
    # Determine horizons to analyze
    if HORIZONS_TO_ANALYZE == "all":
        horizons_to_use = ALL_HORIZONS
    else:
        horizons_to_use = HORIZONS_TO_ANALYZE
    
    logger.info("=" * 80)
    logger.info("MC_ALD Investigation Script")
    logger.info("=" * 80)
    logger.info(f"Regions to analyze: {REGIONS_TO_ANALYZE}")
    logger.info(f"Horizons to analyze: {horizons_to_use}")
    logger.info(f"Save plots: {SAVE_PLOTS}")
    logger.info("=" * 80)

    # Analyze each region
    for region in REGIONS_TO_ANALYZE:
        logger.info(f"\n{'#' * 80}")
        logger.info(f"REGION: {region}")
        logger.info("#" * 80)

        # Determine save directory for this region
        if SAVE_PLOTS and output_dir:
            save_dir = Path(output_dir) / region.lower() / "mc_ald_investigation"
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = None

        all_r2_data = []
        all_raw_data = []  # Store raw data for bias plots
        processed_horizons = []  # Track which horizons have data

        # Analyze each horizon
        for horizon in horizons_to_use:
            data, r2_data = analyze_region_horizon(
                region=region,
                horizon=horizon,
                save_dir=save_dir,
            )

            if r2_data is not None:
                all_r2_data.append(r2_data)
            
            if data is not None:
                all_raw_data.append(data)
                processed_horizons.append(horizon)

        # Create combined bias plot if we have data from multiple horizons
        if len(all_raw_data) > 0:
            bias_plot_path = None
            if save_dir:
                bias_plot_path = save_dir / f"{region.lower()}_bias_all_horizons.png"
            
            fig = plot_bias_all_horizons(all_raw_data, processed_horizons, region, bias_plot_path)
            
            if not SAVE_PLOTS:
                plt.show()
            else:
                plt.close(fig)

        # Create combined horizon plot if multiple horizons analyzed
        if len(all_r2_data) > 1:
            combined_r2 = pd.concat(all_r2_data, ignore_index=True)
            
            horizon_plot_path = None
            if save_dir:
                horizon_plot_path = save_dir / f"{region.lower()}_r2_by_horizon.png"
            
            fig = plot_r2_by_horizon(combined_r2, region, horizon_plot_path)
            
            if not SAVE_PLOTS:
                plt.show()
            else:
                plt.close(fig)

            # Print overall summary
            logger.info(f"\n{'=' * 80}")
            logger.info(f"{region}: Overall Summary Across All Horizons")
            logger.info("=" * 80)
            
            overall_summary = (
                combined_r2.groupby("horizon_num")
                .agg(
                    {
                        "R² Q_loc (original)": ["mean", "median", "std"],
                        "R² Q_MC_ALD": ["mean", "median", "std"],
                        "R² Q_loc (corrected)": ["mean", "median", "std"],
                    }
                )
                .round(3)
            )
            print(overall_summary.to_string())

    logger.info("\n" + "=" * 80)
    logger.info("MC_ALD Investigation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
