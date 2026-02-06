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
HORIZONS_TO_ANALYZE: list[str] | str = [
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

# Whether to save plots to files (True) or display interactively (False)
SAVE_PLOTS: bool = True

# Models to include in ensemble (main output only, no submodel variants)
ENSEMBLE_MODELS = [
    "SM_GBT",
    "SM_GBT_Norm",
    "SM_GBT_LR",
    "LR_SM",
    "LR_Base",
    "LR_SM_ROF",
]

# Model weights for weighted ensemble (GBT models weight 3x, LR models weight 1x)
MODEL_WEIGHTS = {
    "SM_GBT": 3,
    "SM_GBT_Norm": 3,
    "SM_GBT_LR": 3,
    "LR_SM": 1,
    "LR_Base": 1,
    "LR_SM_ROF": 1,
}

# Submodel variants to exclude (use only main model output)
SUBMODEL_SUFFIXES_TO_EXCLUDE = ["_xgb", "_catboost", "_lgbm", "_rf"]

# =============================================================================
# FUNCTIONS
# =============================================================================


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


def calculate_target(obs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates daily observations to monthly means for each code.

    Args:
        obs: DataFrame with columns: date, code, discharge (daily observations)

    Returns:
        DataFrame with columns: code, year, month, Q_obs_monthly (monthly mean discharge)
    """
    # Extract year and month from date
    obs = obs.copy()
    obs["year"] = obs["date"].dt.year
    obs["month"] = obs["date"].dt.month

    # Group by code, year, month and calculate mean discharge
    monthly_obs = (
        obs.groupby(["code", "year", "month"])["discharge"]
        .mean()
        .reset_index()
        .rename(columns={"discharge": "Q_obs_monthly"})
    )

    return monthly_obs


def get_path_config(region: str) -> dict[str, str]:
    """Get path configuration for a specific region."""
    if region == "Kyrgyzstan":
        return kgz_path_config
    elif region == "Tajikistan":
        return taj_path_config
    else:
        raise ValueError(
            f"Unknown region: {region}. Must be 'Kyrgyzstan' or 'Tajikistan'"
        )


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
    monthly_obs: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Load MC_ALD hindcast data for a specific horizon.

    Args:
        pred_dir: Base prediction directory
        horizon: Horizon name (e.g., "month_1")
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly).
                     If provided, Q_obs is merged from monthly observations using valid_from.

    Returns:
        DataFrame with MC_ALD data or None if file not found
    """
    file_path = get_mc_ald_path(pred_dir, horizon)

    if not file_path.exists():
        logger.warning(f"MC_ALD hindcast file not found: {file_path}")
        return None

    try:
        data = pd.read_csv(file_path, parse_dates=["date", "valid_from", "valid_to"])

        # Filter by day of forecast if specified
        forecast_day = day_of_forecast.get(horizon)
        if forecast_day is not None:
            data = data[data["date"].dt.day == forecast_day].copy()

        # Select relevant columns (Q_obs is optional, will be merged if monthly_obs provided)
        required_cols = ["date", "code", "Q_loc", "Q_MC_ALD", "valid_from"]
        available_cols = [col for col in required_cols if col in data.columns]

        if len(available_cols) < len(required_cols):
            missing = set(required_cols) - set(available_cols)
            logger.warning(f"Missing columns in {file_path}: {missing}")
            return None

        # Keep valid_from for merging
        data = data[
            available_cols + ["valid_to"]
            if "valid_to" in data.columns
            else available_cols
        ].copy()
        data = data.dropna(subset=["Q_loc", "Q_MC_ALD"])

        # Extract target month from valid_from (no shift calculation needed)
        data["target_month"] = data["valid_from"].dt.month
        data["target_year"] = data["valid_from"].dt.year
        data["month"] = data["target_month"]  # For compatibility
        data["horizon"] = horizon
        data["horizon_num"] = int(horizon.split("_")[1])

        # Merge with monthly observations if provided
        if monthly_obs is not None:
            # Drop 'month' column before merge to avoid conflict (we have target_month)
            data = data.drop(columns=["month"], errors="ignore")
            data = data.merge(
                monthly_obs,
                left_on=["code", "target_year", "target_month"],
                right_on=["code", "year", "month"],
                how="left",
            )
            data["Q_obs"] = data["Q_obs_monthly"]
            # Drop redundant columns and restore 'month' for compatibility
            data = data.drop(columns=["year", "Q_obs_monthly"], errors="ignore")
            data = data.rename(columns={"month": "month"})  # Keep from monthly_obs
            # Drop rows without observations
            data = data.dropna(subset=["Q_obs"])
        elif "Q_obs" in data.columns:
            # Use Q_obs from file if available
            data = data.dropna(subset=["Q_obs"])
        else:
            logger.warning(
                f"No Q_obs available for {horizon} (not in file and monthly_obs not provided)"
            )
            return None

        logger.info(
            f"Loaded {len(data)} records from {horizon} "
            f"({data['code'].nunique()} codes, months: {sorted(data['month'].unique())})"
        )

        return data

    except Exception as e:
        logger.error(f"Failed to load {file_path}: {e}")
        return None


def load_all_models_data(
    pred_dir: str,
    horizon: str,
    monthly_obs: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """
    Load hindcast data for all available models for a specific horizon.

    Args:
        pred_dir: Base prediction directory
        horizon: Horizon name (e.g., "month_1")
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly).
                     If provided, Q_obs is merged from monthly observations using valid_from.

    Returns:
        DataFrame with all models' predictions or None if no data found
    """
    import re

    base_path = Path(pred_dir)
    horizon_path = base_path / horizon

    if not horizon_path.exists():
        logger.warning(f"Horizon directory not found: {horizon_path}")
        return None

    forecast_day = day_of_forecast.get(horizon)
    all_model_data = []

    # Iterate through model subdirectories
    for model_dir in horizon_path.iterdir():
        if not model_dir.is_dir():
            continue

        model_name = model_dir.name
        hindcast_file = model_dir / f"{model_name}_hindcast.csv"

        if not hindcast_file.exists():
            continue

        try:
            # Parse valid_from for target month extraction
            df = pd.read_csv(
                hindcast_file, parse_dates=["date", "valid_from", "valid_to"]
            )
        except Exception as e:
            logger.warning(f"Failed to read {hindcast_file}: {e}")
            continue

        if df.empty:
            continue

        # Filter by day of forecast
        if forecast_day is not None:
            df = df[df["date"].dt.day == forecast_day].copy()

        if df.empty:
            continue

        # Find prediction columns (Q_* except quantiles and Q_obs)
        quantile_cols = [col for col in df.columns if re.fullmatch(r"Q\d+", col)]
        excluded_cols = ["Q_obs", "Q_Obs", "Q_OBS"] + quantile_cols
        q_cols = [
            c for c in df.columns if c.startswith("Q_") and c not in excluded_cols
        ]

        if not q_cols:
            continue

        # For each prediction column, check if it's a main output or submodel variant
        for q_col in q_cols:
            # Extract model name from column (e.g., Q_SM_GBT -> SM_GBT)
            pred_model_name = q_col[2:]  # Remove 'Q_' prefix

            # Skip submodel variants
            is_submodel = any(
                pred_model_name.endswith(suffix)
                for suffix in SUBMODEL_SUFFIXES_TO_EXCLUDE
            )
            if is_submodel:
                continue

            # Create result dataframe for this prediction
            result_df = pd.DataFrame(
                {
                    "date": df["date"],
                    "code": df["code"],
                    "valid_from": df["valid_from"],
                    "Q_pred": df[q_col],
                    "model": pred_model_name,
                }
            )

            all_model_data.append(result_df)

    if not all_model_data:
        logger.warning(f"No model data found for horizon {horizon}")
        return None

    combined = pd.concat(all_model_data, ignore_index=True)
    combined = combined.dropna(subset=["Q_pred"])

    # Extract target month from valid_from (no shift calculation needed)
    combined["target_month"] = combined["valid_from"].dt.month
    combined["target_year"] = combined["valid_from"].dt.year
    combined["month"] = combined["target_month"]  # For compatibility
    combined["horizon"] = horizon
    combined["horizon_num"] = int(horizon.split("_")[1])

    # Merge with monthly observations if provided
    if monthly_obs is not None:
        # Drop 'month' column before merge to avoid conflict (we have target_month)
        combined = combined.drop(columns=["month"], errors="ignore")
        combined = combined.merge(
            monthly_obs,
            left_on=["code", "target_year", "target_month"],
            right_on=["code", "year", "month"],
            how="left",
        )
        combined["Q_obs"] = combined["Q_obs_monthly"]
        # Drop redundant columns (keep 'month' from monthly_obs for compatibility)
        combined = combined.drop(columns=["year", "Q_obs_monthly"], errors="ignore")
        # Drop rows without observations
        combined = combined.dropna(subset=["Q_obs"])
    else:
        logger.warning(
            f"No monthly_obs provided for horizon {horizon}, Q_obs will be NaN"
        )
        combined["Q_obs"] = np.nan

    logger.info(
        f"Loaded {len(combined)} records from {combined['model'].nunique()} models for {horizon}"
    )
    logger.info(f"  Models: {sorted(combined['model'].unique().tolist())}")

    return combined


def create_weighted_ensemble(
    all_models_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a weighted ensemble from multiple models.

    Weighting: GBT models (SM_GBT, SM_GBT_Norm, SM_GBT_LR) weighted 3x,
               LR models (LR_SM, LR_Base, LR_SM_ROF) weighted 1x.

    Args:
        all_models_df: DataFrame with predictions from multiple models

    Returns:
        DataFrame with weighted ensemble predictions
    """
    df = all_models_df.copy()

    # Filter to only include models in our ensemble configuration
    df_filtered = df[df["model"].isin(MODEL_WEIGHTS.keys())].copy()

    if df_filtered.empty:
        logger.warning("No models available for ensemble creation")
        return pd.DataFrame()

    logger.info(
        f"Creating weighted ensemble from {df_filtered['model'].nunique()} models"
    )
    logger.info(f"  Models: {sorted(df_filtered['model'].unique().tolist())}")
    logger.info(f"  Weights: {MODEL_WEIGHTS}")

    # Add weight column
    df_filtered["weight"] = df_filtered["model"].map(MODEL_WEIGHTS)

    # Calculate weighted mean for each group
    def weighted_mean(group):
        weights = group["weight"].values
        values = group["Q_pred"].values
        return np.average(values, weights=weights)

    # Group by date, code
    grouped = df_filtered.groupby(["date", "code", "month", "horizon", "horizon_num"])

    ensemble_preds = grouped.apply(weighted_mean, include_groups=False).reset_index()
    ensemble_preds.columns = [
        "date",
        "code",
        "month",
        "horizon",
        "horizon_num",
        "Q_Ensemble",
    ]

    # Get Q_obs from the first model in each group
    q_obs = grouped["Q_obs"].first().reset_index()

    ensemble = ensemble_preds.merge(
        q_obs, on=["date", "code", "month", "horizon", "horizon_num"]
    )

    logger.info(f"Created weighted ensemble with {len(ensemble)} records")

    return ensemble


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


def calculate_r2_by_month(
    data: pd.DataFrame, include_ensemble: bool = False
) -> pd.DataFrame:
    """
    Calculate R² scores per month for different predictions vs Q_obs.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, Q_loc_corrected, and optionally Q_Ensemble
        include_ensemble: Whether to include ensemble R² calculation

    Returns:
        DataFrame with R² scores per month
    """
    results = []

    required_cols = ["Q_obs", "Q_loc", "Q_MC_ALD", "Q_loc_corrected"]
    if include_ensemble:
        required_cols.append("Q_Ensemble")

    for month in sorted(data["month"].unique()):
        month_data = data[data["month"] == month].dropna(subset=required_cols)

        if len(month_data) < 2:
            continue

        r2_q_loc = r2_score(month_data["Q_obs"], month_data["Q_loc"])
        r2_mc_ald = r2_score(month_data["Q_obs"], month_data["Q_MC_ALD"])
        r2_corrected = r2_score(month_data["Q_obs"], month_data["Q_loc_corrected"])

        result = {
            "month": month,
            "R² Q_loc (original)": r2_q_loc,
            "R² Q_MC_ALD": r2_mc_ald,
            "R² Q_loc (corrected)": r2_corrected,
            "n_samples": len(month_data),
        }

        if include_ensemble and "Q_Ensemble" in month_data.columns:
            r2_ensemble = r2_score(month_data["Q_obs"], month_data["Q_Ensemble"])
            result["R² Ensemble"] = r2_ensemble

        results.append(result)

    return pd.DataFrame(results)


def calculate_r2_by_code_month(
    data: pd.DataFrame, include_ensemble: bool = False
) -> pd.DataFrame:
    """
    Calculate R² scores per code and month for different predictions vs Q_obs.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, Q_loc_corrected, and optionally Q_Ensemble
        include_ensemble: Whether to include ensemble R² calculation

    Returns:
        DataFrame with R² scores per code and month
    """
    results = []

    required_cols = ["Q_obs", "Q_loc", "Q_MC_ALD", "Q_loc_corrected"]
    if include_ensemble:
        required_cols.append("Q_Ensemble")

    for code in data["code"].unique():
        for month in sorted(data["month"].unique()):
            mask = (data["code"] == code) & (data["month"] == month)
            subset = data[mask].dropna(subset=required_cols)

            if len(subset) < 2:
                continue

            r2_q_loc = r2_score(subset["Q_obs"], subset["Q_loc"])
            r2_mc_ald = r2_score(subset["Q_obs"], subset["Q_MC_ALD"])
            r2_corrected = r2_score(subset["Q_obs"], subset["Q_loc_corrected"])

            result = {
                "code": code,
                "month": month,
                "R² Q_loc (original)": r2_q_loc,
                "R² Q_MC_ALD": r2_mc_ald,
                "R² Q_loc (corrected)": r2_corrected,
                "n_samples": len(subset),
            }

            if include_ensemble and "Q_Ensemble" in subset.columns:
                r2_ensemble = r2_score(subset["Q_obs"], subset["Q_Ensemble"])
                result["R² Ensemble"] = r2_ensemble

            results.append(result)

    return pd.DataFrame(results)


def plot_r2_distribution_by_month(
    r2_df: pd.DataFrame,
    region: str,
    horizon: str,
    save_path: Path | None = None,
    include_ensemble: bool = False,
) -> plt.Figure:
    """
    Plot the distribution of R² scores per month for all methods.

    Args:
        r2_df: DataFrame with R² scores per code and month
        region: Region name for the title
        horizon: Horizon name for the title
        save_path: Optional path to save the figure
        include_ensemble: Whether to include ensemble in the plot

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
        if include_ensemble and "R² Ensemble" in row.index:
            plot_data.append(
                {"month": row["month"], "Method": "Ensemble", "R²": row["R² Ensemble"]}
            )

    plot_df = pd.DataFrame(plot_data)

    # Create the boxplot and violin plot side by side
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Define colors for each method
    palette = {
        "Q_loc (original)": "#3498db",  # Blue
        "Q_MC_ALD": "#e74c3c",  # Red
        "Q_loc (corrected)": "#2ecc71",  # Green
        "Ensemble": "#9b59b6",  # Purple
    }

    # Boxplot
    sns.boxplot(
        x="month", y="R²", hue="Method", data=plot_df, palette=palette, ax=axes[0]
    )
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
        x="month",
        y="R²",
        hue="Method",
        data=plot_df,
        palette=palette,
        inner="box",
        ax=axes[1],
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
    include_ensemble: bool = False,
) -> plt.Figure:
    """
    Plot summary statistics comparing the methods.

    Args:
        r2_df: DataFrame with R² scores per code and month
        region: Region name for the title
        horizon: Horizon name for the title
        save_path: Optional path to save the figure
        include_ensemble: Whether to include ensemble in the summary

    Returns:
        matplotlib Figure object
    """
    # Calculate median R² per month for each method
    agg_dict = {
        "R² Q_loc (original)": ["median", "mean", "std", "count"],
        "R² Q_MC_ALD": ["median", "mean", "std"],
        "R² Q_loc (corrected)": ["median", "mean", "std"],
    }
    if include_ensemble and "R² Ensemble" in r2_df.columns:
        agg_dict["R² Ensemble"] = ["median", "mean", "std"]

    summary = r2_df.groupby("month").agg(agg_dict).round(3)

    logger.info(f"\n{'=' * 80}")
    logger.info(f"{region} - {horizon}: Summary Statistics per Month")
    logger.info("=" * 80)
    print(summary.to_string())

    # Calculate overall statistics
    methods = ["Q_loc (original)", "Q_MC_ALD", "Q_loc (corrected)"]
    median_values = [
        r2_df["R² Q_loc (original)"].median(),
        r2_df["R² Q_MC_ALD"].median(),
        r2_df["R² Q_loc (corrected)"].median(),
    ]
    mean_values = [
        r2_df["R² Q_loc (original)"].mean(),
        r2_df["R² Q_MC_ALD"].mean(),
        r2_df["R² Q_loc (corrected)"].mean(),
    ]
    std_values = [
        r2_df["R² Q_loc (original)"].std(),
        r2_df["R² Q_MC_ALD"].std(),
        r2_df["R² Q_loc (corrected)"].std(),
    ]

    if include_ensemble and "R² Ensemble" in r2_df.columns:
        methods.append("Ensemble")
        median_values.append(r2_df["R² Ensemble"].median())
        mean_values.append(r2_df["R² Ensemble"].mean())
        std_values.append(r2_df["R² Ensemble"].std())

    overall_stats = pd.DataFrame(
        {
            "Method": methods,
            "Median R²": median_values,
            "Mean R²": mean_values,
            "Std R²": std_values,
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

    # Determine number of methods and adjust width
    n_methods = 4 if include_ensemble and "R² Ensemble" in r2_df.columns else 3
    width = 0.8 / n_methods

    medians_orig = r2_df.groupby("month")["R² Q_loc (original)"].median()
    medians_mcald = r2_df.groupby("month")["R² Q_MC_ALD"].median()
    medians_corr = r2_df.groupby("month")["R² Q_loc (corrected)"].median()

    # Median R² bar chart
    if n_methods == 4:
        medians_ens = r2_df.groupby("month")["R² Ensemble"].median()
        offset = width * 1.5
        axes[0].bar(
            x - offset, medians_orig, width, label="Q_loc (original)", color="#3498db"
        )
        axes[0].bar(
            x - width / 2, medians_mcald, width, label="Q_MC_ALD", color="#e74c3c"
        )
        axes[0].bar(
            x + width / 2,
            medians_corr,
            width,
            label="Q_loc (corrected)",
            color="#2ecc71",
        )
        axes[0].bar(x + offset, medians_ens, width, label="Ensemble", color="#9b59b6")
    else:
        axes[0].bar(
            x - width, medians_orig, width, label="Q_loc (original)", color="#3498db"
        )
        axes[0].bar(x, medians_mcald, width, label="Q_MC_ALD", color="#e74c3c")
        axes[0].bar(
            x + width, medians_corr, width, label="Q_loc (corrected)", color="#2ecc71"
        )

    axes[0].set_xlabel("Month", fontsize=11)
    axes[0].set_ylabel("Median R²", fontsize=11)
    axes[0].set_title(
        f"{region} - {horizon}\nMedian R² per Month by Method",
        fontsize=12,
        fontweight="bold",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([month_renaming.get(m, m)[:3] for m in months])
    axes[0].legend(fontsize=9)
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Improvement comparison - compare Ensemble vs MC_ALD and Q_loc
    if n_methods == 4:
        improvement_ens_vs_mcald = medians_ens - medians_mcald
        improvement_ens_vs_loc = medians_ens - medians_orig

        axes[1].bar(
            x - width / 2,
            improvement_ens_vs_loc,
            width,
            label="Ensemble vs Q_loc (original)",
            color="#3498db",
        )
        axes[1].bar(
            x + width / 2,
            improvement_ens_vs_mcald,
            width,
            label="Ensemble vs Q_MC_ALD",
            color="#e74c3c",
        )

        axes[1].set_title(
            f"{region} - {horizon}\nR² Improvement of Ensemble",
            fontsize=12,
            fontweight="bold",
        )
    else:
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
            x + width / 2,
            improvement_vs_mcald,
            width,
            label="vs Q_MC_ALD",
            color="#e74c3c",
        )

        axes[1].set_title(
            f"{region} - {horizon}\nR² Improvement of Corrected Q_loc",
            fontsize=12,
            fontweight="bold",
        )

    axes[1].set_xlabel("Month", fontsize=11)
    axes[1].set_ylabel("R² Improvement", fontsize=11)
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
    include_ensemble: bool = False,
) -> plt.Figure:
    """
    Plot R² vs forecast horizon for all methods.

    Args:
        all_r2_data: DataFrame with R² scores including horizon_num column
        region: Region name for the title
        save_path: Optional path to save the figure
        include_ensemble: Whether to include ensemble in the plot

    Returns:
        matplotlib Figure object
    """
    # Aggregate by horizon
    agg_dict = {
        "R² Q_loc (original)": ["mean", "std"],
        "R² Q_MC_ALD": ["mean", "std"],
        "R² Q_loc (corrected)": ["mean", "std"],
    }
    if include_ensemble and "R² Ensemble" in all_r2_data.columns:
        agg_dict["R² Ensemble"] = ["mean", "std"]

    agg_data = all_r2_data.groupby("horizon_num").agg(agg_dict).reset_index()

    # Flatten column names
    col_names = ["horizon"]
    for col in agg_dict.keys():
        short_name = (
            col.replace("R² ", "").replace(" ", "_").replace("(", "").replace(")", "")
        )
        col_names.extend([f"{short_name}_mean", f"{short_name}_std"])
    agg_data.columns = col_names

    fig, ax = plt.subplots(figsize=(12, 7))

    horizons = agg_data["horizon"].values

    # Plot lines with error bands
    ax.plot(
        horizons,
        agg_data["Q_loc_original_mean"],
        "o-",
        color="#3498db",
        linewidth=2,
        markersize=8,
        label="Q_loc (original)",
    )
    ax.fill_between(
        horizons,
        agg_data["Q_loc_original_mean"] - agg_data["Q_loc_original_std"],
        agg_data["Q_loc_original_mean"] + agg_data["Q_loc_original_std"],
        color="#3498db",
        alpha=0.2,
    )

    ax.plot(
        horizons,
        agg_data["Q_MC_ALD_mean"],
        "s-",
        color="#e74c3c",
        linewidth=2,
        markersize=8,
        label="Q_MC_ALD",
    )
    ax.fill_between(
        horizons,
        agg_data["Q_MC_ALD_mean"] - agg_data["Q_MC_ALD_std"],
        agg_data["Q_MC_ALD_mean"] + agg_data["Q_MC_ALD_std"],
        color="#e74c3c",
        alpha=0.2,
    )

    ax.plot(
        horizons,
        agg_data["Q_loc_corrected_mean"],
        "^-",
        color="#2ecc71",
        linewidth=2,
        markersize=8,
        label="Q_loc (corrected)",
    )
    ax.fill_between(
        horizons,
        agg_data["Q_loc_corrected_mean"] - agg_data["Q_loc_corrected_std"],
        agg_data["Q_loc_corrected_mean"] + agg_data["Q_loc_corrected_std"],
        color="#2ecc71",
        alpha=0.2,
    )

    ax.set_xlabel("Forecast Lead Time (months)", fontsize=12, fontweight="bold")
    # Add Ensemble line if available
    if include_ensemble and "Ensemble_mean" in agg_data.columns:
        ax.plot(
            horizons,
            agg_data["Ensemble_mean"],
            "D-",
            color="#9b59b6",
            linewidth=2,
            markersize=8,
            label="Ensemble",
        )
        ax.fill_between(
            horizons,
            agg_data["Ensemble_mean"] - agg_data["Ensemble_std"],
            agg_data["Ensemble_mean"] + agg_data["Ensemble_std"],
            color="#9b59b6",
            alpha=0.2,
        )

    ax.set_xlabel("Forecast Lead Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Mean R² Score", fontsize=12, fontweight="bold")
    ax.set_title(
        f"{region}: R² vs Forecast Lead Time\n(Mean ± Std across all codes and months)",
        fontsize=14,
        fontweight="bold",
    )
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
        .agg(
            {
                "true_bias_pct": "mean",
                "model_corr_pct": "mean",
            }
        )
        .reset_index()
    )

    # Reshape for plotting
    plot_data = []
    for _, row in avg_bias.iterrows():
        plot_data.append(
            {
                "month": row["month"],
                "Bias Type": "(Q_obs - Q_loc) / Q_obs (true bias)",
                "Bias (%)": row["true_bias_pct"],
            }
        )
        plot_data.append(
            {
                "month": row["month"],
                "Bias Type": "(Q_MC_ALD - Q_loc) / Q_obs (model correction)",
                "Bias (%)": row["model_corr_pct"],
            }
        )

    plot_df = pd.DataFrame(plot_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 6))

    # Define colors
    palette = {
        "(Q_obs - Q_loc) / Q_obs (true bias)": "#2ecc71",  # Green
        "(Q_MC_ALD - Q_loc) / Q_obs (model correction)": "#e74c3c",  # Red
    }

    sns.boxplot(
        x="month", y="Bias (%)", hue="Bias Type", data=plot_df, palette=palette, ax=ax
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

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
    )
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


def diagnose_extreme_year_effect(
    all_models_df: pd.DataFrame,
    region: str,
) -> None:
    """
    Diagnose if extreme years are causing systematic bias.

    If one year has much higher discharge than others, it inflates the global mean/std,
    causing overprediction for all other years.
    """
    df = all_models_df.copy()
    df = df[df["Q_obs"] > 0.01]
    df["year"] = df["date"].dt.year

    # Calculate bias per year
    df["bias_pct"] = (df["Q_obs"] - df["Q_pred"]) / df["Q_obs"] * 100

    # Get mean observed discharge per year (to identify extreme years)
    yearly_obs = (
        df.groupby("year")
        .agg(
            {
                "Q_obs": ["mean", "std", "max"],
                "Q_pred": "mean",
                "bias_pct": ["mean", "median"],
            }
        )
        .round(2)
    )
    yearly_obs.columns = [
        "Q_obs_mean",
        "Q_obs_std",
        "Q_obs_max",
        "Q_pred_mean",
        "bias_mean",
        "bias_median",
    ]

    # Identify potential extreme years (Q_obs_mean > 1.5 * median)
    median_obs = yearly_obs["Q_obs_mean"].median()
    yearly_obs["is_extreme"] = yearly_obs["Q_obs_mean"] > 1.5 * median_obs

    logger.info(f"\n{'=' * 100}")
    logger.info(f"{region}: YEARLY BIAS DIAGNOSTIC (Extreme Year Effect)")
    logger.info("=" * 100)
    logger.info(f"Median yearly Q_obs: {median_obs:.1f}")
    logger.info(
        f"Extreme years (>1.5x median): {yearly_obs[yearly_obs['is_extreme']].index.tolist()}"
    )
    print(yearly_obs.to_string())

    # Check correlation: do years with higher Q_obs have different bias?
    corr = yearly_obs["Q_obs_mean"].corr(yearly_obs["bias_mean"])
    logger.info(f"\nCorrelation (Q_obs_mean vs bias_mean): {corr:.3f}")

    if corr < -0.3:
        logger.warning(
            "⚠️  Negative correlation suggests extreme years cause underprediction,"
        )
        logger.warning(
            "    while normal years get overpredicted (global scaler effect!)"
        )
    elif corr > 0.3:
        logger.info(
            "Positive correlation suggests model struggles with high discharge years"
        )
    else:
        logger.info("Weak correlation - extreme years may not be the main bias source")


def print_bias_statistics(
    all_models_df: pd.DataFrame,
    region: str,
    horizon: str,
) -> pd.DataFrame:
    """
    Print detailed bias statistics comparing mean vs median.

    Args:
        all_models_df: DataFrame with model predictions
        region: Region name
        horizon: Horizon name

    Returns:
        DataFrame with statistics
    """
    df = all_models_df.copy()
    df = df[df["Q_obs"] > 0.01]

    # Calculate bias
    df["bias_pct"] = (df["Q_obs"] - df["Q_pred"]) / df["Q_obs"] * 100

    # Calculate statistics per model (first average per code to avoid sample size bias)
    avg_per_code = df.groupby(["model", "code"])["bias_pct"].mean().reset_index()

    stats = (
        avg_per_code.groupby("model")["bias_pct"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    stats.columns = ["Model", "Mean", "Median", "Std", "Min", "Max", "N_codes"]
    stats["Mean-Median"] = stats["Mean"] - stats["Median"]
    stats = stats.sort_values("Median")

    # Print formatted table
    logger.info(f"\n{'=' * 100}")
    logger.info(f"{region} - {horizon}: BIAS STATISTICS (% of Q_obs)")
    logger.info(f"Negative = OVERESTIMATE, Positive = UNDERESTIMATE")
    logger.info("=" * 100)
    print(stats.round(2).to_string(index=False))

    # Check for outlier effect
    mean_median_diff = (stats["Mean"] - stats["Median"]).abs().mean()
    if mean_median_diff > 5:
        logger.warning(
            f"⚠️  Mean-Median difference averages {mean_median_diff:.1f}% - OUTLIERS likely affecting results!"
        )
    else:
        logger.info(
            f"Mean-Median difference averages {mean_median_diff:.1f}% - outlier effect is small"
        )

    return stats


def plot_model_bias_comparison(
    all_models_df: pd.DataFrame,
    region: str,
    horizon: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot bias comparison across all models as boxplots.

    Shows (Q_obs - Q_pred) / Q_obs * 100 for each model.
    Negative bias = OVERESTIMATE, Positive bias = UNDERESTIMATE.

    Args:
        all_models_df: DataFrame with columns: model, Q_pred, Q_obs, month, code
        region: Region name for the title
        horizon: Horizon name for the title
        save_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Print statistics first
    stats = print_bias_statistics(all_models_df, region, horizon)

    df = all_models_df.copy()

    # Filter valid observations
    df = df[df["Q_obs"] > 0.01]

    # Calculate bias as percentage of observed
    df["bias_pct"] = (df["Q_obs"] - df["Q_pred"]) / df["Q_obs"] * 100

    # Get unique models sorted
    models = sorted(df["model"].unique())
    n_models = len(models)

    # Create figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Overall bias per model (boxplot)
    ax1 = axes[0]
    avg_bias_per_code = df.groupby(["model", "code"])["bias_pct"].mean().reset_index()

    # Calculate mean and median for sorting
    model_stats = (
        avg_bias_per_code.groupby("model")["bias_pct"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    model_stats = model_stats.sort_values("median")  # Sort by median

    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, n_models))
    color_map = {model: colors[i] for i, model in enumerate(model_stats["model"])}

    sns.boxplot(
        data=avg_bias_per_code,
        x="model",
        y="bias_pct",
        order=model_stats["model"],
        palette=[color_map[m] for m in model_stats["model"]],
        ax=ax1,
    )

    ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Bias (% of Q_obs)", fontsize=12)
    ax1.set_title(
        f"Bias Distribution per Model\n(avg per code, n={model_stats['count'].iloc[0]} codes)",
        fontsize=12,
    )
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(-80, 80)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Mean vs Median comparison (bar chart)
    ax2 = axes[1]
    x = np.arange(len(model_stats))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        model_stats["mean"],
        width,
        label="Mean",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x + width / 2,
        model_stats["median"],
        width,
        label="Median",
        color="darkorange",
        alpha=0.8,
    )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Bias (% of Q_obs)", fontsize=12)
    ax2.set_title("Mean vs Median Bias\n(outlier detection)", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_stats["model"], rotation=45, ha="right")
    ax2.legend(loc="best")
    ax2.set_ylim(-50, 30)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars1, model_stats["mean"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    for bar, val in zip(bars2, model_stats["median"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 1,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # Plot 3: Bias by month for key models
    ax3 = axes[2]

    key_models = ["LR_Base", "LR_SM", "SM_GBT", "SM_GBT_Norm", "SM_GBT_LR", "LR_SM_ROF"]
    key_models = [m for m in key_models if m in models][:6]

    df_selected = df[df["model"].isin(key_models)]

    # Use MEDIAN per code and month (more robust to outliers)
    median_by_month = (
        df_selected.groupby(["model", "month", "code"])["bias_pct"].mean().reset_index()
    )

    sns.boxplot(
        data=median_by_month,
        x="month",
        y="bias_pct",
        hue="model",
        hue_order=key_models,
        ax=ax3,
    )

    ax3.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax3.set_xlabel("Month", fontsize=12)
    ax3.set_ylabel("Bias (% of Q_obs)", fontsize=12)
    ax3.set_title("Bias by Month for Key Models", fontsize=12)
    ax3.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax3.set_ylim(-80, 80)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"{region} - {horizon}: Model Bias Comparison\nBias = (Q_obs - Q_pred) / Q_obs × 100%  |  Negative = OVERESTIMATE",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved model bias comparison to {save_path}")

    return fig


def plot_model_bias_all_horizons(
    all_horizons_data: list[pd.DataFrame],
    horizons: list[str],
    region: str,
    save_path: Path | None = None,
) -> plt.Figure:
    """
    Plot model bias comparison aggregated across all horizons.

    Args:
        all_horizons_data: List of DataFrames from load_all_models_data, one per horizon
        horizons: List of horizon names
        region: Region name
        save_path: Optional path to save

    Returns:
        matplotlib Figure
    """
    # Combine all data
    combined = pd.concat(all_horizons_data, ignore_index=True)

    # Filter valid observations
    combined = combined[combined["Q_obs"] > 0.01]

    # Calculate bias
    combined["bias_pct"] = (
        (combined["Q_obs"] - combined["Q_pred"]) / combined["Q_obs"] * 100
    )

    # Get models
    models = sorted(combined["model"].unique())

    # Print combined statistics
    logger.info(f"\n{'=' * 100}")
    logger.info(f"{region}: COMBINED BIAS STATISTICS ACROSS ALL HORIZONS")
    logger.info(f"Negative = OVERESTIMATE, Positive = UNDERESTIMATE")
    logger.info("=" * 100)

    avg_per_code = combined.groupby(["model", "code"])["bias_pct"].mean().reset_index()
    stats = (
        avg_per_code.groupby("model")["bias_pct"]
        .agg(["mean", "median", "std", "min", "max", "count"])
        .reset_index()
    )
    stats.columns = ["Model", "Mean", "Median", "Std", "Min", "Max", "N_codes"]
    stats["Mean-Median"] = stats["Mean"] - stats["Median"]
    stats = stats.sort_values("Median")
    print(stats.round(2).to_string(index=False))

    # Create figure: 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Plot 1: Bias distribution by model (boxplot)
    ax1 = axes[0]
    avg_bias = combined.groupby(["model", "code"])["bias_pct"].mean().reset_index()
    model_order = (
        avg_bias.groupby("model")["bias_pct"].median().sort_values().index.tolist()
    )

    sns.boxplot(
        data=avg_bias,
        x="model",
        y="bias_pct",
        order=model_order,
        palette="RdYlGn",
        ax=ax1,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax1.set_xlabel("Model", fontsize=12)
    ax1.set_ylabel("Bias (% of Q_obs)", fontsize=12)
    ax1.set_title("Bias Distribution by Model\n(sorted by median)", fontsize=12)
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(-80, 80)
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Mean vs Median comparison
    ax2 = axes[1]
    model_stats = stats.set_index("Model").loc[model_order].reset_index()

    x = np.arange(len(model_stats))
    width = 0.35

    bars1 = ax2.bar(
        x - width / 2,
        model_stats["Mean"],
        width,
        label="Mean",
        color="steelblue",
        alpha=0.8,
    )
    bars2 = ax2.bar(
        x + width / 2,
        model_stats["Median"],
        width,
        label="Median",
        color="darkorange",
        alpha=0.8,
    )

    ax2.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax2.set_xlabel("Model", fontsize=12)
    ax2.set_ylabel("Bias (% of Q_obs)", fontsize=12)
    ax2.set_title(
        "Mean vs Median Bias\n(large difference = outlier effect)", fontsize=12
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_stats["Model"], rotation=45, ha="right")
    ax2.legend(loc="best")
    ax2.set_ylim(-50, 30)
    ax2.grid(True, alpha=0.3, axis="y")

    # Plot 3: Bias by horizon for key models
    ax3 = axes[2]

    key_models = ["LR_Base", "LR_SM", "SM_GBT", "SM_GBT_Norm", "SM_GBT_LR", "LR_SM_ROF"]
    key_models = [m for m in key_models if m in models]

    df_key = combined[combined["model"].isin(key_models)]
    avg_by_horizon = (
        df_key.groupby(["model", "horizon_num", "code"])["bias_pct"]
        .mean()
        .reset_index()
    )

    sns.boxplot(
        data=avg_by_horizon,
        x="horizon_num",
        y="bias_pct",
        hue="model",
        hue_order=key_models,
        ax=ax3,
    )
    ax3.axhline(0, color="black", linestyle="--", linewidth=1.5)
    ax3.set_xlabel("Forecast Horizon (months)", fontsize=12)
    ax3.set_ylabel("Bias (% of Q_obs)", fontsize=12)
    ax3.set_title("Bias by Horizon for Key Models", fontsize=12)
    ax3.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax3.set_ylim(-80, 80)
    ax3.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"{region}: Model Bias Across All Horizons\nBias = (Q_obs - Q_pred) / Q_obs × 100%  |  Negative = OVERESTIMATE",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved model bias all horizons to {save_path}")

    return fig


def analyze_region_horizon(
    region: str,
    horizon: str,
    monthly_obs: pd.DataFrame,
    save_dir: Path | None = None,
    include_ensemble: bool = True,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    """
    Analyze MC_ALD for a specific region and horizon.

    Args:
        region: Region name
        horizon: Horizon name (e.g., "month_1")
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly)
        save_dir: Optional directory to save plots
        include_ensemble: Whether to include weighted ensemble comparison

    Returns:
        Tuple of (data_with_corrections, r2_by_code_month, all_models_data) or (None, None, None) if no data
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Analyzing {region} - {horizon}")
    logger.info("=" * 80)

    # Get path configuration
    path_config = get_path_config(region)

    # Load MC_ALD data with monthly observations
    data = load_mc_ald_data(path_config["pred_dir"], horizon, monthly_obs)

    if data is None or data.empty:
        logger.warning(f"No data available for {region} - {horizon}")
        return None, None, None

    logger.info(
        f"Loaded {len(data)} MC_ALD records with {data['code'].nunique()} unique codes"
    )

    # Load all models data (always, for bias comparison)
    logger.info("Loading all models data...")
    all_models_data = load_all_models_data(
        path_config["pred_dir"], horizon, monthly_obs
    )

    # Create weighted ensemble if requested
    ensemble_data = None
    if include_ensemble and all_models_data is not None and not all_models_data.empty:
        logger.info("Creating weighted ensemble...")
        ensemble_df = create_weighted_ensemble(all_models_data)

        if not ensemble_df.empty:
            # Merge ensemble with MC_ALD data
            ensemble_data = ensemble_df[
                ["date", "code", "month", "Q_Ensemble"]
            ].drop_duplicates()
            data = data.merge(ensemble_data, on=["date", "code", "month"], how="left")
            logger.info(
                f"Merged ensemble data: {data['Q_Ensemble'].notna().sum()} records with ensemble predictions"
            )

    # Create model bias comparison plot
    if all_models_data is not None and not all_models_data.empty and save_dir:
        bias_path = save_dir / f"{region.lower()}_{horizon}_model_bias_comparison.png"
        fig_bias = plot_model_bias_comparison(
            all_models_data, region, horizon, bias_path
        )
        if SAVE_PLOTS:
            plt.close(fig_bias)
        else:
            plt.show()

    # Train correction model using LOOCV
    logger.info("Training linear regression correction model with Leave-One-Out CV...")
    data_with_corrections = train_correction_model_loocv(data)

    # Calculate R² scores per code and month
    logger.info("Calculating R² scores per code and month...")
    has_ensemble = (
        "Q_Ensemble" in data_with_corrections.columns
        and data_with_corrections["Q_Ensemble"].notna().any()
    )
    r2_by_code_month = calculate_r2_by_code_month(
        data_with_corrections, include_ensemble=has_ensemble
    )

    if r2_by_code_month.empty:
        logger.error("No R² scores could be calculated. Check your data.")
        return data_with_corrections, None, all_models_data

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

    fig1 = plot_r2_distribution_by_month(
        r2_by_code_month, region, horizon, dist_path, include_ensemble=has_ensemble
    )
    fig2 = plot_summary_statistics(
        r2_by_code_month, region, horizon, summary_path, include_ensemble=has_ensemble
    )

    if not SAVE_PLOTS:
        plt.show()
    else:
        plt.close(fig1)
        plt.close(fig2)

    # Show aggregated R² per month
    logger.info(f"\n{'=' * 80}")
    logger.info(f"{region} - {horizon}: Aggregated R² per Month (all samples pooled)")
    logger.info("=" * 80)
    r2_aggregated = calculate_r2_by_month(
        data_with_corrections, include_ensemble=has_ensemble
    )
    print(r2_aggregated.to_string(index=False))

    return data_with_corrections, r2_by_code_month, all_models_data


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

        # Get path configuration
        path_config = get_path_config(region)

        # Load observations and calculate monthly targets (once per region)
        logger.info(f"Loading observations from {path_config['obs_file']}...")
        obs_df = load_observations(path_config["obs_file"])
        logger.info(f"Loaded {len(obs_df)} daily observations")

        logger.info("Calculating monthly observation targets...")
        monthly_obs = calculate_target(obs_df)
        logger.info(
            f"Calculated {len(monthly_obs)} monthly observations "
            f"({monthly_obs['code'].nunique()} codes)"
        )

        # Determine save directory for this region
        if SAVE_PLOTS and output_dir:
            save_dir = Path(output_dir) / region.lower() / "mc_ald_investigation"
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = None

        all_r2_data = []
        all_raw_data = []  # Store raw data for bias plots
        all_models_data_list = []  # Store all models data for bias comparison
        processed_horizons = []  # Track which horizons have data

        # Analyze each horizon
        for horizon in horizons_to_use:
            data, r2_data, all_models_data = analyze_region_horizon(
                region=region,
                horizon=horizon,
                monthly_obs=monthly_obs,
                save_dir=save_dir,
            )

            if r2_data is not None:
                all_r2_data.append(r2_data)

            if data is not None:
                all_raw_data.append(data)
                processed_horizons.append(horizon)

            if all_models_data is not None:
                all_models_data_list.append(all_models_data)

        # Create combined bias plot if we have data from multiple horizons
        if len(all_raw_data) > 0:
            bias_plot_path = None
            if save_dir:
                bias_plot_path = save_dir / f"{region.lower()}_bias_all_horizons.png"

            fig = plot_bias_all_horizons(
                all_raw_data, processed_horizons, region, bias_plot_path
            )

            if not SAVE_PLOTS:
                plt.show()
            else:
                plt.close(fig)

        # Create model bias comparison across all horizons
        if len(all_models_data_list) > 0:
            logger.info("\nCreating model bias comparison across all horizons...")
            model_bias_path = None
            if save_dir:
                model_bias_path = (
                    save_dir / f"{region.lower()}_model_bias_all_horizons.png"
                )

            fig = plot_model_bias_all_horizons(
                all_models_data_list, processed_horizons, region, model_bias_path
            )

            if not SAVE_PLOTS:
                plt.show()
            else:
                plt.close(fig)

            # Diagnose extreme year effect
            combined_models = pd.concat(all_models_data_list, ignore_index=True)
            diagnose_extreme_year_effect(combined_models, region)

        # Create combined horizon plot if multiple horizons analyzed
        if len(all_r2_data) > 1:
            combined_r2 = pd.concat(all_r2_data, ignore_index=True)

            # Check if ensemble data is available
            has_ensemble = "R² Ensemble" in combined_r2.columns

            horizon_plot_path = None
            if save_dir:
                horizon_plot_path = save_dir / f"{region.lower()}_r2_by_horizon.png"

            fig = plot_r2_by_horizon(
                combined_r2, region, horizon_plot_path, include_ensemble=has_ensemble
            )

            if not SAVE_PLOTS:
                plt.show()
            else:
                plt.close(fig)

            # Print overall summary
            logger.info(f"\n{'=' * 80}")
            logger.info(f"{region}: Overall Summary Across All Horizons")
            logger.info("=" * 80)

            agg_dict = {
                "R² Q_loc (original)": ["mean", "median", "std"],
                "R² Q_MC_ALD": ["mean", "median", "std"],
                "R² Q_loc (corrected)": ["mean", "median", "std"],
            }
            if has_ensemble:
                agg_dict["R² Ensemble"] = ["mean", "median", "std"]

            overall_summary = combined_r2.groupby("horizon_num").agg(agg_dict).round(3)
            print(overall_summary.to_string())

    logger.info("\n" + "=" * 80)
    logger.info("MC_ALD Investigation Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
