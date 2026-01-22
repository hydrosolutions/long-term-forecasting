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
from sklearn.metrics import r2_score, mean_squared_error

# load the .env file
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


day_of_forecast = {
    "month_0": 15,
    "month_1": 25,
    "month_2": 25,
    "month_3": 25,
    "month_4": 25,
    "month_5": 25,
    # "month_6": 25,
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

# Models to exclude from ensemble (includes Q_obs variants as safety measure)
models_not_to_ensemble = [
    "MC_ALD",
    "SM_GBT",
    "SM_GBT_Norm",
    "SM_GBT_LR",
    "LR_SM_ROF",
    "LR_SM",
    "LR_Base",
    "MC_ALD_loc",
    "obs",
    "Obs",
    "OBS",  # Exclude any observation-based "models"
]

models_plot = ["LR_Base", "LR_SM", "MC_ALD"]

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

# Add configuration for which issue months to evaluate
# Set to None to evaluate all months, or specify a list like [3, 4, 5] for Mar-May only
issue_months_to_evaluate: list[int] | None = None  # Evaluate all months

# Flag to control whether to use Q_obs from predictions or load from observations file
# True: Use Q_obs directly from prediction files (if available)
# False: Load observations from file and aggregate to monthly means
use_Q_obs: bool = False

# Flag to control whether to apply climatology-based ratio correction
# True: Calculate ratio = Q_pred / Q_ltm_period, then Q_pred_corrected = ratio * Q_ltm_monthly
# False: Use Q_pred directly without correction
apply_climatology_correction: bool = True


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate various performance metrics.

    1. R2
    2. nRMSE = RMSE / mean(observed)
    3. MAE
    4. nMAE = MAE / mean(observed)
    5. Accuracy : |y_true - y_pred| <= 0.675 * std of y_true -> 1 else 0
    6. Efficiency: std (|y_true - y_pred|) / std(y_true)

    Args:
        y_true: Series of observed values
        y_pred: Series of predicted values

    Returns:
        Dictionary with calculated metrics
    """
    # Drop NaN values
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
            "accuracy": np.nan,
            "efficiency": np.nan,
            "n_samples": len(y_true_clean),
        }

    # Calculate metrics
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))

    mean_obs = y_true_clean.mean()
    std_obs = y_true_clean.std()

    nrmse = rmse / mean_obs if mean_obs != 0 else np.nan
    nmae = mae / mean_obs if mean_obs != 0 else np.nan

    # Accuracy: fraction of predictions within 0.675 * std of observed
    threshold = 0.675 * std_obs
    accuracy = np.mean(np.abs(y_true_clean - y_pred_clean) <= threshold)

    # Efficiency: std(errors) / std(observed)
    errors = np.abs(y_true_clean - y_pred_clean)
    efficiency = errors.std() / std_obs if std_obs != 0 else np.nan

    return {
        "r2": r2,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "nmae": nmae,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "n_samples": len(y_true_clean),
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


def calculate_leave_one_out_monthly_mean_fast(
    monthly_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate leave-one-out long-term mean and std for each code and month (FAST vectorized version).

    For each (code, year, month) combination, calculates the mean and std of all other
    years' values for that month (excluding the current year).

    Uses vectorized operations: LOO_mean = (total_sum - current_value) / (n - 1)

    Args:
        monthly_obs: DataFrame with columns: code, year, month, Q_obs_monthly

    Returns:
        DataFrame with columns: code, year, month, Q_obs_monthly,
                               Q_ltm_monthly (leave-one-out long-term mean),
                               Q_std_monthly (leave-one-out standard deviation)
    """
    df = monthly_obs.copy()

    # Calculate sum, sum of squares, and count for each (code, month) group
    agg = (
        df.groupby(["code", "month"])["Q_obs_monthly"]
        .agg(["sum", "count", lambda x: (x**2).sum()])
        .reset_index()
    )
    agg.columns = ["code", "month", "total_sum", "n_years", "total_sum_sq"]

    # Merge back to get total_sum, total_sum_sq and n_years for each row
    df = df.merge(agg, on=["code", "month"], how="left")

    # Calculate leave-one-out mean: (total_sum - current_value) / (n_years - 1)
    df["Q_ltm_monthly"] = (df["total_sum"] - df["Q_obs_monthly"]) / (df["n_years"] - 1)

    # Calculate leave-one-out std using the formula:
    # LOO_var = (sum_sq - x^2 - (n-1)*LOO_mean^2) / (n-2) for sample variance
    # Or more directly: recalculate from remaining values
    # Using: var = E[X^2] - E[X]^2, adjusted for leave-one-out
    loo_sum = df["total_sum"] - df["Q_obs_monthly"]
    loo_sum_sq = df["total_sum_sq"] - df["Q_obs_monthly"] ** 2
    loo_n = df["n_years"] - 1

    # LOO variance = (sum_sq / n) - mean^2, then multiply by n/(n-1) for sample variance
    # = (loo_sum_sq - loo_sum^2/loo_n) / (loo_n - 1)
    df["Q_std_monthly"] = np.sqrt((loo_sum_sq - (loo_sum**2 / loo_n)) / (loo_n - 1))

    # Handle edge case where n_years == 1 (use the value itself, std = 0)
    df.loc[df["n_years"] == 1, "Q_ltm_monthly"] = df.loc[
        df["n_years"] == 1, "Q_obs_monthly"
    ]
    df.loc[df["n_years"] <= 2, "Q_std_monthly"] = (
        np.nan
    )  # Can't compute std with < 3 samples for LOO

    # Drop helper columns
    df = df.drop(columns=["total_sum", "n_years", "total_sum_sq"])

    logger.info(
        f"Calculated leave-one-out monthly means and stds for {df['code'].nunique()} codes"
    )

    return df


def precompute_daily_climatology(
    daily_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Pre-compute daily climatology statistics for fast leave-one-out calculations.

    Creates a lookup table with sum and count of discharge for each (code, month, day)
    combination across all years, enabling O(1) leave-one-out calculations.

    Args:
        daily_obs: DataFrame with columns: date, code, discharge

    Returns:
        DataFrame with columns: code, month, day, total_sum, n_years, yearly data
    """
    df = daily_obs.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day

    # Aggregate by code, month, day, year first (in case of duplicates)
    daily_by_year = (
        df.groupby(["code", "year", "month", "day"])["discharge"].mean().reset_index()
    )

    # Create summary stats for each (code, month, day)
    daily_stats = (
        daily_by_year.groupby(["code", "month", "day"])
        .agg(
            total_sum=("discharge", "sum"),
            n_years=("discharge", "count"),
        )
        .reset_index()
    )

    # Also create a pivot of year -> discharge for each (code, month, day)
    # This allows us to quickly subtract specific years
    yearly_pivot = daily_by_year.pivot_table(
        index=["code", "month", "day"],
        columns="year",
        values="discharge",
        aggfunc="mean",
    ).reset_index()

    # Merge stats with yearly data
    result = daily_stats.merge(yearly_pivot, on=["code", "month", "day"], how="left")

    # Ensure month and day are integers
    result["month"] = result["month"].astype(int)
    result["day"] = result["day"].astype(int)

    logger.info(f"Pre-computed daily climatology for {result['code'].nunique()} codes")

    return result


def calculate_period_ltm_fast(
    daily_climatology: pd.DataFrame,
    predictions_df: pd.DataFrame,
    daily_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate leave-one-out period means and stds for all predictions using vectorized operations.

    For each prediction, calculates the long-term mean and std for the valid_from to valid_to period,
    excluding all years that overlap with the prediction (valid_from.year, valid_to.year, target_year).

    Args:
        daily_climatology: Pre-computed daily climatology from precompute_daily_climatology
        predictions_df: DataFrame with predictions (must have valid_from, valid_to, code, target_year)
        daily_obs: Original daily observations (for year lookup)

    Returns:
        predictions_df with Q_ltm_period and Q_std_period columns added
    """
    df = predictions_df.copy()

    # Extract period bounds
    df["from_month"] = df["valid_from"].dt.month
    df["from_day"] = df["valid_from"].dt.day
    df["to_month"] = df["valid_to"].dt.month
    df["to_day"] = df["valid_to"].dt.day
    df["from_year"] = df["valid_from"].dt.year
    df["to_year"] = df["valid_to"].dt.year

    # Get unique years in the daily observations
    year_cols = [
        col for col in daily_climatology.columns if isinstance(col, (int, np.integer))
    ]

    # Helper function to convert month/day to day-of-year (for period filtering)
    def to_doy(month: int, day: int) -> int:
        days_in_months = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        return sum(days_in_months[: int(month)]) + min(
            int(day), days_in_months[int(month)]
        )

    # Add day-of-year to climatology (vectorized)
    clim = daily_climatology.copy()
    days_in_months_cumsum = np.array(
        [0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    )
    clim["doy"] = days_in_months_cumsum[clim["month"].values] + clim["day"].values

    # Process in batches by unique (code, valid_from, valid_to, target_year) combinations
    # This is much faster than row-by-row
    unique_periods = df[
        [
            "code",
            "valid_from",
            "valid_to",
            "target_year",
            "from_month",
            "from_day",
            "to_month",
            "to_day",
            "from_year",
            "to_year",
        ]
    ].drop_duplicates()

    logger.info(f"Calculating period LTM for {len(unique_periods)} unique periods...")

    period_ltm_results = []

    for _, period in unique_periods.iterrows():
        code = period["code"]
        from_doy = to_doy(period["from_month"], period["from_day"])
        to_doy_val = to_doy(period["to_month"], period["to_day"])

        # Years to exclude (leave-one-out)
        exclude_years = set(
            [period["from_year"], period["to_year"], period["target_year"]]
        )

        # Filter climatology for this code and period
        code_clim = clim[clim["code"] == code].copy()

        # Filter by day-of-year range
        if from_doy <= to_doy_val:
            period_clim = code_clim[
                (code_clim["doy"] >= from_doy) & (code_clim["doy"] <= to_doy_val)
            ]
        else:
            # Period spans year boundary
            period_clim = code_clim[
                (code_clim["doy"] >= from_doy) | (code_clim["doy"] <= to_doy_val)
            ]

        if period_clim.empty:
            period_ltm_results.append(
                {
                    "code": code,
                    "valid_from": period["valid_from"],
                    "valid_to": period["valid_to"],
                    "target_year": period["target_year"],
                    "Q_ltm_period": np.nan,
                    "Q_std_period": np.nan,
                }
            )
            continue

        # Calculate leave-one-out mean and std
        # Sum all year columns except excluded years
        valid_year_cols = [y for y in year_cols if y not in exclude_years]

        if not valid_year_cols:
            period_ltm_results.append(
                {
                    "code": code,
                    "valid_from": period["valid_from"],
                    "valid_to": period["valid_to"],
                    "target_year": period["target_year"],
                    "Q_ltm_period": np.nan,
                    "Q_std_period": np.nan,
                }
            )
            continue

        # Calculate mean across valid years for each day, then mean across days
        # This properly weights each day equally
        daily_means = period_clim[valid_year_cols].mean(axis=1, skipna=True)
        period_mean = daily_means.mean()

        # Calculate std: for each year, get the period mean, then std across years
        # This gives the inter-annual variability of period means
        year_period_means = []
        for y in valid_year_cols:
            year_data = period_clim[y].dropna()
            if len(year_data) > 0:
                year_period_means.append(year_data.mean())

        if len(year_period_means) >= 2:
            period_std = np.std(year_period_means, ddof=1)  # sample std
        else:
            period_std = np.nan

        period_ltm_results.append(
            {
                "code": code,
                "valid_from": period["valid_from"],
                "valid_to": period["valid_to"],
                "target_year": period["target_year"],
                "Q_ltm_period": period_mean,
                "Q_std_period": period_std,
            }
        )

    period_ltm_df = pd.DataFrame(period_ltm_results)

    # Merge back to predictions
    df = df.merge(
        period_ltm_df, on=["code", "valid_from", "valid_to", "target_year"], how="left"
    )

    # Clean up temporary columns
    df = df.drop(
        columns=["from_month", "from_day", "to_month", "to_day", "from_year", "to_year"]
    )

    logger.info(f"Calculated period LTM for {len(unique_periods)} unique periods")

    return df


def apply_climatology_ratio_correction_fast(
    predictions_df: pd.DataFrame,
    daily_obs: pd.DataFrame,
    monthly_obs_with_ltm: pd.DataFrame,
    region: str,
) -> pd.DataFrame:
    """
    Apply climatology-based z-score correction to predictions (FAST vectorized version).

    For each prediction:
    1. Calculate leave-one-out long-term mean and std for the valid_from to valid_to period
       (excluding ALL years that overlap with valid_from/valid_to to prevent data leakage)
    2. Calculate z-score: z = (Q_pred - Q_ltm_period) / Q_std_period
    3. Clip z-score at 2% and 98% probability levels (z ≈ -2.054 to +2.054)
    4. Final prediction = Q_ltm_monthly + z_clipped * Q_std_monthly

    IMPORTANT: Leave-one-out is applied on a yearly basis to prevent data leakage.

    Args:
        predictions_df: DataFrame with predictions (must have valid_from, valid_to, Q_pred)
        daily_obs: DataFrame with daily observations (date, code, discharge)
        monthly_obs_with_ltm: DataFrame with monthly obs and leave-one-out means/stds
        region: Region name ("Kyrgyzstan" or "Tajikistan")

    Returns:
        DataFrame with added columns:
        - Q_ltm_period: Leave-one-out long-term mean for the valid period
        - Q_std_period: Leave-one-out standard deviation for the valid period
        - Q_ltm_monthly: Leave-one-out long-term mean for the target month
        - Q_std_monthly: Leave-one-out standard deviation for the target month
        - z_score: (Q_pred - Q_ltm_period) / Q_std_period
        - z_score_clipped: z_score clipped at 2% and 98% levels
        - Q_pred_corrected: Q_ltm_monthly + z_score_clipped * Q_std_monthly
    """
    from scipy import stats

    df = predictions_df.copy()

    # Calculate target date based on region
    if region == "Kyrgyzstan":
        # Vectorized: add horizon months
        df["target_date"] = df.apply(
            lambda row: row["issue_date"] + pd.DateOffset(months=int(row["horizon"])),
            axis=1,
        )
    elif region == "Tajikistan":
        df["target_date"] = df.apply(
            lambda row: row["issue_date"]
            + pd.DateOffset(months=int(row["horizon"]) - 1),
            axis=1,
        )
    else:
        raise ValueError(f"Unknown region: {region}")

    df["target_year"] = df["target_date"].dt.year
    df["target_month"] = df["target_date"].dt.month
    df["issue_month"] = df["issue_date"].dt.month

    logger.info(
        "Calculating climatology-based z-score corrections (fast vectorized)..."
    )
    logger.info("  NOTE: Using leave-one-out on yearly basis to prevent data leakage")
    logger.info("  NOTE: Z-scores clipped at 2% and 98% probability levels")

    # Pre-compute daily climatology
    logger.info("  Pre-computing daily climatology...")
    daily_climatology = precompute_daily_climatology(daily_obs)

    # Calculate period LTM and STD for all predictions
    logger.info("  Calculating period leave-one-out means and stds...")
    df = calculate_period_ltm_fast(daily_climatology, df, daily_obs)

    # Merge with monthly leave-one-out means and stds
    logger.info("  Merging with monthly leave-one-out means and stds...")
    df = df.merge(
        monthly_obs_with_ltm[
            ["code", "year", "month", "Q_ltm_monthly", "Q_std_monthly"]
        ],
        left_on=["code", "target_year", "target_month"],
        right_on=["code", "year", "month"],
        how="left",
    )
    df = df.drop(columns=["year", "month"], errors="ignore")

    # Calculate z-score: (Q_pred - Q_ltm_period) / Q_std_period
    logger.info("  Calculating z-scores and corrected predictions...")
    df["z_score"] = (df["Q_pred"] - df["Q_ltm_period"]) / df["Q_std_period"]

    # Clip z-score at 1% and 99% probability levels
    # For a standard normal distribution:
    z_lower = stats.norm.ppf(0.01)
    z_upper = stats.norm.ppf(0.99)
    df["z_score_clipped"] = df["z_score"].clip(lower=z_lower, upper=z_upper)

    # Transform z-score to monthly prediction: Q_ltm_monthly + z_clipped * Q_std_monthly
    df["Q_pred_corrected"] = (
        df["Q_ltm_monthly"] + df["z_score_clipped"] * df["Q_std_monthly"]
    )

    # Handle edge cases (missing std, etc.)
    df.loc[df["Q_std_period"] <= 0, "z_score"] = np.nan
    df.loc[df["Q_std_period"] <= 0, "z_score_clipped"] = np.nan
    df.loc[df["Q_std_period"].isna(), "Q_pred_corrected"] = np.nan
    df.loc[df["Q_std_monthly"].isna(), "Q_pred_corrected"] = np.nan

    # Ensure Q_pred_corrected is non-negative (discharge can't be negative)
    df["Q_pred_corrected"] = df["Q_pred_corrected"].clip(lower=0)

    logger.info(f"Finished processing {len(df)} rows")
    logger.info(f"  Rows with Q_ltm_period: {df['Q_ltm_period'].notna().sum()}")
    logger.info(f"  Rows with Q_std_period: {df['Q_std_period'].notna().sum()}")
    logger.info(f"  Rows with Q_pred_corrected: {df['Q_pred_corrected'].notna().sum()}")
    logger.info(
        f"  Z-score stats: min={df['z_score'].min():.3f}, median={df['z_score'].median():.3f}, max={df['z_score'].max():.3f}"
    )
    logger.info(
        f"  Z-score clipped stats: min={df['z_score_clipped'].min():.3f}, median={df['z_score_clipped'].median():.3f}, max={df['z_score_clipped'].max():.3f}"
    )
    logger.info(f"  Z-score clip bounds: [{z_lower:.3f}, {z_upper:.3f}] (1% - 99%)")

    return df


def create_ensemble(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates ensemble mean across all models except those in models_not_to_ensemble.

    Args:
        predictions_df: DataFrame with predictions containing columns:
            - code, issue_date, valid_from, valid_to, Q_pred, Q_obs, horizon, model

    Returns:
        DataFrame with ensemble predictions added as a new model "Ensemble"
    """
    # Filter models to ensemble
    ensemble_models = predictions_df[
        ~predictions_df["model"].isin(models_not_to_ensemble)
    ].copy()

    logger.info(f"Creating ensemble from {ensemble_models['model'].nunique()} models")
    logger.info(
        f" Names of models included: {ensemble_models['model'].unique().tolist()}"
    )

    if ensemble_models.empty:
        logger.warning("No models available for ensemble creation")
        return predictions_df

    # Group by code, issue_date, horizon and calculate mean prediction
    ensemble = (
        ensemble_models.groupby(
            ["code", "issue_date", "horizon", "valid_from", "valid_to"]
        )
        .agg(
            {
                "Q_pred": "mean",
                "Q_obs": "first",  # Q_obs should be the same for all models
            }
        )
        .reset_index()
    )

    # Add model name
    ensemble["model"] = "Ensemble"

    # Add quantile columns as NaN (ensemble doesn't have quantiles)
    quantile_cols = [
        col for col in predictions_df.columns if re.fullmatch(r"Q\d+", col)
    ]
    for col in quantile_cols:
        ensemble[col] = np.nan

    # Concatenate with original predictions
    combined = pd.concat([predictions_df, ensemble], ignore_index=True)

    logger.info(f"Created ensemble from {ensemble_models['model'].nunique()} models")

    return combined


def aggregate(
    predictions_df: pd.DataFrame, monthly_obs: pd.DataFrame, region: str
) -> pd.DataFrame:
    """
    Merge predictions with monthly aggregated observations based on region-specific logic.

    For Kyrgyzstan: target month = issue_date + horizon (months)
        e.g., 15.4 + month_0 = April, month_1 = May, etc.
    For Tajikistan: target month = issue_date + horizon - 1 (months)
        e.g., 15.4 + month_1 = April, etc.

    Args:
        predictions_df: DataFrame with predictions (code, issue_date, horizon, Q_pred, model, etc.)
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly)
        region: Region name ("Kyrgyzstan" or "Tajikistan")

    Returns:
        DataFrame with merged predictions and monthly observations, including target_month
    """
    df = predictions_df.copy()

    # Calculate target date based on region
    if region == "Kyrgyzstan":
        # Target month = issue_date + horizon
        df["target_date"] = df.apply(
            lambda row: row["issue_date"] + pd.DateOffset(months=int(row["horizon"])),
            axis=1,
        )
    elif region == "Tajikistan":
        # Target month = issue_date + horizon - 1
        df["target_date"] = df.apply(
            lambda row: row["issue_date"]
            + pd.DateOffset(months=int(row["horizon"]) - 1),
            axis=1,
        )
    else:
        raise ValueError(
            f"Unknown region: {region}. Must be 'Kyrgyzstan' or 'Tajikistan'"
        )

    # Extract year and month from target_date
    df["target_year"] = df["target_date"].dt.year
    df["target_month"] = df["target_date"].dt.month
    df["issue_month"] = df["issue_date"].dt.month

    # Merge with monthly observations
    merged = df.merge(
        monthly_obs,
        left_on=["code", "target_year", "target_month"],
        right_on=["code", "year", "month"],
        how="left",
    )

    # Drop redundant columns
    merged = merged.drop(columns=["year", "month"], errors="ignore")

    # Replace Q_obs with Q_obs_monthly if available
    merged["Q_obs"] = merged["Q_obs_monthly"].combine_first(merged["Q_obs"])
    merged = merged.drop(columns=["Q_obs_monthly"], errors="ignore")

    logger.info(f"Aggregated predictions with monthly observations for {region}")
    logger.info(f"  Total records: {len(merged)}")
    logger.info(f"  Records with Q_obs: {merged['Q_obs'].notna().sum()}")

    return merged


def compute_metrics_dataframe(aggregated_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics for all combinations of issue_month, horizon, target_month, model, and code.

    Args:
        aggregated_df: DataFrame with aggregated predictions and observations

    Returns:
        DataFrame with columns: code, model, issue_month, horizon, target_month,
                               r2, rmse, nrmse, mae, nmae, accuracy, efficiency, n_samples
    """
    metrics_list = []

    # Group by code, model, issue_month, horizon, target_month
    for (
        code,
        model,
        issue_month,
        horizon,
        target_month,
    ), group in aggregated_df.groupby(
        ["code", "model", "issue_month", "horizon", "target_month"]
    ):
        # Calculate metrics for this group
        metrics = calculate_metrics(group["Q_obs"], group["Q_pred"])

        metrics_list.append(
            {
                "code": code,
                "model": model,
                "issue_month": issue_month,
                "horizon": horizon,
                "target_month": target_month,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(metrics_list)

    logger.info(f"Computed metrics for {len(metrics_df)} combinations")

    return metrics_df


def load_predictions(
    base_path: str,
    horizons: list[str],
    issue_months: list[int] | None = None,
) -> pd.DataFrame:
    """
    This function loads all the model predictions from the specified directory.

    Args:
        base_path: Base directory containing horizon subdirectories
        horizons: List of horizon identifiers (e.g., ["month_0", "month_1", ...])
        issue_months: Optional list of issue months to filter (1-12). If None, all months are kept.

    Returns:
        DataFrame with concatenated predictions filtered by issue_months if specified.
    """
    base_path = Path(base_path)
    all_predictions = []

    set_codes = set()
    for horizon in horizons:
        horizon_path = base_path / horizon
        if not horizon_path.exists():
            logger.warning(f"Horizon directory not found: {horizon_path}")
            continue

        # Extract horizon number from 'month_X' format
        horizon_num = int(horizon.split("_")[1])
        forecast_day = day_of_forecast[horizon]

        # Iterate through model subdirectories
        for model_dir in horizon_path.iterdir():
            if not model_dir.is_dir():
                continue

            model_name = model_dir.name
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
                logger.debug(f"Empty dataframe in {hindcast_file}")
                continue

            # Convert dates to datetime
            df["date"] = pd.to_datetime(df["date"])
            df["valid_from"] = pd.to_datetime(df["valid_from"])
            df["valid_to"] = pd.to_datetime(df["valid_to"])

            # Convert code to int
            df["code"] = df["code"].astype(int)

            logger.debug(
                f"Loaded {len(df)} rows from {hindcast_file}, date range: {df['date'].min()} to {df['date'].max()}, days: {df['date'].dt.day.unique()}"
            )

            # Keep track of unique codes across all models
            if not set_codes:
                set_codes = set(df["code"].unique().tolist())
            else:
                set_codes.update(df["code"].unique().tolist())

            # Sort by code and date
            df = df.sort_values(["code", "date"]).reset_index(drop=True)

            # Filter to keep only the day of the month matching forecast day
            df = df[df["date"].dt.day == forecast_day].copy()

            if df.empty:
                logger.debug(
                    f"No data after filtering by forecast day {forecast_day} in {hindcast_file}"
                )
                continue

            # Filter by issue months early if specified
            if issue_months is not None:
                df = df[df["date"].dt.month.isin(issue_months)].copy()
                if df.empty:
                    logger.debug(
                        f"No data after filtering by issue months in {hindcast_file}"
                    )
                    continue

            # Find all prediction columns (Q_* except quantiles)
            # Quantile columns match pattern Q followed by digits only (Q5, Q25, Q50, Q75, Q95, etc.)
            quantile_cols = [col for col in df.columns if re.fullmatch(r"Q\d+", col)]
            # Exclude Q_obs (observed discharge) - should not be treated as a prediction
            excluded_cols = ["Q_obs", "Q_Obs", "Q_OBS"]
            q_cols = [
                c
                for c in df.columns
                if c.startswith("Q_")
                and c not in quantile_cols
                and c not in excluded_cols
            ]

            # Find the Q_obs column if it exists
            q_obs_col = next(
                (col for col in df.columns if col.lower() == "q_obs"), None
            )

            if not q_cols:
                logger.warning(f"No prediction column found in {hindcast_file}")
                continue

            # Create a result DataFrame for each Q column (submodel)
            for q_col in q_cols:
                # Extract submodel name from column (e.g., Q_xgb -> xgb, Q_SM_GBT -> SM_GBT)
                submodel_name = q_col[2:]  # Remove 'Q_' prefix

                # Create full model name: model_dir/submodel or just submodel if it matches model_dir
                if submodel_name == model_name:
                    full_model_name = model_name
                else:
                    full_model_name = f"{model_name}_{submodel_name}"

                # Restructure the DataFrame - keep valid_from and valid_to for ratio calculation
                result_df = pd.DataFrame(
                    {
                        "code": df["code"],
                        "issue_date": df["date"],
                        "valid_from": df["valid_from"],
                        "valid_to": df["valid_to"],
                        "Q_pred": df[q_col],
                        "Q_obs": df[q_obs_col] if q_obs_col else np.nan,
                        "horizon": horizon_num,
                        "model": full_model_name,
                    }
                )

                # Add quantile columns if they exist (only for the main model, not submodels)
                for quantile_col in quantile_cols:
                    if quantile_col in df.columns and submodel_name == model_name:
                        result_df[quantile_col] = df[quantile_col].values
                    else:
                        result_df[quantile_col] = np.nan

                all_predictions.append(result_df)

    if not all_predictions:
        logger.error("No predictions loaded from any horizon/model combination.")
        return pd.DataFrame()

    combined_df = pd.concat(all_predictions, ignore_index=True)
    logger.info(
        f"Loaded {len(combined_df)} prediction records from {len(all_predictions)} files."
    )
    logger.info(f"Unique codes found: {sorted(set_codes)}")
    logger.info(
        f"Unique models found: {sorted(combined_df['model'].unique().tolist())}"
    )

    return combined_df


def plot_metric_by_horizon(
    metrics_df: pd.DataFrame,
    horizon: int,
    models: list[str],
    metric: str = "r2",
    y_limits: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot metric distribution by target month for a given forecast horizon.

    Creates a boxplot showing metric distribution for all models across all target months
    for a specific forecast horizon. X-axis = target month (1-12), Y-axis = metric.
    If all points for a model/month fall below the y-axis limit, a visual marker
    showing the percentage of samples below threshold is displayed.

    Args:
        metrics_df: DataFrame with metrics containing columns:
            - code, model, issue_month, horizon, target_month, r2, rmse, nrmse, etc.
        horizon: The forecast horizon to filter by (e.g., 1, 2, 3)
        models: List of model names to include in the plot
        metric: Metric to plot (default: "r2"). Options: "r2", "rmse", "nrmse", "mae", "nmae", "accuracy", "efficiency"
        y_limits: Optional tuple (y_min, y_max) for y-axis limits. If None, uses metric defaults.
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Filter for the specified horizon
    df = metrics_df[metrics_df["horizon"] == horizon].copy()

    if df.empty:
        logger.warning(f"No data found for horizon {horizon}")
        return plt.figure()

    # Filter for specified models
    df = df[df["model"].isin(models)].copy()

    if df.empty:
        logger.warning(f"No data found for models {models} at horizon {horizon}")
        return plt.figure()

    # Set default y-axis limits based on metric
    if y_limits is None:
        if metric == "r2":
            y_limits = (-1.0, 1.0)
        elif metric == "accuracy":
            y_limits = (0.0, 1.0)
        elif metric in ["nrmse", "nmae", "efficiency"]:
            y_limits = (0.0, 2.0)
        else:
            # For rmse, mae - use data range
            y_limits = (df[metric].min() * 0.9, df[metric].max() * 1.1)

    y_min, y_max = y_limits

    # Create a single plot
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Get unique target months and sort them
    target_months = sorted(df["target_month"].unique())

    # Define color palette for models
    model_colors = sns.color_palette("husl", len(models))
    color_map = dict(zip(models, model_colors))

    # Prepare data for grouped boxplot
    positions = []
    data_to_plot = []
    colors_to_use = []
    below_threshold_markers = []  # Store info for markers

    box_width = 0.8 / len(models)

    # Debug: track what we find for each model/month
    missing_combinations = []

    for m_idx, month in enumerate(target_months):
        for model_idx, model in enumerate(models):
            model_month_data = df[
                (df["target_month"] == month) & (df["model"] == model)
            ][metric]

            position = m_idx + (model_idx - len(models) / 2 + 0.5) * box_width

            if len(model_month_data) == 0:
                missing_combinations.append(f"{model} @ month {month}")
                continue

            if len(model_month_data) > 0:
                # Filter out NaN values - they can occur when metrics calculation had insufficient data
                values = model_month_data.dropna().values

                if len(values) == 0:
                    missing_combinations.append(f"{model} @ month {month} (all NaN)")
                    continue

                n_total = len(values)
                n_below = np.sum(values < y_min)
                n_above = np.sum(values > y_max)

                # Clip values to y_limits for plotting, but track outliers
                values_clipped = np.clip(values, y_min, y_max)

                # If ALL values are below threshold, don't plot box, just marker
                if n_below == n_total:
                    below_threshold_markers.append(
                        {
                            "position": position,
                            "pct_below": 100.0,
                            "median": np.median(values),
                            "color": color_map[model],
                            "model": model,
                            "month": month,
                            "direction": "below",
                        }
                    )
                elif n_above == n_total:
                    below_threshold_markers.append(
                        {
                            "position": position,
                            "pct_above": 100.0,
                            "median": np.median(values),
                            "color": color_map[model],
                            "model": model,
                            "month": month,
                            "direction": "above",
                        }
                    )
                else:
                    positions.append(position)
                    data_to_plot.append(values)
                    colors_to_use.append(color_map[model])

                    # Track partial below/above threshold
                    if n_below > 0:
                        below_threshold_markers.append(
                            {
                                "position": position,
                                "pct_below": (n_below / n_total) * 100,
                                "color": color_map[model],
                                "model": model,
                                "month": month,
                                "direction": "below",
                                "partial": True,
                            }
                        )
                    if n_above > 0:
                        below_threshold_markers.append(
                            {
                                "position": position,
                                "pct_above": (n_above / n_total) * 100,
                                "color": color_map[model],
                                "model": model,
                                "month": month,
                                "direction": "above",
                                "partial": True,
                            }
                        )

    # Create boxplots if there's data
    if data_to_plot:
        bp = ax.boxplot(
            data_to_plot,
            positions=positions,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker="o", markersize=3, alpha=0.5),
        )

        # Color the boxes
        for patch, color in zip(bp["boxes"], colors_to_use):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    else:
        logger.warning(f"Horizon {horizon}: No data to plot in boxplot!")

    # Add markers for values below/above threshold
    for marker_info in below_threshold_markers:
        pos = marker_info["position"]
        color = marker_info["color"]

        if marker_info.get("partial", False):
            # Partial: some values outside range - add small annotation
            if marker_info["direction"] == "below":
                pct = marker_info["pct_below"]
                ax.annotate(
                    f"↓{pct:.0f}%",
                    xy=(pos, y_min),
                    xytext=(pos, y_min + 0.02 * (y_max - y_min)),
                    fontsize=7,
                    color=color,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )
            else:
                pct = marker_info["pct_above"]
                ax.annotate(
                    f"↑{pct:.0f}%",
                    xy=(pos, y_max),
                    xytext=(pos, y_max - 0.02 * (y_max - y_min)),
                    fontsize=7,
                    color=color,
                    ha="center",
                    va="top",
                    fontweight="bold",
                )
        else:
            # All values outside range - prominent marker
            if marker_info["direction"] == "below":
                median = marker_info["median"]
                ax.scatter(
                    [pos],
                    [y_min + 0.03 * (y_max - y_min)],
                    marker="v",
                    s=80,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=10,
                )
                ax.annotate(
                    f"100%↓\n(med={median:.2f})",
                    xy=(pos, y_min + 0.03 * (y_max - y_min)),
                    xytext=(pos, y_min + 0.12 * (y_max - y_min)),
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=color,
                    ),
                )
            else:
                median = marker_info["median"]
                ax.scatter(
                    [pos],
                    [y_max - 0.03 * (y_max - y_min)],
                    marker="^",
                    s=80,
                    color=color,
                    edgecolors="black",
                    linewidths=0.5,
                    zorder=10,
                )
                ax.annotate(
                    f"100%↑\n(med={median:.2f})",
                    xy=(pos, y_max - 0.03 * (y_max - y_min)),
                    xytext=(pos, y_max - 0.12 * (y_max - y_min)),
                    fontsize=8,
                    color=color,
                    ha="center",
                    va="top",
                    fontweight="bold",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        alpha=0.8,
                        edgecolor=color,
                    ),
                )

    # Customize the plot
    ax.set_xlabel("Target Month", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        metric.upper() if len(metric) <= 4 else metric.capitalize(),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"Forecast Horizon {horizon} - {metric.upper()} by Target Month",
        fontsize=14,
        fontweight="bold",
    )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(target_months)))
    ax.set_xticklabels(
        [month_renaming[m][:3] for m in target_months], rotation=45, ha="right"
    )

    # Set y-axis limits
    ax.set_ylim(y_min, y_max)

    # Add horizontal reference line
    if metric == "r2":
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    elif metric in ["nrmse", "nmae", "efficiency"]:
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    elif metric == "accuracy":
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.grid(axis="y", alpha=0.3)

    # Create legend
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color_map[model], alpha=0.7) for model in models
    ]
    ax.legend(legend_handles, models, loc="upper right", framealpha=0.9, fontsize=10)

    # Log missing combinations
    if missing_combinations:
        logger.warning(
            f"Horizon {horizon}: Missing data for {len(missing_combinations)} combinations: {missing_combinations[:10]}..."
        )

    # Log below/above threshold statistics
    n_full_below = len(
        [
            m
            for m in below_threshold_markers
            if not m.get("partial", False) and m["direction"] == "below"
        ]
    )
    n_full_above = len(
        [
            m
            for m in below_threshold_markers
            if not m.get("partial", False) and m["direction"] == "above"
        ]
    )
    if n_full_below > 0 or n_full_above > 0:
        logger.info(
            f"Horizon {horizon}: {n_full_below} model/month combos have 100% values below y_min={y_min}, {n_full_above} have 100% above y_max={y_max}"
        )
        # Print details of which ones
        for m in below_threshold_markers:
            if not m.get("partial", False):
                logger.info(
                    f"  -> {m['model']} @ month {m['month']}: direction={m['direction']}, median={m.get('median', 'N/A')}"
                )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(
            f"Saved {metric} by target month plot for horizon {horizon} to {output_path}"
        )

    return fig


def plot_metric_by_lead_time(
    metrics_df: pd.DataFrame,
    models: list[str],
    metric: str = "r2",
    y_limits: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
    aggregate_by: str = "target_month",
) -> plt.Figure:
    """
    Plot metric vs forecasting lead time (horizon) for all months.

    Creates a line plot showing metric values for each model across different
    forecast horizons, with separate lines for each month (target or issue).
    X-axis = forecast lead time (horizon), Y-axis = metric.

    Args:
        metrics_df: DataFrame with metrics containing columns:
            - code, model, issue_month, horizon, target_month, r2, rmse, nrmse, etc.
        models: List of model names to include in the plot
        metric: Metric to plot (default: "r2"). Options: "r2", "rmse", "nrmse", "mae", "nmae", "accuracy", "efficiency"
        y_limits: Optional tuple (y_min, y_max) for y-axis limits. If None, uses metric defaults.
        output_path: Optional path to save the figure
        aggregate_by: Which month to use for grouping - "target_month" or "issue_month"

    Returns:
        matplotlib Figure object
    """
    # Filter for specified models
    df = metrics_df[metrics_df["model"].isin(models)].copy()

    if df.empty:
        logger.warning(f"No data found for models {models}")
        return plt.figure()

    # Set default y-axis limits based on metric
    if y_limits is None:
        if metric == "r2":
            y_limits = (-0.5, 1.0)
        elif metric == "accuracy":
            y_limits = (0.0, 1.0)
        elif metric in ["nrmse", "nmae", "efficiency"]:
            y_limits = (0.0, 2.0)
        else:
            y_limits = (df[metric].min() * 0.9, df[metric].max() * 1.1)

    y_min, y_max = y_limits

    # Get unique horizons and months
    horizons = sorted(df["horizon"].unique())
    months = sorted(df[aggregate_by].unique())

    # Create subplots - one per model
    n_models = len(models)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False
    )
    axes_flat = axes.flatten()

    # Define color palette for months (12 distinct colors)
    month_colors = sns.color_palette("husl", 12)
    color_map = {m: month_colors[m - 1] for m in range(1, 13)}

    for model_idx, model in enumerate(models):
        ax = axes_flat[model_idx]
        model_data = df[df["model"] == model]

        if model_data.empty:
            ax.set_title(f"{model}\n(No data)", fontsize=12, fontweight="bold")
            ax.set_visible(True)
            continue

        # Aggregate metrics by horizon and month (mean across codes)
        agg_data = (
            model_data.groupby(["horizon", aggregate_by])[metric]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        agg_data.columns = ["horizon", "month", "mean", "std", "count"]

        # Plot line for each month
        for month in months:
            month_data = agg_data[agg_data["month"] == month].sort_values("horizon")

            if month_data.empty:
                continue

            color = color_map.get(month, "gray")
            month_name = month_renaming.get(month, str(month))[:3]

            # Plot mean line with error bands (± 1 std / sqrt(n) for SEM)
            ax.plot(
                month_data["horizon"],
                month_data["mean"],
                marker="o",
                markersize=6,
                linewidth=2,
                color=color,
                label=month_name,
                alpha=0.8,
            )

            # Add shaded error band (standard error of the mean)
            sem = month_data["std"] / np.sqrt(month_data["count"])
            ax.fill_between(
                month_data["horizon"],
                month_data["mean"] - sem,
                month_data["mean"] + sem,
                color=color,
                alpha=0.15,
            )

        # Customize subplot
        ax.set_xlabel("Forecast Lead Time (months)", fontsize=11, fontweight="bold")
        ax.set_ylabel(
            metric.upper() if len(metric) <= 4 else metric.capitalize(),
            fontsize=11,
            fontweight="bold",
        )
        ax.set_title(f"{model}", fontsize=12, fontweight="bold")
        ax.set_xticks(horizons)
        ax.set_xticklabels([str(h) for h in horizons])
        ax.set_ylim(y_min, y_max)

        # Add horizontal reference line
        if metric == "r2":
            ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        elif metric in ["nrmse", "nmae", "efficiency"]:
            ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        elif metric == "accuracy":
            ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.grid(axis="both", alpha=0.3)
        ax.legend(
            loc="best",
            fontsize=8,
            ncol=3,
            framealpha=0.9,
            title=aggregate_by.replace("_", " ").title(),
        )

    # Hide unused subplots
    for idx in range(n_models, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    # Add overall title
    fig.suptitle(
        f"{metric.upper()} vs Forecast Lead Time (by {aggregate_by.replace('_', ' ').title()})",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {metric} by lead time plot to {output_path}")

    return fig


def plot_metric_by_lead_time_combined(
    metrics_df: pd.DataFrame,
    models: list[str],
    metric: str = "r2",
    y_limits: tuple[float, float] | None = None,
    output_path: str | Path | None = None,
    aggregate_by: str = "target_month",
) -> plt.Figure:
    """
    Plot metric vs forecasting lead time (horizon) for all months in a single combined plot.

    Creates a single plot showing metric values across different forecast horizons,
    with separate grouped bars/lines for each model and month combination.
    X-axis = forecast lead time (horizon), Y-axis = metric.

    Args:
        metrics_df: DataFrame with metrics containing columns:
            - code, model, issue_month, horizon, target_month, r2, rmse, nrmse, etc.
        models: List of model names to include in the plot
        metric: Metric to plot (default: "r2")
        y_limits: Optional tuple (y_min, y_max) for y-axis limits
        output_path: Optional path to save the figure
        aggregate_by: Which month to use for grouping - "target_month" or "issue_month"

    Returns:
        matplotlib Figure object
    """
    # Filter for specified models
    df = metrics_df[metrics_df["model"].isin(models)].copy()

    if df.empty:
        logger.warning(f"No data found for models {models}")
        return plt.figure()

    # Set default y-axis limits based on metric
    if y_limits is None:
        if metric == "r2":
            y_limits = (-0.5, 1.0)
        elif metric == "accuracy":
            y_limits = (0.0, 1.0)
        elif metric in ["nrmse", "nmae", "efficiency"]:
            y_limits = (0.0, 2.0)
        else:
            y_limits = (df[metric].min() * 0.9, df[metric].max() * 1.1)

    y_min, y_max = y_limits

    # Aggregate metrics by model, horizon, and month (mean across codes)
    agg_data = (
        df.groupby(["model", "horizon", aggregate_by])[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg_data.columns = ["model", "horizon", "month", "mean", "std", "count"]

    # Calculate overall mean across all months for each model and horizon
    overall_mean = (
        agg_data.groupby(["model", "horizon"])
        .agg(
            mean=("mean", "mean"),
            std=("mean", "std"),  # std of monthly means
            count=("count", "sum"),
        )
        .reset_index()
    )

    # Get unique horizons
    horizons = sorted(df["horizon"].unique())

    # Create a single plot
    fig, ax = plt.subplots(figsize=(12, 7))

    # Define color palette for models
    model_colors = sns.color_palette("husl", len(models))
    color_map = dict(zip(models, model_colors))

    # Define line styles for different visualization
    line_styles = ["-", "--", "-.", ":"]

    # Plot line for each model (showing mean across all months with individual month points)
    for model_idx, model in enumerate(models):
        model_overall = overall_mean[overall_mean["model"] == model].sort_values(
            "horizon"
        )
        model_monthly = agg_data[agg_data["model"] == model]

        if model_overall.empty:
            continue

        color = color_map[model]

        # Plot mean line across all months
        ax.plot(
            model_overall["horizon"],
            model_overall["mean"],
            marker="o",
            markersize=10,
            linewidth=3,
            color=color,
            label=model,
            alpha=0.9,
            linestyle=line_styles[model_idx % len(line_styles)],
        )

        # Add shaded error band (std across months)
        ax.fill_between(
            model_overall["horizon"],
            model_overall["mean"] - model_overall["std"],
            model_overall["mean"] + model_overall["std"],
            color=color,
            alpha=0.15,
        )

        # Plot individual month points as scatter (smaller, semi-transparent)
        for horizon in horizons:
            month_points = model_monthly[model_monthly["horizon"] == horizon]
            if not month_points.empty:
                # Jitter x positions slightly for visibility
                jitter = (model_idx - len(models) / 2) * 0.05
                ax.scatter(
                    [horizon + jitter] * len(month_points),
                    month_points["mean"],
                    color=color,
                    alpha=0.4,
                    s=30,
                    marker="o",
                )

    # Customize plot
    ax.set_xlabel("Forecast Lead Time (months)", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        metric.upper() if len(metric) <= 4 else metric.capitalize(),
        fontsize=12,
        fontweight="bold",
    )
    ax.set_title(
        f"{metric.upper()} vs Forecast Lead Time\n(Lines = mean across all {aggregate_by.replace('_', ' ')}s, points = individual months)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(horizons)
    ax.set_xticklabels([str(h) for h in horizons])
    ax.set_ylim(y_min, y_max)

    # Add horizontal reference line
    if metric == "r2":
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    elif metric in ["nrmse", "nmae", "efficiency"]:
        ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    elif metric == "accuracy":
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.grid(axis="both", alpha=0.3)
    ax.legend(loc="best", fontsize=11, framealpha=0.9, title="Model")

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {metric} by lead time combined plot to {output_path}")

    return fig


def main():
    region = "Kyrgyzstan"  # Options: "Kyrgyzstan" or "Tajikistan"
    save_dir = Path(output_dir) / f"{region.lower()}"
    # Configuration for plotting
    metric_to_plot = (
        "r2"  # Options: "r2", "rmse", "nrmse", "mae", "nmae", "accuracy", "efficiency"
    )

    # Load predictions
    if region == "Tajikistan":
        pred_config = taj_path_config
    else:
        pred_config = kgz_path_config

    logger.info(f"Loading predictions for {region}...")
    predictions_df = load_predictions(
        base_path=pred_config["pred_dir"],
        horizons=horizons,
        issue_months=issue_months_to_evaluate,
    )

    if predictions_df.empty:
        logger.error("No predictions loaded. Exiting.")
        return

    logger.info(f"Loaded {len(predictions_df)} prediction records")
    logger.info(f"Available models: {predictions_df['model'].unique().tolist()}")
    logger.info(f"Available horizons: {predictions_df['horizon'].unique().tolist()}")

    # Create ensemble
    logger.info("Creating ensemble...")
    predictions_df = create_ensemble(predictions_df)

    # Handle observations based on use_Q_obs flag
    if use_Q_obs:
        logger.info("Using Q_obs from prediction files...")
        # Check if Q_obs is available
        if predictions_df["Q_obs"].isna().all():
            logger.error(
                "Q_obs not available in prediction files. Set use_Q_obs=False to load from observations file."
            )
            return

        # Add target_month, issue_month based on region logic
        if region == "Kyrgyzstan":
            predictions_df["target_date"] = predictions_df.apply(
                lambda row: row["issue_date"]
                + pd.DateOffset(months=int(row["horizon"])),
                axis=1,
            )
        elif region == "Tajikistan":
            predictions_df["target_date"] = predictions_df.apply(
                lambda row: row["issue_date"]
                + pd.DateOffset(months=int(row["horizon"]) - 1),
                axis=1,
            )

        predictions_df["target_month"] = predictions_df["target_date"].dt.month
        predictions_df["issue_month"] = predictions_df["issue_date"].dt.month
        aggregated_df = predictions_df

    else:
        logger.info(f"Loading observations from {pred_config['obs_file']}...")
        obs_df = load_observations(pred_config["obs_file"])

        # Calculate monthly targets
        logger.info("Calculating monthly observation targets...")
        monthly_obs = calculate_target(obs_df)

        # Apply climatology-based ratio correction if enabled
        if apply_climatology_correction:
            logger.info("Applying climatology-based ratio correction...")

            # Calculate leave-one-out monthly means (fast vectorized version)
            monthly_obs_with_ltm = calculate_leave_one_out_monthly_mean_fast(
                monthly_obs
            )

            # Apply ratio correction to predictions (fast vectorized version)
            predictions_df = apply_climatology_ratio_correction_fast(
                predictions_df=predictions_df,
                daily_obs=obs_df,
                monthly_obs_with_ltm=monthly_obs_with_ltm,
                region=region,
            )

            # Use corrected predictions as Q_pred for evaluation
            # Keep original Q_pred as Q_pred_original
            predictions_df["Q_pred_original"] = predictions_df["Q_pred"]
            predictions_df["Q_pred"] = predictions_df["Q_pred_corrected"]

            logger.info(
                "Ratio correction applied. Q_pred now contains corrected values."
            )

        # Aggregate predictions with observations
        logger.info("Aggregating predictions with observations...")
        aggregated_df = aggregate(predictions_df, monthly_obs, region)

    # Compute metrics dataframe
    logger.info("Computing metrics...")
    metrics_df = compute_metrics_dataframe(aggregated_df)

    logger.info(f"Metrics computed for {len(metrics_df)} combinations")
    logger.info(f"\nMetrics DataFrame preview:")
    print(metrics_df.head(20))

    # Diagnostic: Print coverage per model and horizon
    print("\n" + "=" * 80)
    print(
        "DIAGNOSTIC: Data coverage per model and horizon (number of target months with data)"
    )
    print("=" * 80)
    for model in models_plot:
        print(f"\n{model}:")
        model_data = metrics_df[metrics_df["model"] == model]
        if model_data.empty:
            print("  NO DATA FOUND!")
            # Check if it's in the predictions_df
            pred_model_data = aggregated_df[aggregated_df["model"] == model]
            if not pred_model_data.empty:
                print(f"  But found {len(pred_model_data)} rows in aggregated_df")
                print(
                    f"  Horizons in aggregated_df: {sorted(pred_model_data['horizon'].unique())}"
                )
                print(
                    f"  Target months in aggregated_df: {sorted(pred_model_data['target_month'].unique())}"
                )
        else:
            for horizon in sorted(model_data["horizon"].unique()):
                horizon_data = model_data[model_data["horizon"] == horizon]
                target_months = sorted(horizon_data["target_month"].unique())
                n_codes = horizon_data["code"].nunique()
                print(
                    f"  Horizon {horizon}: {len(target_months)} target months {target_months}, {n_codes} codes, {len(horizon_data)} records"
                )

    # Also print raw predictions coverage for comparison
    print("\n" + "=" * 80)
    print("DIAGNOSTIC: Raw predictions coverage (before metrics calculation)")
    print("=" * 80)
    for model in models_plot:
        print(f"\n{model}:")
        model_data = aggregated_df[aggregated_df["model"] == model]
        if model_data.empty:
            print("  NO DATA FOUND in aggregated_df!")
        else:
            for horizon in sorted(model_data["horizon"].unique()):
                horizon_data = model_data[model_data["horizon"] == horizon]
                target_months = sorted(horizon_data["target_month"].unique())
                n_records = len(horizon_data)
                n_with_obs = horizon_data["Q_obs"].notna().sum()
                n_with_pred = horizon_data["Q_pred"].notna().sum()
                print(
                    f"  Horizon {horizon}: {n_records} records, {n_with_obs} with Q_obs, {n_with_pred} with Q_pred, target_months: {target_months}"
                )
    print("=" * 80 + "\n")

    # Add Ensemble to models_plot if not already there
    models_to_plot = (
        models_plot + ["Ensemble"] if "Ensemble" not in models_plot else models_plot
    )

    # Plot metric by target month for each forecast horizon
    available_horizons = sorted(metrics_df["horizon"].unique())
    logger.info(f"\nPlotting {metric_to_plot} by target month for each horizon...")

    for horizon in available_horizons:
        logger.info(f"  Plotting horizon {horizon}...")
        fig = plot_metric_by_horizon(
            metrics_df=metrics_df,
            horizon=horizon,
            models=models_to_plot,
            metric=metric_to_plot,
            output_path=Path(save_dir)
            / f"{metric_to_plot}_horizon_{horizon}_by_target_month.png"
            if save_dir
            else None,
        )
        # plt.show()
        plt.close(fig)  # Close to avoid memory issues

    # Plot metric by lead time (horizon) for all months
    logger.info(f"\nPlotting {metric_to_plot} by forecast lead time...")

    # Plot with subplots per model (showing each month as a separate line)
    fig = plot_metric_by_lead_time(
        metrics_df=metrics_df,
        models=models_to_plot,
        metric=metric_to_plot,
        output_path=Path(save_dir) / f"{metric_to_plot}_by_lead_time_per_model.png"
        if save_dir
        else None,
        aggregate_by="target_month",
    )
    plt.close(fig)

    # Plot combined view (all models on one plot, mean across months)
    fig = plot_metric_by_lead_time_combined(
        metrics_df=metrics_df,
        models=models_to_plot,
        metric=metric_to_plot,
        output_path=Path(save_dir) / f"{metric_to_plot}_by_lead_time_combined.png"
        if save_dir
        else None,
        aggregate_by="target_month",
    )
    plt.close(fig)

    logger.info("Processing complete!")


if __name__ == "__main__":
    main()
