import os
import re
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
from sklearn.metrics import r2_score, mean_squared_error

# load the .env file
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


day_of_forecast = {
    # "month_0": 10,
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

# Models to exclude from ensemble (includes Q_obs variants as safety measure)
models_not_to_ensemble = [
    "MC_ALD",
    "SM_GBT",
    "SM_GBT_Norm",
    "SM_GBT_LR",
    "MC_ALD_loc",
    "obs",
    "Obs",
    "OBS",  # Exclude any observation-based "models"
]

models_plot = ["LR_Base", "LR_SM", "Ensemble", "MC_ALD"]

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


def metric_pipeline(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantiles_pred: dict[str, np.ndarray] | None = None,
) -> pd.DataFrame:
    """
    Pipeline to evaluate operational metrics for forecasting models.

    Metrics computed:
    R2
    RMSE
    MSE
    MAE
    Accuracy
    Efficiency
    Coverage (if quantiles provided)
    """

    nan_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true_clean = y_true[nan_mask]
    y_pred_clean = y_pred[nan_mask]

    if len(y_true_clean) == 0:
        logger.warning(
            "No valid data points after removing NaNs for metric computation."
        )
        return {}

    # Need at least 2 samples for meaningful metrics
    if len(y_true_clean) < 2:
        logger.warning(
            f"Only {len(y_true_clean)} data point(s) - insufficient for metric computation."
        )
        return {}

    for quantile in quantiles_pred.keys() if quantiles_pred is not None else []:
        quantiles_pred[quantile] = quantiles_pred[quantile][nan_mask]

    metrics = {}

    # R2
    metrics["R2"] = r2_score(y_true_clean, y_pred_clean)
    # RMSE
    metrics["RMSE"] = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    # MSE
    metrics["MSE"] = mean_squared_error(y_true_clean, y_pred_clean)
    # MAE
    metrics["MAE"] = np.mean(np.abs(y_true_clean - y_pred_clean))

    obs_mean = np.mean(y_true_clean)
    metrics["nRMSE"] = metrics["RMSE"] / obs_mean if obs_mean != 0 else np.nan
    metrics["nMAE"] = metrics["MAE"] / obs_mean if obs_mean != 0 else np.nan

    # Accuracy
    sigma_obs = np.std(y_true_clean)
    abs_errors = np.abs(y_true_clean - y_pred_clean)
    # if abs error smaller than 0.674 * sigma_obs, count as accurate -> 1 else 0
    if sigma_obs > 0:
        accurate_preds = abs_errors < (0.674 * sigma_obs)
        metrics["Accuracy"] = np.mean(accurate_preds)
    else:
        metrics["Accuracy"] = np.nan

    std_abs_errors = np.std(abs_errors)
    metrics["Efficiency"] = std_abs_errors / sigma_obs if sigma_obs > 0 else np.nan

    # Coverage for quantiles, 90% interval and 50% interval
    if quantiles_pred is not None:

        def coverage(y_true, lower_bound, upper_bound):
            return np.mean((y_true >= lower_bound) & (y_true <= upper_bound))

        if "Q5" in quantiles_pred and "Q95" in quantiles_pred:
            metrics["Coverage_90"] = coverage(
                y_true_clean, quantiles_pred["Q5"], quantiles_pred["Q95"]
            )
        if "Q25" in quantiles_pred and "Q75" in quantiles_pred:
            metrics["Coverage_50"] = coverage(
                y_true_clean, quantiles_pred["Q25"], quantiles_pred["Q75"]
            )

    return pd.DataFrame([metrics])


def compute_long_term_means(
    observations_df: pd.DataFrame,
    daily_obs_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute long-term means for:
    1. Each calendar month (for target month normalization)
    2. Daily observations (for period-based normalization)

    Args:
        observations_df: Monthly aggregated observations with columns ["code", "date", "discharge"]
        daily_obs_path: Path to daily observation file

    Returns:
        Tuple of:
        - monthly_means: DataFrame with ["code", "month", "discharge_ltm"]
        - daily_obs: DataFrame with daily observations ["code", "date", "discharge", "day_of_year"]
    """
    # Compute monthly long-term means from aggregated monthly data
    obs_copy = observations_df.copy()
    obs_copy["month"] = obs_copy["date"].dt.month

    monthly_means = (
        obs_copy.groupby(["code", "month"])
        .agg(discharge_ltm=("discharge", "mean"))
        .reset_index()
    )

    logger.info(
        f"Computed monthly long-term means for {monthly_means['code'].nunique()} stations"
    )

    # Load daily observations for period-based calculations
    daily_obs_path = Path(daily_obs_path)
    daily_df = pd.read_csv(daily_obs_path)

    # Find date column
    date_col = None
    for candidate in ["date", "Date", "DATE", "time", "Time"]:
        if candidate in daily_df.columns:
            date_col = candidate
            break
    if date_col is None:
        date_col = daily_df.columns[0]

    daily_df["date"] = pd.to_datetime(daily_df[date_col])

    # Find discharge column
    discharge_col = None
    for candidate in ["discharge", "Discharge", "Q", "q", "runoff", "Runoff"]:
        if candidate in daily_df.columns:
            discharge_col = candidate
            break
    if discharge_col is None:
        non_date_cols = [
            c for c in daily_df.columns if c not in [date_col, "code", "Code"]
        ]
        if non_date_cols:
            discharge_col = non_date_cols[0]

    # Find code column
    code_col = None
    for candidate in ["code", "Code", "CODE", "station_id", "basin_id"]:
        if candidate in daily_df.columns:
            code_col = candidate
            break

    daily_df["code"] = daily_df[code_col].astype(int)
    daily_df["discharge"] = pd.to_numeric(daily_df[discharge_col], errors="coerce")
    daily_df["day_of_year"] = daily_df["date"].dt.dayofyear
    daily_df["month"] = daily_df["date"].dt.month
    daily_df["day"] = daily_df["date"].dt.day

    logger.info(
        f"Loaded {len(daily_df)} daily observations for long-term mean calculation"
    )

    return monthly_means, daily_df


def compute_period_long_term_mean(
    daily_obs: pd.DataFrame,
    code: int,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> float:
    """
    Compute the long-term mean for a specific period pattern (e.g., Jul 25 - Aug 25) across all years.

    Args:
        daily_obs: DataFrame with daily observations
        code: Station code
        start_month: Start month of the period (1-12)
        start_day: Start day of the period
        end_month: End month of the period (1-12)
        end_day: End day of the period

    Returns:
        Long-term mean discharge for the specified period pattern
    """
    station_data = daily_obs[daily_obs["code"] == code]

    if station_data.empty:
        return np.nan

    # Handle period within same month or crossing month boundary
    if start_month == end_month:
        # Same month: filter by month and day range
        mask = (
            (station_data["month"] == start_month)
            & (station_data["day"] >= start_day)
            & (station_data["day"] <= end_day)
        )
    elif end_month == start_month + 1 or (start_month == 12 and end_month == 1):
        # Crosses one month boundary
        mask = (
            (station_data["month"] == start_month) & (station_data["day"] >= start_day)
        ) | ((station_data["month"] == end_month) & (station_data["day"] <= end_day))
    else:
        # Spans multiple months
        if start_month < end_month:
            mask = (
                (
                    (station_data["month"] == start_month)
                    & (station_data["day"] >= start_day)
                )
                | (
                    (station_data["month"] > start_month)
                    & (station_data["month"] < end_month)
                )
                | (
                    (station_data["month"] == end_month)
                    & (station_data["day"] <= end_day)
                )
            )
        else:
            # Crosses year boundary
            mask = (
                (
                    (station_data["month"] == start_month)
                    & (station_data["day"] >= start_day)
                )
                | (station_data["month"] > start_month)
                | (station_data["month"] < end_month)
                | (
                    (station_data["month"] == end_month)
                    & (station_data["day"] <= end_day)
                )
            )

    period_data = station_data[mask]

    if period_data.empty:
        return np.nan

    return period_data["discharge"].mean()


def _get_period_mask(
    station_data: pd.DataFrame,
    start_month: int,
    start_day: int,
    end_month: int,
    end_day: int,
) -> pd.Series:
    """
    Create a boolean mask for observations falling within a specific period pattern.

    Handles periods within the same month, crossing month boundaries, and crossing year boundaries.
    """
    if start_month == end_month:
        # Same month: filter by month and day range
        mask = (
            (station_data["month"] == start_month)
            & (station_data["day"] >= start_day)
            & (station_data["day"] <= end_day)
        )
    elif end_month == start_month + 1 or (start_month == 12 and end_month == 1):
        # Crosses one month boundary
        mask = (
            (station_data["month"] == start_month) & (station_data["day"] >= start_day)
        ) | ((station_data["month"] == end_month) & (station_data["day"] <= end_day))
    else:
        # Spans multiple months
        if start_month < end_month:
            mask = (
                (
                    (station_data["month"] == start_month)
                    & (station_data["day"] >= start_day)
                )
                | (
                    (station_data["month"] > start_month)
                    & (station_data["month"] < end_month)
                )
                | (
                    (station_data["month"] == end_month)
                    & (station_data["day"] <= end_day)
                )
            )
        else:
            # Crosses year boundary
            mask = (
                (
                    (station_data["month"] == start_month)
                    & (station_data["day"] >= start_day)
                )
                | (station_data["month"] > start_month)
                | (station_data["month"] < end_month)
                | (
                    (station_data["month"] == end_month)
                    & (station_data["day"] <= end_day)
                )
            )
    return mask


def precompute_period_ltms_loo(
    predictions_raw: pd.DataFrame,
    daily_obs: pd.DataFrame,
) -> dict[tuple[int, int, int, int, int, int], float]:
    """
    Precompute leave-one-out (LOO) long-term means for period patterns.

    For each unique (code, start_month, start_day, end_month, end_day, exclude_year) combination,
    computes the mean discharge excluding the specified year. This prevents data leakage
    when evaluating hindcasts.

    Efficiency: Uses vectorized per-year sum/count aggregation, then computes LOO means
    in O(1) per lookup by subtracting the excluded year's contribution.

    Args:
        predictions_raw: DataFrame with raw predictions including valid_from, valid_to
        daily_obs: DataFrame with daily observations

    Returns:
        Dictionary mapping (code, start_month, start_day, end_month, end_day, exclude_year) to LOO period LTM
    """
    predictions_raw = predictions_raw.copy()
    predictions_raw["start_month"] = predictions_raw["valid_from"].dt.month
    predictions_raw["start_day"] = predictions_raw["valid_from"].dt.day
    predictions_raw["end_month"] = predictions_raw["valid_to"].dt.month
    predictions_raw["end_day"] = predictions_raw["valid_to"].dt.day
    predictions_raw["target_year"] = predictions_raw["valid_to"].dt.year

    # Get unique period patterns (without year)
    unique_periods = predictions_raw[
        ["code", "start_month", "start_day", "end_month", "end_day"]
    ].drop_duplicates()

    # Get all years we need to exclude
    all_years = predictions_raw["target_year"].unique()

    logger.info(
        f"Precomputing LOO period LTMs for {len(unique_periods)} patterns × {len(all_years)} years"
    )

    # Add year column to daily_obs if not present
    if "year" not in daily_obs.columns:
        daily_obs = daily_obs.copy()
        daily_obs["year"] = daily_obs["date"].dt.year

    period_ltm_loo_cache: dict[tuple[int, int, int, int, int, int], float] = {}

    for _, period_row in unique_periods.iterrows():
        code = period_row["code"]
        start_month = period_row["start_month"]
        start_day = period_row["start_day"]
        end_month = period_row["end_month"]
        end_day = period_row["end_day"]

        # Filter station data once
        station_data = daily_obs[daily_obs["code"] == code]
        if station_data.empty:
            continue

        # Apply period mask
        mask = _get_period_mask(
            station_data, start_month, start_day, end_month, end_day
        )
        period_data = station_data[mask]

        if period_data.empty:
            continue

        # Compute per-year sum and count (vectorized)
        yearly_stats = period_data.groupby("year")["discharge"].agg(["sum", "count"])
        total_sum = yearly_stats["sum"].sum()
        total_count = yearly_stats["count"].sum()

        # For each year that needs to be excluded, compute LOO mean
        for exclude_year in all_years:
            cache_key = (
                code,
                start_month,
                start_day,
                end_month,
                end_day,
                int(exclude_year),
            )

            if exclude_year in yearly_stats.index:
                year_sum = yearly_stats.loc[exclude_year, "sum"]
                year_count = yearly_stats.loc[exclude_year, "count"]
                loo_sum = total_sum - year_sum
                loo_count = total_count - year_count
            else:
                # Year not in data, use full stats
                loo_sum = total_sum
                loo_count = total_count

            if loo_count > 0:
                period_ltm_loo_cache[cache_key] = loo_sum / loo_count
            else:
                period_ltm_loo_cache[cache_key] = np.nan

    logger.info(f"Precomputed {len(period_ltm_loo_cache)} LOO period LTMs")

    return period_ltm_loo_cache


def precompute_monthly_ltms_loo(
    observations_df: pd.DataFrame,
) -> dict[tuple[int, int, int], float]:
    """
    Precompute leave-one-out monthly long-term means.

    For each (code, month, exclude_year) combination, computes the mean monthly discharge
    excluding the specified year.

    Args:
        observations_df: Monthly aggregated observations with columns ["code", "date", "discharge"]

    Returns:
        Dictionary mapping (code, month, exclude_year) to LOO monthly LTM
    """
    obs_copy = observations_df.copy()
    obs_copy["month"] = obs_copy["date"].dt.month
    obs_copy["year"] = obs_copy["date"].dt.year

    all_years = obs_copy["year"].unique()

    # Compute per-year, per-month sum and count (vectorized)
    yearly_monthly_stats = (
        obs_copy.groupby(["code", "month", "year"])["discharge"]
        .agg(["sum", "count"])
        .reset_index()
    )

    # Compute total sum and count per (code, month)
    totals = (
        yearly_monthly_stats.groupby(["code", "month"])
        .agg(total_sum=("sum", "sum"), total_count=("count", "sum"))
        .reset_index()
    )

    # Create lookup for yearly stats
    yearly_lookup: dict[tuple[int, int, int], tuple[float, int]] = {}
    for _, row in yearly_monthly_stats.iterrows():
        key = (int(row["code"]), int(row["month"]), int(row["year"]))
        yearly_lookup[key] = (row["sum"], row["count"])

    # Create lookup for totals
    totals_lookup: dict[tuple[int, int], tuple[float, int]] = {}
    for _, row in totals.iterrows():
        key = (int(row["code"]), int(row["month"]))
        totals_lookup[key] = (row["total_sum"], row["total_count"])

    monthly_ltm_loo_cache: dict[tuple[int, int, int], float] = {}

    unique_code_months = totals[["code", "month"]].values

    for code, month in unique_code_months:
        code = int(code)
        month = int(month)
        total_sum, total_count = totals_lookup[(code, month)]

        for exclude_year in all_years:
            exclude_year = int(exclude_year)
            cache_key = (code, month, exclude_year)

            yearly_key = (code, month, exclude_year)
            if yearly_key in yearly_lookup:
                year_sum, year_count = yearly_lookup[yearly_key]
                loo_sum = total_sum - year_sum
                loo_count = total_count - year_count
            else:
                loo_sum = total_sum
                loo_count = total_count

            if loo_count > 0:
                monthly_ltm_loo_cache[cache_key] = loo_sum / loo_count
            else:
                monthly_ltm_loo_cache[cache_key] = np.nan

    logger.info(
        f"Precomputed {len(monthly_ltm_loo_cache)} LOO monthly LTMs for "
        f"{len(unique_code_months)} (code, month) combinations"
    )

    return monthly_ltm_loo_cache


def transform_predictions_with_ratio(
    predictions_raw: pd.DataFrame,
    observations_df: pd.DataFrame,
    daily_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Transform predictions using the ratio approach with leave-one-out (LOO) normalization.

    Uses LOO long-term means to prevent data leakage during hindcast evaluation:
    1. Calculate ratio = raw_prediction / LOO_long_term_mean(valid_period, exclude_target_year)
    2. Determine target month from issue_date + horizon
    3. Final prediction = LOO_target_month_long_term_mean(exclude_target_year) * ratio

    Args:
        predictions_raw: DataFrame with raw predictions including valid_from, valid_to
        observations_df: Monthly aggregated observations for computing LOO monthly LTMs
        daily_obs: DataFrame with daily observations for period-based LOO LTM calculation

    Returns:
        DataFrame with transformed predictions using unbiased LOO normalization
    """
    # Precompute all LOO period LTMs (efficient: O(patterns × years) precomputation)
    period_ltm_loo_cache = precompute_period_ltms_loo(predictions_raw, daily_obs)

    # Precompute all LOO monthly LTMs
    monthly_ltm_loo_cache = precompute_monthly_ltms_loo(observations_df)

    # Vectorized extraction of date components for fast lookup
    predictions_raw = predictions_raw.copy()
    predictions_raw["start_month"] = predictions_raw["valid_from"].dt.month
    predictions_raw["start_day"] = predictions_raw["valid_from"].dt.day
    predictions_raw["end_month"] = predictions_raw["valid_to"].dt.month
    predictions_raw["end_day"] = predictions_raw["valid_to"].dt.day
    predictions_raw["target_year"] = predictions_raw["valid_to"].dt.year

    # Pre-extract arrays for faster iteration
    codes = predictions_raw["code"].values
    issue_dates = predictions_raw["issue_date"].values
    horizons = predictions_raw["horizon"].values
    q_preds_raw = predictions_raw["Q_pred"].values
    models = predictions_raw["model"].values
    start_months = predictions_raw["start_month"].values
    start_days = predictions_raw["start_day"].values
    end_months = predictions_raw["end_month"].values
    end_days = predictions_raw["end_day"].values
    target_years = predictions_raw["target_year"].values

    # Pre-extract quantile arrays if they exist
    quantile_arrays = {}
    for q_col in ["Q5", "Q25", "Q75", "Q95"]:
        if q_col in predictions_raw.columns:
            quantile_arrays[q_col] = predictions_raw[q_col].values

    result_rows = []
    skipped_period = 0
    skipped_monthly = 0

    for i in range(len(predictions_raw)):
        code = int(codes[i])
        issue_date = pd.Timestamp(issue_dates[i])
        horizon = int(horizons[i])
        q_pred_raw = q_preds_raw[i]
        model = models[i]
        target_year = int(target_years[i])

        # Build cache key for LOO period LTM lookup
        period_cache_key = (
            code,
            int(start_months[i]),
            int(start_days[i]),
            int(end_months[i]),
            int(end_days[i]),
            target_year,
        )
        period_ltm = period_ltm_loo_cache.get(period_cache_key, np.nan)

        if pd.isna(period_ltm) or period_ltm == 0:
            skipped_period += 1
            continue

        # Calculate ratio using LOO period LTM
        ratio = q_pred_raw / period_ltm

        # Determine target month from issue_date + horizon
        issue_period = issue_date.to_period("M")
        target_period = issue_period + horizon
        target_month = target_period.month
        target_date = target_period.to_timestamp()

        # Get LOO target month long-term mean (exclude target year)
        monthly_cache_key = (code, target_month, target_year)
        target_ltm = monthly_ltm_loo_cache.get(monthly_cache_key, np.nan)

        if pd.isna(target_ltm) or target_ltm == 0:
            skipped_monthly += 1
            continue

        # Calculate final prediction using LOO LTMs
        q_pred_transformed = target_ltm * ratio

        result_row = {
            "code": code,
            "issue_date": issue_date,
            "target_date": target_date,
            "Q_pred": q_pred_transformed,
            "Q_pred_raw": q_pred_raw,
            "ratio": ratio,
            "period_ltm": period_ltm,
            "target_ltm": target_ltm,
            "horizon": horizon,
            "model": model,
        }

        # Transform quantile columns using the same LOO LTMs
        for q_col in ["Q5", "Q25", "Q75", "Q95"]:
            if q_col in quantile_arrays and pd.notna(quantile_arrays[q_col][i]):
                q_ratio = quantile_arrays[q_col][i] / period_ltm
                result_row[q_col] = target_ltm * q_ratio
            else:
                result_row[q_col] = np.nan

        result_rows.append(result_row)

    if skipped_period > 0 or skipped_monthly > 0:
        logger.debug(
            f"Skipped {skipped_period} rows due to missing period LTM, "
            f"{skipped_monthly} rows due to missing monthly LTM"
        )

    if not result_rows:
        logger.error("No predictions transformed successfully")
        return pd.DataFrame()

    result_df = pd.DataFrame(result_rows)
    logger.info(
        f"Transformed {len(result_df)} predictions using LOO ratio approach "
        f"(unbiased evaluation)"
    )

    return result_df


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

            # Convert dates to datetime
            df["date"] = pd.to_datetime(df["date"])
            df["valid_from"] = pd.to_datetime(df["valid_from"])
            df["valid_to"] = pd.to_datetime(df["valid_to"])

            # Convert code to int
            df["code"] = df["code"].astype(int)

            # update the set code to only have codes which are present in all models
            set_codes = (
                set_codes.intersection(set(df["code"].unique().tolist()))
                if set_codes
                else set(df["code"].unique().tolist())
            )

            # Keep track of unique codes
            set_codes.update(df["code"].unique().tolist())

            # Sort by code and date
            df = df.sort_values(["code", "date"]).reset_index(drop=True)

            # Filter to keep only the day of the month matching forecast day
            df = df[df["date"].dt.day == forecast_day].copy()

            # Filter by issue months early if specified
            if issue_months is not None:
                df = df[df["date"].dt.month.isin(issue_months)].copy()
                if df.empty:
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

    # Filter combined_df to only include codes present in all models
    combined_df = combined_df[combined_df["code"].isin(set_codes)].copy()

    return combined_df


def load_ground_truth(path_obs: str) -> pd.DataFrame:
    """
    This functions loads the daily observed discharge data from the specified path.
    1. convert dates to datetime
    2. convert "code" to int
    3. sort by code and date
    4. aggregate to monthly data (mean, min_obs = 20 days per month)
    Returns a pandas DataFrame with columns: ["code", "date", "discharge"]
    """
    path_obs = Path(path_obs)

    if not path_obs.exists():
        raise FileNotFoundError(f"Observation file not found: {path_obs}")

    try:
        df = pd.read_csv(path_obs)
    except Exception as e:
        logger.error(f"Failed to read observation file {path_obs}: {e}")
        raise

    # Convert date column to datetime
    # Try common date column names
    date_col = None
    for candidate in ["date", "Date", "DATE", "time", "Time"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        # Assume first column is date
        date_col = df.columns[0]

    df["date"] = pd.to_datetime(df[date_col])

    # filter to only include dates from 2000 onwards
    df = df[df["date"].dt.year >= 2000].copy()

    # Find discharge column
    discharge_col = None
    for candidate in ["discharge", "Discharge", "Q", "q", "runoff", "Runoff"]:
        if candidate in df.columns:
            discharge_col = candidate
            break

    if discharge_col is None:
        # Assume second column after date and code is discharge
        non_date_cols = [c for c in df.columns if c not in [date_col, "code", "Code"]]
        if non_date_cols:
            discharge_col = non_date_cols[0]
        else:
            raise ValueError("Could not identify discharge column in observation file.")

    # Find code column
    code_col = None
    for candidate in ["code", "Code", "CODE", "station_id", "basin_id"]:
        if candidate in df.columns:
            code_col = candidate
            break

    if code_col is None:
        raise ValueError("Could not identify code/station column in observation file.")

    # Convert code to int
    df["code"] = df[code_col].astype(int)
    df["discharge"] = pd.to_numeric(df[discharge_col], errors="coerce")

    # Sort by code and date
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    # Create year-month column for aggregation
    df["year_month"] = df["date"].dt.to_period("M")

    # Aggregate to monthly data with minimum 20 days requirement
    monthly_agg = (
        df.groupby(["code", "year_month"])
        .agg(discharge_mean=("discharge", "mean"), day_count=("discharge", "count"))
        .reset_index()
    )

    # Filter for months with at least 20 valid days
    monthly_agg = monthly_agg[monthly_agg["day_count"] >= 20].copy()

    # Convert period back to timestamp (first day of month)
    monthly_agg["date"] = monthly_agg["year_month"].dt.to_timestamp()

    # Select and rename columns
    result_df = monthly_agg[["code", "date", "discharge_mean"]].copy()
    result_df = result_df.rename(columns={"discharge_mean": "discharge"})

    logger.info(f"Loaded {len(result_df)} monthly observation records.")

    return result_df


def create_ensemble(
    predictions_df: pd.DataFrame,
    models_to_exclude: list[str] | None = None,
    ensemble_name: str = "Ensemble",
) -> pd.DataFrame:
    """
    Create ensemble predictions by averaging all models for each code, horizon, and issue_date.

    For each unique combination of (code, horizon, issue_date, target_date), this function
    computes the mean of Q_pred and quantile columns across all models not in the exclusion list,
    then adds these ensemble predictions as new rows with model="Ensemble".

    Args:
        predictions_df: DataFrame with predictions from multiple models.
            Expected columns: ["code", "issue_date", "target_date", "Q_pred",
                             "Q5", "Q25", "Q75", "Q95", "horizon", "model"]
        models_to_exclude: List of model names to exclude from ensemble averaging.
            If None, all models are included.
        ensemble_name: Name to assign to the ensemble model. Defaults to "Ensemble".

    Returns:
        DataFrame with original predictions plus ensemble predictions appended.

    Raises:
        ValueError: If predictions_df is empty or missing required columns.
    """
    if predictions_df.empty:
        logger.warning("Empty predictions DataFrame provided for ensemble creation.")
        return predictions_df

    required_cols = ["code", "issue_date", "target_date", "Q_pred", "horizon", "model"]
    missing_cols = [col for col in required_cols if col not in predictions_df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns for ensemble creation: {missing_cols}"
        )

    if models_to_exclude is None:
        models_to_exclude = []

    # Filter out excluded models for ensemble calculation
    ensemble_eligible_df = predictions_df[
        ~predictions_df["model"].isin(models_to_exclude)
    ].copy()

    if ensemble_eligible_df.empty:
        logger.warning("No models remaining after exclusion for ensemble creation.")
        return predictions_df

    # Define columns to average
    numeric_cols = ["Q_pred"]
    quantile_cols = ["Q5", "Q25", "Q75", "Q95"]
    for q_col in quantile_cols:
        if q_col in ensemble_eligible_df.columns:
            numeric_cols.append(q_col)

    # Group by code, horizon, issue_date, and target_date
    grouping_cols = ["code", "horizon", "issue_date", "target_date"]

    # Compute ensemble averages
    ensemble_df = ensemble_eligible_df.groupby(grouping_cols, as_index=False).agg(
        {col: "mean" for col in numeric_cols}
    )

    # Add model name
    ensemble_df["model"] = ensemble_name

    # Ensure all quantile columns exist (fill with NaN if not present in original)
    for q_col in quantile_cols:
        if q_col not in ensemble_df.columns:
            ensemble_df[q_col] = np.nan

    # Reorder columns to match original DataFrame
    column_order = [
        "code",
        "issue_date",
        "target_date",
        "Q_pred",
        "Q5",
        "Q25",
        "Q75",
        "Q95",
        "horizon",
        "model",
    ]
    existing_cols = [c for c in column_order if c in ensemble_df.columns]
    ensemble_df = ensemble_df[existing_cols]

    # Concatenate with original predictions
    combined_df = pd.concat([predictions_df, ensemble_df], ignore_index=True)

    logger.info(
        f"Created {len(ensemble_df)} ensemble predictions from "
        f"{ensemble_eligible_df['model'].nunique()} models."
    )

    return combined_df


def evaluate(
    predictions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    1. merges predictions with observations based on the code and the target_date
    2. computes the metrics for each model and each basin and each forecast horizon
    3. returns a pandas DataFrame with the metrics with the format
        ["code", "horizon", "issue_month", "model", "R2", "RMSE", "MSE", "MAE", "nRMSE", "nMAE", "Accuracy", "Efficiency", "Coverage_90", "Coverage_50"]
    """
    if predictions_df.empty or observations_df.empty:
        logger.warning("Empty predictions or observations DataFrame provided.")
        return pd.DataFrame()

    # Merge predictions with observations
    # Match on code and target_date (predictions) with date (observations)
    merged_df = predictions_df.merge(
        observations_df,
        left_on=["code", "target_date"],
        right_on=["code", "date"],
        how="inner",
        suffixes=("_pred", "_obs"),
    )

    if merged_df.empty:
        logger.warning(
            "No matching records found between predictions and observations."
        )
        return pd.DataFrame()

    logger.info(f"Merged {len(merged_df)} prediction-observation pairs.")

    # Extract issue_month from issue_date
    merged_df["issue_month"] = merged_df["issue_date"].dt.month

    # Group by code, horizon, issue_month, and model
    grouping_cols = ["code", "horizon", "issue_month", "model"]
    all_metrics = []

    for group_keys, group_df in merged_df.groupby(grouping_cols):
        code, horizon, issue_month, model = group_keys

        y_true = group_df["discharge"].values
        y_pred = group_df["Q_pred"].values

        # Prepare quantiles if available
        quantiles_pred = {}
        for q_col in ["Q5", "Q25", "Q75", "Q95"]:
            if q_col in group_df.columns and group_df[q_col].notna().any():
                quantiles_pred[q_col] = group_df[q_col].values

        if not quantiles_pred:
            quantiles_pred = None

        # Compute metrics
        metrics_df = metric_pipeline(y_true, y_pred, quantiles_pred)

        if isinstance(metrics_df, pd.DataFrame) and not metrics_df.empty:
            metrics_dict = metrics_df.iloc[0].to_dict()
        elif isinstance(metrics_df, dict):
            metrics_dict = metrics_df
        else:
            continue

        # Add grouping information
        metrics_dict["code"] = code
        metrics_dict["horizon"] = horizon
        metrics_dict["issue_month"] = issue_month
        metrics_dict["model"] = model

        all_metrics.append(metrics_dict)

    if not all_metrics:
        logger.warning("No metrics computed for any group.")
        return pd.DataFrame()

    result_df = pd.DataFrame(all_metrics)

    # Reorder columns
    column_order = [
        "code",
        "horizon",
        "issue_month",
        "model",
        "R2",
        "RMSE",
        "MSE",
        "MAE",
        "nRMSE",
        "nMAE",
        "Accuracy",
        "Efficiency",
        "Coverage_90",
        "Coverage_50",
    ]

    # Only include columns that exist
    existing_cols = [c for c in column_order if c in result_df.columns]
    result_df = result_df[existing_cols]

    # rename the issue_month to month name
    result_df["issue_month"] = result_df["issue_month"].map(month_renaming)

    logger.info(f"Computed metrics for {len(result_df)} groups.")

    return result_df


def draw_overall_plot(
    metrics_df: pd.DataFrame,
    models: list[str],
    metric_name: str,
    start_month: str,
    output_path: str,
) -> None:
    """
    Create a boxplot showing metric performance across forecast horizons for multiple models.

    Filters the metrics_df for the specified models and start_month (issue_month),
    then plots the specified metric_name across forecast horizons for each model.
    The plot shows transparent boxplots with individual data points in the background.

    Args:
        metrics_df: DataFrame containing evaluation metrics with columns
            ["code", "horizon", "issue_month", "model", metric_name, ...].
        models: List of model names to include in the plot.
        metric_name: Name of the metric column to plot (e.g., "R2", "nRMSE").
        start_month: Issue month name to filter on (e.g., "March").
        output_path: File path where the plot will be saved.

    Raises:
        ValueError: If metric_name is not found in metrics_df columns.
        ValueError: If no data remains after filtering.
    """
    if metric_name not in metrics_df.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found in DataFrame columns: "
            f"{metrics_df.columns.tolist()}"
        )

    # Filter for specified models and start_month
    filtered_df = metrics_df[
        (metrics_df["model"].isin(models)) & (metrics_df["issue_month"] == start_month)
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No data found for models {models} and issue_month {start_month}. "
            f"Available models: {metrics_df['model'].unique().tolist()}, "
            f"Available issue_months: {metrics_df['issue_month'].unique().tolist()}"
        )

    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot individual points in the background using stripplot
    sns.stripplot(
        data=filtered_df,
        x="horizon",
        y=metric_name,
        hue="model",
        hue_order=models,
        dodge=True,
        alpha=0.4,
        size=4,
        jitter=0.15,
        ax=ax,
        legend=False,
    )

    # Plot transparent boxplots on top
    sns.boxplot(
        data=filtered_df,
        x="horizon",
        y=metric_name,
        hue="model",
        hue_order=models,
        boxprops={"alpha": 0.6},
        whiskerprops={"alpha": 0.6},
        capprops={"alpha": 0.6},
        medianprops={"alpha": 0.9, "color": "black"},
        flierprops={"alpha": 0},  # Hide outlier points (already shown by stripplot)
        ax=ax,
    )

    # Set y-axis limits based on metric
    if metric_name == "R2":
        ax.set_ylim(-0.5, 1.0)
        # set y=0 line
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
    elif metric_name == "Accuracy":
        ax.set_ylim(0.0, 1.0)

    elif metric_name == "Efficiency":
        ax.set_ylim(0.0, 2.0)
        ax.axhline(0.6, color="black", linestyle="-", linewidth=1)

    # Customize the plot
    ax.set_xlabel("Forecast Horizon (months)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"{metric_name} by Forecast Horizon (Issue Month: {start_month})",
        fontsize=14,
    )

    # Adjust legend
    ax.legend(title="Model", loc="best", framealpha=0.9)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved overall plot to {output_path}")


def draw_single_basin_plot(
    metrics_df: pd.DataFrame,
    basin_codes: list[int],
    models: list[str],
    metric_name: str,
    output_path: str,
) -> None:
    """
    Create a 4x3 grid figure showing metric vs horizon for specified basins across all months.

    Each subplot represents one issue month, with lines for each basin showing the metric
    value across forecast horizons. Different models are shown as separate lines.

    Args:
        metrics_df: DataFrame containing evaluation metrics with columns
            ["code", "horizon", "issue_month", "model", metric_name, ...].
        basin_codes: List of basin codes to include in the plot (e.g., [16936, 15194, 16100]).
        models: List of model names to include in the plot.
        metric_name: Name of the metric column to plot (e.g., "R2", "nRMSE").
        output_path: File path where the plot will be saved.

    Raises:
        ValueError: If metric_name is not found in metrics_df columns.
        ValueError: If no data remains after filtering.
    """
    if metric_name not in metrics_df.columns:
        raise ValueError(
            f"Metric '{metric_name}' not found in DataFrame columns: "
            f"{metrics_df.columns.tolist()}"
        )

    # Filter for specified basins and models
    filtered_df = metrics_df[
        (metrics_df["code"].isin(basin_codes)) & (metrics_df["model"].isin(models))
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No data found for basins {basin_codes} and models {models}. "
            f"Available codes: {metrics_df['code'].unique().tolist()[:10]}..., "
            f"Available models: {metrics_df['model'].unique().tolist()}"
        )

    # Get ordered list of months for consistent subplot arrangement
    month_order = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    available_months = [
        m for m in month_order if m in filtered_df["issue_month"].unique()
    ]

    # Set up 4x3 grid figure
    fig, axes = plt.subplots(4, 3, figsize=(16, 14), sharey=True)
    axes_flat = axes.flatten()

    # Define colors and markers for basins and line styles for models
    basin_colors = plt.cm.tab10.colors[: len(basin_codes)]
    model_linestyles = ["-", "--", "-.", ":"]
    model_markers = ["o", "s", "^", "D"]

    # Create subplot for each month
    for idx, month_name in enumerate(month_order):
        ax = axes_flat[idx]

        month_data = filtered_df[filtered_df["issue_month"] == month_name]

        if month_data.empty:
            ax.set_title(month_name, fontsize=11, fontweight="bold")
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="gray",
            )
            ax.set_xlabel("Horizon")
            ax.set_ylabel(metric_name if idx % 3 == 0 else "")
            continue

        # Plot each combination of basin and model
        for basin_idx, basin_code in enumerate(basin_codes):
            for model_idx, model_name in enumerate(models):
                basin_model_data = month_data[
                    (month_data["code"] == basin_code)
                    & (month_data["model"] == model_name)
                ].sort_values("horizon")

                if basin_model_data.empty:
                    continue

                # Use basin color, model linestyle/marker
                color = basin_colors[basin_idx]
                linestyle = model_linestyles[model_idx % len(model_linestyles)]
                marker = model_markers[model_idx % len(model_markers)]

                label = f"{basin_code} - {model_name}"
                ax.plot(
                    basin_model_data["horizon"],
                    basin_model_data[metric_name],
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=5,
                    linewidth=1.5,
                    alpha=0.8,
                    label=label,
                )

        ax.set_title(month_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Horizon")
        ax.set_ylabel(metric_name if idx % 3 == 0 else "")
        ax.grid(True, linestyle="--", alpha=0.3)

        # Set y-axis limits based on metric
        if metric_name == "R2":
            ax.set_ylim(-0.5, 1.0)
            ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        elif metric_name == "Accuracy":
            ax.set_ylim(0.0, 1.0)
        elif metric_name == "Efficiency":
            ax.set_ylim(0.0, 2.0)
            ax.axhline(0.6, color="black", linestyle="-", linewidth=0.8)

    # Create a single legend for all subplots
    # Get handles and labels from the last non-empty subplot
    handles, labels = [], []
    for ax in axes_flat:
        h, l = ax.get_legend_handles_labels()
        if h:
            handles, labels = h, l
            break

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Add legend at the bottom of the figure
    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        ncol=min(len(unique_labels), 4),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Add overall title
    fig.suptitle(
        f"{metric_name} vs Forecast Horizon for Basins {basin_codes}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save the plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved single basin plot to {output_path}")


def compute_seasonal_aggregates(
    predictions_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    target_months: list[int],
    issue_months: list[int],
) -> pd.DataFrame:
    """
    Compute aggregated (mean) predictions and observations for a seasonal period.

    For each unique combination of (code, model, issue_month, year), this function
    computes the mean of predictions and observations over the specified target months
    (e.g., April-September).

    Args:
        predictions_df: DataFrame with predictions including columns
            ["code", "issue_date", "target_date", "Q_pred", "model"].
        observations_df: DataFrame with observations including columns
            ["code", "date", "discharge"].
        target_months: List of target months to aggregate (e.g., [4, 5, 6, 7, 8, 9] for Apr-Sep).
        issue_months: List of issue months to filter on (e.g., [1, 2, 3] for Jan-Mar).

    Returns:
        DataFrame with aggregated seasonal values per year:
            ["code", "model", "issue_month", "year", "Q_pred_seasonal", "Q_obs_seasonal"]
    """
    if predictions_df.empty or observations_df.empty:
        logger.warning("Empty predictions or observations DataFrame provided.")
        return pd.DataFrame()

    # Merge predictions with observations
    merged_df = predictions_df.merge(
        observations_df,
        left_on=["code", "target_date"],
        right_on=["code", "date"],
        how="inner",
        suffixes=("_pred", "_obs"),
    )

    if merged_df.empty:
        logger.warning(
            "No matching records found between predictions and observations."
        )
        return pd.DataFrame()

    # Extract issue_month and target_month
    merged_df["issue_month"] = merged_df["issue_date"].dt.month
    merged_df["target_month"] = merged_df["target_date"].dt.month
    merged_df["year"] = merged_df["issue_date"].dt.year

    # Filter for specified issue months and target months
    filtered_df = merged_df[
        (merged_df["issue_month"].isin(issue_months))
        & (merged_df["target_month"].isin(target_months))
    ].copy()

    if filtered_df.empty:
        logger.warning(
            f"No data found for issue months {issue_months} and target months {target_months}."
        )
        return pd.DataFrame()

    logger.info(
        f"Filtered {len(filtered_df)} records for seasonal aggregation "
        f"(issue months: {issue_months}, target months: {target_months})."
    )

    # Group by code, model, issue_month, year and compute mean
    grouping_cols = ["code", "model", "issue_month", "year"]
    aggregated_df = filtered_df.groupby(grouping_cols, as_index=False).agg(
        Q_pred_seasonal=("Q_pred", "mean"),
        Q_obs_seasonal=("discharge", "mean"),
        n_months=("target_month", "nunique"),
    )

    # Only keep aggregates with all target months present
    expected_months = len(target_months)
    complete_aggregates = aggregated_df[
        aggregated_df["n_months"] == expected_months
    ].copy()

    if complete_aggregates.empty:
        logger.warning(
            f"No complete seasonal aggregates found with all {expected_months} months."
        )
        # Fall back to partial aggregates with at least half the months
        min_months = expected_months // 2
        complete_aggregates = aggregated_df[
            aggregated_df["n_months"] >= min_months
        ].copy()

    # Map issue_month to month name
    complete_aggregates["issue_month_name"] = complete_aggregates["issue_month"].map(
        month_renaming
    )

    logger.info(f"Computed {len(complete_aggregates)} seasonal aggregates.")

    return complete_aggregates


def draw_seasonal_obs_vs_pred_plot(
    seasonal_df: pd.DataFrame,
    basin_codes: list[int],
    models: list[str],
    issue_months: list[str],
    output_path: str,
) -> None:
    """
    Create a 3x3 grid of obs vs pred scatter plots for seasonal aggregates.

    Rows represent issue months (e.g., January, February, March).
    Columns represent basins.
    Each subplot shows observed vs predicted seasonal mean with different models as colors.

    Args:
        seasonal_df: DataFrame with seasonal aggregates from compute_seasonal_aggregates.
            Expected columns: ["code", "model", "issue_month_name", "year",
                             "Q_pred_seasonal", "Q_obs_seasonal"].
        basin_codes: List of basin codes to plot (one per column).
        models: List of model names to include in the plot.
        issue_months: List of issue month names (e.g., ["January", "February", "March"]).
        output_path: File path where the plot will be saved.

    Raises:
        ValueError: If no data is available for plotting.
    """
    # Filter for specified basins and models
    filtered_df = seasonal_df[
        (seasonal_df["code"].isin(basin_codes))
        & (seasonal_df["model"].isin(models))
        & (seasonal_df["issue_month_name"].isin(issue_months))
    ].copy()

    if filtered_df.empty:
        raise ValueError(
            f"No data found for basins {basin_codes}, models {models}, "
            f"and issue months {issue_months}."
        )

    # Set up 3x3 grid figure (3 issue months x 3 basins)
    n_rows = len(issue_months)
    n_cols = len(basin_codes)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))

    # Ensure axes is 2D array
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Define colors for models
    model_colors = dict(zip(models, plt.cm.tab10.colors[: len(models)]))

    # Create subplot for each combination
    for row_idx, issue_month in enumerate(issue_months):
        for col_idx, basin_code in enumerate(basin_codes):
            ax = axes[row_idx, col_idx]

            subplot_data = filtered_df[
                (filtered_df["issue_month_name"] == issue_month)
                & (filtered_df["code"] == basin_code)
            ]

            if subplot_data.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=10,
                    color="gray",
                )
                ax.set_title(f"Basin {basin_code}\n(Issue: {issue_month})", fontsize=10)
                continue

            # Plot each model
            for model_name in models:
                model_data = subplot_data[subplot_data["model"] == model_name]
                if model_data.empty:
                    continue

                ax.scatter(
                    model_data["Q_obs_seasonal"],
                    model_data["Q_pred_seasonal"],
                    c=[model_colors[model_name]],
                    label=model_name,
                    alpha=0.7,
                    s=50,
                    edgecolors="white",
                    linewidth=0.5,
                )

            # Add 1:1 line
            all_values = pd.concat(
                [
                    subplot_data["Q_obs_seasonal"],
                    subplot_data["Q_pred_seasonal"],
                ]
            )
            min_val = all_values.min() * 0.9
            max_val = all_values.max() * 1.1
            ax.plot(
                [min_val, max_val], [min_val, max_val], "k--", linewidth=1, alpha=0.5
            )

            # Compute R2 for each model and display
            r2_texts = []
            for model_name in models:
                model_data = subplot_data[subplot_data["model"] == model_name]
                if len(model_data) >= 2:  # Need at least 2 points for R2
                    r2_model = r2_score(
                        model_data["Q_obs_seasonal"],
                        model_data["Q_pred_seasonal"],
                    )
                    r2_texts.append(f"{model_name}: {r2_model:.2f}")

            if r2_texts:
                r2_display = "R²\n" + "\n".join(r2_texts)
                ax.text(
                    0.05,
                    0.95,
                    r2_display,
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment="top",
                    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
                )

            ax.set_xlim(min_val, max_val)
            ax.set_ylim(min_val, max_val)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, linestyle="--", alpha=0.3)

            # Set titles and labels
            if row_idx == 0:
                ax.set_title(f"Basin {basin_code}", fontsize=11, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{issue_month}\nPredicted (m³/s)", fontsize=10)
            else:
                ax.set_ylabel("")
            if row_idx == n_rows - 1:
                ax.set_xlabel("Observed (m³/s)", fontsize=10)
            else:
                ax.set_xlabel("")

    # Create a single legend for all subplots
    handles, labels = [], []
    for ax_row in axes:
        for ax in ax_row:
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            break

    # Remove duplicates while preserving order
    unique_labels = []
    unique_handles = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_labels.append(label)
            unique_handles.append(handle)

    # Add legend at the bottom of the figure
    fig.legend(
        unique_handles,
        unique_labels,
        loc="lower center",
        ncol=len(unique_labels),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
    )

    # Add overall title
    fig.suptitle(
        "Seasonal Mean (Apr-Sep) Observed vs Predicted Discharge",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save the plot
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Saved seasonal obs vs pred plot to {output_path}")


def main():
    region = "taj"  # or "taj"
    if region == "kgz":
        path_config = kgz_path_config
        region_output_dir = os.path.join(output_dir, "kgz")
    elif region == "taj":
        path_config = taj_path_config
        region_output_dir = os.path.join(output_dir, "taj")
    else:
        raise ValueError("Invalid region specified. Choose 'kgz' or 'taj'.")

    # Load raw predictions (with valid_from, valid_to) - filtered by issue months
    predictions_raw = load_predictions(
        base_path=path_config["pred_dir"],
        horizons=horizons,
        issue_months=issue_months_to_evaluate,
    )

    # Debug: Check which months are in the raw predictions
    logger.info(
        f"Issue months in raw predictions: "
        f"{sorted(predictions_raw['issue_date'].dt.month.unique().tolist())}"
    )

    # Load observations
    observations_df = load_ground_truth(path_obs=path_config["obs_file"])

    # Load daily observations for LOO LTM calculation
    _, daily_obs = compute_long_term_means(
        observations_df=observations_df,
        daily_obs_path=path_config["obs_file"],
    )

    # Transform predictions using LOO ratio approach (unbiased evaluation)
    predictions_df = transform_predictions_with_ratio(
        predictions_raw=predictions_raw,
        observations_df=observations_df,
        daily_obs=daily_obs,
    )

    # Create ensemble predictions
    predictions_df = create_ensemble(
        predictions_df=predictions_df,
        models_to_exclude=models_not_to_ensemble,
        ensemble_name="Ensemble",
    )

    # Create snow mapper ensemble
    models_no_snow = [m for m in predictions_df["model"].unique() if "SM" not in m]
    predictions_df = create_ensemble(
        predictions_df=predictions_df,
        models_to_exclude=models_no_snow,
        ensemble_name="SM_Ensemble",
    )

    # Evaluate
    metrics_df = evaluate(
        predictions_df=predictions_df,
        observations_df=observations_df,
    )
    print(metrics_df.head())

    # Save metrics to CSV
    metrics_output_path = os.path.join(region_output_dir, "operational_metrics.csv")
    os.makedirs(region_output_dir, exist_ok=True)
    metrics_df.to_csv(metrics_output_path, index=False)
    logger.info(f"Saved operational metrics to {metrics_output_path}")

    # Generate plots for different metrics and start months
    metrics_to_plot = ["R2", "Accuracy", "Efficiency"]

    # Use all available issue months from the metrics, or specify a subset
    available_months = metrics_df["issue_month"].unique().tolist()
    start_months_to_plot = available_months  # Plot all available months

    # Or if you want specific months, filter to only those that exist:
    # desired_months = ["January", "March", "April", "May", "June", "July"]
    # start_months_to_plot = [m for m in desired_months if m in available_months]

    for metric in metrics_to_plot:
        if metric not in metrics_df.columns:
            logger.warning(f"Metric {metric} not found in results, skipping.")
            continue
        for start_month in start_months_to_plot:
            try:
                draw_overall_plot(
                    metrics_df=metrics_df,
                    models=models_plot,
                    metric_name=metric,
                    start_month=start_month,
                    output_path=os.path.join(
                        region_output_dir, f"overall_{metric}_month{start_month}.png"
                    ),
                )
            except ValueError as e:
                logger.warning(
                    f"Could not generate plot for {metric}, month {start_month}: {e}"
                )

    # Generate single basin plots for specified basins
    basins_to_plot = [16936, 15194, 16100]
    for metric in metrics_to_plot:
        if metric not in metrics_df.columns:
            logger.warning(
                f"Metric {metric} not found in results, skipping single basin plot."
            )
            continue
        try:
            draw_single_basin_plot(
                metrics_df=metrics_df,
                basin_codes=basins_to_plot,
                models=models_plot,
                metric_name=metric,
                output_path=os.path.join(
                    region_output_dir, f"single_basin_{metric}_all_months.png"
                ),
            )
        except ValueError as e:
            logger.warning(f"Could not generate single basin plot for {metric}: {e}")

    # Generate seasonal aggregated obs vs pred plot
    # Aggregate April-September predictions/observations issued in January, February, March
    seasonal_target_months = [4, 5, 6, 7, 8, 9]  # April to September
    seasonal_issue_months = [1, 2, 3]  # January, February, March

    try:
        seasonal_aggregates = compute_seasonal_aggregates(
            predictions_df=predictions_df,
            observations_df=observations_df,
            target_months=seasonal_target_months,
            issue_months=seasonal_issue_months,
        )

        if not seasonal_aggregates.empty:
            draw_seasonal_obs_vs_pred_plot(
                seasonal_df=seasonal_aggregates,
                basin_codes=basins_to_plot,
                models=models_plot,
                issue_months=["January", "February", "March"],
                output_path=os.path.join(
                    region_output_dir, "seasonal_obs_vs_pred_apr_sep.png"
                ),
            )

            # Save seasonal aggregates to CSV
            seasonal_output_path = os.path.join(
                region_output_dir, "seasonal_aggregates_apr_sep.csv"
            )
            seasonal_aggregates.to_csv(seasonal_output_path, index=False)
            logger.info(f"Saved seasonal aggregates to {seasonal_output_path}")
    except ValueError as e:
        logger.warning(f"Could not generate seasonal obs vs pred plot: {e}")


if __name__ == "__main__":
    main()
