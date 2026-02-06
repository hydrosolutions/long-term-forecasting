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
    "month_0": 10,
    "month_1": 25,
    "month_2": 25,
    "month_3": 25,
    "month_4": 25,
    "month_5": 25,
    "month_6": 25,
    "month_7": 25,
    "month_8": 25,
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
    "MC_ALD_loc",
    "obs",
    "Obs",
    "OBS",  # Exclude any observation-based "models"
]

# Submodel variants to exclude (use only main model output)
submodel_suffixes_to_exclude = ["_xgb", "_catboost", "_lgbm", "_rf"]

# Model weights for weighted ensemble (GBT models weight 3x, LR models weight 1x)
model_weights = {
    "SM_GBT": 3,
    "SM_GBT_Norm": 3,
    "SM_GBT_LR": 3,
    "LR_SM": 1,
    "LR_Base": 1,
    "LR_SM_ROF": 1,
}

models_plot = ["LR_Base", "LR_SM", "MC_ALD", "Ensemble"]

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


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Calculate various performance metrics.

    1. R2
    2. nRMSE = RMSE / mean(observed)
    3. MAE
    4. nMAE = MAE / mean(observed)
    5. Accuracy : |y_true - y_pred| <= 0.675 * std of y_true -> 1 else 0
    6. Efficiency: std (|y_true - y_pred|) / std(y_true)
    7. PBIAS: 100 * sum(predicted - observed) / sum(observed)

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
            "pbias": np.nan,
            "n_samples": len(y_true_clean),
        }

    # Calculate metrics
    r2 = r2_score(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))

    mean_obs = y_true_clean.mean()
    std_obs = y_true_clean.std()
    sum_obs = y_true_clean.sum()

    nrmse = rmse / mean_obs if mean_obs != 0 else np.nan
    nmae = mae / mean_obs if mean_obs != 0 else np.nan

    # Accuracy: fraction of predictions within 0.675 * std of observed
    threshold = 0.675 * std_obs
    accuracy = np.mean(np.abs(y_true_clean - y_pred_clean) <= threshold)

    # Efficiency: std(errors) / std(observed)
    errors = np.abs(y_true_clean - y_pred_clean)
    efficiency = errors.std() / std_obs if std_obs != 0 else np.nan

    # PBIAS: Percent Bias - positive means over-prediction, negative means under-prediction
    pbias = (
        100 * (y_pred_clean - y_true_clean).sum() / sum_obs if sum_obs != 0 else np.nan
    )

    return {
        "r2": r2,
        "rmse": rmse,
        "nrmse": nrmse,
        "mae": mae,
        "nmae": nmae,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "pbias": pbias,
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


def create_ensemble(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates weighted ensemble across selected models.

    Weighting: GBT models (SM_GBT, SM_GBT_Norm, SM_GBT_LR) weighted 3x,
               LR models (LR_SM, LR_Base, LR_SM_ROF) weighted 1x.

    Only uses main model outputs (excludes _xgb, _catboost, _lgbm variants).

    Args:
        predictions_df: DataFrame with predictions containing columns:
            - code, issue_date, valid_from, valid_to, Q_pred, Q_obs, horizon, model

    Returns:
        DataFrame with ensemble predictions added as a new model "Ensemble"
    """
    # Filter out excluded models and submodel variants
    df = predictions_df.copy()

    # Exclude models in the exclusion list
    df_filtered = df[~df["model"].isin(models_not_to_ensemble)].copy()

    # Exclude submodel variants (e.g., model_xgb, model_catboost)
    for suffix in submodel_suffixes_to_exclude:
        df_filtered = df_filtered[~df_filtered["model"].str.endswith(suffix)]

    # Keep only models that have weights defined
    df_filtered = df_filtered[df_filtered["model"].isin(model_weights.keys())]

    logger.info(
        f"Creating weighted ensemble from {df_filtered['model'].nunique()} models"
    )
    logger.info(f"  Models included: {df_filtered['model'].unique().tolist()}")
    logger.info(f"  Weights: {model_weights}")

    if df_filtered.empty:
        logger.warning("No models available for ensemble creation")
        return predictions_df

    # Add weight column based on model
    df_filtered["weight"] = df_filtered["model"].map(model_weights)

    # Calculate weighted mean for each group
    def weighted_mean(group):
        weights = group["weight"].values
        values = group["Q_pred"].values
        return np.average(values, weights=weights)

    # Group by code, issue_date, horizon and calculate weighted mean prediction
    grouped = df_filtered.groupby(
        ["code", "issue_date", "horizon", "valid_from", "valid_to"]
    )

    ensemble_preds = grouped.apply(weighted_mean, include_groups=False).reset_index()
    ensemble_preds.columns = [
        "code",
        "issue_date",
        "horizon",
        "valid_from",
        "valid_to",
        "Q_pred",
    ]

    # Get Q_obs from the first model in each group
    q_obs = grouped["Q_obs"].first().reset_index().rename(columns={"Q_obs": "Q_obs"})

    ensemble = ensemble_preds.merge(
        q_obs, on=["code", "issue_date", "horizon", "valid_from", "valid_to"]
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

    logger.info(
        f"Created weighted ensemble from {df_filtered['model'].nunique()} models"
    )

    return combined


def aggregate(predictions_df: pd.DataFrame, monthly_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Merge predictions with monthly aggregated observations.

    Target month is extracted directly from valid_from (no shift calculation needed).

    Args:
        predictions_df: DataFrame with predictions (code, issue_date, valid_from, horizon, Q_pred, model, etc.)
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly)

    Returns:
        DataFrame with merged predictions and monthly observations, including target_month
    """
    df = predictions_df.copy()

    # Target month directly from valid_from (no shift calculation)
    df["target_month"] = df["valid_from"].dt.month
    df["target_year"] = df["valid_from"].dt.year
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

    logger.info("Aggregated predictions with monthly observations")
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


def create_horizon_metrics_dataframe(
    aggregated_df: pd.DataFrame,
    horizon: int,
    models: list[str] | None = None,
    month_column: str = "target_month",
) -> pd.DataFrame:
    """
    Create metrics DataFrame for a specific forecast horizon.

    Args:
        aggregated_df: DataFrame with aggregated predictions and observations
        horizon: Forecast horizon to filter by
        models: Optional list of models to include. If None, all models are included.
        month_column: Column to use for month grouping ("target_month" or "issue_month")

    Returns:
        DataFrame with columns: code, model, month, R2, Accuracy, Efficiency, PBIAS,
                               MAE, nMAE, nRMSE, n_samples
    """
    # Filter by horizon
    df = aggregated_df[aggregated_df["horizon"] == horizon].copy()

    if df.empty:
        logger.warning(f"No data found for horizon {horizon}")
        return pd.DataFrame()

    # Filter by models if specified
    if models is not None:
        df = df[df["model"].isin(models)]
        if df.empty:
            logger.warning(f"No data found for horizon {horizon} with specified models")
            return pd.DataFrame()

    metrics_list = []

    # Group by (code, model, month)
    for (code, model, month), group in df.groupby(["code", "model", month_column]):
        metrics = calculate_metrics(group["Q_obs"], group["Q_pred"])

        metrics_list.append(
            {
                "code": code,
                "model": model,
                "month": month,
                "R2": metrics["r2"],
                "Accuracy": metrics["accuracy"],
                "Efficiency": metrics["efficiency"],
                "PBIAS": metrics["pbias"],
                "MAE": metrics["mae"],
                "nMAE": metrics["nmae"],
                "nRMSE": metrics["nrmse"],
                "n_samples": metrics["n_samples"],
            }
        )

    metrics_df = pd.DataFrame(metrics_list)

    if not metrics_df.empty:
        # Sort for consistent output
        metrics_df = metrics_df.sort_values(["code", "model", "month"]).reset_index(
            drop=True
        )

    logger.debug(
        f"Created horizon {horizon} metrics DataFrame with {len(metrics_df)} rows"
    )

    return metrics_df


def calculate_quantile_exceedance_rates(
    observed: np.ndarray, quantile_preds: dict[str, np.ndarray]
) -> dict[str, float]:
    """
    Calculate exceedance rate for each quantile.

    For a well-calibrated forecast:
    - Q5 should exceed observed ~5% of the time
    - Q50 should exceed observed ~50% of the time
    - Q95 should exceed observed ~95% of the time

    Formula: exceedance_rate(p) = P(Q_p > observed)

    Args:
        observed: Array of observed values
        quantile_preds: Dictionary mapping quantile names (e.g., "Q5", "Q50")
                       to arrays of predicted quantile values

    Returns:
        Dictionary with exceedance rates for each quantile
    """
    results = {}

    for quantile_name, pred_values in quantile_preds.items():
        # Create mask for valid (non-NaN) pairs
        mask = ~(np.isnan(observed) | np.isnan(pred_values))
        obs_clean = observed[mask]
        pred_clean = pred_values[mask]

        if len(obs_clean) == 0:
            results[f"{quantile_name}_exceedance"] = np.nan
            continue

        # Calculate exceedance rate: fraction where quantile prediction > observed
        exceedance_rate = np.mean(pred_clean > obs_clean)
        results[f"{quantile_name}_exceedance"] = exceedance_rate

    return results


def create_quantile_exceedance_dataframe(
    aggregated_df: pd.DataFrame,
    model: str = "MC_ALD",
    horizons: list[int] | None = None,
    month_column: str = "target_month",
) -> pd.DataFrame:
    """
    Create quantile exceedance DataFrame for probabilistic model.

    For each combination of code, month, and horizon, calculates the empirical
    exceedance rates for each quantile (Q5, Q10, Q25, Q50, Q75, Q90, Q95) and
    compares them to expected rates.

    Args:
        aggregated_df: DataFrame with aggregated predictions and observations
        model: Model name to filter for (default: "MC_ALD")
        horizons: Optional list of horizons to include. If None, all horizons are included.
        month_column: Column to use for month grouping ("target_month" or "issue_month")

    Returns:
        DataFrame with columns:
            code, month, horizon, Q5_exceedance, Q10_exceedance, Q25_exceedance,
            Q50_exceedance, Q75_exceedance, Q90_exceedance, Q95_exceedance,
            Q5_bias, Q10_bias, ..., n_samples
    """
    # Expected exceedance rates for each quantile
    expected_rates = {
        "Q5": 0.05,
        "Q10": 0.10,
        "Q25": 0.25,
        "Q50": 0.50,
        "Q75": 0.75,
        "Q90": 0.90,
        "Q95": 0.95,
    }

    # Filter for specified model
    df = aggregated_df[aggregated_df["model"] == model].copy()

    if df.empty:
        logger.warning(f"No data found for model {model}")
        return pd.DataFrame()

    # Filter by horizons if specified
    if horizons is not None:
        df = df[df["horizon"].isin(horizons)]
        if df.empty:
            logger.warning(f"No data found for model {model} with specified horizons")
            return pd.DataFrame()

    # Identify available quantile columns
    quantile_cols = [col for col in df.columns if re.fullmatch(r"Q\d+", col)]
    available_quantiles = {
        col: expected_rates.get(col) for col in quantile_cols if col in expected_rates
    }

    if not available_quantiles:
        logger.warning(f"No quantile columns found for model {model}")
        return pd.DataFrame()

    logger.info(f"Found quantile columns: {list(available_quantiles.keys())}")

    exceedance_list = []

    # Group by (code, month, horizon)
    for (code, month, horizon), group in df.groupby(["code", month_column, "horizon"]):
        observed = group["Q_obs"].values

        # Build quantile predictions dictionary
        quantile_preds = {}
        for q_col in available_quantiles.keys():
            if q_col in group.columns:
                quantile_preds[q_col] = group[q_col].values

        # Calculate exceedance rates
        exceedance_rates = calculate_quantile_exceedance_rates(observed, quantile_preds)

        # Build result row
        row = {
            "code": code,
            "month": month,
            "horizon": horizon,
        }

        # Add exceedance rates and biases
        for q_col, expected_rate in available_quantiles.items():
            exc_key = f"{q_col}_exceedance"
            if exc_key in exceedance_rates:
                row[f"{q_col}_exc"] = exceedance_rates[exc_key]
                # Bias = actual exceedance - expected exceedance
                if not np.isnan(exceedance_rates[exc_key]):
                    row[f"{q_col}_bias"] = exceedance_rates[exc_key] - expected_rate
                else:
                    row[f"{q_col}_bias"] = np.nan

        # Count valid samples (where both Q_obs and at least one quantile are not NaN)
        mask = ~np.isnan(observed)
        for q_col in available_quantiles.keys():
            if q_col in group.columns:
                mask = mask & ~group[q_col].isna().values
        row["n_samples"] = mask.sum()

        exceedance_list.append(row)

    exceedance_df = pd.DataFrame(exceedance_list)

    if not exceedance_df.empty:
        # Sort for consistent output
        exceedance_df = exceedance_df.sort_values(
            ["code", "month", "horizon"]
        ).reset_index(drop=True)

    logger.info(
        f"Created quantile exceedance DataFrame for {model} with {len(exceedance_df)} rows"
    )

    return exceedance_df


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

            # Convert dates to datetime (use mixed format to handle both date and datetime strings)
            df["date"] = pd.to_datetime(df["date"], format="mixed")
            df["valid_from"] = pd.to_datetime(df["valid_from"], format="mixed")
            df["valid_to"] = pd.to_datetime(df["valid_to"], format="mixed")

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

        # Add target_month, issue_month directly from valid_from
        predictions_df["target_month"] = predictions_df["valid_from"].dt.month
        predictions_df["issue_month"] = predictions_df["issue_date"].dt.month
        aggregated_df = predictions_df

    else:
        logger.info(f"Loading observations from {pred_config['obs_file']}...")
        obs_df = load_observations(pred_config["obs_file"])

        # Calculate monthly targets
        logger.info("Calculating monthly observation targets...")
        monthly_obs = calculate_target(obs_df)

        # Aggregate predictions with observations
        logger.info("Aggregating predictions with observations...")
        aggregated_df = aggregate(predictions_df, monthly_obs)

    # Compute metrics dataframe
    logger.info("Computing metrics...")
    metrics_df = compute_metrics_dataframe(aggregated_df)

    logger.info(f"Metrics computed for {len(metrics_df)} combinations")
    logger.info("\nMetrics DataFrame preview:")
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

    # ==========================================================================
    # Create structured metrics DataFrames
    # ==========================================================================

    # Create deterministic metrics DataFrames (one per horizon)
    logger.info("\nCreating horizon-specific metrics DataFrames...")
    horizon_metrics: dict[int, pd.DataFrame] = {}
    for horizon in available_horizons:
        horizon_metrics[horizon] = create_horizon_metrics_dataframe(
            aggregated_df,
            horizon=horizon,
            models=models_to_plot,
            month_column="target_month",
        )
        logger.info(f"  Horizon {horizon}: {len(horizon_metrics[horizon])} rows")

    # Create probabilistic evaluation DataFrame for MC_ALD
    quantile_exceedance_df = pd.DataFrame()
    if "MC_ALD" in aggregated_df["model"].unique():
        logger.info("\nCreating quantile exceedance DataFrame for MC_ALD...")
        quantile_exceedance_df = create_quantile_exceedance_dataframe(
            aggregated_df,
            model="MC_ALD",
            horizons=available_horizons,
            month_column="target_month",
        )

        # Print summary statistics for quantile calibration
        if not quantile_exceedance_df.empty:
            print("\n" + "=" * 80)
            print("QUANTILE CALIBRATION SUMMARY (MC_ALD)")
            print("Expected vs Actual Exceedance Rates (aggregated across all data)")
            print("=" * 80)
            quantile_cols = [
                col for col in quantile_exceedance_df.columns if col.endswith("_exc")
            ]
            for q_col in quantile_cols:
                q_name = q_col.replace("_exc", "")
                expected = {
                    "Q5": 0.05,
                    "Q10": 0.10,
                    "Q25": 0.25,
                    "Q50": 0.50,
                    "Q75": 0.75,
                    "Q90": 0.90,
                    "Q95": 0.95,
                }.get(q_name, np.nan)
                actual = quantile_exceedance_df[q_col].mean()
                bias = actual - expected if not np.isnan(expected) else np.nan
                print(
                    f"  {q_name}: Expected={expected:.2f}, Actual={actual:.3f}, Bias={bias:+.3f}"
                )
            print("=" * 80 + "\n")
    else:
        logger.warning("MC_ALD model not found - skipping quantile exceedance analysis")

    # ==========================================================================

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
