import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


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
    "month_1": 27,
    "month_2": 27,
    "month_3": 27,
    "month_4": 27,
    "month_5": 27,
    "month_6": 27,
    "month_7": 27,
    "month_8": 27,
    "month_9": 27,
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

models_not_to_ensemble = ["MC_ALD"]


models_plot = ["LR_Base", "SM_GBT", "Ensemble", "MC_ALD"]

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
    accurate_preds = abs_errors < (0.674 * sigma_obs)
    metrics["Accuracy"] = np.mean(accurate_preds)

    std_abs_errors = np.std(abs_errors)
    metrics["Efficiency"] = std_abs_errors / sigma_obs

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


def load_predictions(
    base_path: str,
    horizons: list[str],
) -> pd.DataFrame:
    """
    This function loads all the model predictions from the specified directory.
    The structure is:
    base_path/
        horizon_1/
                A/
                    {model_name}_hindcast.csv # model_name is A
                B/
                    {model_name}_hindcast.csv # model_name is B
        horizon_2/
                A/
                    {model_name}_hindcast.csv # model_name is A
        ...
    the column we want to extract are:
    date (yyyy-mm-dd), code (int), Q_{model_name} (float), Optional Q5, Q25, Q75, Q95 (float), valid_from (yyyy-mm-dd), valid_to (yyyy-mm-dd)
    1. convert dates to datetime
    2. convert "code" to int
    3. sort by code and date
    4. only keep the day of the month according to day_of_forecast
    5. Restructures the data to have following columns:
         ["code","issue_date", "target_date", "Q_pred", "Q5", "Q25", "Q75", "Q95", "horizon", "model"]
         where target_date is just the month and year of the valid_from date indicating the forecasted month
         issue_date is the date when the forecast was issued -> date column
         horizon is the forecast horizon in months (0-9)
    6. concatenate all model predictions into a single DataFrame
    Returns a pandas DataFrame with the concatenated predictions.
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

            # Find the prediction column (Q_{model_name})
            q_col = f"Q_{model_name}"
            if q_col not in df.columns:
                # Try to find any column starting with 'Q_' that's not a quantile
                q_cols = [
                    c
                    for c in df.columns
                    if c.startswith("Q_") and c not in ["Q5", "Q25", "Q75", "Q95"]
                ]
                if q_cols:
                    q_col = q_cols[0]
                else:
                    logger.warning(f"No prediction column found in {hindcast_file}")
                    continue

            # Restructure the DataFrame
            result_df = pd.DataFrame(
                {
                    "code": df["code"],
                    "issue_date": df["date"],
                    "target_date": df["valid_from"].dt.to_period("M").dt.to_timestamp(),
                    "Q_pred": df[q_col],
                    "horizon": horizon_num,
                    "model": model_name,
                }
            )

            # Add quantile columns if they exist
            for quantile_col in ["Q5", "Q25", "Q75", "Q95"]:
                if quantile_col in df.columns:
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
    Create a bar plot showing metric performance across forecast horizons for multiple models.

    Filters the metrics_df for the specified models and start_month (issue_month),
    then plots the specified metric_name across forecast horizons for each model.
    The plot shows the median as bars and the 25th/75th percentiles as error bars.

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

    # Use seaborn barplot with median and percentile error bars
    sns.barplot(
        data=filtered_df,
        x="horizon",
        y=metric_name,
        hue="model",
        hue_order=models,
        estimator="median",
        errorbar=("pi", 50),  # 50% percentile interval (25th to 75th)
        capsize=0.1,
        ax=ax,
    )

    # Customize the plot
    ax.set_xlabel("Forecast Horizon (months)", fontsize=12)
    ax.set_ylabel(metric_name, fontsize=12)
    ax.set_title(
        f"{metric_name} by Forecast Horizon (Issue Month: {start_month})\n"
        f"Median with 25th-75th Percentile Error Bars",
        fontsize=14,
    )

    # Adjust legend
    ax.legend(title="Model", loc="best", framealpha=0.9)

    # Add grid for better readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
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
    basin_id: str,
    models: list,
    metric_name: str,
    start_month: str,
    output_path: str,
):
    pass


def main():
    region = "kgz"  # or "taj"
    if region == "kgz":
        path_config = kgz_path_config
        region_output_dir = os.path.join(output_dir, "kgz")
    elif region == "taj":
        path_config = taj_path_config
        region_output_dir = os.path.join(output_dir, "taj")
    else:
        raise ValueError("Invalid region specified. Choose 'kgz' or 'taj'.")

    predictions_df = load_predictions(
        base_path=path_config["pred_dir"],
        horizons=horizons,
    )

    # Create ensemble predictions
    predictions_df = create_ensemble(
        predictions_df=predictions_df,
        models_to_exclude=models_not_to_ensemble,
        ensemble_name="Ensemble",
    )

    observations_df = load_ground_truth(path_obs=path_config["obs_file"])
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
    metrics_to_plot = ["R2", "Accuracy"]
    start_months_to_plot = ["March", "April", "May"]  # Example: March, April, May

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


if __name__ == "__main__":
    main()
