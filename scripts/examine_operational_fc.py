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
     "month_0": 10,
    "month_1": 10,
    "month_2": 10,
    "month_3": 10,
    "month_4": 10,
    "month_5": 10,
    "month_6": 10,
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

models_plot = ["LR_Base", "LR_SM",  "MC_ALD"]

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
use_Q_obs: bool = True


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
    
    if ensemble_models.empty:
        logger.warning("No models available for ensemble creation")
        return predictions_df
    
    # Group by code, issue_date, horizon and calculate mean prediction
    ensemble = (
        ensemble_models.groupby(["code", "issue_date", "horizon", "valid_from", "valid_to"])
        .agg({
            "Q_pred": "mean",
            "Q_obs": "first",  # Q_obs should be the same for all models
        })
        .reset_index()
    )
    
    # Add model name
    ensemble["model"] = "Ensemble"
    
    # Add quantile columns as NaN (ensemble doesn't have quantiles)
    quantile_cols = [col for col in predictions_df.columns if re.fullmatch(r"Q\d+", col)]
    for col in quantile_cols:
        ensemble[col] = np.nan
    
    # Concatenate with original predictions
    combined = pd.concat([predictions_df, ensemble], ignore_index=True)
    
    logger.info(f"Created ensemble from {ensemble_models['model'].nunique()} models")
    
    return combined


def aggregate(
    predictions_df: pd.DataFrame, 
    monthly_obs: pd.DataFrame, 
    region: str
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
            axis=1
        )
    elif region == "Tajikistan":
        # Target month = issue_date + horizon - 1
        df["target_date"] = df.apply(
            lambda row: row["issue_date"] + pd.DateOffset(months=int(row["horizon"]) - 1),
            axis=1
        )
    else:
        raise ValueError(f"Unknown region: {region}. Must be 'Kyrgyzstan' or 'Tajikistan'")
    
    # Extract year and month from target_date
    df["target_year"] = df["target_date"].dt.year
    df["target_month"] = df["target_date"].dt.month
    df["issue_month"] = df["issue_date"].dt.month
    
    # Merge with monthly observations
    merged = df.merge(
        monthly_obs,
        left_on=["code", "target_year", "target_month"],
        right_on=["code", "year", "month"],
        how="left"
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
    for (code, model, issue_month, horizon, target_month), group in aggregated_df.groupby(
        ["code", "model", "issue_month", "horizon", "target_month"]
    ):
        # Calculate metrics for this group
        metrics = calculate_metrics(group["Q_obs"], group["Q_pred"])
        
        metrics_list.append({
            "code": code,
            "model": model,
            "issue_month": issue_month,
            "horizon": horizon,
            "target_month": target_month,
            **metrics
        })
    
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

            logger.debug(f"Loaded {len(df)} rows from {hindcast_file}, date range: {df['date'].min()} to {df['date'].max()}, days: {df['date'].dt.day.unique()}")

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
                logger.debug(f"No data after filtering by forecast day {forecast_day} in {hindcast_file}")
                continue

            # Filter by issue months early if specified
            if issue_months is not None:
                df = df[df["date"].dt.month.isin(issue_months)].copy()
                if df.empty:
                    logger.debug(f"No data after filtering by issue months in {hindcast_file}")
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
    logger.info(f"Unique models found: {sorted(combined_df['model'].unique().tolist())}")

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
            model_month_data = df[(df["target_month"] == month) & (df["model"] == model)][metric]

            position = m_idx + (model_idx - len(models)/2 + 0.5) * box_width

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
                    below_threshold_markers.append({
                        "position": position,
                        "pct_below": 100.0,
                        "median": np.median(values),
                        "color": color_map[model],
                        "model": model,
                        "month": month,
                        "direction": "below",
                    })
                elif n_above == n_total:
                    below_threshold_markers.append({
                        "position": position,
                        "pct_above": 100.0,
                        "median": np.median(values),
                        "color": color_map[model],
                        "model": model,
                        "month": month,
                        "direction": "above",
                    })
                else:
                    positions.append(position)
                    data_to_plot.append(values)
                    colors_to_use.append(color_map[model])
                    
                    # Track partial below/above threshold
                    if n_below > 0:
                        below_threshold_markers.append({
                            "position": position,
                            "pct_below": (n_below / n_total) * 100,
                            "color": color_map[model],
                            "model": model,
                            "month": month,
                            "direction": "below",
                            "partial": True,
                        })
                    if n_above > 0:
                        below_threshold_markers.append({
                            "position": position,
                            "pct_above": (n_above / n_total) * 100,
                            "color": color_map[model],
                            "model": model,
                            "month": month,
                            "direction": "above",
                            "partial": True,
                        })

    # Create boxplots if there's data
    if data_to_plot:
        bp = ax.boxplot(
            data_to_plot,
            positions=positions,
            widths=box_width * 0.85,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=3, alpha=0.5),
        )

        # Color the boxes
        for patch, color in zip(bp['boxes'], colors_to_use):
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
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                )
            else:
                pct = marker_info["pct_above"]
                ax.annotate(
                    f"↑{pct:.0f}%",
                    xy=(pos, y_max),
                    xytext=(pos, y_max - 0.02 * (y_max - y_min)),
                    fontsize=7,
                    color=color,
                    ha='center',
                    va='top',
                    fontweight='bold',
                )
        else:
            # All values outside range - prominent marker
            if marker_info["direction"] == "below":
                median = marker_info["median"]
                ax.scatter([pos], [y_min + 0.03 * (y_max - y_min)], 
                          marker='v', s=80, color=color, edgecolors='black', 
                          linewidths=0.5, zorder=10)
                ax.annotate(
                    f"100%↓\n(med={median:.2f})",
                    xy=(pos, y_min + 0.03 * (y_max - y_min)),
                    xytext=(pos, y_min + 0.12 * (y_max - y_min)),
                    fontsize=8,
                    color=color,
                    ha='center',
                    va='bottom',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color),
                )
            else:
                median = marker_info["median"]
                ax.scatter([pos], [y_max - 0.03 * (y_max - y_min)], 
                          marker='^', s=80, color=color, edgecolors='black',
                          linewidths=0.5, zorder=10)
                ax.annotate(
                    f"100%↑\n(med={median:.2f})",
                    xy=(pos, y_max - 0.03 * (y_max - y_min)),
                    xytext=(pos, y_max - 0.12 * (y_max - y_min)),
                    fontsize=8,
                    color=color,
                    ha='center',
                    va='top',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor=color),
                )

    # Customize the plot
    ax.set_xlabel("Target Month", fontsize=12, fontweight="bold")
    ax.set_ylabel(metric.upper() if len(metric) <= 4 else metric.capitalize(), fontsize=12, fontweight="bold")
    ax.set_title(
        f"Forecast Horizon {horizon} - {metric.upper()} by Target Month",
        fontsize=14,
        fontweight="bold"
    )

    # Set x-axis ticks and labels
    ax.set_xticks(range(len(target_months)))
    ax.set_xticklabels([month_renaming[m][:3] for m in target_months], rotation=45, ha="right")

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
    legend_handles = [plt.Rectangle((0, 0), 1, 1, fc=color_map[model], alpha=0.7) for model in models]
    ax.legend(legend_handles, models, loc='upper right', framealpha=0.9, fontsize=10)
    
    # Log missing combinations
    if missing_combinations:
        logger.warning(f"Horizon {horizon}: Missing data for {len(missing_combinations)} combinations: {missing_combinations[:10]}...")
    
    # Log below/above threshold statistics
    n_full_below = len([m for m in below_threshold_markers if not m.get("partial", False) and m["direction"] == "below"])
    n_full_above = len([m for m in below_threshold_markers if not m.get("partial", False) and m["direction"] == "above"])
    if n_full_below > 0 or n_full_above > 0:
        logger.info(f"Horizon {horizon}: {n_full_below} model/month combos have 100% values below y_min={y_min}, {n_full_above} have 100% above y_max={y_max}")
        # Print details of which ones
        for m in below_threshold_markers:
            if not m.get("partial", False):
                logger.info(f"  -> {m['model']} @ month {m['month']}: direction={m['direction']}, median={m.get('median', 'N/A')}")

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved {metric} by target month plot for horizon {horizon} to {output_path}")

    return fig


def plot_r2_by_month(
    predictions_df: pd.DataFrame,
    horizon: int,
    models: list[str],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot R2 score distribution across different months of the year for given models.

    Creates a boxplot showing R2 score distribution for each model across all months
    of the year (inferred from valid_from.month) for a specific forecast horizon.
    R2 is calculated separately for each basin (code).

    Args:
        predictions_df: DataFrame with predictions containing columns:
            - Q_pred: predicted discharge
            - Q_obs: observed discharge
            - valid_from: validity start date (used to extract month)
            - horizon: forecast horizon
            - model: model name
            - code: basin code
        horizon: The forecast horizon to filter by (e.g., 1, 2, 3)
        models: List of model names to include in the plot
        output_path: Optional path to save the figure

    Returns:
        matplotlib Figure object
    """
    # Filter for the specified horizon
    df = predictions_df[predictions_df["horizon"] == horizon].copy()

    if df.empty:
        logger.warning(f"No data found for horizon {horizon}")
        return plt.figure()

    # Filter for specified models
    df = df[df["model"].isin(models)].copy()

    if df.empty:
        logger.warning(f"No data found for models {models} at horizon {horizon}")
        return plt.figure()

    # Extract month from issue_date + horizon months
    df["valid_month"] = (df["issue_date"] + pd.DateOffset(months=horizon)).dt.month

    # Calculate R2 for each model, month, and code combination
    r2_results = []
    for model in models:
        model_df = df[df["model"] == model]
        for month in range(1, 13):
            month_df = model_df[model_df["valid_month"] == month]

            # Calculate R2 for each code separately
            for code in month_df["code"].unique():
                code_df = month_df[month_df["code"] == code]

                # Drop rows with NaN values in Q_obs or Q_pred
                code_df = code_df.dropna(subset=["Q_obs", "Q_pred"])

                if len(code_df) < 2:
                    continue

                try:
                    r2 = r2_score(code_df["Q_obs"], code_df["Q_pred"])
                    r2_results.append(
                        {
                            "model": model,
                            "month": month,
                            "month_name": month_renaming[month],
                            "code": code,
                            "r2": r2,
                            "n_samples": len(code_df),
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not compute R2 for {model}, month {month}, code {code}: {e}"
                    )

    if not r2_results:
        logger.warning("No R2 scores could be computed")
        return plt.figure()

    r2_df = pd.DataFrame(r2_results)

    # Create the plot - one subplot per model
    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 5 * n_models), squeeze=False)
    axes = axes.flatten()

    # Define color palette for months
    month_colors = sns.color_palette("husl", 12)

    for idx, model in enumerate(models):
        ax = axes[idx]
        model_data = r2_df[r2_df["model"] == model]

        if model_data.empty:
            ax.text(0.5, 0.5, f"No data for {model}", ha="center", va="center")
            ax.set_title(f"{model}")
            continue

        # Create boxplot
        sns.boxplot(
            data=model_data,
            x="month",
            y="r2",
            ax=ax,
            palette=month_colors,
            order=range(1, 13),
        )

        # Customize the subplot
        ax.set_xlabel("Month", fontsize=11)
        ax.set_ylabel("R² Score", fontsize=11)
        ax.set_title(f"{model} - Horizon {horizon}", fontsize=12, fontweight="bold")
        ax.set_xticklabels(
            [month_renaming[m][:3] for m in range(1, 13)], rotation=45, ha="right"
        )
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylim(-10.0, 1.0)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved R2 by month plot to {output_path}")

    return fig


def main():
    
    region = "Tajikistan"  # Options: "Kyrgyzstan" or "Tajikistan"
    save_dir = Path(output_dir) / f"{region.lower()}"
    # Configuration for plotting
    metric_to_plot = "r2"  # Options: "r2", "rmse", "nrmse", "mae", "nmae", "accuracy", "efficiency"
    
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
            logger.error("Q_obs not available in prediction files. Set use_Q_obs=False to load from observations file.")
            return
        
        # Add target_month, issue_month based on region logic
        if region == "Kyrgyzstan":
            predictions_df["target_date"] = predictions_df.apply(
                lambda row: row["issue_date"] + pd.DateOffset(months=int(row["horizon"])),
                axis=1
            )
        elif region == "Tajikistan":
            predictions_df["target_date"] = predictions_df.apply(
                lambda row: row["issue_date"] + pd.DateOffset(months=int(row["horizon"]) - 1),
                axis=1
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
    print("DIAGNOSTIC: Data coverage per model and horizon (number of target months with data)")
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
                print(f"  Horizons in aggregated_df: {sorted(pred_model_data['horizon'].unique())}")
                print(f"  Target months in aggregated_df: {sorted(pred_model_data['target_month'].unique())}")
        else:
            for horizon in sorted(model_data["horizon"].unique()):
                horizon_data = model_data[model_data["horizon"] == horizon]
                target_months = sorted(horizon_data["target_month"].unique())
                n_codes = horizon_data["code"].nunique()
                print(f"  Horizon {horizon}: {len(target_months)} target months {target_months}, {n_codes} codes, {len(horizon_data)} records")
    
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
                print(f"  Horizon {horizon}: {n_records} records, {n_with_obs} with Q_obs, {n_with_pred} with Q_pred, target_months: {target_months}")
    print("=" * 80 + "\n")
    
    # Add Ensemble to models_plot if not already there
    models_to_plot = models_plot + ["Ensemble"] if "Ensemble" not in models_plot else models_plot
    
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
            output_path=Path(save_dir) / f"{metric_to_plot}_horizon_{horizon}_by_target_month.png"
            if save_dir
            else None,
        )
        #plt.show()
        plt.close(fig)  # Close to avoid memory issues
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()