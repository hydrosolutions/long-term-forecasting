import os
import sys
import argparse
import json
import traceback
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import forecast models
from lt_forecasting.forecast_models.LINEAR_REGRESSION import LinearRegressionModel
from lt_forecasting.forecast_models.SciRegressor import SciRegressor
from lt_forecasting.forecast_models.deep_models.uncertainty_mixture import (
    UncertaintyMixtureModel,
)
from lt_forecasting.scr import data_loading as dl


# Setup logging
from lt_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Force ALL logging to INFO level
logging.getLogger().setLevel(logging.INFO)  # Root logger
for handler in logging.getLogger().handlers:
    handler.setLevel(logging.INFO)


# import environment variables saved in .env file
from dotenv import load_dotenv

load_dotenv()

PATH_TO_DISCHARGE = os.getenv("path_discharge")
PATH_TO_FORCING_ERA5 = os.getenv("PATH_TO_FORCING_ERA5")
PATH_TO_FORCING_OPERATIONAL = os.getenv("path_forcing_operational")
PATH_SWE_00003 = os.getenv("path_SWE_00003")
PATH_SWE_500m = os.getenv("PATH_SWE_500m")
PATH_ROF_00003 = os.getenv("path_ROF_00003")
PATH_ROF_500m = os.getenv("PATH_ROF_500m")
PATH_TO_SHP = os.getenv("path_to_shp")
PATH_TO_STATIC = os.getenv("PATH_TO_STATIC")

MODELS_OPERATIONAL = {
    "BaseCase": [
        ("LR", "LR_Q_T_P"),
        ("SciRegressor", "GBT_simple"),
    ],
    "SnowMapper_Based": [
        ("LR", "LR_Snowmapper"),
        ("LR", "LR_Snowmapper_DT"),
        ("SciRegressor", "Snow_GBT"),
        ("SciRegressor", "Snow_GBT_LR"),
        # ("SciRegressor", "Snow_GBT_Norm"),
    ],
    "Uncertainty": [
        ("UncertaintyMixtureMLP", "UncertaintyMixtureMLP"),
    ],
}

MODELS_DIR = "../monthly_forecasting_models"
MOCK_OUTPUT_DIR = "../monthly_forecasting_results/mock_operational"


### Load the Data Function
def load_discharge():
    df_discharge = pd.read_csv(PATH_TO_DISCHARGE, parse_dates=["date"])
    df_discharge["date"] = pd.to_datetime(df_discharge["date"], format="%Y-%m-%d")
    df_discharge["code"] = df_discharge["code"].astype(int)

    # esnure that the discharge df is continuous -> for all dates from min to max date is an entry for each code
    min_date = df_discharge["date"].min()
    max_date = df_discharge["date"].max()

    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    all_codes = df_discharge["code"].unique()

    continuous_df = pd.DataFrame()

    for code in all_codes:
        code_df = df_discharge[df_discharge["code"] == code].copy()
        code_df = code_df.set_index("date").reindex(all_dates).reset_index()
        code_df["code"] = code
        continuous_df = pd.concat([continuous_df, code_df], ignore_index=True)

    continuous_df.rename(columns={"index": "date"}, inplace=True)

    return continuous_df


def load_forcing():
    operational_T = pd.read_csv(
        os.path.join(PATH_TO_FORCING_OPERATIONAL, "00003_T_control_member.csv")
    )
    operational_P = pd.read_csv(
        os.path.join(PATH_TO_FORCING_OPERATIONAL, "00003_P_control_member.csv")
    )

    operational_T["date"] = pd.to_datetime(operational_T["date"], format="%Y-%m-%d")
    operational_P["date"] = pd.to_datetime(operational_P["date"], format="%Y-%m-%d")
    operational_T["code"] = operational_T["code"].astype(int)
    operational_P["code"] = operational_P["code"].astype(int)

    operational_data = pd.merge(operational_T, operational_P, on=["date", "code"])
    logger.info("Operational forcing data loaded successfully.")

    hindcast_T = pd.read_csv(
        os.path.join(PATH_TO_FORCING_ERA5, "00003_T_reanalysis.csv")
    )
    hindcast_P = pd.read_csv(
        os.path.join(PATH_TO_FORCING_ERA5, "00003_P_reanalysis.csv")
    )
    hindcast_T["date"] = pd.to_datetime(hindcast_T["date"], format="%Y-%m-%d")
    hindcast_P["date"] = pd.to_datetime(hindcast_P["date"], format="%Y-%m-%d")
    hindcast_T["code"] = hindcast_T["code"].astype(int)
    hindcast_P["code"] = hindcast_P["code"].astype(int)
    hindcast_data = pd.merge(hindcast_T, hindcast_P, on=["date", "code"])

    logger.info("Hindcast forcing data loaded successfully.")

    operationl_data = pd.concat([hindcast_data, operational_data], ignore_index=True)
    operationl_data = operationl_data.sort_values(by=["date", "code"]).reset_index(
        drop=True
    )

    # drop duplicates based on date and code - keep last occurrence
    operational_data = operationl_data.drop_duplicates(
        subset=["date", "code"], keep="last"
    )

    return operational_data


def load_snowmapper():
    swe_00003 = pd.read_csv(PATH_SWE_00003, parse_dates=["date"])
    swe_00003["date"] = pd.to_datetime(swe_00003["date"], format="%Y-%m-%d")
    swe_00003["code"] = swe_00003["code"].astype(int)

    logger.info("SWE 00003 columns: %s", swe_00003.columns.tolist())

    swe_500m = pd.read_csv(PATH_SWE_500m, parse_dates=["date"])
    swe_500m["date"] = pd.to_datetime(swe_500m["date"], format="%Y-%m-%d")
    swe_500m["code"] = swe_500m["code"].astype(int)

    logger.info("SWE 500m columns: %s", swe_500m.columns.tolist())

    rof_00003 = pd.read_csv(PATH_ROF_00003, parse_dates=["date"])
    rof_00003["date"] = pd.to_datetime(rof_00003["date"], format="%Y-%m-%d")
    rof_00003["code"] = rof_00003["code"].astype(int)

    logger.info("ROF 00003 columns: %s", rof_00003.columns.tolist())
    # replace RoF with ROF
    rof_00003.rename(columns={"RoF": "ROF"}, inplace=True)

    rof_500m = pd.read_csv(PATH_ROF_500m, parse_dates=["date"])
    rof_500m["date"] = pd.to_datetime(rof_500m["date"], format="%Y-%m-%d")
    rof_500m["code"] = rof_500m["code"].astype(int)

    # replace RoF with ROF
    for col in rof_500m.columns:
        if col.startswith("RoF"):
            new_col_name = col.replace("RoF", "ROF")
            rof_500m.rename(columns={col: new_col_name}, inplace=True)

    logger.info("ROF 500m columns: %s", rof_500m.columns.tolist())

    return swe_00003, swe_500m, rof_00003, rof_500m


def load_static_data():
    df_static = pd.read_csv(PATH_TO_STATIC)

    if "CODE" in df_static.columns:
        df_static.rename(columns={"CODE": "code"}, inplace=True)

    df_static["code"] = df_static["code"].astype(int)

    return df_static


def create_data_frame(config: Dict[str, Any], debug: bool = False) -> pd.DataFrame:
    data = load_discharge()

    forcing = load_forcing()

    swe_00003, swe_500m, rof_00003, rof_500m = load_snowmapper()

    static_data = load_static_data()
    logger.info(f"Static Data Columns: {static_data.columns.tolist()}")
    logger.info("Data loading complete. Merging data...")

    # Only keep code which are in static data
    static_codes = static_data["code"].unique()
    data = data[data["code"].isin(static_codes)].copy()

    data = data.merge(forcing, on=["date", "code"], how="left")

    SWE_HRU = config.get("HRU_SWE", None)

    logger.info(f" Number of nan values before merging SWE: {data.isna().sum()}")

    if SWE_HRU == "HRU_00003" or SWE_HRU == "00003":
        data = data.merge(swe_00003, on=["date", "code"], how="left")

    elif SWE_HRU == "KGZ500m" or SWE_HRU == "500m":
        data = data.merge(swe_500m, on=["date", "code"], how="left")

    else:
        logger.info("No SWE data merged.")

    logger.info(f" Number of nan values after merging SWE: {data.isna().sum()}")

    # Merge ROF data
    logger.info(f" Number of nan values before merging ROF: {data.isna().sum()}")

    ROF_HRU = config.get("HRU_ROF", None)
    if ROF_HRU == "HRU_00003" or ROF_HRU == "00003":
        data = data.merge(rof_00003, on=["date", "code"], how="left")
    elif ROF_HRU == "KGZ500m" or ROF_HRU == "500m":
        data = data.merge(rof_500m, on=["date", "code"], how="left")
    else:
        logger.info("No ROF data merged.")

    logger.info(f" Number of nan values after merging ROF: {data.isna().sum()}")

    # Run debug analysis if requested
    if debug:
        debug_snow_data_alignment(data, static_data, config)

    if "dayofyear_x" in data.columns:
        data.drop(columns=["dayofyear_x"], inplace=True)
    if "dayofyear_y" in data.columns:
        data.drop(columns=["dayofyear_y"], inplace=True)

    return data, static_data


def debug_snow_data_alignment(
    data: pd.DataFrame,
    static_data: pd.DataFrame,
    config: Dict[str, Any],
    debug_output_dir: str = "debug_snow_analysis",
) -> None:
    """
    Debug function to analyze SWE/discharge alignment and temporal patterns.

    Args:
        data: Merged DataFrame with discharge, forcing, and snow data
        static_data: Static basin characteristics
        config: Configuration dictionary
        debug_output_dir: Directory to save debug outputs
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from pathlib import Path

    # Create output directory
    output_dir = Path(debug_output_dir)
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("STARTING SNOW DATA ALIGNMENT DEBUG ANALYSIS")
    logger.info("=" * 60)

    # Get SWE and ROF column names
    swe_cols = [col for col in data.columns if "SWE" in col]
    rof_cols = [col for col in data.columns if "ROF" in col]

    logger.info(f"SWE columns found: {swe_cols}")
    logger.info(f"ROF columns found: {rof_cols}")

    # 1. Overall data quality assessment
    logger.info("\n1. OVERALL DATA QUALITY ASSESSMENT")
    logger.info("-" * 40)

    total_records = len(data)
    discharge_missing = data["discharge"].isna().sum()

    logger.info(f"Total records: {total_records:,}")
    logger.info(
        f"Discharge missing: {discharge_missing:,} ({discharge_missing / total_records * 100:.1f}%)"
    )

    if swe_cols:
        for swe_col in swe_cols:
            swe_missing = data[swe_col].isna().sum()
            logger.info(
                f"{swe_col} missing: {swe_missing:,} ({swe_missing / total_records * 100:.1f}%)"
            )

    if rof_cols:
        for rof_col in rof_cols:
            rof_missing = data[rof_col].isna().sum()
            logger.info(
                f"{rof_col} missing: {rof_missing:,} ({rof_missing / total_records * 100:.1f}%)"
            )

    # 2. Temporal coverage analysis
    logger.info("\n2. TEMPORAL COVERAGE ANALYSIS")
    logger.info("-" * 40)

    date_range = data["date"].agg(["min", "max"])
    logger.info(
        f"Data time range: {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}"
    )

    # Recent 2-year window analysis
    recent_cutoff = data["date"].max() - pd.DateOffset(years=2)
    recent_data = data[data["date"] >= recent_cutoff]

    logger.info(f"Recent 2-year window: {len(recent_data):,} records")
    logger.info(f"Recent discharge missing: {recent_data['discharge'].isna().sum()}")

    if swe_cols:
        for swe_col in swe_cols:
            recent_swe_missing = recent_data[swe_col].isna().sum()
            logger.info(f"Recent {swe_col} missing: {recent_swe_missing}")

    # 3. Basin-level coverage analysis
    logger.info("\n3. BASIN-LEVEL COVERAGE ANALYSIS")
    logger.info("-" * 40)

    basin_coverage = []
    for code in data["code"].unique():
        basin_data = data[data["code"] == code]
        recent_basin_data = recent_data[recent_data["code"] == code]

        coverage_stats = {
            "code": code,
            "total_records": len(basin_data),
            "recent_records": len(recent_basin_data),
            "discharge_coverage": 1 - basin_data["discharge"].isna().mean(),
            "recent_discharge_coverage": 1
            - recent_basin_data["discharge"].isna().mean()
            if len(recent_basin_data) > 0
            else 0,
        }

        if swe_cols:
            for swe_col in swe_cols:
                coverage_stats[f"{swe_col}_coverage"] = (
                    1 - basin_data[swe_col].isna().mean()
                )
                coverage_stats[f"recent_{swe_col}_coverage"] = (
                    1 - recent_basin_data[swe_col].isna().mean()
                    if len(recent_basin_data) > 0
                    else 0
                )

        basin_coverage.append(coverage_stats)

    coverage_df = pd.DataFrame(basin_coverage)

    # Select 5 representative basins for detailed analysis
    logger.info("\n4. SELECTING REPRESENTATIVE BASINS FOR DETAILED ANALYSIS")
    logger.info("-" * 40)

    # Basin selection strategy
    if swe_cols:
        main_swe_col = swe_cols[0]  # Use first SWE column for selection

        # Sort basins by SWE coverage
        coverage_df_sorted = coverage_df.sort_values(
            f"{main_swe_col}_coverage", ascending=False
        )

        selected_basins = []
        # 2 basins with best SWE coverage
        selected_basins.extend(coverage_df_sorted.head(2)["code"].tolist())
        # 2 basins with worst SWE coverage (but some data)
        worst_with_data = coverage_df_sorted[
            coverage_df_sorted[f"{main_swe_col}_coverage"] > 0
        ].tail(2)
        selected_basins.extend(worst_with_data["code"].tolist())
        # 1 basin with medium coverage
        if len(coverage_df_sorted) > 4:
            middle_idx = len(coverage_df_sorted) // 2
            selected_basins.append(coverage_df_sorted.iloc[middle_idx]["code"])

        # Take only first 5 if we have more
        selected_basins = selected_basins[:5]
    else:
        # If no SWE data, just select first 5 basins
        selected_basins = coverage_df.head(5)["code"].tolist()

    logger.info(f"Selected basins for detailed analysis: {selected_basins}")

    # Log coverage stats for selected basins
    for basin_code in selected_basins:
        basin_stats = coverage_df[coverage_df["code"] == basin_code].iloc[0]
        logger.info(f"Basin {basin_code}:")
        logger.info(f"  Discharge coverage: {basin_stats['discharge_coverage']:.2%}")
        if swe_cols:
            for swe_col in swe_cols:
                logger.info(
                    f"  {swe_col} coverage: {basin_stats[f'{swe_col}_coverage']:.2%}"
                )
        logger.info(
            f"  Recent discharge coverage: {basin_stats['recent_discharge_coverage']:.2%}"
        )
        if swe_cols:
            for swe_col in swe_cols:
                logger.info(
                    f"  Recent {swe_col} coverage: {basin_stats[f'recent_{swe_col}_coverage']:.2%}"
                )

    # 5. Create time series plots for selected basins
    logger.info("\n5. GENERATING TIME SERIES PLOTS")
    logger.info("-" * 40)

    plt.style.use("default")
    fig, axes = plt.subplots(
        len(selected_basins), 1, figsize=(15, 4 * len(selected_basins))
    )
    if len(selected_basins) == 1:
        axes = [axes]

    for i, basin_code in enumerate(selected_basins):
        basin_data = data[data["code"] == basin_code].sort_values("date")

        ax = axes[i]

        # Plot discharge
        ax.plot(
            basin_data["date"],
            basin_data["discharge"],
            label="Discharge",
            color="blue",
            alpha=0.7,
            linewidth=1,
        )

        # Plot SWE if available
        if swe_cols:
            ax2 = ax.twinx()
            for j, swe_col in enumerate(swe_cols):
                ax2.plot(
                    basin_data["date"],
                    basin_data[swe_col],
                    label=swe_col,
                    color=f"C{j + 1}",
                    alpha=0.7,
                    linewidth=1,
                )

        # Highlight recent 2-year window
        ax.axvline(
            x=recent_cutoff,
            color="red",
            linestyle="--",
            alpha=0.8,
            label="2-year cutoff",
        )

        # Formatting
        ax.set_title(f"Basin {basin_code} - Discharge vs SWE Time Series")
        ax.set_xlabel("Date")
        ax.set_ylabel("Discharge (mm/day)", color="blue")
        ax.tick_params(axis="y", labelcolor="blue")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

        if swe_cols:
            ax2.set_ylabel("SWE (mm)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")
            ax2.legend(loc="upper right")

        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plot_file = output_dir / "basin_timeseries_analysis.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Time series plot saved to: {plot_file}")

    # 6. Save coverage statistics
    coverage_file = output_dir / "basin_coverage_stats.csv"
    coverage_df.to_csv(coverage_file, index=False)
    logger.info(f"Coverage statistics saved to: {coverage_file}")

    # 7. Correlation analysis for recent period
    if swe_cols:
        logger.info("\n6. CORRELATION ANALYSIS (Recent 2-year period)")
        logger.info("-" * 40)

        corr_data = []
        for basin_code in selected_basins:
            basin_recent = recent_data[recent_data["code"] == basin_code]
            if len(basin_recent) > 10:  # Need sufficient data points
                for swe_col in swe_cols:
                    if (
                        not basin_recent[swe_col].isna().all()
                        and not basin_recent["discharge"].isna().all()
                    ):
                        corr = basin_recent["discharge"].corr(basin_recent[swe_col])
                        if not pd.isna(corr):
                            corr_data.append(
                                {
                                    "basin": basin_code,
                                    "swe_variable": swe_col,
                                    "correlation": corr,
                                    "n_points": len(
                                        basin_recent.dropna(
                                            subset=["discharge", swe_col]
                                        )
                                    ),
                                }
                            )

        if corr_data:
            corr_df = pd.DataFrame(corr_data)
            logger.info("\nDischarge-SWE Correlations (Recent Period):")
            for _, row in corr_df.iterrows():
                logger.info(
                    f"  Basin {row['basin']} - {row['swe_variable']}: r={row['correlation']:.3f} (n={row['n_points']})"
                )

            # Save correlation results
            corr_file = output_dir / "discharge_swe_correlations.csv"
            corr_df.to_csv(corr_file, index=False)
            logger.info(f"Correlation analysis saved to: {corr_file}")

    logger.info("\n" + "=" * 60)
    logger.info("SNOW DATA ALIGNMENT DEBUG ANALYSIS COMPLETED")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


def load_operational_configs(
    model_type: str, family_name: str, model_name: str
) -> Dict[str, Any]:
    """
    Load all configuration files for an operational model.

    Args:
        model_type: Type of model ('LR' or 'SciRegressor')
        model_name: Name of the model configuration

    Returns:
        Dictionary containing all configurations
    """
    # Determine configuration directory based on model type and name
    config_dir = Path(MODELS_DIR) / family_name / model_name

    if not config_dir.exists():
        # Try alternative structure
        raise FileNotFoundError(
            f"Configuration directory not found: {config_dir}. "
            "Please check the model family and name."
        )

    # Load required configuration files
    config_files = {
        "general_config": "general_config.json",
        "model_config": "model_config.json",
        "feature_config": "feature_config.json",
        "data_config": "data_config.json",
        "path_config": "data_paths.json",
    }

    configs = {}

    for config_name, config_file in config_files.items():
        config_path = config_dir / config_file

        if config_path.exists():
            with open(config_path, "r") as f:
                configs[config_name] = json.load(f)
            logger.info(f"Loaded {config_name} from {config_path}")
        else:
            logger.warning(f"Configuration file not found: {config_path}")
            configs[config_name] = {}

    # Set model type in general config
    if configs["general_config"]:
        configs["general_config"]["model_type"] = (
            "linear_regression" if model_type == "LR" else "sciregressor"
        )

    # we need to change the path where the load the base models if there are any
    if "path_to_lr_predictors" in configs["path_config"]:
        # the path have the format: ../monthly_forecasting_results/FAMILY/MODEL/PREDICTIONS.CSV
        # we need to change it to OUTPUT_DIR/FAMILY/MODEL/PREDICTIONS.CSV
        lr_paths = configs["path_config"]["path_to_lr_predictors"]
        for i, path in enumerate(lr_paths):
            # replace ../monthly_forecasting_results with MOCK_OUTPUT_DIR
            lr_paths[i] = path.replace(
                "../monthly_forecasting_results", MOCK_OUTPUT_DIR
            )

    return configs


def shift_data_to_current_year(
    data_df: pd.DataFrame, shift_years: int = 1
) -> pd.DataFrame:
    """
    Shift data dates by specified years to mock current year.

    Args:
        data_df: DataFrame with 'date' column
        shift_years: Number of years to shift forward

    Returns:
        DataFrame with shifted dates
    """
    data_df_shifted = data_df.copy()
    data_df_shifted["date"] = data_df_shifted["date"] + pd.DateOffset(years=shift_years)
    return data_df_shifted


def create_model_instance(
    model_type: str,
    model_name: str,
    configs: Dict[str, Any],
    data: pd.DataFrame,
    static_data: pd.DataFrame,
):
    """
    Create the appropriate model instance based on the model type.

    Args:
        model_type: 'LR' or 'SciRegressor'
        model_name: Name of the model configuration
        configs: All configuration dictionaries
        data: Time series data
        static_data: Static basin characteristics

    Returns:
        Model instance
    """
    general_config = configs["general_config"]
    model_config = configs["model_config"]
    feature_config = configs["feature_config"]
    path_config = configs["path_config"]

    # Set model name in general config
    general_config["model_name"] = model_name

    # Create model instance based on type
    if model_type == "LR":
        model = LinearRegressionModel(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    elif model_type == "SciRegressor":
        model = SciRegressor(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    elif model_type == "UncertaintyMixture":
        model = UncertaintyMixtureModel(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model


def calculate_average_discharge(
    data: pd.DataFrame, valid_from: str, valid_to: str
) -> pd.DataFrame:
    """
    Calculate average discharge for each basin from valid_from to valid_to period.

    Args:
        data: DataFrame with discharge data
        valid_from: Start date for averaging period
        valid_to: End date for averaging period

    Returns:
        DataFrame with basin codes and average discharge
    """
    # Filter data for the specified period
    mask = (data["date"] >= valid_from) & (data["date"] <= valid_to)
    period_data = data[mask]

    # count number of valid observations per basin
    valid_counts = period_data.groupby("code")["discharge"].count().reset_index()

    # if a basin has less than 25 valid observations, we skip it
    basin_to_nan = valid_counts[valid_counts["discharge"] < 25]["code"]

    # Calculate average discharge per basin
    avg_discharge = period_data.groupby("code")["discharge"].mean().reset_index()
    avg_discharge.columns = ["code", "observed_avg_discharge"]

    avg_discharge.loc[
        avg_discharge["code"].isin(basin_to_nan), "observed_avg_discharge"
    ] = np.nan

    return avg_discharge


def run_operational_prediction(
    today: datetime.datetime = None, debug: bool = False
) -> Dict[str, Any]:
    """
    Main operational prediction workflow.

    Args:
        today (datetime.datetime, optional): Date to use as "today" for prediction.
            If None, uses current datetime.
        debug (bool, optional): Enable snow data debugging analysis.

    Returns:
        Dictionary with all results and metrics
    """
    if today is None:
        today = datetime.datetime.now()

    logger.info(
        f"Starting operational prediction workflow for {today.strftime('%Y-%m-%d')}..."
    )

    # Initialize results storage
    all_predictions = {}
    all_metrics = []
    timing_results = {}

    ensemble_predictions = None

    # Start overall timing
    overall_start_time = datetime.datetime.now()

    # Process each model family
    for family_name, models in MODELS_OPERATIONAL.items():
        logger.info(f"Processing model family: {family_name}")

        for model_type, model_name in models:
            logger.info(f"-" * 50)
            logger.info(f"-" * 50)
            logger.info(f"Processing model: {model_type} - {model_name}")

            # Start model timing
            model_start_time = datetime.datetime.now()

            try:
                # Load configurations
                configs = load_operational_configs(
                    model_type, family_name=family_name, model_name=model_name
                )

                # Load and prepare data
                # Use environment variables for data loading since configs might not have data_config
                data, static_data = create_data_frame(
                    config=configs["path_config"], debug=debug
                )

                logger.info(f"Data columns after loading: {data.columns.tolist()}")
                # DEBUG: Check available basins in data
                available_basins = set(data["code"].unique())
                logger.info(
                    f"Available basins in data for {model_name}: {len(available_basins)}"
                )

                # Shift data to current year for operational prediction
                data = shift_data_to_current_year(data)
                today_dt = pd.to_datetime(today.date(), format="%Y-%m-%d")

                today_plus_15 = today_dt + pd.DateOffset(days=15)

                data_model = data[data["date"] <= today_plus_15].copy()
                # set discharge to nan for dates after today
                data_model.loc[data_model["date"] > today_dt, "discharge"] = np.nan

                # Create model instance
                model = create_model_instance(
                    model_type, model_name, configs, data_model, static_data
                )

                # Run operational prediction
                raw_predictions = model.predict_operational(today=today)

                raw_predictions["date"] = raw_predictions["forecast_date"]

                # Save the model
                save_dir = Path(MOCK_OUTPUT_DIR) / family_name / model_name
                if not save_dir.exists():
                    save_dir.mkdir(parents=True, exist_ok=True)

                raw_predictions.to_csv(save_dir / f"predictions.csv", index=False)

                logger.info(
                    f"Raw predictions for {model_name}:\n{raw_predictions.head()}"
                )

                pred_cols = [
                    col for col in raw_predictions.columns if col.startswith("Q_")
                ]

                valid_from = raw_predictions["valid_from"].min()
                valid_to = raw_predictions["valid_to"].max()

                logger.info(f"Valid period for predictions: {valid_from} to {valid_to}")

                # use original data - we have "future data"
                observed_avg_discharge = calculate_average_discharge(
                    data=data, valid_from=valid_from, valid_to=valid_to
                )

                logger.info(
                    f"Unique Basins in predictions: {raw_predictions['code'].nunique()}"
                )
                logger.info(
                    f"Unique Basins in observed data: {observed_avg_discharge['code'].nunique()}"
                )

                matching_basin_codes = set(
                    raw_predictions["code"].unique()
                ).intersection(set(observed_avg_discharge["code"].unique()))
                logger.info(
                    "Precentage of matching basins: %.2f%%",
                    100
                    * len(matching_basin_codes)
                    / len(raw_predictions["code"].unique()),
                )

                logger.info(
                    f"Number on nan values in raw predictions: {raw_predictions.isna().sum()}"
                )
                logger.info(
                    f"Number on nan values in observed data: {observed_avg_discharge.isna().sum()}"
                )
                raw_predictions = raw_predictions.merge(
                    observed_avg_discharge, on="code", how="left"
                )

                logger.info(
                    f"Number of nan values after merging observed data: {raw_predictions.isna().sum()}"
                )

                for pred_col in pred_cols:
                    exact_type = pred_col.split("_")[-1]
                    this_model_name = f"{family_name}_{model_name}_{exact_type}"

                    all_predictions[this_model_name] = raw_predictions[
                        [
                            "code",
                            pred_col,
                            "valid_from",
                            "valid_to",
                            "observed_avg_discharge",
                        ]
                    ].rename(columns={pred_col: f"Q_pred"})

                    if ensemble_predictions is None:
                        ensemble_predictions = all_predictions[this_model_name].copy()
                    else:
                        ensemble_predictions = pd.merge(
                            ensemble_predictions,
                            all_predictions[this_model_name],
                            on=["code"],
                            how="outer",
                            suffixes=("", f"_{this_model_name}"),
                        )

            except Exception as e:
                logger.error(f"Error processing {model_type} - {model_name}: {str(e)}")
                logger.error(f"Full traceback:\n{traceback.format_exc()}")
                continue

    ensemble_pred_cols = [
        col for col in ensemble_predictions.columns if col.startswith("Q_pred")
    ]

    ensemble_predictions["Q_Ensemble"] = ensemble_predictions[ensemble_pred_cols].mean(
        axis=1, skipna=True
    )

    ensemble_predictions.drop(columns=ensemble_pred_cols, inplace=True)

    all_predictions["Ensemble"] = ensemble_predictions.rename(
        columns={"Q_Ensemble": "Q_pred"}
    )

    # Calculate overall timing
    overall_end_time = datetime.datetime.now()
    overall_duration = (overall_end_time - overall_start_time).total_seconds()
    timing_results["overall_duration"] = overall_duration

    metrics_df = pd.DataFrame()
    for model_name, predictions in all_predictions.items():
        if predictions.empty:
            logger.warning(f"No predictions for model: {model_name}")
            continue

        # Calculate metrics for this model
        metrics, nan_basins = calculate_metrics(predictions)

        # for basin_code in nan_basins:
        #    plot_time_series(data=data, code=basin_code)

        if metrics.empty:
            logger.warning(f"No valid metrics calculated for model: {model_name}")
            continue

        # Add model name to metrics
        metrics["model"] = model_name

        # Append to overall metrics DataFrame
        metrics_df = pd.concat([metrics_df, metrics], ignore_index=True)

    logger.info(f"Completed operational prediction workflow in {overall_duration:.2f}s")

    return {
        "predictions": all_predictions,
        "timing": timing_results,
        "metrics": metrics_df,
    }


def relative_error(predicted: pd.Series, observed: pd.Series) -> float:
    """
    Calculate the absolute relative error between predicted and observed values.

    Args:
        predicted (pd.Series): Predicted values.
        observed (pd.Series): Observed values.

    Returns:
        pd.Series: Relative error values.
    """
    return np.abs((predicted - observed) / (observed + 1e-4))


def absolute_mean_error(predicted: pd.Series, observed: pd.Series) -> float:
    """
    Calculate the absolute mean error between predicted and observed values.

    Args:
        predicted (pd.Series): Predicted values.
        observed (pd.Series): Observed values.

    Returns:
        float: Absolute mean error.
    """
    return np.mean(np.abs(predicted - observed))


def calculate_metrics(
    df: pd.DataFrame,
    pred_col: str = "Q_pred",
    obs_col: str = "observed_avg_discharge",
) -> pd.DataFrame:
    """
    Calculate performance metrics for each basin in the DataFrame.

    Args:
        df: DataFrame containing predictions and observations
        pred_col: Column name for predictions
        obs_col: Column name for observations

    Returns:
        DataFrame with metrics for each basin
    """
    metrics_df = pd.DataFrame()
    nan_basins = []

    for code in df["code"].unique():
        basin_data = df[df["code"] == code].copy()

        if basin_data.empty:
            continue

        # Get prediction and observation values
        predicted = basin_data[pred_col].iloc[0] if len(basin_data) > 0 else np.nan
        observed = basin_data[obs_col].iloc[0] if len(basin_data) > 0 else np.nan

        # Skip if either value is NaN
        if pd.isna(predicted) or pd.isna(observed):
            logger.warning(
                f"Skipping basin {code}: predicted={predicted}, observed={observed}"
            )
            nan_basins.append(code)
            continue

        # Calculate relative error (scalar value)
        relative_error_value = relative_error(
            pd.Series([predicted]), pd.Series([observed])
        ).iloc[0]

        # Calculate absolute mean error (scalar value)
        abs_mean_error = absolute_mean_error(
            pd.Series([predicted]), pd.Series([observed])
        )

        # Create metrics row with scalar values
        metrics_row = {
            "code": code,
            "abs_mean_error": abs_mean_error,
            "relative_error_basin": relative_error_value,
        }

        metrics_row = pd.DataFrame([metrics_row], index=[0])
        metrics_df = pd.concat([metrics_df, metrics_row], ignore_index=True)

    return metrics_df, nan_basins


def plot_metric_boxplot(
    metrics_df: pd.DataFrame,
    metric_col: str = "abs_mean_error",
):
    import seaborn as sns
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=metrics_df,
        x="model",
        y=metric_col,
        hue="model",
        palette="Set2",
    )

    plt.title(f"Boxplot of {metric_col} by Model")
    plt.xlabel("Model")
    plt.ylabel(metric_col.replace("_", " ").title())
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


def print_performance_model(
    df: pd.DataFrame, model_name: str, metric_col: str = "relative_error_basin"
) -> None:
    """
    Print performance metrics for a specific model.

    Args:
        df (pd.DataFrame): DataFrame containing metrics.
        model_name (str): Name of the model to filter by.
    """
    model_metrics = df[df["model"] == model_name].copy()

    if model_metrics.empty:
        logger.warning(f"No metrics found for model: {model_name}")
        return

    logger.info(f"-" * 50)
    logger.info(f"Performance metrics for model: {model_name}")

    # sort by the metric column
    model_metrics = model_metrics.sort_values(by=metric_col, ascending=True)

    # print the top 10 basins with the lowest relative error
    logger.info(f"Top 10 basins with lowest {metric_col}:")
    logger.info(model_metrics.head(10))

    # print the worst 10 basins with the highest relative error
    logger.info(f"Worst 10 basins with highest {metric_col}:")
    logger.info(model_metrics.tail(10))

    mean_metric = model_metrics[metric_col].mean()
    median_metric = model_metrics[metric_col].median()
    max_metric = model_metrics[metric_col].max()
    min_metric = model_metrics[metric_col].min()
    std_metric = model_metrics[metric_col].std()

    logger.info(f"Performance metrics for model {model_name}:")
    logger.info(f"Mean {metric_col}: {mean_metric:.4f}")
    logger.info(f"Median {metric_col}: {median_metric:.4f}")
    logger.info(f"Max {metric_col}: {max_metric:.4f}")
    logger.info(f"Min {metric_col}: {min_metric:.4f}")
    logger.info(f"Standard Deviation {metric_col}: {std_metric:.4f}")

    logger.info(f"-" * 50)


def calculate_aggregated_metrics(all_metrics: list[pd.DataFrame]) -> dict[str, Any]:
    """
    Calculate aggregated performance metrics across all evaluation periods.

    Args:
        all_metrics: List of DataFrame objects containing metrics for each period

    Returns:
        Dictionary with aggregated metrics by model
    """
    if not all_metrics:
        return {}

    # Combine all metrics DataFrames
    combined_metrics = pd.concat(all_metrics, ignore_index=True)

    # Calculate aggregated statistics by model
    aggregated = {}
    for model in combined_metrics["model"].unique():
        model_data = combined_metrics[combined_metrics["model"] == model]

        aggregated[model] = {
            "count": len(model_data),
            "mean_relative_error": model_data["relative_error_basin"].mean(),
            "median_relative_error": model_data["relative_error_basin"].median(),
            "std_relative_error": model_data["relative_error_basin"].std(),
            "mean_abs_error": model_data["abs_mean_error"].mean(),
            "median_abs_error": model_data["abs_mean_error"].median(),
            "std_abs_error": model_data["abs_mean_error"].std(),
        }

    return aggregated


def generate_synthetic_evaluation_report(
    all_metrics: list[pd.DataFrame],
    aggregated_metrics: dict[str, Any],
    evaluation_dates: list[datetime.datetime],
    output_path: Path,
) -> None:
    """
    Generate a comprehensive evaluation report.

    Args:
        all_metrics: List of metrics DataFrames
        aggregated_metrics: Aggregated performance metrics
        evaluation_dates: List of evaluation dates
        output_path: Directory to save the report
    """
    report_file = output_path / "synthetic_evaluation_report.txt"

    with open(report_file, "w") as f:
        f.write("SYNTHETIC EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Evaluation Period: {len(evaluation_dates)} dates\n")
        f.write(f"Date Range: {min(evaluation_dates)} to {max(evaluation_dates)}\n\n")

        f.write("AGGREGATED PERFORMANCE METRICS\n")
        f.write("-" * 30 + "\n")

        for model, metrics in aggregated_metrics.items():
            f.write(f"\nModel: {model}\n")
            f.write(f"  Predictions: {metrics['count']}\n")
            f.write(f"  Mean Relative Error: {metrics['mean_relative_error']:.4f}\n")
            f.write(
                f"  Median Relative Error: {metrics['median_relative_error']:.4f}\n"
            )
            f.write(f"  Mean Absolute Error: {metrics['mean_abs_error']:.4f}\n")

    logger.info(f"Saved evaluation report to {report_file}")


def generate_plots(
    combined_predictions: pd.DataFrame,
    all_metrics: list[pd.DataFrame],
    output_path: Path,
    target_basins: list[int] | None = None,
    create_plots: bool = True,
) -> None:
    """
    Generate performance plots for synthetic evaluation.

    Args:
        combined_predictions: Combined prediction results
        all_metrics: List of metrics DataFrames
        output_path: Directory to save plots
        target_basins: Specific basins to plot
        create_plots: Whether to create plots
    """
    if not create_plots:
        logger.info("Skipping plot generation")
        return

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        # Set up plotting style
        plt.style.use("default")
        sns.set_palette("husl")

        # Combine all metrics for plotting
        combined_metrics = (
            pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
        )

        if combined_metrics.empty:
            logger.warning("No metrics data available for plotting")
            return

        # Create plots directory
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)

        logger.info(f"Generating individual plots in {plots_dir}")

        # Plot 1: Relative error by model (boxplot)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=combined_metrics, x="model", y="relative_error_basin")
        plt.xticks(rotation=45, ha="right")
        plt.title(
            "Relative Error Distribution by Model", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Relative Error", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = plots_dir / "relative_error_by_model.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved relative error boxplot to {plot_file}")

        # Plot 2: Absolute error by model (boxplot)
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=combined_metrics, x="model", y="abs_mean_error")
        plt.xticks(rotation=45, ha="right")
        plt.title(
            "Absolute Error Distribution by Model", fontsize=14, fontweight="bold"
        )
        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Absolute Mean Error", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = plots_dir / "absolute_error_by_model.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved absolute error boxplot to {plot_file}")

        # Plot 3: Error distribution histograms
        plt.figure(figsize=(14, 8))
        models = combined_metrics["model"].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

        for i, (model, color) in enumerate(zip(models, colors)):
            model_data = combined_metrics[combined_metrics["model"] == model]
            plt.hist(
                model_data["relative_error_basin"],
                alpha=0.7,
                label=model,
                bins=30,
                color=color,
                density=True,
            )

        plt.xlabel("Relative Error", fontsize=12)
        plt.ylabel("Density", fontsize=12)
        plt.title(
            "Relative Error Distribution by Model", fontsize=14, fontweight="bold"
        )
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plot_file = plots_dir / "error_distribution_histogram.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved error distribution histogram to {plot_file}")

        # Plot 4: Model performance summary (bar chart)
        plt.figure(figsize=(12, 8))
        model_stats = (
            combined_metrics.groupby("model")
            .agg(
                {
                    "relative_error_basin": ["mean", "std"],
                    "abs_mean_error": ["mean", "std"],
                }
            )
            .round(4)
        )

        # Flatten column names
        model_stats.columns = ["_".join(col).strip() for col in model_stats.columns]

        # Plot mean relative error with error bars
        x_pos = np.arange(len(model_stats.index))
        bars = plt.bar(
            x_pos,
            model_stats["relative_error_basin_mean"],
            yerr=model_stats["relative_error_basin_std"],
            capsize=5,
            alpha=0.8,
            color=colors[: len(model_stats)],
        )

        plt.xlabel("Model", fontsize=12)
        plt.ylabel("Mean Relative Error ± Std", fontsize=12)
        plt.title("Model Performance Comparison", fontsize=14, fontweight="bold")
        plt.xticks(x_pos, model_stats.index, rotation=45, ha="right")
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, mean_val, std_val in zip(
            bars,
            model_stats["relative_error_basin_mean"],
            model_stats["relative_error_basin_std"],
        ):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + std_val + 0.01,
                f"{mean_val:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        plot_file = plots_dir / "model_performance_summary.png"
        plt.savefig(plot_file, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved model performance summary to {plot_file}")

        # Plot 5: Basin 16936 specific analysis - Relative error vs evaluation date
        basin_16936_data = combined_metrics[combined_metrics["code"] == 16936].copy()

        if not basin_16936_data.empty:
            plt.figure(figsize=(14, 8))

            # Convert evaluation_date to datetime for proper plotting
            basin_16936_data["eval_date"] = pd.to_datetime(
                basin_16936_data["evaluation_date"]
            )

            # Plot each model separately
            for model in basin_16936_data["model"].unique():
                model_data = basin_16936_data[basin_16936_data["model"] == model]
                plt.plot(
                    model_data["eval_date"],
                    model_data["relative_error_basin"],
                    marker="o",
                    linewidth=2,
                    markersize=6,
                    label=model,
                    alpha=0.8,
                )

            plt.xlabel("Evaluation Date", fontsize=12)
            plt.ylabel("Relative Error", fontsize=12)
            plt.title(
                "Basin 16936: Relative Error vs Evaluation Date",
                fontsize=14,
                fontweight="bold",
            )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)

            # Add summary statistics
            overall_mean = basin_16936_data["relative_error_basin"].mean()
            plt.axhline(
                y=overall_mean,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"Overall Mean: {overall_mean:.3f}",
            )

            plt.tight_layout()

            plot_file = plots_dir / "basin_16936_error_timeseries.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved basin 16936 timeseries plot to {plot_file}")

            # Additional analysis for basin 16936
            logger.info(f"Basin 16936 Analysis:")
            logger.info(f"  Number of predictions: {len(basin_16936_data)}")
            logger.info(
                f"  Mean relative error: {basin_16936_data['relative_error_basin'].mean():.4f}"
            )
            logger.info(
                f"  Std relative error: {basin_16936_data['relative_error_basin'].std():.4f}"
            )
            logger.info(
                f"  Min relative error: {basin_16936_data['relative_error_basin'].min():.4f}"
            )
            logger.info(
                f"  Max relative error: {basin_16936_data['relative_error_basin'].max():.4f}"
            )

            # Best and worst performance dates
            best_idx = basin_16936_data["relative_error_basin"].idxmin()
            worst_idx = basin_16936_data["relative_error_basin"].idxmax()

            logger.info(
                f"  Best performance: {basin_16936_data.loc[best_idx, 'evaluation_date']} "
                f"(error: {basin_16936_data.loc[best_idx, 'relative_error_basin']:.4f}, "
                f"model: {basin_16936_data.loc[best_idx, 'model']})"
            )
            logger.info(
                f"  Worst performance: {basin_16936_data.loc[worst_idx, 'evaluation_date']} "
                f"(error: {basin_16936_data.loc[worst_idx, 'relative_error_basin']:.4f}, "
                f"model: {basin_16936_data.loc[worst_idx, 'model']})"
            )
        else:
            logger.warning("No data found for basin 16936")

        # Plot 6: Model comparison heatmap
        plt.figure(figsize=(12, 8))

        # Create pivot table for heatmap
        heatmap_data = (
            combined_metrics.groupby(["model", "evaluation_date"])[
                "relative_error_basin"
            ]
            .mean()
            .unstack()
        )

        if not heatmap_data.empty:
            sns.heatmap(
                heatmap_data,
                annot=False,
                cmap="RdYlBu_r",
                center=heatmap_data.mean().mean(),
                cbar_kws={"label": "Mean Relative Error"},
            )
            plt.title(
                "Model Performance Heatmap by Evaluation Date",
                fontsize=14,
                fontweight="bold",
            )
            plt.xlabel("Evaluation Date", fontsize=12)
            plt.ylabel("Model", fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()

            plot_file = plots_dir / "model_performance_heatmap.png"
            plt.savefig(plot_file, dpi=300, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved performance heatmap to {plot_file}")

        logger.info(f"All plots saved successfully in {plots_dir}")

    except ImportError as e:
        logger.warning(f"Required plotting libraries not available: {e}")
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        import traceback

        logger.error(f"Full traceback:\n{traceback.format_exc()}")


def print_synthetic_evaluation_summary(
    aggregated_metrics: dict[str, Any],
    evaluation_dates: list[datetime.datetime],
) -> None:
    """
    Print a summary of synthetic evaluation results.

    Args:
        aggregated_metrics: Aggregated performance metrics
        evaluation_dates: List of evaluation dates
    """
    logger.info("=" * 60)
    logger.info("SYNTHETIC EVALUATION SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"Evaluated {len(evaluation_dates)} dates from {min(evaluation_dates)} to {max(evaluation_dates)}"
    )

    if aggregated_metrics:
        logger.info("\nModel Performance Rankings (by mean relative error):")

        # Sort models by mean relative error
        sorted_models = sorted(
            aggregated_metrics.items(), key=lambda x: x[1]["mean_relative_error"]
        )

        for i, (model, metrics) in enumerate(sorted_models, 1):
            logger.info(
                f"{i:2d}. {model:20s} - "
                f"Rel.Err: {metrics['mean_relative_error']:.4f} ± {metrics['std_relative_error']:.4f}, "
                f"Abs.Err: {metrics['mean_abs_error']:.4f} ± {metrics['std_abs_error']:.4f} "
                f"({metrics['count']} predictions)"
            )

    logger.info("=" * 60)


def plot_time_series(
    data: pd.DataFrame,
    code: int,
) -> None:
    df_code = data[data["code"] == code].copy()
    if df_code.empty:
        logger.warning(f"No data available for basin code {code}")
        return

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_code["date"],
        df_code["discharge"],
        marker="o",
        linestyle="-",
        label="Discharge",
    )
    plt.title(f"Discharge Time Series for Basin {code}")
    plt.xlabel("Date")
    plt.ylabel("Discharge (m³/s)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def run_synthetic_evaluation(
    start_month: int = 4,
    end_month: int = 9,
    year: int = 2025,
    forecast_days: list = [10, 20, -1],  # -1 represents last day of month
    output_dir: str = "synthetic_test",
    target_basins: list = None,
    create_plots: bool = True,
) -> Dict[str, Any]:
    """
    Run synthetic evaluation over multiple dates to assess model performance.

    Args:
        start_month (int): Starting month (1-12)
        end_month (int): Ending month (1-12)
        year (int): Year for evaluation
        forecast_days (list): List of days to forecast on (-1 for last day)
        output_dir (str): Directory to save evaluation results
        target_basins (list): List of specific basin codes to plot (None for auto-selection)
        create_plots (bool): Whether to create interactive plots

    Returns:
        Dictionary with combined results and performance metrics
    """
    logger.info("=" * 60)
    logger.info("SYNTHETIC EVALUATION WORKFLOW")
    logger.info("=" * 60)
    logger.info(f"Evaluating from {start_month:02d}/{year} to {end_month:02d}/{year}")
    logger.info(f"Forecast days: {forecast_days}")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Initialize combined results
    all_predictions = []
    all_metrics = []
    evaluation_dates = []

    # Generate evaluation dates
    for month in range(start_month, end_month + 1):
        for day in forecast_days:
            if day == -1:
                # Last day of month
                last_day = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
                eval_date = datetime.datetime(year, month, last_day.day)
            else:
                # Specific day
                try:
                    eval_date = datetime.datetime(year, month, day)
                except ValueError:
                    # Skip invalid dates (e.g., Feb 30th)
                    continue

            evaluation_dates.append(eval_date)

    logger.info(f"Generated {len(evaluation_dates)} evaluation dates")

    # Run predictions for each date
    for i, eval_date in enumerate(evaluation_dates):
        logger.info(
            f"Processing date {i + 1}/{len(evaluation_dates)}: {eval_date.strftime('%Y-%m-%d')}"
        )

        try:
            # Run prediction for this date
            results = run_operational_prediction(today=eval_date)

            # Process predictions - results["predictions"] is a dict of DataFrames
            if results["predictions"]:
                period_predictions = []
                for model_name, pred_df in results["predictions"].items():
                    if not pred_df.empty:
                        pred_df_copy = pred_df.copy()
                        pred_df_copy["model"] = model_name
                        pred_df_copy["evaluation_date"] = eval_date.strftime("%Y-%m-%d")
                        period_predictions.append(pred_df_copy)

                if period_predictions:
                    period_combined = pd.concat(period_predictions, ignore_index=True)
                    all_predictions.append(period_combined)

            # Process metrics - results["metrics"] is already a DataFrame
            if not results["metrics"].empty:
                metrics_copy = results["metrics"].copy()
                metrics_copy["evaluation_date"] = eval_date.strftime("%Y-%m-%d")
                all_metrics.append(metrics_copy)

        except Exception as e:
            logger.error(
                f"Error processing date {eval_date.strftime('%Y-%m-%d')}: {str(e)}"
            )
            continue

    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        logger.info(
            f"Combined {len(combined_predictions)} predictions from {len(evaluation_dates)} dates"
        )
    else:
        combined_predictions = pd.DataFrame()
        logger.warning("No predictions generated during synthetic evaluation")

    # Save metrics
    if all_metrics:
        combined_metrics = pd.concat(all_metrics, ignore_index=True)
        metrics_file = output_path / "synthetic_metrics.csv"
        combined_metrics.to_csv(metrics_file, index=False)
        logger.info(f"Saved synthetic metrics to {metrics_file}")

    # Calculate aggregated performance metrics
    aggregated_metrics = calculate_aggregated_metrics(all_metrics)

    # Generate synthetic evaluation report
    generate_synthetic_evaluation_report(
        all_metrics, aggregated_metrics, evaluation_dates, output_path
    )

    # Save results
    if not combined_predictions.empty:
        predictions_file = output_path / "synthetic_predictions.csv"
        combined_predictions.to_csv(predictions_file, index=False)
        logger.info(f"Saved synthetic predictions to {predictions_file}")

    # Generate plots
    generate_plots(
        combined_predictions, all_metrics, output_path, target_basins, create_plots
    )

    # Print summary
    print_synthetic_evaluation_summary(aggregated_metrics, evaluation_dates)

    return {
        "predictions": combined_predictions,
        "metrics": all_metrics,
        "aggregated_metrics": aggregated_metrics,
        "evaluation_dates": evaluation_dates,
    }


def run_operational(today: datetime.datetime = None, debug: bool = False):
    """
    Main entry point for operational prediction workflow.

    Args:
        today (datetime.datetime, optional): Date to use as "today" for prediction.
            If None, uses current datetime.
        debug (bool, optional): Enable snow data debugging analysis.
    """
    try:
        logger.info("=" * 50)
        logger.info("OPERATIONAL PREDICTION WORKFLOW")
        logger.info("=" * 50)

        # Run the prediction workflow
        results = run_operational_prediction(today=today, debug=debug)

        # only compare basins which are present in all models
        if results["predictions"]:
            common_basin_codes = set.intersection(
                *[
                    set(pred_df["code"].unique())
                    for pred_df in results["predictions"].values()
                ]
            )
            logger.info(f"Common basins across all models: {len(common_basin_codes)}")

        else:
            logger.warning("No predictions available for operational workflow.")
            return

        logger.info(f"Common basin codes: {common_basin_codes}")
        results["metrics"] = results["metrics"][
            results["metrics"]["code"].isin(common_basin_codes)
        ]

        for model, model_metrics in results["metrics"].groupby("model"):
            print_performance_model(
                model_metrics, model_name=model, metric_col="relative_error_basin"
            )

            # print also the observed vs predicted values
            predictions = results["predictions"][model]
            if not predictions.empty:
                # sort by code
                predictions = predictions.sort_values(
                    by="code", ascending=False
                ).reset_index(drop=True)
                logger.info(f"Observed vs Predicted for model {model}:")
                logger.info(
                    predictions[["code", "Q_pred", "observed_avg_discharge"]].head(10)
                )

        plot_metric_boxplot(
            results["metrics"],
            metric_col="relative_error_basin",
        )

        logger.info("Operational prediction workflow completed successfully!")

    except Exception as e:
        logger.error(f"Error in operational prediction workflow: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run operational prediction workflow")
    parser.add_argument(
        "--mode",
        choices=["operational", "synthetic"],
        default="operational",
        help="Mode to run: operational (single date) or synthetic (multiple dates)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="../monthly_forecasting_results/synthetic_test",
        help="Output directory for synthetic evaluation",
    )
    parser.add_argument(
        "--target-basins",
        nargs="+",
        type=int,
        default=None,
        help="Specific basin codes to plot (default: auto-select top 10)",
    )
    parser.add_argument(
        "--create-plots",
        action="store_true",
        default=True,
        help="Create interactive HTML plots",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        default=False,
        help="Skip plot generation",
    )
    parser.add_argument(
        "--start-month",
        type=int,
        default=3,
        help="Starting month for synthetic evaluation (default: 4 for April)",
    )
    parser.add_argument(
        "--end-month",
        type=int,
        default=8,
        help="Ending month for synthetic evaluation (default: 9 for September)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year for synthetic evaluation (default: 2025)",
    )
    parser.add_argument(
        "--forecast-days",
        nargs="+",
        type=str,
        default=["-1"],
        help="Forecast days for synthetic evaluation (default: 10 20 -1)",
    )
    parser.add_argument(
        "--debug-snow",
        action="store_true",
        default=False,
        help="Enable snow data alignment debugging analysis",
    )

    args = parser.parse_args()

    # Convert forecast_days from strings to integers, handling -1 for end of month
    forecast_days = []
    for day in args.forecast_days:
        forecast_days.append(int(day))

    year = args.year
    start_month = args.start_month
    end_month = args.end_month

    # Parse today argument if provided
    today = datetime.datetime.now()  # Default to current date
    try:
        today = pd.to_datetime(
            today, format="%Y-%m-%d"
        )  # Ensure today is a datetime object
    except ValueError:
        logger.error(f"Invalid date format: {today}. Use YYYY-MM-DD format.")
        sys.exit(1)

    if args.mode == "operational":
        run_operational(today=today, debug=args.debug_snow)
    elif args.mode == "synthetic":
        create_plots = args.create_plots and not args.no_plots
        run_synthetic_evaluation(
            start_month=start_month,
            end_month=end_month,
            year=year,
            forecast_days=forecast_days,
            output_dir=args.output_dir,
            target_basins=args.target_basins,
            create_plots=create_plots,
        )
