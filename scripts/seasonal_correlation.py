"""
Seasonal Correlation Analysis Script.

Analyzes correlations between predictors (discharge, P, T, SWE) at different lead times
and seasonal (Apr-Sep) mean discharge target.
"""

import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import ConstantInputWarning, pearsonr

warnings.filterwarnings("ignore", category=ConstantInputWarning)

load_dotenv()

# Configuration
PREDICTOR_CONFIGS = {
    "discharge": {
        "operation": "mean",
        "windows": [30],
    },
    "P": {
        "operation": "sum",
        "windows": [30, 60, 90],
    },
    "T": {
        "operation": "mean",
        "windows": [30, 60],
    },
    "SWE": {
        "operation": "mean",
        "windows": [10, 30],
    },
}

ANALYSIS_MONTHS = [10, 11, 12, 1, 2, 3, 4]  # Oct-Apr
TARGET_MONTHS = [4, 5, 6, 7, 8, 9]  # Apr-Sep
MONTH_LABELS = ["Oct", "Nov", "Dec", "Jan", "Feb", "Mar", "Apr"]


def get_output_dir() -> Path:
    base_dir = os.getenv("out_dir_op_lt")
    if base_dir is None:
        raise ValueError("Missing environment variable: out_dir_op_lt")
    return Path(base_dir) / "seasonal" / "correlations"


def load_discharge(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["code"] = df["code"].astype(int)
    return df[["date", "code", "discharge"]]


def load_forcing(path_dir: str, hru: str = "00003") -> pd.DataFrame:
    path_p = os.path.join(path_dir, f"{hru}_P_reanalysis.csv")
    path_t = os.path.join(path_dir, f"{hru}_T_reanalysis.csv")

    df_p = pd.read_csv(path_p)
    df_t = pd.read_csv(path_t)

    df_p["date"] = pd.to_datetime(df_p["date"], format="mixed")
    df_t["date"] = pd.to_datetime(df_t["date"], format="mixed")
    df_p["code"] = df_p["code"].astype(int)
    df_t["code"] = df_t["code"].astype(int)

    df = df_p[["date", "code", "P"]].merge(
        df_t[["date", "code", "T"]], on=["date", "code"], how="outer"
    )
    return df


def load_swe(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["code"] = df["code"].astype(int)
    return df[["date", "code", "SWE"]]


def load_predictor_data() -> pd.DataFrame:
    path_discharge = os.getenv("kgz_path_base_pred")
    path_forcing = os.getenv("PATH_TO_FORCING_ERA5")
    path_swe = os.getenv("path_SWE_00003")

    if not all([path_discharge, path_forcing, path_swe]):
        raise ValueError(
            "Missing environment variables: kgz_path_base_pred, "
            "PATH_TO_FORCING_ERA5, or path_SWE_00003"
        )

    discharge_df = load_discharge(path_discharge)
    forcing_df = load_forcing(path_forcing)
    swe_df = load_swe(path_swe)

    df = discharge_df.merge(forcing_df, on=["date", "code"], how="outer")
    df = df.merge(swe_df, on=["date", "code"], how="outer")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    return df


def calculate_seasonal_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    target_df = df[df["month"].isin(TARGET_MONTHS)].copy()
    target_df = (
        target_df.groupby(["code", "year"])["discharge"]
        .mean()
        .reset_index()
        .rename(columns={"discharge": "Q_target", "year": "target_year"})
    )
    return target_df


def get_month_end_dates(df: pd.DataFrame, month: int) -> pd.DataFrame:
    df = df.copy()
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    month_data = df[df["month"] == month].copy()
    month_ends = month_data.groupby(["code", "year"])["date"].max().reset_index()
    month_ends.rename(columns={"date": "month_end_date"}, inplace=True)
    return month_ends


def create_predictor_features_at_month_end(
    df: pd.DataFrame,
) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    results = []

    for month in ANALYSIS_MONTHS:
        month_ends = get_month_end_dates(df, month)

        for _, row in month_ends.iterrows():
            code = row["code"]
            year = row["year"]
            month_end_date = row["month_end_date"]

            # Determine target year based on month
            if month >= 10:  # Oct, Nov, Dec -> next year's target
                target_year = year + 1
            else:  # Jan-Apr -> same year's target
                target_year = year

            basin_df = df[df["code"] == code].set_index("date").sort_index()

            for predictor, config in PREDICTOR_CONFIGS.items():
                for window in config["windows"]:
                    # Get window of data ending at month_end_date
                    start_date = month_end_date - pd.Timedelta(days=window - 1)
                    window_data = basin_df.loc[start_date:month_end_date, predictor]

                    if len(window_data) < window * 0.8:  # Require 80% data
                        value = np.nan
                    elif config["operation"] == "mean":
                        value = window_data.mean()
                    elif config["operation"] == "sum":
                        value = window_data.sum()
                    else:
                        value = np.nan

                    results.append(
                        {
                            "code": code,
                            "target_year": target_year,
                            "month": month,
                            "predictor": predictor,
                            "window": window,
                            "value": value,
                        }
                    )

    return pd.DataFrame(results)


def calculate_predictor_correlations(
    features_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    merged = features_df.merge(target_df, on=["code", "target_year"], how="inner")

    results = []

    for code in merged["code"].unique():
        code_data = merged[merged["code"] == code]

        for predictor in PREDICTOR_CONFIGS:
            for window in PREDICTOR_CONFIGS[predictor]["windows"]:
                for month in ANALYSIS_MONTHS:
                    subset = code_data[
                        (code_data["predictor"] == predictor)
                        & (code_data["window"] == window)
                        & (code_data["month"] == month)
                    ]

                    x = subset["value"].dropna()
                    y = subset.loc[x.index, "Q_target"]

                    if len(x) >= 5:  # Require at least 5 data points
                        corr, _ = pearsonr(x, y)
                    else:
                        corr = np.nan

                    results.append(
                        {
                            "code": code,
                            "predictor": predictor,
                            "window": window,
                            "month": month,
                            "correlation": corr,
                        }
                    )

    return pd.DataFrame(results)


def plot_predictor_correlation_figure(
    corr_df: pd.DataFrame, predictor: str, output_path: Path
) -> None:
    config = PREDICTOR_CONFIGS[predictor]
    windows = config["windows"]
    n_windows = len(windows)

    fig, axes = plt.subplots(n_windows, 1, figsize=(10, 4 * n_windows), squeeze=False)

    pred_data = corr_df[corr_df["predictor"] == predictor]

    for i, window in enumerate(windows):
        ax = axes[i, 0]
        window_data = pred_data[pred_data["window"] == window]

        box_data = []
        for month in ANALYSIS_MONTHS:
            month_corrs = window_data[window_data["month"] == month]["correlation"]
            box_data.append(month_corrs.dropna().values)

        bp = ax.boxplot(box_data, labels=MONTH_LABELS, patch_artist=True)

        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylim(-1, 1)
        ax.set_ylabel("Pearson Correlation")
        ax.set_xlabel("Month (Lead Time)")
        ax.set_title(f"{predictor.upper()} - {window}d rolling window")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{predictor.upper()} Correlation with Apr-Sep Discharge\n"
        "(Distribution across basins)",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    output_dir = get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading predictor data...")
    df = load_predictor_data()
    print(f"  Loaded {len(df)} records, {df['code'].nunique()} basins")

    print("Calculating seasonal target (Apr-Sep mean discharge)...")
    target_df = calculate_seasonal_target(df)
    print(f"  Calculated targets for {len(target_df)} basin-years")

    print("Creating predictor features at month ends...")
    features_df = create_predictor_features_at_month_end(df)
    print(f"  Created {len(features_df)} feature records")

    print("Calculating correlations per basin...")
    corr_df = calculate_predictor_correlations(features_df, target_df)
    print(f"  Calculated {len(corr_df)} correlation values")

    print("Generating correlation figures...")
    for predictor in PREDICTOR_CONFIGS:
        output_path = output_dir / f"{predictor}_correlations.png"
        plot_predictor_correlation_figure(corr_df, predictor, output_path)
        print(f"  Saved {output_path.name}")

    print("\nDone! Figures saved to:", output_dir)


if __name__ == "__main__":
    main()
