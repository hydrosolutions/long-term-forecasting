#!/usr/bin/env python3
"""
Analyze MC_ALD Ensemble Behavior

Investigates whether the MC_ALD model uses ensemble spread/skew statistics
to make intelligent adjustments (e.g., shrinking toward climatology when
ensemble uncertainty is high).

Analysis questions:
1. Does prediction deviate from ensemble when spread is high?
2. When ensemble mean != median (skewed), does model correct?
3. Do spread statistics correlate with prediction behavior?

This helps determine if MC_ALD is doing something genuinely useful
or just passing through ensemble predictions.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# Import from examine_operational_fc.py (in same directory)
from examine_operational_fc import (
    load_predictions,
    load_observations,
    calculate_target,
    aggregate,
    kgz_path_config,
    taj_path_config,
    day_of_forecast,
    output_dir as default_output_dir,
)

# Import style configuration
from dev_tools.visualization.style_config import set_global_plot_style

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Base models that form the ensemble (excluding MC_ALD)
BASE_MODELS = [
    "LR_Base",
    "LR_SM",
    "SM_GBT",
    "SM_GBT_Norm",
    "SM_GBT_LR",
]

# Agricultural period months (April - September)
AGRICULTURAL_MONTHS = [4, 5, 6, 7, 8, 9]


def compute_ensemble_statistics(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ensemble statistics from base model predictions.

    For each (code, issue_date, horizon), computes:
    - ensemble_mean
    - ensemble_median
    - ensemble_std
    - ensemble_min
    - ensemble_max
    - ensemble_spread (max - min)
    - ensemble_skew

    Args:
        predictions_df: DataFrame with predictions from all models

    Returns:
        DataFrame with ensemble statistics per prediction instance
    """
    # Filter to base models only
    base_df = predictions_df[predictions_df["model"].isin(BASE_MODELS)].copy()

    if base_df.empty:
        logger.warning("No base model predictions found")
        return pd.DataFrame()

    # Pivot to wide format: each model as a column
    pivot_df = base_df.pivot_table(
        index=["code", "issue_date", "valid_from", "valid_to", "horizon"],
        columns="model",
        values="Q_pred",
        aggfunc="first",
    ).reset_index()

    # Get model columns that exist
    model_cols = [col for col in BASE_MODELS if col in pivot_df.columns]

    if not model_cols:
        logger.warning("No model columns found after pivot")
        return pd.DataFrame()

    logger.info(
        f"Computing ensemble statistics from {len(model_cols)} models: {model_cols}"
    )

    # Compute ensemble statistics
    pivot_df["ensemble_mean"] = pivot_df[model_cols].mean(axis=1)
    pivot_df["ensemble_median"] = pivot_df[model_cols].median(axis=1)
    pivot_df["ensemble_std"] = pivot_df[model_cols].std(axis=1)
    pivot_df["ensemble_min"] = pivot_df[model_cols].min(axis=1)
    pivot_df["ensemble_max"] = pivot_df[model_cols].max(axis=1)
    pivot_df["ensemble_spread"] = pivot_df["ensemble_max"] - pivot_df["ensemble_min"]
    pivot_df["ensemble_skew"] = pivot_df[model_cols].skew(axis=1)
    pivot_df["mean_median_diff"] = (
        pivot_df["ensemble_mean"] - pivot_df["ensemble_median"]
    )
    pivot_df["num_valid_models"] = pivot_df[model_cols].notna().sum(axis=1)

    # Drop individual model columns
    result_df = pivot_df.drop(columns=model_cols)

    logger.info(
        f"Computed ensemble statistics for {len(result_df)} prediction instances"
    )

    return result_df


def compute_climatology(monthly_obs: pd.DataFrame) -> pd.DataFrame:
    """
    Compute monthly climatology (mean discharge) for each basin.

    Args:
        monthly_obs: DataFrame with monthly observations (code, year, month, Q_obs_monthly)

    Returns:
        DataFrame with columns: code, month, climatology
    """
    climatology = (
        monthly_obs.groupby(["code", "month"])["Q_obs_monthly"]
        .mean()
        .reset_index()
        .rename(columns={"Q_obs_monthly": "climatology"})
    )

    logger.info(f"Computed climatology for {climatology['code'].nunique()} basins")

    return climatology


def merge_analysis_data(
    predictions_df: pd.DataFrame,
    ensemble_stats: pd.DataFrame,
    climatology: pd.DataFrame,
    monthly_obs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge MC_ALD predictions with ensemble statistics, climatology, and observations.

    Args:
        predictions_df: Full predictions DataFrame
        ensemble_stats: Ensemble statistics DataFrame
        climatology: Climatology DataFrame
        monthly_obs: Monthly observations DataFrame

    Returns:
        Merged DataFrame with all analysis columns
    """
    # Get MC_ALD predictions only
    mcald_df = predictions_df[predictions_df["model"] == "MC_ALD"].copy()

    if mcald_df.empty:
        logger.warning("No MC_ALD predictions found")
        return pd.DataFrame()

    logger.info(f"Found {len(mcald_df)} MC_ALD predictions")

    # Add target month
    mcald_df["target_month"] = mcald_df["valid_from"].dt.month

    # Merge with ensemble statistics
    merge_keys = ["code", "issue_date", "valid_from", "valid_to", "horizon"]
    analysis_df = mcald_df.merge(ensemble_stats, on=merge_keys, how="left")

    # Add year for climatology merge
    analysis_df["year"] = analysis_df["valid_from"].dt.year

    # Merge with observations
    analysis_df = analysis_df.merge(
        monthly_obs[["code", "year", "month", "Q_obs_monthly"]],
        left_on=["code", "year", "target_month"],
        right_on=["code", "year", "month"],
        how="left",
    )

    # Merge with climatology
    analysis_df = analysis_df.merge(
        climatology,
        left_on=["code", "target_month"],
        right_on=["code", "month"],
        how="left",
        suffixes=("", "_clim"),
    )

    # Compute analysis metrics
    # Use Q50 as the main prediction (median of ALD)
    if "Q50" in analysis_df.columns:
        pred_col = "Q50"
    else:
        pred_col = "Q_pred"
        logger.warning("Q50 not found, using Q_pred instead")

    analysis_df["pred_vs_ensemble"] = (
        analysis_df[pred_col] - analysis_df["ensemble_mean"]
    )
    analysis_df["pred_vs_clim"] = analysis_df[pred_col] - analysis_df["climatology"]
    analysis_df["ensemble_vs_clim"] = (
        analysis_df["ensemble_mean"] - analysis_df["climatology"]
    )

    # Normalized versions (by climatology)
    analysis_df["pred_vs_ensemble_norm"] = (
        analysis_df["pred_vs_ensemble"] / analysis_df["climatology"]
    )
    analysis_df["pred_vs_clim_norm"] = (
        analysis_df["pred_vs_clim"] / analysis_df["climatology"]
    )
    analysis_df["ensemble_spread_norm"] = (
        analysis_df["ensemble_spread"] / analysis_df["climatology"]
    )

    # Absolute distance metrics
    analysis_df["abs_pred_vs_clim"] = np.abs(analysis_df["pred_vs_clim"])
    analysis_df["abs_ensemble_vs_clim"] = np.abs(analysis_df["ensemble_vs_clim"])

    # Did MC_ALD move prediction closer to climatology than ensemble?
    analysis_df["moved_toward_clim"] = (
        analysis_df["abs_pred_vs_clim"] < analysis_df["abs_ensemble_vs_clim"]
    )

    logger.info(f"Final analysis DataFrame: {len(analysis_df)} rows")

    return analysis_df


def run_correlation_analysis(analysis_df: pd.DataFrame) -> dict:
    """
    Run correlation analysis to test hypotheses about MC_ALD behavior.

    Returns:
        Dictionary with correlation results
    """
    results = {}

    # Drop rows with NaN values in key columns
    cols_for_analysis = [
        "ensemble_spread",
        "ensemble_spread_norm",
        "pred_vs_clim",
        "pred_vs_clim_norm",
        "pred_vs_ensemble",
        "mean_median_diff",
        "ensemble_skew",
        "moved_toward_clim",
    ]
    df = analysis_df.dropna(
        subset=[c for c in cols_for_analysis if c in analysis_df.columns]
    )

    if df.empty:
        logger.warning("No valid data for correlation analysis")
        return results

    logger.info(f"\n{'=' * 80}")
    logger.info("CORRELATION ANALYSIS RESULTS")
    logger.info(f"{'=' * 80}")

    # 1. High spread → prediction closer to climatology?
    logger.info(
        "\n1. Does high ensemble spread lead to predictions closer to climatology?"
    )
    logger.info(
        "   (Hypothesis: When models disagree, MC_ALD shrinks toward climatology)"
    )

    corr_spread_clim = df[["ensemble_spread_norm", "pred_vs_clim_norm"]].corr()
    r_spread_clim = corr_spread_clim.iloc[0, 1]
    results["spread_vs_pred_clim_corr"] = r_spread_clim

    # Spearman correlation for robustness
    spearman_r, spearman_p = stats.spearmanr(
        df["ensemble_spread_norm"], np.abs(df["pred_vs_clim_norm"])
    )
    results["spread_vs_abs_pred_clim_spearman"] = spearman_r
    results["spread_vs_abs_pred_clim_pvalue"] = spearman_p

    logger.info(f"   Pearson r(spread_norm, pred_vs_clim_norm): {r_spread_clim:.4f}")
    logger.info(
        f"   Spearman r(spread_norm, |pred_vs_clim_norm|): {spearman_r:.4f} (p={spearman_p:.4e})"
    )

    # Percent of high-spread cases where MC_ALD moved toward climatology
    high_spread_mask = df["ensemble_spread_norm"] > df["ensemble_spread_norm"].median()
    pct_moved_high_spread = df.loc[high_spread_mask, "moved_toward_clim"].mean() * 100
    pct_moved_low_spread = df.loc[~high_spread_mask, "moved_toward_clim"].mean() * 100
    results["pct_moved_toward_clim_high_spread"] = pct_moved_high_spread
    results["pct_moved_toward_clim_low_spread"] = pct_moved_low_spread

    logger.info(
        f"   % predictions moved toward climatology (high spread): {pct_moved_high_spread:.1f}%"
    )
    logger.info(
        f"   % predictions moved toward climatology (low spread): {pct_moved_low_spread:.1f}%"
    )

    # 2. When mean != median (skewed ensemble), does model correct?
    logger.info("\n2. When ensemble is skewed (mean != median), does MC_ALD correct?")
    logger.info("   (Hypothesis: MC_ALD uses skew information to adjust predictions)")

    corr_skew_pred = df[["mean_median_diff", "pred_vs_ensemble"]].corr()
    r_skew_pred = corr_skew_pred.iloc[0, 1]
    results["mean_median_diff_vs_pred_deviation"] = r_skew_pred

    spearman_skew, p_skew = stats.spearmanr(
        df["mean_median_diff"], df["pred_vs_ensemble"]
    )
    results["mean_median_diff_vs_pred_spearman"] = spearman_skew
    results["mean_median_diff_vs_pred_pvalue"] = p_skew

    logger.info(f"   Pearson r(mean_median_diff, pred_vs_ensemble): {r_skew_pred:.4f}")
    logger.info(f"   Spearman r: {spearman_skew:.4f} (p={p_skew:.4e})")

    # 3. Ensemble skew correlation
    logger.info("\n3. Correlation between ensemble_skew and prediction adjustment:")

    corr_eskew_pred = df[["ensemble_skew", "pred_vs_ensemble"]].corr()
    r_eskew_pred = corr_eskew_pred.iloc[0, 1]
    results["ensemble_skew_vs_pred_deviation"] = r_eskew_pred

    logger.info(f"   Pearson r(ensemble_skew, pred_vs_ensemble): {r_eskew_pred:.4f}")

    # 4. Overall: Does MC_ALD beat ensemble on average?
    logger.info("\n4. Overall performance comparison:")

    # Mean absolute error vs climatology
    mae_mcald = np.abs(df["pred_vs_clim"]).mean()
    mae_ensemble = np.abs(df["ensemble_vs_clim"]).mean()
    improvement_pct = (mae_ensemble - mae_mcald) / mae_ensemble * 100
    results["mae_mcald_vs_clim"] = mae_mcald
    results["mae_ensemble_vs_clim"] = mae_ensemble
    results["improvement_pct"] = improvement_pct

    logger.info(f"   MAE (MC_ALD vs climatology): {mae_mcald:.2f}")
    logger.info(f"   MAE (Ensemble vs climatology): {mae_ensemble:.2f}")
    logger.info(f"   Improvement: {improvement_pct:.1f}%")

    # Also check against actual observations if available
    if "Q_obs_monthly" in df.columns:
        df_obs = df.dropna(subset=["Q_obs_monthly"])
        if not df_obs.empty:
            pred_col = "Q50" if "Q50" in df_obs.columns else "Q_pred"
            mae_mcald_obs = np.abs(df_obs[pred_col] - df_obs["Q_obs_monthly"]).mean()
            mae_ensemble_obs = np.abs(
                df_obs["ensemble_mean"] - df_obs["Q_obs_monthly"]
            ).mean()
            mae_clim_obs = np.abs(
                df_obs["climatology"] - df_obs["Q_obs_monthly"]
            ).mean()

            results["mae_mcald_vs_obs"] = mae_mcald_obs
            results["mae_ensemble_vs_obs"] = mae_ensemble_obs
            results["mae_clim_vs_obs"] = mae_clim_obs

            logger.info(f"\n   Against actual observations (n={len(df_obs)}):")
            logger.info(f"   MAE (MC_ALD): {mae_mcald_obs:.2f}")
            logger.info(f"   MAE (Ensemble): {mae_ensemble_obs:.2f}")
            logger.info(f"   MAE (Climatology): {mae_clim_obs:.2f}")

    # 5. Horizon-specific analysis: Does spread predict outcome magnitude?
    logger.info(
        "\n5. Horizon-specific: Does ensemble spread predict outcome magnitude?"
    )
    logger.info(
        "   (Testing if spread correlates with |observed - climatology| at each lead time)"
    )

    if "Q_obs_monthly" in df.columns and "horizon" in df.columns:
        df_obs = df.dropna(subset=["Q_obs_monthly", "climatology", "ensemble_spread"])

        if not df_obs.empty:
            # Compute absolute anomaly (deviation from climatology)
            df_obs = df_obs.copy()
            df_obs["abs_anomaly"] = np.abs(
                df_obs["Q_obs_monthly"] - df_obs["climatology"]
            )
            df_obs["abs_anomaly_norm"] = df_obs["abs_anomaly"] / df_obs["climatology"]

            horizons = sorted(df_obs["horizon"].unique())

            logger.info(
                "\n   Horizon | n     | r(spread, |anomaly|) | r(spread_norm, |anomaly_norm|)"
            )
            logger.info("   " + "-" * 70)

            for horizon in horizons:
                h_df = df_obs[df_obs["horizon"] == horizon]

                if len(h_df) < 10:
                    continue

                # Raw correlation
                corr_raw = h_df["ensemble_spread"].corr(h_df["abs_anomaly"])

                # Normalized correlation
                corr_norm = h_df["ensemble_spread_norm"].corr(h_df["abs_anomaly_norm"])

                # Spearman for robustness
                spearman_r, spearman_p = stats.spearmanr(
                    h_df["ensemble_spread_norm"], h_df["abs_anomaly_norm"]
                )

                results[f"spread_vs_anomaly_h{horizon}_pearson"] = corr_norm
                results[f"spread_vs_anomaly_h{horizon}_spearman"] = spearman_r
                results[f"spread_vs_anomaly_h{horizon}_n"] = len(h_df)

                logger.info(
                    f"   {horizon:7d} | {len(h_df):5d} | {corr_raw:20.3f} | {corr_norm:.3f} (spearman: {spearman_r:.3f}, p={spearman_p:.2e})"
                )

            # Also compute prediction error correlation with spread per horizon
            logger.info("\n   Does spread predict prediction error at each lead time?")
            logger.info(
                "   Horizon | r(spread, |pred_error|) | r(spread, |ensemble_error|)"
            )
            logger.info("   " + "-" * 65)

            pred_col = "Q50" if "Q50" in df_obs.columns else "Q_pred"

            for horizon in horizons:
                h_df = df_obs[df_obs["horizon"] == horizon].copy()

                if len(h_df) < 10:
                    continue

                h_df["abs_pred_error"] = np.abs(h_df[pred_col] - h_df["Q_obs_monthly"])
                h_df["abs_ensemble_error"] = np.abs(
                    h_df["ensemble_mean"] - h_df["Q_obs_monthly"]
                )

                corr_pred = h_df["ensemble_spread_norm"].corr(
                    h_df["abs_pred_error"] / h_df["climatology"]
                )
                corr_ens = h_df["ensemble_spread_norm"].corr(
                    h_df["abs_ensemble_error"] / h_df["climatology"]
                )

                results[f"spread_vs_pred_error_h{horizon}"] = corr_pred
                results[f"spread_vs_ens_error_h{horizon}"] = corr_ens

                logger.info(f"   {horizon:7d} | {corr_pred:24.3f} | {corr_ens:.3f}")

    logger.info(f"\n{'=' * 80}\n")

    return results


def create_analysis_plots(
    analysis_df: pd.DataFrame,
    output_dir: Path,
    agricultural_period: bool = False,
) -> None:
    """
    Create visualization plots for the analysis.

    Args:
        analysis_df: Analysis DataFrame
        output_dir: Directory to save plots
        agricultural_period: If True, filter to April-September only
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = analysis_df.copy()

    if agricultural_period:
        df = df[df["target_month"].isin(AGRICULTURAL_MONTHS)]
        suffix = "_agricultural"
    else:
        suffix = ""

    # Drop NaN for plotting
    plot_cols = [
        "ensemble_spread_norm",
        "pred_vs_clim_norm",
        "mean_median_diff",
        "pred_vs_ensemble",
        "ensemble_skew",
    ]
    df = df.dropna(subset=[c for c in plot_cols if c in df.columns])

    if df.empty:
        logger.warning("No data available for plotting")
        return

    # 1. Spread vs prediction deviation from climatology
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter plot
    ax = axes[0]
    ax.scatter(
        df["ensemble_spread_norm"],
        np.abs(df["pred_vs_clim_norm"]),
        alpha=0.3,
        s=20,
        c="steelblue",
    )
    # Add regression line
    slope, intercept, r, p, se = stats.linregress(
        df["ensemble_spread_norm"], np.abs(df["pred_vs_clim_norm"])
    )
    x_line = np.array([0, df["ensemble_spread_norm"].max()])
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r={r:.3f}")
    ax.set_xlabel("Ensemble Spread (normalized)", fontsize=11)
    ax.set_ylabel("|Prediction - Climatology| (normalized)", fontsize=11)
    ax.set_title("High Spread → Closer to Climatology?", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Boxplot by spread quartiles
    ax = axes[1]
    df["spread_quartile"] = pd.qcut(
        df["ensemble_spread_norm"], q=4, labels=["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"]
    )
    df["moved_toward_clim_int"] = df["moved_toward_clim"].astype(int) * 100
    sns.barplot(
        data=df,
        x="spread_quartile",
        y="moved_toward_clim_int",
        ax=ax,
        color="steelblue",
        errorbar=("ci", 95),
    )
    ax.set_xlabel("Ensemble Spread Quartile", fontsize=11)
    ax.set_ylabel("% Moved Toward Climatology", fontsize=11)
    ax.set_title(
        "Does High Spread → Move Toward Climatology?", fontsize=12, fontweight="bold"
    )
    ax.axhline(y=50, color="gray", linestyle="--", alpha=0.7)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"spread_vs_climatology{suffix}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info(f"Saved spread_vs_climatology{suffix}.png")

    # 2. Mean-median difference vs prediction adjustment
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(
        df["mean_median_diff"],
        df["pred_vs_ensemble"],
        alpha=0.3,
        s=20,
        c="steelblue",
    )
    # Regression line
    slope, intercept, r, p, se = stats.linregress(
        df["mean_median_diff"], df["pred_vs_ensemble"]
    )
    x_range = df["mean_median_diff"].max() - df["mean_median_diff"].min()
    x_line = np.array([df["mean_median_diff"].min(), df["mean_median_diff"].max()])
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r={r:.3f}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Ensemble Mean - Median", fontsize=11)
    ax.set_ylabel("MC_ALD Prediction - Ensemble Mean", fontsize=11)
    ax.set_title("Skew Detection: Does MC_ALD Correct?", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    # Ensemble skew vs prediction adjustment
    ax = axes[1]
    ax.scatter(
        df["ensemble_skew"],
        df["pred_vs_ensemble"],
        alpha=0.3,
        s=20,
        c="steelblue",
    )
    slope, intercept, r, p, se = stats.linregress(
        df["ensemble_skew"], df["pred_vs_ensemble"]
    )
    x_line = np.array([df["ensemble_skew"].min(), df["ensemble_skew"].max()])
    ax.plot(x_line, slope * x_line + intercept, "r-", linewidth=2, label=f"r={r:.3f}")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.7)
    ax.set_xlabel("Ensemble Skewness", fontsize=11)
    ax.set_ylabel("MC_ALD Prediction - Ensemble Mean", fontsize=11)
    ax.set_title("Skewness vs Prediction Adjustment", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(
        output_dir / f"skew_correction{suffix}.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    logger.info(f"Saved skew_correction{suffix}.png")

    # 3. Performance comparison by horizon
    if "horizon" in df.columns and "Q_obs_monthly" in df.columns:
        df_obs = df.dropna(subset=["Q_obs_monthly"])
        if not df_obs.empty:
            pred_col = "Q50" if "Q50" in df_obs.columns else "Q_pred"

            fig, ax = plt.subplots(figsize=(10, 6))

            horizons = sorted(df_obs["horizon"].unique())
            mae_mcald = []
            mae_ensemble = []
            mae_clim = []

            for h in horizons:
                h_df = df_obs[df_obs["horizon"] == h]
                mae_mcald.append(np.abs(h_df[pred_col] - h_df["Q_obs_monthly"]).mean())
                mae_ensemble.append(
                    np.abs(h_df["ensemble_mean"] - h_df["Q_obs_monthly"]).mean()
                )
                mae_clim.append(
                    np.abs(h_df["climatology"] - h_df["Q_obs_monthly"]).mean()
                )

            x = np.arange(len(horizons))
            width = 0.25

            ax.bar(x - width, mae_mcald, width, label="MC_ALD", color="steelblue")
            ax.bar(x, mae_ensemble, width, label="Ensemble", color="darkorange")
            ax.bar(x + width, mae_clim, width, label="Climatology", color="gray")

            ax.set_xlabel("Forecast Horizon (months)", fontsize=11)
            ax.set_ylabel("Mean Absolute Error", fontsize=11)
            ax.set_title("MAE Comparison by Horizon", fontsize=12, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(horizons)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            fig.savefig(
                output_dir / f"mae_by_horizon{suffix}.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig)
            logger.info(f"Saved mae_by_horizon{suffix}.png")

            # 3b. Spread vs anomaly/error correlation by horizon
            df_obs = df_obs.copy()
            df_obs["abs_anomaly"] = np.abs(
                df_obs["Q_obs_monthly"] - df_obs["climatology"]
            )
            df_obs["abs_anomaly_norm"] = df_obs["abs_anomaly"] / df_obs["climatology"]
            df_obs["abs_pred_error"] = np.abs(
                df_obs[pred_col] - df_obs["Q_obs_monthly"]
            )
            df_obs["abs_pred_error_norm"] = (
                df_obs["abs_pred_error"] / df_obs["climatology"]
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Correlation: spread vs |anomaly| by horizon
            corr_anomaly = []
            corr_error = []
            valid_horizons = []

            for h in horizons:
                h_df = df_obs[df_obs["horizon"] == h]
                if len(h_df) >= 10:
                    valid_horizons.append(h)
                    corr_anomaly.append(
                        h_df["ensemble_spread_norm"].corr(h_df["abs_anomaly_norm"])
                    )
                    corr_error.append(
                        h_df["ensemble_spread_norm"].corr(h_df["abs_pred_error_norm"])
                    )

            if valid_horizons:
                ax = axes[0]
                ax.bar(valid_horizons, corr_anomaly, color="steelblue", alpha=0.8)
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
                ax.set_xlabel("Forecast Horizon (months)", fontsize=11)
                ax.set_ylabel("Correlation", fontsize=11)
                ax.set_title(
                    "Spread vs |Observed - Climatology| by Horizon",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xticks(valid_horizons)
                ax.grid(axis="y", alpha=0.3)

                ax = axes[1]
                ax.bar(valid_horizons, corr_error, color="darkorange", alpha=0.8)
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
                ax.set_xlabel("Forecast Horizon (months)", fontsize=11)
                ax.set_ylabel("Correlation", fontsize=11)
                ax.set_title(
                    "Spread vs |Prediction Error| by Horizon",
                    fontsize=12,
                    fontweight="bold",
                )
                ax.set_xticks(valid_horizons)
                ax.grid(axis="y", alpha=0.3)

                plt.tight_layout()
                fig.savefig(
                    output_dir / f"spread_correlations_by_horizon{suffix}.png",
                    dpi=150,
                    bbox_inches="tight",
                )
                plt.close(fig)
                logger.info(f"Saved spread_correlations_by_horizon{suffix}.png")

    # 4. Summary correlation heatmap
    corr_cols = [
        "ensemble_spread_norm",
        "ensemble_std",
        "mean_median_diff",
        "ensemble_skew",
        "pred_vs_ensemble",
        "pred_vs_clim_norm",
    ]
    corr_cols = [c for c in corr_cols if c in df.columns]

    if len(corr_cols) >= 2:
        fig, ax = plt.subplots(figsize=(8, 6))
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdBu_r",
            center=0,
            ax=ax,
            square=True,
        )
        ax.set_title(
            "Correlation Matrix: Ensemble Stats vs Prediction Behavior",
            fontsize=12,
            fontweight="bold",
        )
        # Note: skip tight_layout() here as it conflicts with seaborn heatmap colorbar
        fig.savefig(
            output_dir / f"correlation_heatmap{suffix}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        logger.info(f"Saved correlation_heatmap{suffix}.png")

    # 5. Correction analysis: How far does Q50 deviate from ensemble mean?
    if "Q_obs_monthly" in df.columns and "horizon" in df.columns:
        df_corr = df.dropna(
            subset=["Q_obs_monthly", "ensemble_mean", "ensemble_spread", "climatology"]
        ).copy()

        if not df_corr.empty:
            pred_col = "Q50" if "Q50" in df_corr.columns else "Q_pred"
            df_corr["correction"] = df_corr[pred_col] - df_corr["ensemble_mean"]
            df_corr["correction_norm"] = df_corr["correction"] / df_corr["climatology"]
            df_corr["error_reduction"] = (
                df_corr["Q_obs_monthly"] - df_corr["ensemble_mean"]
            )

            # Select representative horizons
            all_horizons = sorted(df_corr["horizon"].unique())
            if len(all_horizons) >= 3:
                selected_horizons = [
                    all_horizons[0],
                    all_horizons[len(all_horizons) // 2],
                    all_horizons[-1],
                ]
            else:
                selected_horizons = all_horizons

            # Log correction statistics
            logger.info("\nCorrection Analysis (Q50 - ensemble_mean):")
            logger.info("-" * 60)
            for h in selected_horizons:
                h_df = df_corr[df_corr["horizon"] == h]
                if len(h_df) < 10:
                    continue
                logger.info(f"Horizon {h}:")
                logger.info(
                    f"  Mean |correction|: {h_df['correction'].abs().mean():.3f}"
                )
                logger.info(
                    f"  Correction vs spread: {h_df['correction'].corr(h_df['ensemble_spread']):.3f}"
                )
                logger.info(
                    f"  Correction vs error_reduction: {h_df['correction'].corr(h_df['error_reduction']):.3f}"
                )

            # Plot correction analysis
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            for i, h in enumerate(selected_horizons[:3]):
                ax = axes[i]
                h_df = df_corr[df_corr["horizon"] == h]

                if len(h_df) < 10:
                    ax.text(
                        0.5,
                        0.5,
                        f"Insufficient data\nHorizon {h}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Scatter: correction vs spread
                ax.scatter(
                    h_df["ensemble_spread_norm"],
                    h_df["correction_norm"],
                    alpha=0.3,
                    s=20,
                    c="steelblue",
                )

                # Regression line
                valid = h_df[["ensemble_spread_norm", "correction_norm"]].dropna()
                if len(valid) > 2:
                    slope, intercept, r, p, se = stats.linregress(
                        valid["ensemble_spread_norm"], valid["correction_norm"]
                    )
                    x_line = np.array([0, valid["ensemble_spread_norm"].max()])
                    ax.plot(
                        x_line,
                        slope * x_line + intercept,
                        "r-",
                        linewidth=2,
                        label=f"r={r:.3f}",
                    )
                    ax.legend(loc="upper right")

                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
                ax.set_xlabel("Ensemble Spread (norm)", fontsize=10)
                ax.set_ylabel("Correction (Q50 - Ensemble) / Clim", fontsize=10)
                ax.set_title(
                    f"Horizon {h}: Correction vs Spread", fontsize=11, fontweight="bold"
                )
                ax.grid(alpha=0.3)

            plt.tight_layout()
            fig.savefig(
                output_dir / f"correction_vs_spread{suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            logger.info(f"Saved correction_vs_spread{suffix}.png")

    # 6. Skill by spread quintiles: R² comparison in different spread bins
    if "Q_obs_monthly" in df.columns and "horizon" in df.columns:
        df_skill = df.dropna(
            subset=["Q_obs_monthly", "ensemble_mean", "ensemble_spread_norm"]
        ).copy()

        if not df_skill.empty:
            pred_col = "Q50" if "Q50" in df_skill.columns else "Q_pred"

            # Select representative horizons
            all_horizons = sorted(df_skill["horizon"].unique())
            if len(all_horizons) >= 3:
                selected_horizons = [
                    all_horizons[0],
                    all_horizons[len(all_horizons) // 2],
                    all_horizons[-1],
                ]
            else:
                selected_horizons = all_horizons

            fig, axes = plt.subplots(
                1, len(selected_horizons), figsize=(5 * len(selected_horizons), 5)
            )
            if len(selected_horizons) == 1:
                axes = [axes]

            for i, h in enumerate(selected_horizons):
                ax = axes[i]
                h_df = df_skill[df_skill["horizon"] == h].copy()

                if len(h_df) < 50:
                    ax.text(
                        0.5,
                        0.5,
                        f"Insufficient data\nHorizon {h}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Calculate overall mean for proper R² calculation
                # (must use overall mean, not bin mean, for SS_tot)
                overall_mean = h_df["Q_obs_monthly"].mean()

                # Create spread quintiles
                try:
                    h_df["spread_bin"] = pd.qcut(
                        h_df["ensemble_spread_norm"],
                        q=5,
                        labels=["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"],
                    )
                except ValueError:
                    # Not enough unique values for quintiles
                    h_df["spread_bin"] = pd.cut(
                        h_df["ensemble_spread_norm"],
                        bins=5,
                        labels=["Q1\n(low)", "Q2", "Q3", "Q4", "Q5\n(high)"],
                    )

                # Calculate R² for each bin
                r2_mcald = []
                r2_ensemble = []
                bin_labels = []

                for bin_label in h_df["spread_bin"].cat.categories:
                    bin_df = h_df[h_df["spread_bin"] == bin_label]
                    if len(bin_df) < 5:
                        continue

                    y_true = bin_df["Q_obs_monthly"].values
                    y_pred_mcald = bin_df[pred_col].values
                    y_pred_ens = bin_df["ensemble_mean"].values

                    # R² calculation (using overall mean for proper SS_tot)
                    ss_tot = np.sum((y_true - overall_mean) ** 2)
                    if ss_tot > 0:
                        ss_res_mcald = np.sum((y_true - y_pred_mcald) ** 2)
                        ss_res_ens = np.sum((y_true - y_pred_ens) ** 2)
                        r2_mcald.append(1 - ss_res_mcald / ss_tot)
                        r2_ensemble.append(1 - ss_res_ens / ss_tot)
                        bin_labels.append(bin_label)

                if bin_labels:
                    x = np.arange(len(bin_labels))
                    width = 0.35

                    ax.bar(
                        x - width / 2,
                        r2_mcald,
                        width,
                        label="MC_ALD",
                        color="steelblue",
                    )
                    ax.bar(
                        x + width / 2,
                        r2_ensemble,
                        width,
                        label="Ensemble",
                        color="darkorange",
                    )

                    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7)
                    ax.set_xlabel("Spread Quintile", fontsize=10)
                    ax.set_ylabel("R²", fontsize=10)
                    ax.set_title(
                        f"Horizon {h}: Skill by Spread", fontsize=11, fontweight="bold"
                    )
                    ax.set_xticks(x)
                    ax.set_xticklabels(bin_labels)
                    ax.legend(loc="lower left")
                    ax.grid(axis="y", alpha=0.3)

            plt.tight_layout()
            fig.savefig(
                output_dir / f"skill_by_spread_quintile{suffix}.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig)
            logger.info(f"Saved skill_by_spread_quintile{suffix}.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze MC_ALD ensemble behavior and spread/skew utilization."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["Kyrgyzstan", "Tajikistan"],
        default="Kyrgyzstan",
        help="Region to analyze (default: Kyrgyzstan)",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=list(range(9)),
        help="List of horizons to analyze (default: 0-8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--agricultural-period",
        action="store_true",
        help="Only analyze months April-September",
    )

    args = parser.parse_args()

    # Set global plot style
    set_global_plot_style()

    # Determine paths based on region
    if args.region == "Tajikistan":
        pred_config = taj_path_config
    else:
        pred_config = kgz_path_config

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(default_output_dir) / args.region.lower() / "mcald_analysis"

    # Convert horizons to horizon strings
    horizon_strings = [
        f"month_{h}" for h in args.horizons if f"month_{h}" in day_of_forecast
    ]

    logger.info(f"Analyzing MC_ALD behavior for {args.region}")
    logger.info(f"Horizons: {args.horizons}")
    if args.agricultural_period:
        logger.info("Agricultural period filter enabled (April-September only)")

    # Load predictions (all models)
    logger.info("Loading predictions...")
    predictions_df = load_predictions(
        base_path=pred_config["pred_dir"],
        horizons=horizon_strings,
    )

    if predictions_df.empty:
        logger.error("No predictions loaded. Exiting.")
        return

    logger.info(f"Loaded {len(predictions_df)} prediction records")
    logger.info(f"Available models: {predictions_df['model'].unique().tolist()}")

    # Check if MC_ALD is present
    if "MC_ALD" not in predictions_df["model"].unique():
        logger.error("MC_ALD model not found in predictions. Exiting.")
        return

    # Load observations
    logger.info(f"Loading observations from {pred_config['obs_file']}...")
    obs_df = load_observations(pred_config["obs_file"])
    monthly_obs = calculate_target(obs_df)

    # Compute ensemble statistics
    logger.info("Computing ensemble statistics...")
    ensemble_stats = compute_ensemble_statistics(predictions_df)

    if ensemble_stats.empty:
        logger.error("Failed to compute ensemble statistics. Exiting.")
        return

    # Compute climatology
    logger.info("Computing climatology...")
    climatology = compute_climatology(monthly_obs)

    # Merge all data for analysis
    logger.info("Merging analysis data...")
    analysis_df = merge_analysis_data(
        predictions_df=predictions_df,
        ensemble_stats=ensemble_stats,
        climatology=climatology,
        monthly_obs=monthly_obs,
    )

    if analysis_df.empty:
        logger.error("Failed to create analysis DataFrame. Exiting.")
        return

    # Filter for agricultural period if requested
    if args.agricultural_period:
        analysis_df = analysis_df[analysis_df["target_month"].isin(AGRICULTURAL_MONTHS)]
        logger.info(f"Filtered to agricultural period: {len(analysis_df)} rows")

    # Run correlation analysis
    logger.info("Running correlation analysis...")
    results = run_correlation_analysis(analysis_df)

    # Create visualization plots
    logger.info("Creating analysis plots...")
    create_analysis_plots(
        analysis_df=analysis_df,
        output_dir=output_dir,
        agricultural_period=args.agricultural_period,
    )

    # Save analysis results
    results_df = pd.DataFrame([results])
    results_path = output_dir / "analysis_results.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"Saved analysis results to {results_path}")

    # Save full analysis DataFrame for further investigation
    analysis_path = output_dir / "analysis_data.parquet"
    analysis_df.to_parquet(analysis_path, index=False)
    logger.info(f"Saved full analysis data to {analysis_path}")

    logger.info(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
