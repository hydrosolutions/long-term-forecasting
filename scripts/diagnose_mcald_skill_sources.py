#!/usr/bin/env python3
"""
Diagnose MCALD Skill Sources

This script investigates whether the MCALD R² ≈ 0.2 plateau at long lead times
represents real forecasting skill or is an artifact of:
1. Q_obs statistics features providing climatology information directly
2. Contaminated baselines (climatology computed from all data including test)

Key diagnostic outputs:
- R² per (code, month) across years - the TRUE forecasting skill metric
- Comparison of MCALD R² vs proper out-of-sample climatology R²
- Skill decomposition by horizon

The proper evaluation approach:
1. For each (code, month) combination, compute R² across years
2. Average across all (code, month) combinations
This isolates the actual forecasting skill: predicting year-to-year deviations
from the expected seasonal pattern for each basin.

Interpretation:
- If MCALD R² - Climatology R² < 0.02 at long leads → skill is climatology artifact
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

# Add project root to path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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

# Models to analyze
MODELS_TO_ANALYZE = ["MC_ALD", "Ensemble"]

# Base models for ensemble
BASE_MODELS = [
    "LR_Base",
    "LR_SM",
    "SM_GBT",
    "SM_GBT_Norm",
    "SM_GBT_LR",
]


def compute_climatology_oos(
    monthly_obs: pd.DataFrame, test_years: List[int]
) -> pd.DataFrame:
    """
    Compute out-of-sample climatology excluding test years.

    This ensures the climatology baseline is not contaminated by test data.

    Args:
        monthly_obs: DataFrame with columns: code, year, month, Q_obs_monthly
        test_years: List of years to exclude (test set)

    Returns:
        DataFrame with columns: code, month, climatology (mean from train years only)
    """
    # Filter to training years only
    train_obs = monthly_obs[~monthly_obs["year"].isin(test_years)].copy()

    if train_obs.empty:
        logger.warning("No training data after excluding test years!")
        return pd.DataFrame()

    logger.info(
        f"Computing OOS climatology from {train_obs['year'].nunique()} training years "
        f"(excluded {len(test_years)} test years: {sorted(test_years)})"
    )

    # Compute mean per (code, month) from training data only
    climatology = (
        train_obs.groupby(["code", "month"])["Q_obs_monthly"]
        .mean()
        .reset_index()
        .rename(columns={"Q_obs_monthly": "climatology"})
    )

    logger.info(f"Computed climatology for {climatology['code'].nunique()} basins")

    return climatology


def compute_r2_per_code_month(
    df: pd.DataFrame, pred_col: str = "Q_pred", obs_col: str = "Q_obs"
) -> pd.DataFrame:
    """
    Compute R² for each (code, month) combination across years.

    This is the CORRECT approach for evaluating forecasting skill:
    - Removes inter-basin variance (different basins have different mean flows)
    - Removes seasonal variance (different months have different mean flows)
    - Measures only year-to-year prediction skill

    Args:
        df: DataFrame with predictions and observations, must have 'code' and 'target_month'
        pred_col: Column name for predictions
        obs_col: Column name for observations

    Returns:
        DataFrame with R² per (code, month): code, month, r2, n_samples
    """
    results = []

    for (code, month), group in df.groupby(["code", "target_month"]):
        valid = group.dropna(subset=[pred_col, obs_col])

        if len(valid) < 3:  # Need at least 3 years for meaningful R²
            continue

        y_true = valid[obs_col].values
        y_pred = valid[pred_col].values

        # R² calculation
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = np.nan

        results.append(
            {"code": code, "month": month, "r2": r2, "n_samples": len(valid)}
        )

    return pd.DataFrame(results)


def compute_global_r2(
    df: pd.DataFrame, pred_col: str = "Q_pred", obs_col: str = "Q_obs"
) -> float:
    """
    Compute global R² across all data pooled together.

    NOTE: This metric is inflated by inter-basin and inter-month variance!
    Only useful for understanding the scale of variance being explained.

    Args:
        df: DataFrame with predictions and observations
        pred_col: Column name for predictions
        obs_col: Column name for observations

    Returns:
        Global R² value
    """
    valid = df.dropna(subset=[pred_col, obs_col])

    if len(valid) < 2:
        return np.nan

    y_true = valid[obs_col].values
    y_pred = valid[pred_col].values

    return r2_score(y_true, y_pred)


def create_ensemble_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create simple ensemble mean from base models.

    Args:
        predictions_df: DataFrame with all model predictions

    Returns:
        DataFrame with ensemble predictions added
    """
    # Filter to base models only
    base_df = predictions_df[predictions_df["model"].isin(BASE_MODELS)].copy()

    if base_df.empty:
        logger.warning("No base model predictions found for ensemble")
        return predictions_df

    # Pivot to get each model's prediction as a column
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
        return predictions_df

    logger.info(f"Creating ensemble from {len(model_cols)} models: {model_cols}")

    # Compute ensemble mean
    pivot_df["Q_pred"] = pivot_df[model_cols].mean(axis=1)
    pivot_df["model"] = "Ensemble"

    # Get Q_obs from any base model
    q_obs_df = (
        base_df.groupby(["code", "issue_date", "valid_from", "valid_to", "horizon"])[
            "Q_obs"
        ]
        .first()
        .reset_index()
    )

    # Merge Q_obs
    ensemble_df = pivot_df.drop(columns=model_cols).merge(
        q_obs_df, on=["code", "issue_date", "valid_from", "valid_to", "horizon"]
    )

    # Add quantile columns as NaN
    quantile_cols = [
        col
        for col in predictions_df.columns
        if col.startswith("Q") and col[1:].isdigit()
    ]
    for col in quantile_cols:
        if col not in ensemble_df.columns:
            ensemble_df[col] = np.nan

    # Ensure same columns as original
    for col in predictions_df.columns:
        if col not in ensemble_df.columns:
            ensemble_df[col] = np.nan

    # Concatenate
    combined = pd.concat(
        [predictions_df, ensemble_df[predictions_df.columns]], ignore_index=True
    )

    logger.info(f"Created ensemble with {len(ensemble_df)} predictions")

    return combined


def diagnose_skill_sources(
    aggregated_df: pd.DataFrame,
    monthly_obs: pd.DataFrame,
    test_years: List[int],
    models: List[str] = MODELS_TO_ANALYZE,
) -> pd.DataFrame:
    """
    Main diagnostic function to compare skill sources by horizon.

    Computes R² per (code, month) across years - the TRUE forecasting skill.
    This removes both inter-basin and seasonal variance.

    Args:
        aggregated_df: DataFrame with merged predictions and observations
        monthly_obs: DataFrame with monthly observations
        test_years: Years used for testing (to exclude from climatology)
        models: List of models to analyze

    Returns:
        DataFrame with skill diagnostics by horizon and model
    """
    results = []

    # Compute out-of-sample climatology
    climatology = compute_climatology_oos(monthly_obs, test_years)

    if climatology.empty:
        logger.error("Failed to compute climatology")
        return pd.DataFrame()

    # Add target month for merging
    aggregated_df = aggregated_df.copy()
    aggregated_df["target_month"] = aggregated_df["valid_from"].dt.month

    # Merge climatology into aggregated data
    df_with_clim = aggregated_df.merge(
        climatology,
        left_on=["code", "target_month"],
        right_on=["code", "month"],
        how="left",
    )

    # Get unique horizons
    horizons = sorted(df_with_clim["horizon"].unique())

    logger.info(f"\nAnalyzing {len(horizons)} horizons: {horizons}")
    logger.info(f"Models to analyze: {models}")

    for horizon in horizons:
        horizon_df = df_with_clim[df_with_clim["horizon"] == horizon].copy()

        for model in models:
            model_df = horizon_df[horizon_df["model"] == model].copy()

            if model_df.empty:
                logger.warning(f"No data for model {model} at horizon {horizon}")
                continue

            # Get the prediction column (Q50 for MC_ALD, Q_pred for others)
            if model == "MC_ALD" and "Q50" in model_df.columns:
                pred_col = "Q50"
            else:
                pred_col = "Q_pred"

            # Drop rows with missing values
            valid_df = model_df.dropna(subset=[pred_col, "Q_obs", "climatology"])

            if len(valid_df) < 10:
                logger.warning(
                    f"Insufficient data for {model} at horizon {horizon}: {len(valid_df)} samples"
                )
                continue

            n_basins = valid_df["code"].nunique()
            n_months = valid_df["target_month"].nunique()
            n_samples = len(valid_df)

            # ----------------------------------------------------------------
            # 1. R² per (code, month) - THE CORRECT METRIC
            # ----------------------------------------------------------------
            # Model R² per (code, month)
            r2_code_month = compute_r2_per_code_month(
                valid_df, pred_col=pred_col, obs_col="Q_obs"
            )
            mean_r2_code_month = r2_code_month["r2"].mean()
            median_r2_code_month = r2_code_month["r2"].median()
            std_r2_code_month = r2_code_month["r2"].std()
            n_code_month_groups = len(r2_code_month)

            # Climatology R² per (code, month)
            clim_r2_code_month = compute_r2_per_code_month(
                valid_df, pred_col="climatology", obs_col="Q_obs"
            )
            mean_clim_r2 = clim_r2_code_month["r2"].mean()
            median_clim_r2 = clim_r2_code_month["r2"].median()

            # ----------------------------------------------------------------
            # 2. Skill improvement over climatology
            # ----------------------------------------------------------------
            skill_vs_clim_mean = mean_r2_code_month - mean_clim_r2
            skill_vs_clim_median = median_r2_code_month - median_clim_r2

            # ----------------------------------------------------------------
            # 3. Global R² for reference (inflated by inter-basin variance)
            # ----------------------------------------------------------------
            global_r2_model = compute_global_r2(
                valid_df, pred_col=pred_col, obs_col="Q_obs"
            )
            global_clim_r2 = compute_global_r2(
                valid_df, pred_col="climatology", obs_col="Q_obs"
            )

            results.append(
                {
                    "horizon": horizon,
                    "model": model,
                    "n_basins": n_basins,
                    "n_months": n_months,
                    "n_code_month_groups": n_code_month_groups,
                    "n_samples": n_samples,
                    # R² per (code, month) - THE CORRECT METRICS
                    "r2_mean": mean_r2_code_month,
                    "r2_median": median_r2_code_month,
                    "r2_std": std_r2_code_month,
                    # Climatology baseline (per code, month)
                    "r2_clim_mean": mean_clim_r2,
                    "r2_clim_median": median_clim_r2,
                    # Skill improvement over climatology
                    "skill_vs_clim_mean": skill_vs_clim_mean,
                    "skill_vs_clim_median": skill_vs_clim_median,
                    # Global R² for reference (inflated)
                    "r2_global": global_r2_model,
                    "r2_clim_global": global_clim_r2,
                }
            )

    return pd.DataFrame(results)


def print_diagnostic_table(results_df: pd.DataFrame) -> None:
    """
    Print a nicely formatted diagnostic table.

    Args:
        results_df: DataFrame with diagnostic results
    """
    print("\n" + "=" * 130)
    print("MCALD SKILL SOURCE DIAGNOSTIC TABLE")
    print("R² computed per (code, month) across years - the TRUE forecasting skill")
    print("=" * 130)
    print()
    print("Metric Explanation:")
    print("  - r2_mean/median: Mean/median R² across all (code, month) groups")
    print("    This measures year-to-year prediction skill, removing seasonal patterns")
    print("  - r2_clim_*: Climatology baseline (predicting with train-set mean)")
    print("  - skill_vs_clim: Model R² - Climatology R² (should be > 0 for real skill)")
    print("  - r2_global: Reference only - inflated by inter-basin/seasonal variance")
    print()
    print(
        "Interpretation: If skill_vs_clim < 0.02 at long leads → skill is likely artifact"
    )
    print("=" * 130)
    print()

    # Group by model and display
    for model in results_df["model"].unique():
        model_df = results_df[results_df["model"] == model].sort_values("horizon")

        print(f"\n{'=' * 50}")
        print(f"MODEL: {model}")
        print(f"{'=' * 50}")

        # Header
        print(
            f"{'Horizon':>8} | {'n_groups':>8} | {'n_samples':>9} | "
            f"{'R2_mean':>10} | {'R2_median':>10} | {'R2_std':>8} | "
            f"{'Clim_mean':>10} | {'Clim_med':>10} | "
            f"{'Skill_mean':>10} | {'Skill_med':>10} | "
            f"{'R2_global':>10}"
        )
        print("-" * 130)

        for _, row in model_df.iterrows():
            print(
                f"{row['horizon']:>8} | {row['n_code_month_groups']:>8} | {row['n_samples']:>9} | "
                f"{row['r2_mean']:>10.4f} | {row['r2_median']:>10.4f} | {row['r2_std']:>8.4f} | "
                f"{row['r2_clim_mean']:>10.4f} | {row['r2_clim_median']:>10.4f} | "
                f"{row['skill_vs_clim_mean']:>10.4f} | {row['skill_vs_clim_median']:>10.4f} | "
                f"{row['r2_global']:>10.4f}"
            )

        print()

    # Summary statistics
    print("\n" + "=" * 90)
    print("SUMMARY: Average across horizons (using mean R² per code/month)")
    print("=" * 90)

    summary = results_df.groupby("model").agg(
        {
            "r2_mean": "mean",
            "r2_median": "mean",
            "r2_clim_mean": "mean",
            "skill_vs_clim_mean": "mean",
            "skill_vs_clim_median": "mean",
        }
    )
    summary.columns = [
        "R2_mean",
        "R2_median",
        "Clim_mean",
        "Skill_mean",
        "Skill_median",
    ]

    print(summary.to_string())
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Diagnose MCALD skill sources and check for artifacts."
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=["Kyrgyzstan", "Tajikistan"],
        default="Kyrgyzstan",
        help="Region to analyze (default: Kyrgyzstan)",
    )
    parser.add_argument(
        "--test-years",
        type=int,
        nargs="+",
        default=None,
        help="Test years to exclude from climatology. If not specified, uses 2020-2024.",
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

    args = parser.parse_args()

    # Determine test years
    if args.test_years is None:
        # Default: assume recent years are test set
        test_years = [2020, 2021, 2022, 2023, 2024]
    else:
        test_years = args.test_years

    logger.info(f"Test years (excluded from climatology): {test_years}")

    # Determine paths based on region
    if args.region == "Tajikistan":
        pred_config = taj_path_config
    else:
        pred_config = kgz_path_config

    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = (
            Path(default_output_dir) / args.region.lower() / "skill_diagnostics"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert horizons to horizon strings
    horizon_strings = [
        f"month_{h}" for h in args.horizons if f"month_{h}" in day_of_forecast
    ]

    logger.info(f"Analyzing skill sources for {args.region}")
    logger.info(f"Horizons: {args.horizons}")

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

    # Create ensemble if not present
    if "Ensemble" not in predictions_df["model"].unique():
        logger.info("Creating ensemble predictions...")
        predictions_df = create_ensemble_predictions(predictions_df)

    # Load observations
    logger.info(f"Loading observations from {pred_config['obs_file']}...")
    obs_df = load_observations(pred_config["obs_file"])
    monthly_obs = calculate_target(obs_df)

    # Determine actual test years from the data
    all_years = sorted(monthly_obs["year"].unique())
    available_test_years = [y for y in test_years if y in all_years]

    if not available_test_years:
        logger.warning(
            f"None of the specified test years {test_years} found in data. "
            f"Available years: {all_years}. Using last 3 years as test set."
        )
        available_test_years = all_years[-3:]

    logger.info(f"Using test years: {available_test_years}")
    logger.info(
        f"Training years: {[y for y in all_years if y not in available_test_years]}"
    )

    # Aggregate predictions with observations
    logger.info("Aggregating predictions with observations...")
    aggregated_df = aggregate(predictions_df, monthly_obs)

    # Run diagnostics
    logger.info("\nRunning skill source diagnostics...")
    results_df = diagnose_skill_sources(
        aggregated_df=aggregated_df,
        monthly_obs=monthly_obs,
        test_years=available_test_years,
        models=["MC_ALD", "Ensemble"],
    )

    if results_df.empty:
        logger.error("No diagnostic results generated. Exiting.")
        return

    # Print diagnostic table
    print_diagnostic_table(results_df)

    # Save results
    results_path = output_dir / "skill_diagnostics.csv"
    results_df.to_csv(results_path, index=False)
    logger.info(f"\nSaved diagnostic results to {results_path}")

    # ================================================================
    # Additional analysis: Compare at long lead times specifically
    # ================================================================
    print("\n" + "=" * 90)
    print("CRITICAL FINDING: Long Lead Time Analysis (Horizons 5-8)")
    print("=" * 90)

    long_lead_df = results_df[results_df["horizon"] >= 5]

    if not long_lead_df.empty:
        for model in long_lead_df["model"].unique():
            model_long = long_lead_df[long_lead_df["model"] == model]

            avg_r2 = model_long["r2_mean"].mean()
            avg_r2_median = model_long["r2_median"].mean()
            avg_clim = model_long["r2_clim_mean"].mean()
            avg_skill = model_long["skill_vs_clim_mean"].mean()

            print(f"\n{model}:")
            print(f"  Average R² (mean per code/month): {avg_r2:.4f}")
            print(f"  Average R² (median per code/month): {avg_r2_median:.4f}")
            print(f"  Average Climatology R²: {avg_clim:.4f}")
            print(f"  Average Skill vs Climatology: {avg_skill:.4f}")

            if avg_skill < 0.02:
                print(
                    f"  *** WARNING: Skill vs climatology < 0.02 → Likely artifact! ***"
                )
            elif avg_skill < 0.05:
                print(f"  ! MARGINAL: Skill is small but potentially real")
            else:
                print(f"  OK: Model shows genuine skill over climatology")

    print("\n" + "=" * 90)
    print("CONCLUSION")
    print("=" * 90)

    if not long_lead_df.empty:
        mcald_long = long_lead_df[long_lead_df["model"] == "MC_ALD"]
        ensemble_long = long_lead_df[long_lead_df["model"] == "Ensemble"]

        if not mcald_long.empty and not ensemble_long.empty:
            mcald_skill = mcald_long["skill_vs_clim_mean"].mean()
            ensemble_skill = ensemble_long["skill_vs_clim_mean"].mean()
            mcald_r2 = mcald_long["r2_mean"].mean()
            ensemble_r2 = ensemble_long["r2_mean"].mean()

            print(f"\nAt long lead times (h >= 5):")
            print(
                f"  MC_ALD:    R² = {mcald_r2:.4f}, Skill vs Clim = {mcald_skill:.4f}"
            )
            print(
                f"  Ensemble:  R² = {ensemble_r2:.4f}, Skill vs Clim = {ensemble_skill:.4f}"
            )
            print(f"  MC_ALD improvement over Ensemble: {mcald_r2 - ensemble_r2:.4f}")

            if mcald_skill < 0.02 and ensemble_skill < 0.02:
                print(
                    "\n  VERDICT: BOTH models show essentially NO skill at long leads."
                )
                print("  The apparent R² comes from:")
                print("    - Predicting seasonal patterns (captured by climatology)")
                print(
                    "    - Q_obs statistics features likely encode climatology directly"
                )
                print("\n  Recommendation: ")
                print("    1. Remove Q_obs features from MCALD and retrain")
                print("    2. If skill drops significantly, confirms leakage")
            elif mcald_skill > ensemble_skill + 0.02:
                print("\n  VERDICT: MC_ALD shows improvement over Ensemble.")
                print("  But verify this isn't from Q_obs feature leakage.")
            else:
                print(
                    "\n  VERDICT: Marginal or no improvement of MC_ALD over Ensemble."
                )

    logger.info(f"\nDiagnostics complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
