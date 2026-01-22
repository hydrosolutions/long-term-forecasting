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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

# load the .env file
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")


path_mc_ald = "/Users/sandrohunziker/hydrosolutions Dropbox/Sandro Hunziker/SAPPHIRE_Central_Asia_Technical_Work/data/kyg_data_forecast_tools/intermediate_data/long_term_predictions/month_6/MC_ALD/MC_ALD_hindcast.csv"


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


def calculate_r2_by_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate R² scores per month for different predictions vs Q_obs.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, and Q_loc_corrected

    Returns:
        DataFrame with R² scores per month
    """
    results = []

    for month in sorted(data["month"].unique()):
        month_data = data[data["month"] == month].dropna(
            subset=["Q_obs", "Q_loc", "Q_MC_ALD", "Q_loc_corrected"]
        )

        if len(month_data) < 2:
            continue

        r2_q_loc = r2_score(month_data["Q_obs"], month_data["Q_loc"])
        r2_mc_ald = r2_score(month_data["Q_obs"], month_data["Q_MC_ALD"])
        r2_corrected = r2_score(month_data["Q_obs"], month_data["Q_loc_corrected"])

        results.append(
            {
                "month": month,
                "R² Q_loc (original)": r2_q_loc,
                "R² Q_MC_ALD": r2_mc_ald,
                "R² Q_loc (corrected)": r2_corrected,
                "n_samples": len(month_data),
            }
        )

    return pd.DataFrame(results)


def calculate_r2_by_code_month(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate R² scores per code and month for different predictions vs Q_obs.

    Args:
        data: DataFrame with Q_obs, Q_loc, Q_MC_ALD, and Q_loc_corrected

    Returns:
        DataFrame with R² scores per code and month
    """
    results = []

    for code in data["code"].unique():
        for month in sorted(data["month"].unique()):
            mask = (data["code"] == code) & (data["month"] == month)
            subset = data[mask].dropna(
                subset=["Q_obs", "Q_loc", "Q_MC_ALD", "Q_loc_corrected"]
            )

            if len(subset) < 2:
                continue

            r2_q_loc = r2_score(subset["Q_obs"], subset["Q_loc"])
            r2_mc_ald = r2_score(subset["Q_obs"], subset["Q_MC_ALD"])
            r2_corrected = r2_score(subset["Q_obs"], subset["Q_loc_corrected"])

            results.append(
                {
                    "code": code,
                    "month": month,
                    "R² Q_loc (original)": r2_q_loc,
                    "R² Q_MC_ALD": r2_mc_ald,
                    "R² Q_loc (corrected)": r2_corrected,
                    "n_samples": len(subset),
                }
            )

    return pd.DataFrame(results)


def plot_r2_distribution_by_month(r2_df: pd.DataFrame) -> None:
    """
    Plot the distribution of R² scores per month for all three methods.

    Args:
        r2_df: DataFrame with R² scores per code and month
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

    plot_df = pd.DataFrame(plot_data)

    # Create the boxplot
    plt.figure(figsize=(14, 8))

    # Define colors for each method
    palette = {
        "Q_loc (original)": "#3498db",  # Blue
        "Q_MC_ALD": "#e74c3c",  # Red
        "Q_loc (corrected)": "#2ecc71",  # Green
    }

    sns.boxplot(x="month", y="R²", hue="Method", data=plot_df, palette=palette)

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.title(
        "R² Distribution per Month: Comparison of Prediction Methods\n(Leave-One-Out CV)",
        fontsize=14,
    )
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.legend(title="Method", loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Create violin plot for additional insight
    plt.figure(figsize=(14, 8))

    sns.violinplot(
        x="month", y="R²", hue="Method", data=plot_df, palette=palette, inner="box"
    )

    plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
    plt.title("R² Distribution per Month: Violin Plot\n(Leave-One-Out CV)", fontsize=14)
    plt.xlabel("Month", fontsize=12)
    plt.ylabel("R² Score", fontsize=12)
    plt.legend(title="Method", loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_summary_statistics(r2_df: pd.DataFrame) -> None:
    """
    Plot summary statistics comparing the three methods.

    Args:
        r2_df: DataFrame with R² scores per code and month
    """
    # Calculate median R² per month for each method
    summary = (
        r2_df.groupby("month")
        .agg(
            {
                "R² Q_loc (original)": ["median", "mean", "std", "count"],
                "R² Q_MC_ALD": ["median", "mean", "std"],
                "R² Q_loc (corrected)": ["median", "mean", "std"],
            }
        )
        .round(3)
    )

    logger.info("\n" + "=" * 80)
    logger.info("Summary Statistics per Month:")
    logger.info("=" * 80)
    print(summary.to_string())

    # Calculate overall statistics
    overall_stats = pd.DataFrame(
        {
            "Method": ["Q_loc (original)", "Q_MC_ALD", "Q_loc (corrected)"],
            "Median R²": [
                r2_df["R² Q_loc (original)"].median(),
                r2_df["R² Q_MC_ALD"].median(),
                r2_df["R² Q_loc (corrected)"].median(),
            ],
            "Mean R²": [
                r2_df["R² Q_loc (original)"].mean(),
                r2_df["R² Q_MC_ALD"].mean(),
                r2_df["R² Q_loc (corrected)"].mean(),
            ],
            "Std R²": [
                r2_df["R² Q_loc (original)"].std(),
                r2_df["R² Q_MC_ALD"].std(),
                r2_df["R² Q_loc (corrected)"].std(),
            ],
        }
    ).round(3)

    logger.info("\n" + "=" * 80)
    logger.info("Overall Statistics:")
    logger.info("=" * 80)
    print(overall_stats.to_string(index=False))

    # Plot bar chart of median R² per month
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    months = sorted(r2_df["month"].unique())
    x = np.arange(len(months))
    width = 0.25

    medians_orig = r2_df.groupby("month")["R² Q_loc (original)"].median()
    medians_mcald = r2_df.groupby("month")["R² Q_MC_ALD"].median()
    medians_corr = r2_df.groupby("month")["R² Q_loc (corrected)"].median()

    # Median R² bar chart
    axes[0].bar(
        x - width, medians_orig, width, label="Q_loc (original)", color="#3498db"
    )
    axes[0].bar(x, medians_mcald, width, label="Q_MC_ALD", color="#e74c3c")
    axes[0].bar(
        x + width, medians_corr, width, label="Q_loc (corrected)", color="#2ecc71"
    )

    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Median R²")
    axes[0].set_title("Median R² per Month by Method")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(months)
    axes[0].legend()
    axes[0].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3)

    # Improvement comparison
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
        x + width / 2, improvement_vs_mcald, width, label="vs Q_MC_ALD", color="#e74c3c"
    )

    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("R² Improvement")
    axes[1].set_title("R² Improvement of Corrected Q_loc vs Other Methods")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(months)
    axes[1].legend()
    axes[1].axhline(0, color="gray", linestyle="--", alpha=0.5)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main() -> None:
    # Load data
    data = pd.read_csv(path_mc_ald, parse_dates=["date"])
    data = data[["date", "code", "Q_loc", "Q_MC_ALD", "Q_obs"]].dropna()
    data["month"] = data["date"].dt.month

    logger.info(
        f"Loaded {len(data)} records with {data['code'].nunique()} unique codes"
    )
    logger.info(f"Months in data: {sorted(data['month'].unique())}")

    # Train correction model using LOOCV
    logger.info("Training linear regression correction model with Leave-One-Out CV...")
    data_with_corrections = train_correction_model_loocv(data)

    # Calculate R² scores per code and month
    logger.info("Calculating R² scores per code and month...")
    r2_by_code_month = calculate_r2_by_code_month(data_with_corrections)

    if r2_by_code_month.empty:
        logger.error("No R² scores could be calculated. Check your data.")
        return

    logger.info(f"Calculated R² for {len(r2_by_code_month)} code-month combinations")

    # Plot R² distribution
    plot_r2_distribution_by_month(r2_by_code_month)

    # Plot summary statistics
    plot_summary_statistics(r2_by_code_month)

    # Calculate and show aggregated R² per month (using all data points)
    logger.info("\n" + "=" * 80)
    logger.info("Aggregated R² per Month (all samples pooled):")
    logger.info("=" * 80)
    r2_aggregated = calculate_r2_by_month(data_with_corrections)
    print(r2_aggregated.to_string(index=False))

    # Show sample of corrected data
    logger.info("\n" + "=" * 80)
    logger.info("Sample of corrected data:")
    logger.info("=" * 80)
    sample_cols = [
        "date",
        "code",
        "month",
        "Q_loc",
        "Q_MC_ALD",
        "Q_loc_corrected",
        "Q_obs",
    ]
    print(data_with_corrections[sample_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
