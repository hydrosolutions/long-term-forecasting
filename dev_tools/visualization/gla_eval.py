import os
import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import seaborn as sns
import matplotlib.pyplot as plt

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from data_handlers import PredictionDataHandler, MetricsDataHandler
from style_config import set_global_plot_style

# set global plt and sns styles
set_global_plot_style()

# get data handlers
metric_handler = MetricsDataHandler()
prediction_handler = PredictionDataHandler()

month_mapping = {
    12: "January",
    1: "February",
    2: "March",
    3: "April",
    4: "May",
    5: "June",
    6: "July",
    7: "August",
    8: "September",
    9: "October",
    10: "November",
    11: "December",
}

model_colors = {
    "Linear Regression Base": "#787878",  # gray
    "Base Case Ensemble": "#787878",  # orange
    "SnowMapper Ensemble": "#1F77B4",  # orange
    "MC ALD": "#2CA02C",  # green,
    "Glacier Mapper Ensemble": "#0ECFFF",  # red
    "Gla MC ALD": "#7BDBAB",  # red
}

metric_renamer = {
    "nse": "NSE [-]",
    "rmse": "RMSE [m³/s]",
    "mae": "MAE [m³/s]",
    "r2": "R² [-]",
    "pbias": "PBIAS [-]",
    "kge": "KGE [-]",
}


def config_plotting():
    available_models = metric_handler.available_models
    available_codes = metric_handler.available_codes
    available_metrics = metric_handler.available_metrics

    print("Available models:", available_models)
    print("Available codes:", available_codes)
    print("Available metrics:", available_metrics)

    return available_models, available_codes, available_metrics


def plot_monthly_overall(
    df: pd.DataFrame,
    metric: str,
    models: List[str],
    per_month: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot bar chart of mean and standard deviation of a metric for given models.

    Args:
        df: DataFrame containing metrics with columns ['model', 'code', 'month', metric]
        metric: Metric to plot (e.g., 'nse', 'rmse')
        models: List of model names to include
        per_month: If True, plot per month; otherwise aggregate over all months
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes.
    Returns:
        Matplotlib Axes with the plot
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    plot_df = df[df["model"].isin(models)]

    if per_month:
        plot_df = plot_df[plot_df["level"] == "per_code_month"].copy()

        # sort so that months are in calendar order
        plot_df["month"] = pd.Categorical(
            plot_df["month"], categories=list(month_mapping.values()), ordered=True
        )
        plot_df = plot_df.sort_values("month")

        # order of models in legend
        plot_df["model"] = pd.Categorical(
            plot_df["model"], categories=models, ordered=True
        )
        plot_df = plot_df.sort_values("model")

        sns.barplot(
            data=plot_df,
            x="month",
            y=metric,
            hue="model",
            ax=ax,
            estimator=np.median,
            errorbar=("pi", 50),
            capsize=0.1,
            palette=model_colors,
        )

        """sns.boxplot(
            data=plot_df, x='month',
            y=metric, hue='model',
            ax=ax, 
            palette=model_colors,
        )"""

        ax.set_xlabel("Month")
        # rotate x-tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(0, 1)  # Adjust y-limits as needed
    else:
        plot_df = plot_df[plot_df["level"] == "per_code"].copy()

        # order of models in legend
        plot_df["model"] = pd.Categorical(
            plot_df["model"], categories=models, ordered=True
        )
        plot_df = plot_df.sort_values("model")

        sns.boxplot(
            data=plot_df, x="model", y=metric, ax=ax, palette=model_colors, legend=False
        )
        ax.set_ylim(0.4, 1.0)  # Adjust y-limits as needed

        ax.set_xlabel("")

    ax.set_ylabel(metric_renamer.get(metric, metric))
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    return ax


def create_monthly_and_overall_performance_plots(
    df_metrics: pd.DataFrame,
    metric_to_plot: str,
    models_to_plot: List[str],
    rename_dict: Dict[str, str],
    save_dir: str,
):
    """Create and save monthly and overall performance plots for selected models and metrics."""
    fig, ax = plt.subplots()
    ax = plot_monthly_overall(
        df_metrics,
        metric=metric_to_plot,
        models=list(rename_dict.values()),
        per_month=True,
        ax=ax,
    )
    # draw black border around the whole figure
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out = Path(save_dir) / f"bar_{metric_to_plot}_per_month.png"
    fig.savefig(out)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_monthly_overall(
        df_metrics,
        metric=metric_to_plot,
        models=list(rename_dict.values()),
        per_month=False,
        ax=ax,
    )
    # draw black border around the whole figure
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out = Path(save_dir) / f"box_{metric_to_plot}_overall.png"
    fig.savefig(out)
    plt.show()


def read_feature_importance_gl_fr(
    folders: List[str], feature_name: str = "gl_fr"
) -> pd.DataFrame:
    """
    Read feature importance files from multiple model folders and extract
    the relative importance (position) of a specific feature.

    Args:
        folders: List of folder paths containing feature importance CSV files
        feature_name: Name of the feature to analyze (default: 'gl_fr')

    Returns:
        DataFrame with columns:
            - folder: Folder path
            - model_class: Model class (Base case, SnowMapper, Glacier Mapper)
            - model_name: Short model name
            - model_type: Type of model (catboost, lgbm, xgb)
            - feature_position: Position of gl_fr in importance ranking (1-indexed)
            - total_features: Total number of features
            - relative_importance: Position / total * 100 (percentage)
            - importance_value: Actual importance value from the model
    """
    results = []

    for folder in folders:
        folder_path = Path(folder)

        if not folder_path.exists():
            print(f"Warning: Folder does not exist: {folder}")
            continue

        # Determine model class based on folder structure
        if "GlacierMapper" in str(folder):
            model_class = "Glacier Mapper"
        elif "SnowMapper" in str(folder):
            model_class = "SnowMapper"
        elif "BaseCase" in str(folder):
            model_class = "Base case"
        else:
            model_class = "Unknown"

        # Extract short model name from folder path
        model_name = folder_path.name

        # Check for each model type's feature importance file
        for model_type in ["catboost", "lgbm", "xgb"]:
            feature_importance_file = (
                folder_path / f"{model_type}_feature_importance.csv"
            )

            if not feature_importance_file.exists():
                continue

            try:
                # Read feature importance CSV
                df_importance = pd.read_csv(feature_importance_file)

                # Check if required columns exist
                if (
                    "feature" not in df_importance.columns
                    or "importance" not in df_importance.columns
                ):
                    print(
                        f"Warning: Missing required columns in {feature_importance_file}"
                    )
                    continue

                # Sort by importance (descending) to get proper ranking
                df_importance = df_importance.sort_values(
                    "importance", ascending=False
                ).reset_index(drop=True)

                # Find the feature
                feature_rows = df_importance[df_importance["feature"] == feature_name]

                if len(feature_rows) == 0:
                    print(
                        f"Info: Feature '{feature_name}' not found in {feature_importance_file}"
                    )
                    continue

                # Get position (1-indexed for interpretability)
                feature_position = feature_rows.index[0] + 1
                total_features = len(df_importance)
                importance_value = feature_rows["importance"].values[0]

                # Calculate relative importance as percentage
                relative_importance = (feature_position / total_features) * 100

                results.append(
                    {
                        "folder": str(folder),
                        "model_class": model_class,
                        "model_name": model_name,
                        "model_type": model_type,
                        "feature_position": feature_position,
                        "total_features": total_features,
                        "relative_importance": relative_importance,
                        "importance_value": importance_value,
                    }
                )

            except Exception as e:
                print(f"Error reading {feature_importance_file}: {e}")
                continue

    return pd.DataFrame(results)


def plot_gl_fr_importance(
    df_results: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Visualize the relative importance of gl_fr feature across different models.

    Args:
        df_results: DataFrame from read_feature_importance_gl_fr()
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    if df_results.empty:
        print("Warning: No data to plot")
        return None

    # Define colors for model classes
    class_colors = {
        "Base case": "#787878",
        "SnowMapper": "#1F77B4",
        "Glacier Mapper": "#0ECFFF",
    }

    # Define markers for model types
    model_markers = {
        "xgb": "o",  # circle
        "lgbm": "X",  # X
        "catboost": "^",  # triangle
    }

    # Create single figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Prepare data
    df_plot = df_results.copy()

    # Sort by model class for grouped display
    class_order = ["Base case", "SnowMapper", "Glacier Mapper"]
    df_plot["model_class"] = pd.Categorical(
        df_plot["model_class"], categories=class_order, ordered=True
    )
    df_plot = df_plot.sort_values(["model_class", "model_name"])

    # Create x-axis positions for each model configuration
    unique_models = df_plot.groupby(["model_class", "model_name"]).ngroups
    x_positions = []
    x_labels = []
    current_pos = 0

    for (model_class, model_name), group in df_plot.groupby(
        ["model_class", "model_name"], sort=False
    ):
        x_positions.extend([current_pos] * len(group))
        x_labels.append(model_name)
        current_pos += 1

    df_plot["x_pos"] = x_positions

    # Plot each point with appropriate marker and color
    for model_type in ["xgb", "lgbm", "catboost"]:
        df_type = df_plot[df_plot["model_type"] == model_type]
        if not df_type.empty:
            for model_class in class_order:
                df_subset = df_type[df_type["model_class"] == model_class]
                if not df_subset.empty:
                    ax.scatter(
                        df_subset["x_pos"],
                        df_subset["relative_importance"],
                        marker=model_markers[model_type],
                        color=class_colors[model_class],
                        s=150,
                        edgecolors="black",
                        linewidth=1.0,
                        label=f"{model_class} - {model_type}"
                        if model_class == class_order[0]
                        else "",
                        alpha=0.8,
                    )

    # Customize plot
    ax.set_xlabel("Model Configuration", fontsize=12)
    ax.set_ylabel("Relative Importance Position [%]", fontsize=12)
    ax.set_title(
        "gl_fr Feature Importance Across Models\n(Lower % = More Important)",
        fontsize=13,
    )
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_ylim(0, 100)

    # Create custom legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    # Legend for model classes (colors)
    class_legend = [
        Patch(facecolor=class_colors[cls], edgecolor="black", label=cls)
        for cls in class_order
        if cls in df_plot["model_class"].values
    ]

    # Legend for model types (markers)
    marker_legend = [
        Line2D(
            [0],
            [0],
            marker=model_markers[mtype],
            color="gray",
            linestyle="",
            markersize=8,
            markeredgecolor="black",
            label=mtype.upper(),
            markeredgewidth=1.0,
        )
        for mtype in ["xgb", "lgbm", "catboost"]
        if mtype in df_plot["model_type"].values
    ]

    # Combine legends
    first_legend = ax.legend(
        handles=class_legend, title="Model Class", loc="upper left", frameon=True
    )
    ax.add_artist(first_legend)
    ax.legend(
        handles=marker_legend, title="Model Type", loc="upper right", frameon=True
    )

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return fig


def plot_model_difference_per_month(
    metrics: pd.DataFrame,
    model_1: str,
    model_2: str,
    static_df: pd.DataFrame,
    metric: str,
    ax: plt.Axes | None = None,
    threshold: float = 0.1,
) -> tuple[plt.Axes, pd.DataFrame]:
    """
    Plot the difference between two models per month as boxplot with individual points.
    Also calculates and prints statistics about performance improvements.

    Args:
        metrics: DataFrame with columns ['model', 'code', 'month', metric]
        model_1: Name of first model
        model_2: Name of second model
        static_df: DataFrame with static features including 'code' and 'gl_fr'
        metric: Metric to compare (e.g., 'nse', 'r2')
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes.
        threshold: Threshold for considering improvement/decrease (default: 0.1)

    Returns:
        Tuple of (Matplotlib Axes with the plot, DataFrame with monthly statistics)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Filter data for the two models at per_code_month level
    df_model1 = metrics[
        (metrics["model"] == model_1) & (metrics["level"] == "per_code_month")
    ].copy()
    df_model2 = metrics[
        (metrics["model"] == model_2) & (metrics["level"] == "per_code_month")
    ].copy()

    # Merge on code and month to align the data
    df_merged = df_model1[["code", "month", metric]].merge(
        df_model2[["code", "month", metric]],
        on=["code", "month"],
        suffixes=("_model1", "_model2"),
    )

    # Calculate difference (model_1 - model_2)
    df_merged["difference"] = (
        df_merged[f"{metric}_model1"] - df_merged[f"{metric}_model2"]
    )

    # Sort months in calendar order
    df_merged["month"] = pd.Categorical(
        df_merged["month"], categories=list(month_mapping.values()), ordered=True
    )
    df_merged = df_merged.sort_values("month")

    # Get unique months that actually exist in the data
    unique_months = [
        m for m in df_merged["month"].cat.categories if m in df_merged["month"].values
    ]

    # Categorize performance changes
    df_merged["performance_change"] = pd.cut(
        df_merged["difference"],
        bins=[-np.inf, -threshold, threshold, np.inf],
        labels=["Decreased", "No Change", "Improved"],
    )

    # Calculate statistics per month
    monthly_stats = []
    for month in unique_months:
        month_data = df_merged[df_merged["month"] == month]
        total_basins = len(month_data)

        n_improved = (month_data["difference"] > threshold).sum()
        n_no_change = (
            (month_data["difference"] >= -threshold)
            & (month_data["difference"] <= threshold)
        ).sum()
        n_decreased = (month_data["difference"] < -threshold).sum()

        monthly_stats.append(
            {
                "month": month,
                "total_basins": total_basins,
                "improved": n_improved,
                "no_change": n_no_change,
                "decreased": n_decreased,
                "pct_improved": (n_improved / total_basins * 100)
                if total_basins > 0
                else 0,
                "pct_no_change": (n_no_change / total_basins * 100)
                if total_basins > 0
                else 0,
                "pct_decreased": (n_decreased / total_basins * 100)
                if total_basins > 0
                else 0,
                "mean_difference": month_data["difference"].mean(),
                "median_difference": month_data["difference"].median(),
            }
        )

    stats_df = pd.DataFrame(monthly_stats)

    # Print statistics
    print("\n" + "=" * 80)
    print(f"Performance Comparison: {model_1} vs {model_2}")
    print(f"Metric: {metric_renamer.get(metric, metric)}")
    print(f"Threshold: ±{threshold}")
    print("=" * 80)
    print(
        f"\n{'Month':<12} {'Total':>7} {'Improved':>10} {'No Change':>11} {'Decreased':>11} {'Mean Δ':>10}"
    )
    print("-" * 80)

    for _, row in stats_df.iterrows():
        print(
            f"{row['month']:<12} {row['total_basins']:>7} "
            f"{row['improved']:>6} ({row['pct_improved']:>5.1f}%) "
            f"{row['no_change']:>6} ({row['pct_no_change']:>5.1f}%) "
            f"{row['decreased']:>6} ({row['pct_decreased']:>5.1f}%) "
            f"{row['mean_difference']:>9.3f}"
        )

    # Overall statistics
    total_comparisons = len(df_merged)
    overall_improved = (df_merged["difference"] > threshold).sum()
    overall_no_change = (
        (df_merged["difference"] >= -threshold) & (df_merged["difference"] <= threshold)
    ).sum()
    overall_decreased = (df_merged["difference"] < -threshold).sum()

    print("-" * 80)
    print(
        f"{'OVERALL':<12} {total_comparisons:>7} "
        f"{overall_improved:>6} ({overall_improved / total_comparisons * 100:>5.1f}%) "
        f"{overall_no_change:>6} ({overall_no_change / total_comparisons * 100:>5.1f}%) "
        f"{overall_decreased:>6} ({overall_decreased / total_comparisons * 100:>5.1f}%) "
        f"{df_merged['difference'].mean():>9.3f}"
    )
    print("=" * 80 + "\n")

    # Create bar plot showing percentage of improved and decreased basins
    x_positions = np.arange(len(unique_months))
    width = 0.35

    improved_pcts = stats_df["pct_improved"].values
    decreased_pcts = stats_df["pct_decreased"].values

    # Plot bars
    bars1 = ax.bar(
        x_positions - width / 2,
        improved_pcts,
        width,
        label="Improved (>+0.1)",
        color="#2CA02C",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars2 = ax.bar(
        x_positions + width / 2,
        decreased_pcts,
        width,
        label="Decreased (<-0.1)",
        color="#D62728",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only show label if there's a value
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Customize plot
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(f"Percentage of Basins [%]", fontsize=12)
    ax.set_title(
        f"Basin Performance Changes by Month\n{model_1} vs {model_2} (threshold = ±{threshold})",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(unique_months, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    return ax, stats_df


def plot_model_difference_magnitude(
    metrics: pd.DataFrame,
    model_1: str,
    model_2: str,
    metric: str,
    ax: plt.Axes | None = None,
    threshold: float = 0.1,
) -> tuple[plt.Axes, pd.DataFrame]:
    """
    Plot the average magnitude of improvement/decrease between two models per month.
    Shows average with error bars representing minimum and maximum values.

    Args:
        metrics: DataFrame with columns ['model', 'code', 'month', metric]
        model_1: Name of first model
        model_2: Name of second model
        metric: Metric to compare (e.g., 'nse', 'r2')
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes.
        threshold: Threshold for considering improvement/decrease (default: 0.1)

    Returns:
        Tuple of (Matplotlib Axes with the plot, DataFrame with monthly magnitude statistics)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # Filter data for the two models at per_code_month level
    df_model1 = metrics[
        (metrics["model"] == model_1) & (metrics["level"] == "per_code_month")
    ].copy()
    df_model2 = metrics[
        (metrics["model"] == model_2) & (metrics["level"] == "per_code_month")
    ].copy()

    # Merge on code and month to align the data
    df_merged = df_model1[["code", "month", metric]].merge(
        df_model2[["code", "month", metric]],
        on=["code", "month"],
        suffixes=("_model1", "_model2"),
    )

    # Calculate difference (model_1 - model_2)
    df_merged["difference"] = (
        df_merged[f"{metric}_model1"] - df_merged[f"{metric}_model2"]
    )

    # Sort months in calendar order
    df_merged["month"] = pd.Categorical(
        df_merged["month"], categories=list(month_mapping.values()), ordered=True
    )
    df_merged = df_merged.sort_values("month")

    # Get unique months that actually exist in the data
    unique_months = [
        m for m in df_merged["month"].cat.categories if m in df_merged["month"].values
    ]

    # Calculate statistics per month for improved and decreased basins
    monthly_magnitude_stats = []
    for month in unique_months:
        month_data = df_merged[df_merged["month"] == month]

        # Improved basins (difference > threshold)
        improved_data = month_data[month_data["difference"] > threshold]["difference"]
        if len(improved_data) > 0:
            improved_mean = improved_data.mean()
            improved_min = improved_data.min()
            improved_max = improved_data.max()
        else:
            improved_mean = 0
            improved_min = 0
            improved_max = 0

        # Decreased basins (difference < -threshold)
        decreased_data = month_data[month_data["difference"] < -threshold]["difference"]
        if len(decreased_data) > 0:
            decreased_mean = abs(
                decreased_data.mean()
            )  # Take absolute value for plotting
            decreased_min = abs(
                decreased_data.max()
            )  # Note: max of negative values (closest to 0)
            decreased_max = abs(
                decreased_data.min()
            )  # Note: min of negative values (furthest from 0)
        else:
            decreased_mean = 0
            decreased_min = 0
            decreased_max = 0

        monthly_magnitude_stats.append(
            {
                "month": month,
                "improved_mean": improved_mean,
                "improved_min": improved_min,
                "improved_max": improved_max,
                "improved_error_lower": improved_mean - improved_min,
                "improved_error_upper": improved_max - improved_mean,
                "decreased_mean": decreased_mean,
                "decreased_min": decreased_min,
                "decreased_max": decreased_max,
                "decreased_error_lower": decreased_mean - decreased_min,
                "decreased_error_upper": decreased_max - decreased_mean,
            }
        )

    stats_df = pd.DataFrame(monthly_magnitude_stats)

    # Print statistics
    print("\n" + "=" * 80)
    print(f"Performance Change Magnitude: {model_1} vs {model_2}")
    print(f"Metric: {metric_renamer.get(metric, metric)}")
    print(f"Threshold: ±{threshold}")
    print("=" * 80)
    print(
        f"\n{'Month':<12} {'Improved Δ':>12} {'Min-Max':>20} {'Decreased Δ':>12} {'Min-Max':>20}"
    )
    print("-" * 80)

    for _, row in stats_df.iterrows():
        improved_str = (
            f"{row['improved_mean']:.3f}" if row["improved_mean"] > 0 else "-"
        )
        improved_range = (
            f"[{row['improved_min']:.3f}, {row['improved_max']:.3f}]"
            if row["improved_mean"] > 0
            else "-"
        )
        decreased_str = (
            f"{row['decreased_mean']:.3f}" if row["decreased_mean"] > 0 else "-"
        )
        decreased_range = (
            f"[{row['decreased_min']:.3f}, {row['decreased_max']:.3f}]"
            if row["decreased_mean"] > 0
            else "-"
        )

        print(
            f"{row['month']:<12} {improved_str:>12} {improved_range:>20} {decreased_str:>12} {decreased_range:>20}"
        )
    print("=" * 80 + "\n")

    # Create bar plot with error bars
    x_positions = np.arange(len(unique_months))
    width = 0.35

    improved_means = stats_df["improved_mean"].values
    decreased_means = stats_df["decreased_mean"].values

    # Error bars (asymmetric)
    improved_errors = [
        stats_df["improved_error_lower"].values,
        stats_df["improved_error_upper"].values,
    ]
    decreased_errors = [
        stats_df["decreased_error_lower"].values,
        stats_df["decreased_error_upper"].values,
    ]

    # Plot bars with error bars
    bars1 = ax.bar(
        x_positions - width / 2,
        improved_means,
        width,
        label="Avg Improvement (>+0.1)",
        color="#2CA02C",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
        yerr=improved_errors,
        capsize=4,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )

    bars2 = ax.bar(
        x_positions + width / 2,
        decreased_means,
        width,
        label="Avg Decrease (<-0.1)",
        color="#D62728",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
        yerr=decreased_errors,
        capsize=4,
        error_kw={"linewidth": 1.5, "ecolor": "black"},
    )

    # Add value labels on bars
    for bars, values in [(bars1, improved_means), (bars2, decreased_means)]:
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height > 0.01:  # Only show label if there's a meaningful value
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    # Customize plot
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel(f"Average Δ {metric_renamer.get(metric, metric)}", fontsize=12)
    ax.set_title(
        f"Average Performance Change Magnitude by Month\n{model_1} vs {model_2} (error bars: Min-Max)",
        fontsize=13,
        fontweight="bold",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(unique_months, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Add border around the plot
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.5)

    return ax, stats_df


def collect_and_plot_top_features(
    folders: List[str],
    top_n: int = 20,
    save_path: str | None = None,
) -> tuple[pd.DataFrame, plt.Figure]:
    """
    Collect top N features from each GBT model and plot them in a 2-column layout.

    Args:
        folders: List of folder paths containing feature importance CSV files
        top_n: Number of top features to collect per model (default: 20)
        save_path: Optional path to save the figure

    Returns:
        Tuple of (DataFrame with all top features, Matplotlib Figure)
    """
    all_top_features = []

    # Collect top features from each model
    for folder in folders:
        folder_path = Path(folder)

        if not folder_path.exists():
            print(f"Warning: Folder does not exist: {folder}")
            continue

        # Determine model class and name
        if "GlacierMapper" in str(folder):
            model_class = "Glacier Mapper"
        elif "SnowMapper" in str(folder):
            model_class = "SnowMapper"
            continue
        elif "BaseCase" in str(folder):
            model_class = "Base case"
            continue
        else:
            model_class = "Unknown"
            continue

        model_name = folder_path.name

        # Check each model type
        for model_type in ["catboost", "lgbm", "xgb"]:
            feature_importance_file = (
                folder_path / f"{model_type}_feature_importance.csv"
            )

            if not feature_importance_file.exists():
                continue

            try:
                # Read and sort feature importance
                df_importance = pd.read_csv(feature_importance_file)

                if (
                    "feature" not in df_importance.columns
                    or "importance" not in df_importance.columns
                ):
                    print(f"Warning: Missing columns in {feature_importance_file}")
                    continue

                # Sort by importance and get top N
                df_sorted = df_importance.sort_values(
                    "importance", ascending=False
                ).head(top_n)

                # Add model information
                df_sorted["model_class"] = model_class
                df_sorted["model_name"] = model_name
                df_sorted["model_type"] = model_type
                df_sorted["full_model_name"] = f"{model_name}_{model_type}"
                df_sorted["rank"] = range(1, len(df_sorted) + 1)

                all_top_features.append(df_sorted)

            except Exception as e:
                print(f"Error reading {feature_importance_file}: {e}")
                continue

    if not all_top_features:
        print("No feature importance data found!")
        return pd.DataFrame(), None

    # Combine all data
    df_all = pd.concat(all_top_features, ignore_index=True)

    # Create plot with 2 columns
    unique_models = df_all["full_model_name"].unique()
    n_models = len(unique_models)
    n_cols = 2
    n_rows = (n_models + 1) // 2  # Ceiling division

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))

    # Flatten axes array for easier iteration
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    # Color mapping for model classes
    class_colors = {
        "Base case": "#787878",
        "SnowMapper": "#1F77B4",
        "Glacier Mapper": "#0ECFFF",
    }

    # Plot each model
    for idx, model_name in enumerate(unique_models):
        ax = axes[idx]
        df_model = df_all[df_all["full_model_name"] == model_name].sort_values("rank")

        # Get model class for coloring
        model_class = df_model["model_class"].iloc[0]
        color = class_colors.get(model_class, "gray")

        # Create horizontal bar plot
        bars = ax.barh(
            range(len(df_model)),
            df_model["importance"],
            color=color,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.0,
        )

        # Set y-tick labels to feature names
        ax.set_yticks(range(len(df_model)))
        ax.set_yticklabels(df_model["feature"], fontsize=9)
        ax.invert_yaxis()  # Highest importance at top

        # Formatting
        ax.set_xlabel("Feature Importance", fontsize=10)
        ax.set_title(
            f"{df_model['model_name'].iloc[0]} - {df_model['model_type'].iloc[0].upper()}",
            fontsize=11,
            fontweight="bold",
            color=color,
        )
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, df_model["importance"])):
            ax.text(
                bar.get_width(),
                bar.get_y() + bar.get_height() / 2,
                f" {importance:.0f}",
                va="center",
                ha="left",
                fontsize=8,
                fontweight="bold",
            )

    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].set_visible(False)

    # Add overall title
    fig.suptitle(
        f"Top {top_n} Feature Importance Across GBT Models",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.99])

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    return df_all, fig


def draw_feature_importance(folders_to_check):
    """Main function to analyze and visualize gl_fr feature importance."""

    print("=" * 60)
    print("Analyzing gl_fr feature importance across models")
    print("=" * 60)

    # Read feature importance data
    df_results = read_feature_importance_gl_fr(folders_to_check, feature_name="gl_fr")

    if df_results.empty:
        print("\nNo feature importance data found!")
        return None

    # Display summary statistics
    print(f"\nFound feature importance data for {len(df_results)} model configurations")
    print(f"\nSummary by model class:")
    summary = (
        df_results.groupby("model_class")
        .agg(
            {
                "relative_importance": ["mean", "std", "min", "max"],
                "feature_position": ["mean", "min", "max"],
            }
        )
        .round(2)
    )
    print(summary)

    # Display detailed results
    print("\n" + "=" * 60)
    print("Detailed results:")
    print("=" * 60)
    display_df = df_results[
        [
            "model_class",
            "model_name",
            "model_type",
            "feature_position",
            "total_features",
            "relative_importance",
        ]
    ].copy()
    display_df["relative_importance"] = display_df["relative_importance"].round(1)
    display_df = display_df.sort_values(["model_class", "relative_importance"])
    print(display_df.to_string(index=False))

    # Save results to CSV
    save_dir = Path("../monthly_forecasting_results/figures/KGZ_Glacier_Eval")
    save_dir.mkdir(parents=True, exist_ok=True)

    csv_path = save_dir / "gl_fr_feature_importance_analysis.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")

    # Create visualization
    fig_path = save_dir / "gl_fr_feature_importance.png"
    fig = plot_gl_fr_importance(df_results, save_path=str(fig_path))

    print(f"✓ Visualization saved to: {fig_path}")
    print("=" * 60)

    return df_results


# run with uv run python dev_tools/visualization/gla_eval.py
def main():
    save_dir = "../monthly_forecasting_results/figures/KGZ_Glacier_Eval"
    os.makedirs(save_dir, exist_ok=True)
    config_plotting()

    folders_to_check = [
        "../monthly_forecasting_models/GlacierMapper_Based/Gla_GBT",
        "../monthly_forecasting_models/GlacierMapper_Based/Gla_GBT_Snow",
        "../monthly_forecasting_models/GlacierMapper_Based/Gla_GBT_Norm",
        "../monthly_forecasting_models/SnowMapper_Based/Snow_GBT",
        "../monthly_forecasting_models/SnowMapper_Based/Snow_GBT_Norm",
        "../monthly_forecasting_models/SnowMapper_Based/Snow_GBT_LR",
        "../monthly_forecasting_models/BaseCase/GBT",
    ]
    # Analyze gl_fr feature importance across models
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    draw_feature_importance(folders_to_check=folders_to_check)

    # Plot top 20 features for all GBT models
    print("\n" + "=" * 60)
    print("TOP 20 FEATURES ACROSS GBT MODELS")
    print("=" * 60)

    top_features_save_path = Path(save_dir) / "top_20_features_all_models.png"
    df_top_features, fig_top_features = collect_and_plot_top_features(
        folders=folders_to_check, top_n=20, save_path=str(top_features_save_path)
    )

    if not df_top_features.empty:
        # Save detailed feature importance data
        csv_save_path = Path(save_dir) / "top_20_features_all_models.csv"
        df_top_features.to_csv(csv_save_path, index=False)
        print(f"✓ Top features data saved to: {csv_save_path}")

        # Display summary statistics
        print(
            f"\nTotal models analyzed: {df_top_features['full_model_name'].nunique()}"
        )
        print(f"\nMost common features across all models:")
        feature_counts = df_top_features["feature"].value_counts().head(15)
        print(feature_counts)

    plt.show()
    plt.close("all")

    # Continue with existing analysis
    print("\n" + "=" * 60)
    print("PERFORMANCE METRICS ANALYSIS")
    print("=" * 60)

    static_data = "/Users/sandrohunziker/hydrosolutions Dropbox/Sandro Hunziker/SAPPHIRE_Central_Asia_Technical_Work/data/kyg_data_forecast_tools/config/models_and_scalers/static_features/ML_basin_attributes_v2.csv"
    static_df = pd.read_csv(static_data)

    # if CODE in static_df.columns:
    if "CODE" in static_df.columns:
        static_df = static_df.rename(columns={"CODE": "code"})
        # to int
        static_df["code"] = static_df["code"].astype(int)

    models_to_plot = [
        # "BaseCase_Ensemble",
        "SnowMapper_Based_Ensemble",
        "Uncertainty_MC_ALD",
        "GlacierMapper_Based_Ensemble",
        "Uncertainty_Gla_MC_ALD",
    ]

    rename_dict = {
        # "BaseCase_LR_Base": "Linear Regression Base",
        "BaseCase_Ensemble": "Base Case Ensemble",
        "SnowMapper_Based_Ensemble": "SnowMapper Ensemble",
        "Uncertainty_MC_ALD": "MC ALD",
        "GlacierMapper_Based_Ensemble": "Glacier Mapper Ensemble",
        "Uncertainty_Gla_MC_ALD": "Gla MC ALD",
    }

    df_metrics = metric_handler.get_filtered_data()
    df_metrics["model"] = (
        df_metrics["model"].map(rename_dict).fillna(df_metrics["model"])
    )

    # rename months
    df_metrics["month"] = df_metrics["month"].replace(month_mapping)

    metric_to_plot = "r2"

    create_monthly_and_overall_performance_plots(
        df_metrics=df_metrics,
        metric_to_plot=metric_to_plot,
        models_to_plot=models_to_plot,
        rename_dict=rename_dict,
        save_dir=save_dir,
    )

    plt.close("all")

    combinations = [
        ("Glacier Mapper Ensemble", "SnowMapper Ensemble", "Gla_Snow"),
        ("Glacier Mapper Ensemble", "Base Case Ensemble", "Gla_Base"),
        ("SnowMapper Ensemble", "Base Case Ensemble", "Snow_Base"),
        ("Gla MC ALD", "MC ALD", "Gla_MC"),
    ]

    for model_1, model_2, combo_name in combinations:
        # Model difference per month plot
        fig, ax = plt.subplots()
        ax, stats_df = plot_model_difference_per_month(
            metrics=df_metrics,
            model_1=model_1,
            model_2=model_2,
            static_df=static_df,
            metric="r2",
            threshold=0.1,
        )
        fig = ax.get_figure()
        # Save figure
        out = Path(save_dir) / f"{combo_name}_model_difference_per_month.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")

        # Save statistics to CSV
        stats_out = Path(save_dir) / f"{combo_name}_model_difference_statistics.csv"
        stats_df.to_csv(stats_out, index=False)
        print(f"✓ Model difference statistics saved to: {stats_out}")

        plt.show()
        plt.close("all")

        # Model difference magnitude plot
        fig, ax = plt.subplots()
        ax, magnitude_stats_df = plot_model_difference_magnitude(
            metrics=df_metrics,
            model_1=model_1,
            model_2=model_2,
            metric="r2",
            threshold=0.1,
        )
        fig = ax.get_figure()

        # Save figure
        out = Path(save_dir) / f"{combo_name}_model_difference_magnitude.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")

        # Save statistics to CSV
        magnitude_stats_out = (
            Path(save_dir) / f"{combo_name}_model_difference_magnitude_statistics.csv"
        )
        magnitude_stats_df.to_csv(magnitude_stats_out, index=False)
        print(
            f"✓ Model difference magnitude statistics saved to: {magnitude_stats_out}"
        )

        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()
