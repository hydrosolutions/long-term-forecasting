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
metric_handler = MetricsDataHandler(
    metrics_path="../monthly_forecasting_results/evaluation/TJK/metrics.csv"
)
prediction_handler = PredictionDataHandler(
    results_dir="/Users/sandrohunziker/hydrosolutions Dropbox/Sandro Hunziker/SAPPHIRE_Central_Asia_Technical_Work/data/taj_data_forecast_tools/intermediate_data/long_term_predictions/monthly"
)

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
    "Linear Regression Base": "#787878",  # blue
    "SM GBT LR": "#1F77B4",  # orange
    "MC ALD": "#2CA02C",  # green
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


def plot_time_series_with_uncertainty(
    df: pd.DataFrame,
    code: str,
    start_date: str,
    end_date: str,
    normalize: bool = False,
):
    """Plot time series of observed and predicted values with uncertainty bounds.

    Args:
        df: DataFrame containing columns ['date', 'observed', 'predicted', 'lower_bound', 'upper_bound']
        code: Code of the location to plot
        start_date: Start date for the plot (YYYY-MM-DD)
        end_date: End date for the plot (YYYY-MM-DD)
    """
    code = int(code)
    plot_df = df.copy()

    plot_df["code"] = plot_df["code"].astype(int)
    plot_df["date"] = pd.to_datetime(plot_df["date"])
    plot_df = plot_df[plot_df["code"] == code]
    plot_df = plot_df[(plot_df["date"] >= start_date) & (plot_df["date"] <= end_date)]

    cols_start_with_Q = [col for col in plot_df.columns if col.startswith("Q")]
    print("Columns in DataFrame:", plot_df.columns.tolist())
    print("Columns starting with 'Q':", cols_start_with_Q)

    if normalize:
        obs_min = plot_df["Q_obs"].min()
        obs_max = plot_df["Q_obs"].max()
        plot_df[cols_start_with_Q] = (plot_df[cols_start_with_Q] - obs_min) / (
            obs_max - obs_min
        )

    print(plot_df.head())

    fig, ax = plt.subplots(figsize=(10, 4))

    # Observed series
    ax.plot(
        plot_df["date"],
        plot_df["Q_obs"],
        label="Observed",
        color="black",
        linewidth=1.8,
    )

    # Predicted central estimate with dashed line + markers
    ax.plot(
        plot_df["date"],
        plot_df["Q_pred"],
        linestyle="--",
        marker="o",
        markersize=3,
        color="#2CA02C",
        linewidth=1.2,
        label="Predicted",
    )

    # 90% prediction interval as error bars (Q5 - Q95)
    lower_err_90 = plot_df["Q_pred"] - plot_df["Q5"]
    upper_err_90 = plot_df["Q95"] - plot_df["Q_pred"]
    ax.errorbar(
        plot_df["date"],
        plot_df["Q_pred"],
        yerr=[lower_err_90, upper_err_90],
        fmt="none",
        ecolor="#2CA02C",
        elinewidth=0.9,
        capsize=2,
        alpha=0.5,
        label="90% PI",
    )

    """# (Optional) add a narrower 50% PI; comment out if not needed
    lower_err_50 = plot_df['Q_pred'] - plot_df['Q25']
    upper_err_50 = plot_df['Q75'] - plot_df['Q_pred']
    ax.errorbar(
        plot_df['date'],
        plot_df['Q_pred'],
        yerr=[lower_err_50, upper_err_50],
        fmt='none',
        ecolor='#2CA02C',
        elinewidth=1.4,
        capsize=2,
        alpha=0.9,
        label='50% PI',
    )"""

    ax.set_xlabel("Date")
    if normalize:
        ax.set_ylabel("Scaled Discharge [-]")
    else:
        ax.set_ylabel("Discharge [m³/s]")
    ax.legend()
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()

    return fig, ax


def plot_uncertainty_exceedance(
    df: pd.DataFrame,
    overall: bool,
    ax: plt.Axes | None = None,
):
    """Plot uncertainty exceedance for a given model.
    On the x-axis is the nominal exceedance probability. On the y-axis is the empirical exceedance probability.
    The exceedance probability for Q10 should be 0.9 - so we actually plot the non-exceedance probability (1 - exceedance).
    this should align with the 1:1 line for a well-calibrated model with the quantiles.

    If it is overall, we plot the overall exceedance for all basins. If not, we compute the per-basin statistics and plot a boxplot of the statistic values.
    Args:
        df: DataFrame containing columns ['model', 'code', 'month', 'Q_obs', Q5, Q25, ... Q75, Q95']
        model: Model name to plot
        overall: If True, plot overall exceedance for all basins; otherwise per basin
        ax: Matplotlib Axes to plot on. If None, creates a new figure and axes.
    Returns:
        Matplotlib Axes with the plot
    """
    from dev_tools.eval_scr.metric_functions import prob_exceedance

    if ax is None:
        fig, ax = plt.subplots()

    quantile_cols = [
        col for col in df.columns if col.startswith("Q") and col[1:].isdigit()
    ]
    quantiles = sorted([int(col[1:]) for col in quantile_cols])

    if overall:
        # overall exceedance
        overall_exceedance = {}
        for q in quantiles:
            col_name = f"Q{q}"
            overall_exceedance[q] = prob_exceedance(df["Q_obs"], df[col_name])

        # plot overall exceedance
        ax.plot(
            [q / 100 for q in quantiles],
            [overall_exceedance[q] for q in quantiles],
            marker="o",
            linestyle="-",
            color="#2CA02C",
            label="Overall Exceedance",
        )
    else:
        df_results = []
        for code, group in df.groupby("code"):
            code_exceedance = {}
            for q in quantiles:
                col_name = f"Q{q}"
                code_exceedance[q] = prob_exceedance(group["Q_obs"], group[col_name])
            df_code = pd.DataFrame(
                {
                    "quantile": [q / 100 for q in quantiles],
                    "empirical_non_exceedance": [code_exceedance[q] for q in quantiles],
                    "code": code,
                }
            )
            df_results.append(df_code)
        plot_df = pd.concat(df_results, ignore_index=True)
        print(plot_df.head())
        ax.boxplot(
            x=[
                plot_df[plot_df["quantile"] == q / 100]["empirical_non_exceedance"]
                for q in quantiles
            ],
            positions=[q / 100 for q in quantiles],
            widths=0.02,
            boxprops=dict(color="#2CA02C"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="#2CA02C"),
            capprops=dict(color="#2CA02C"),
            flierprops=dict(
                markerfacecolor="#2CA02C", marker="o", markersize=3, alpha=0.5
            ),
        )

        # Provide a small margin so edge quantile boxes (e.g. 0.05 / 0.95) are fully visible
        ax.set_xlim(-0.02, 1.02)

    # 1:1 line
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="1:1 Line",
    )

    ax.set_xlabel("Nominal Non-Exceedance Probability")
    ax.set_ylabel("Empirical Non-Exceedance Probability")
    ax.set_title("Uncertainty Exceedance Plot")
    # set the x ticks to be the same as the quantiles
    ax.set_xticks([q / 100 for q in quantiles])
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    return ax


def plot_metric_per_month_basin(
    df: pd.DataFrame,
    metric: str,
    code: int,
    models: List[str],
    save_dir: str,
):
    """
    Plots the metric per month and basin.
    for the models in models_to_plot.
    - uses a line plot with markers for each month.
    """

    df = df[df["model"].isin(models)]
    df = df[df["code"] == code]

    # sort so that months are in calendar order
    df["month"] = pd.Categorical(
        df["month"], categories=list(month_mapping.values()), ordered=True
    )

    df = df.sort_values("month")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        data=df,
        x="month",
        y=metric,
        hue="model",
        marker="o",
        ax=ax,
        palette=model_colors,
    )

    ax.set_xlabel("Month")
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_ylabel(metric_renamer.get(metric, metric))
    ax.set_ylim(0, 1)  # Adjust y-limits as needed
    ax.legend(title="Model", loc="lower right")
    plt.tight_layout()
    out = Path(save_dir) / f"line_{metric}_code_{code}.png"
    fig.savefig(out)
    plt.show()


if __name__ == "__main__":
    save_dir = "../monthly_forecasting_results/figures/TJK"
    os.makedirs(save_dir, exist_ok=True)
    config_plotting()

    models_to_plot = [
        "all_models_LR_Base",
        "all_models_SM_GBT_LR",
        "Uncertainty_MC_ALD",
    ]

    rename_dict = {
        "all_models_LR_Base": "Linear Regression Base",
        "all_models_SM_GBT_LR": "SM GBT LR",
        "all_models_MC_ALD": "MC ALD",
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

    # plot per month and basin for selected basins
    possible_codes = ["17084", "17288", "17050"]
    for code in possible_codes:
        plot_metric_per_month_basin(
            df=df_metrics,
            metric=metric_to_plot,
            code=int(code),
            models=list(rename_dict.values()),
            save_dir=save_dir,
        )

    prediction_handler._load_all_predictions()

    all_predictions = prediction_handler._all_predictions

    print(all_predictions.keys())

    df_predictions = all_predictions["all_models_MC_ALD"]

    unique_codes = df_predictions["code"].unique()
    print(f"Unique codes in predictions: {unique_codes}")
    print(f"Number of unique codes: {len(unique_codes)}")

    # uncertainty exceedance plot one ax for overall and one for per-basin
    fig, ax = plt.subplots()
    ax = plot_uncertainty_exceedance(df_predictions, overall=False, ax=ax)
    plt.tight_layout()
    out = Path(save_dir) / f"uncertainty_exceedance_MC_ALD.png"
    fig.savefig(out)
    plt.show()

    possible_codes = ["15149", "15283", "16936", "16510"]
    for code in possible_codes:
        fig, ax = plot_time_series_with_uncertainty(
            df=df_predictions,
            code=code,
            start_date="2023-01-01",
            end_date="2025-09-30",
            normalize=False,
        )

        out = Path(save_dir) / f"time_series_with_uncertainty_code_{code}.png"
        fig.savefig(out)
        plt.close(fig)
