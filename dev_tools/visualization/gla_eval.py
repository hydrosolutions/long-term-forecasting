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
    "Linear Regression Base": "#787878",  # blue
    "SnowMapper Ensemble": "#1F77B4",  # orange
    "MC ALD": "#2CA02C",  # green,
    "Glacier Mapper Ensemble": "#0ECFFF",  # red
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

def draw_feature_importance():
    folders_to_check = [
        "../monthly_forecasting_models/GlacierMapper_Based/Gla_GBT",
        "../monthly_forecasting_models/GlacierMapper_Based/Gla_GBT_Snow",
        "../monthly_forecasting_models/GlacierMapper_Based/Gla_GBT_SnowNorm",
        "../monthly_forecasting_models/SnowMapper_Based/Snow_GBT",
        "../monthly_forecasting_models/SnowMapper_Based/Snow_GBT_Norm",
        "../monthly_forecasting_models/SnowMapper_Based/Snow_GBT_LR",
        "../monthly_forecasting_models/BaseCase/GBT"
    ]
def main():

    save_dir = "../monthly_forecasting_results/figures/KGZ_Glacier_Eval"
    os.makedirs(save_dir, exist_ok=True)
    config_plotting()

    static_data = "/Users/sandrohunziker/hydrosolutions Dropbox/Sandro Hunziker/SAPPHIRE_Central_Asia_Technical_Work/data/kyg_data_forecast_tools/config/models_and_scalers/static_features/ML_basin_attributes_v2.csv"
    static_df = pd.read_csv(static_data)

    # if CODE in static_df.columns:
    if "CODE" in static_df.columns:
        static_df = static_df.rename(columns={"CODE": "code"})
        # to int
        static_df["code"] = static_df["code"].astype(int)


    models_to_plot = [
        "BaseCase_LR_Base",
        "SnowMapper_Based_Ensemble",
        "Uncertainty_MC_ALD",
        "GlacierMapper_Based_Ensemble",
    ]

    rename_dict = {
        "BaseCase_LR_Base": "Linear Regression Base",
        "SnowMapper_Based_Ensemble": "SnowMapper Ensemble",
        "Uncertainty_MC_ALD": "MC ALD",
        "GlacierMapper_Based_Ensemble": "Glacier Mapper Ensemble",
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

if __name__ == "__main__":
    main()