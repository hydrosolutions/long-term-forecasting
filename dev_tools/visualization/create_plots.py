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
    12: 'January', 1: 'February', 2: 'March', 3: 'April',
    4: 'May', 5: 'June', 6: 'July', 7: 'August',
    8: 'September', 9: 'October', 10: 'November', 11: 'December'
}

model_colors = {
    'Linear Regression Base': "#787878",  # blue
    'SnowMapper Ensemble': "#1F77B4",  # orange
    'MC ALD': "#2CA02C",  # green
}

metric_renamer = {
    'nse': 'NSE [-]',
    'rmse': 'RMSE [m³/s]',
    'mae': 'MAE [m³/s]',
    'r2': 'R² [-]',
    'pbias': 'PBIAS [-]',
    'kge': 'KGE [-]',
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

    plot_df = df[df['model'].isin(models)]

    if per_month:
        plot_df = plot_df[plot_df['level'] == "per_code_month"].copy()

        # sort so that months are in calendar order
        plot_df['month'] = pd.Categorical(plot_df['month'], categories=list(month_mapping.values()), ordered=True)
        plot_df = plot_df.sort_values('month')

        # order of models in legend
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=models, ordered=True)
        plot_df = plot_df.sort_values('model')

        sns.barplot(
            data=plot_df, x='month', 
            y=metric, hue='model', 
            ax=ax, estimator=np.median, 
            errorbar=('pi', 50), 
            capsize=0.1,
            palette=model_colors)

        """sns.boxplot(
            data=plot_df, x='month',
            y=metric, hue='model',
            ax=ax, 
            palette=model_colors,
        )"""
        
        ax.set_xlabel('Month')
        # rotate x-tick labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45)
        ax.set_ylim(0, 1)  # Adjust y-limits as needed
    else:
        plot_df = plot_df[plot_df['level'] == "per_code"].copy()

        # order of models in legend
        plot_df['model'] = pd.Categorical(plot_df['model'], categories=models, ordered=True)
        plot_df = plot_df.sort_values('model')

        sns.boxplot(
            data=plot_df, x='model',
            y=metric, 
            ax=ax, 
            palette=model_colors,
            legend=False
        )
        ax.set_ylim(0.4, 1.0)  # Adjust y-limits as needed

        ax.set_xlabel('')
        
    ax.set_ylabel(metric_renamer.get(metric, metric))
    ax.legend(title='Model', loc='lower right')
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
    ax = plot_monthly_overall(df_metrics, 
                           metric=metric_to_plot, 
                           models=list(rename_dict.values()), 
                           per_month=True, ax=ax)
    # draw black border around the whole figure
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)

    plt.tight_layout()
    out = Path(save_dir) / f"bar_{metric_to_plot}_per_month.png"
    fig.savefig(out)
    plt.show()

    fig, ax = plt.subplots()
    ax = plot_monthly_overall(df_metrics, 
                           metric=metric_to_plot, 
                           models=list(rename_dict.values()), 
                           per_month=False, ax=ax)
    # draw black border around the whole figure
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
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
    
    plot_df['code'] = plot_df['code'].astype(int)
    plot_df['date'] = pd.to_datetime(plot_df['date'])
    plot_df = plot_df[plot_df['code'] == code]
    plot_df = plot_df[(plot_df['date'] >= start_date) & (plot_df['date'] <= end_date)]

    cols_start_with_Q = [col for col in plot_df.columns if col.startswith('Q')]
    print("Columns in DataFrame:", plot_df.columns.tolist())
    print("Columns starting with 'Q':", cols_start_with_Q)

    if normalize:
        obs_min = plot_df['Q_obs'].min()
        obs_max = plot_df['Q_obs'].max()
        plot_df[cols_start_with_Q] = (plot_df[cols_start_with_Q] - obs_min) / (obs_max - obs_min)


    print(plot_df.head())

    fig, ax = plt.subplots(figsize=(10, 4))

    # Observed series
    ax.plot(
        plot_df['date'],
        plot_df['Q_obs'],
        label='Observed',
        color='black',
        linewidth=1.8,
    )

    # Predicted central estimate with dashed line + markers
    ax.plot(
        plot_df['date'],
        plot_df['Q_pred'],
        linestyle='--',
        marker='o',
        markersize=3,
        color='#2CA02C',
        linewidth=1.2,
        label='Predicted',
    )

    # 90% prediction interval as error bars (Q5 - Q95)
    lower_err_90 = plot_df['Q_pred'] - plot_df['Q5']
    upper_err_90 = plot_df['Q95'] - plot_df['Q_pred']
    ax.errorbar(
        plot_df['date'],
        plot_df['Q_pred'],
        yerr=[lower_err_90, upper_err_90],
        fmt='none',
        ecolor='#2CA02C',
        elinewidth=0.9,
        capsize=2,
        alpha=0.5,
        label='90% PI',
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

    ax.set_xlabel('Date')
    ax.set_ylabel('Scaled Discharge [-]')
    ax.legend()
    ax.margins(x=0)
    plt.tight_layout()
    plt.show()

    return fig, ax





if __name__ == "__main__":

    save_dir = "../monthly_forecasting_results/figures"

    config_plotting()

    models_to_plot = ['BaseCase_LR_Base',
                      'SnowMapper_Based_Ensemble',
                      'Uncertainty_MC_ALD']
    
    rename_dict = {
        'BaseCase_LR_Base': 'Linear Regression Base',
        'SnowMapper_Based_Ensemble': 'SnowMapper Ensemble',
        'Uncertainty_MC_ALD': 'MC ALD'
    }

    df_metrics = metric_handler.get_filtered_data()
    df_metrics['model'] = df_metrics['model'].map(rename_dict).fillna(df_metrics['model'])

    #rename months
    df_metrics['month'] = df_metrics['month'].replace(month_mapping)

    metric_to_plot = 'r2'

    """create_monthly_and_overall_performance_plots(
        df_metrics=df_metrics,
        metric_to_plot=metric_to_plot,
        models_to_plot=models_to_plot,
        rename_dict=rename_dict,
        save_dir=save_dir,
    )"""

    prediction_handler._load_all_predictions()

    all_predictions = prediction_handler._all_predictions

    print(all_predictions.keys())

    df_predictions = all_predictions['Uncertainty_MC_ALD']

    possible_codes = ["15149", "15194", "16936", "16510"]
    for code in possible_codes:
        fig, ax = plot_time_series_with_uncertainty(
            df=df_predictions,
            code=code,
            start_date='2019-01-01',
            end_date='2022-12-31',
            normalize = True
        )

        out = Path(save_dir) / f"time_series_with_uncertainty_code_{code}.png"
        fig.savefig(out)
        plt.close(fig)






