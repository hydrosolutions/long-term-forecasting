"""
Plotting utilities for the model evaluation dashboard.

This module provides consistent color schemes, styling functions, and plotting helpers
for creating uniform visualizations across the dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Model family color scheme
FAMILY_COLORS = {
    "BaseCase": "#1f77b4",  # Blue
    "SCA_Based": "#2ca02c",  # Green
    "SnowMapper_Based": "#ff7f0e",  # Orange
    "Other": "#d62728",  # Red (for models not in defined families)
}

# Individual model colors (within families)
MODEL_COLORS = {
    # BaseCase models (blue shades)
    "DeviationLR": "#1f77b4",
    "LR_Q_T_P": "#4c9bd6",
    "PerBasinScalingLR": "#7ab8e8",
    "ShortTermLR": "#a8d5f7",
    "ShortTerm_Features": "#d6ecff",
    # SCA_Based models (green shades)
    "LR_Q_SCA": "#2ca02c",
    "LR_Q_T_SCA": "#5fbf5f",
    # SnowMapper_Based models (orange shades)
    "CondenseLR": "#ff7f0e",
    "LR_Q_SWE": "#ff9933",
    "LR_Q_SWE_T": "#ffb366",
    "LR_Q_T_P_SWE": "#ffcc99",
    "LR_Q_dSWEdt_T_P": "#ffe6cc",
    "LongTermLR": "#ffa500",
    # Ensemble models (purple shades)
    "Ensemble_Mean": "#9467bd",
    "Ensemble_Median": "#b49ddb",
    "Ensemble_Weighted": "#d4c5f9",
    "Ensemble_Best3": "#8c564b",
    "Ensemble_Stacking": "#c49c94",
}

# Metric display names and formatting
METRIC_INFO = {
    "r2": {"display_name": "R2", "format": ".3f", "higher_is_better": True},
    "rmse": {"display_name": "RMSE", "format": ".2f", "higher_is_better": False},
    "nrmse": {"display_name": "NRMSE", "format": ".3f", "higher_is_better": False},
    "mae": {"display_name": "MAE", "format": ".2f", "higher_is_better": False},
    "mape": {"display_name": "MAPE (%)", "format": ".1f", "higher_is_better": False},
    "nse": {"display_name": "NSE", "format": ".3f", "higher_is_better": True},
    "kge": {"display_name": "KGE", "format": ".3f", "higher_is_better": True},
    "bias": {"display_name": "Bias", "format": ".2f", "higher_is_better": False},
    "pbias": {"display_name": "PBIAS (%)", "format": ".1f", "higher_is_better": False},
}

# Month names
MONTH_NAMES = {
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
    -1: "All Months",
}

# Month Names - Shifted as we predict on the last day of the previous month for the next month
MONTH_NAMES = {
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
    12: "January",
    -1: "All Months",
}


def get_model_color(model_name: str, model_family: Optional[str] = None) -> str:
    """
    Get the color for a specific model.

    Args:
        model_name: Name of the model
        model_family: Family of the model (optional)

    Returns:
        Hex color code
    """
    # First check if we have a specific color for this model
    if model_name in MODEL_COLORS:
        return MODEL_COLORS[model_name]

    # Otherwise use the family color
    if model_family and model_family in FAMILY_COLORS:
        return FAMILY_COLORS[model_family]

    # Default color
    return FAMILY_COLORS.get("Other", "#808080")


def get_color_mapping(df: pd.DataFrame, color_by: str = "model") -> Dict[str, str]:
    """
    Get color mapping for all unique values in a column.

    Args:
        df: DataFrame containing the data
        color_by: Column to create color mapping for

    Returns:
        Dictionary mapping values to colors
    """
    color_map = {}

    if color_by == "model":
        for model in df["model"].unique():
            family = (
                df[df["model"] == model]["model_family"].iloc[0]
                if "model_family" in df.columns
                else None
            )
            color_map[model] = get_model_color(model, family)
    else:
        # For other columns, use a default color palette
        unique_values = df[color_by].unique()
        colors = px.colors.qualitative.Set3[: len(unique_values)]
        color_map = dict(zip(unique_values, colors))

    return color_map


def apply_default_layout(fig: go.Figure, title: str = None) -> go.Figure:
    """
    Apply default layout settings to a figure.

    Args:
        fig: Plotly figure
        title: Optional title to set

    Returns:
        Updated figure
    """
    fig.update_layout(
        template="plotly_white",
        font=dict(size=12),
        title=dict(text=title, font=dict(size=16)) if title else None,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        margin=dict(l=50, r=150, t=50, b=50),
        hovermode="closest",
    )

    return fig


def format_metric_value(value: float, metric: str) -> str:
    """
    Format a metric value for display.

    Args:
        value: The metric value
        metric: The metric name

    Returns:
        Formatted string
    """
    if pd.isna(value):
        return "N/A"

    metric_fmt = METRIC_INFO.get(metric, {}).get("format", ".3f")
    return f"{value:{metric_fmt}}"


def create_performance_annotation(metrics_dict: Dict[str, float]) -> str:
    """
    Create a formatted annotation string for performance metrics.

    Args:
        metrics_dict: Dictionary of metric names and values

    Returns:
        Formatted annotation string
    """
    lines = []
    for metric, value in metrics_dict.items():
        if metric in METRIC_INFO:
            display_name = METRIC_INFO[metric]["display_name"]
            formatted_value = format_metric_value(value, metric)
            lines.append(f"{display_name}: {formatted_value}")

    return "<br>".join(lines)


def highlight_best_performers(
    df: pd.DataFrame, metric: str, n_best: int = 3
) -> List[str]:
    """
    Identify the best performing models for a given metric.

    Args:
        df: DataFrame with model performance data
        metric: Metric to evaluate
        n_best: Number of best performers to return

    Returns:
        List of best performing model names
    """
    if metric not in df.columns:
        return []

    # Remove rows with missing values for the metric
    df_clean = df.dropna(subset=[metric])

    if df_clean.empty:
        return []

    # Sort based on whether higher is better
    higher_is_better = METRIC_INFO.get(metric, {}).get("higher_is_better", True)
    df_sorted = df_clean.sort_values(metric, ascending=not higher_is_better)

    # Get unique models (in case of multiple entries per model)
    best_models = []
    for model in df_sorted["model"].values:
        if model not in best_models:
            best_models.append(model)
        if len(best_models) >= n_best:
            break

    return best_models


def create_hover_template(include_metrics: List[str] = None) -> str:
    """
    Create a hover template for plots.

    Args:
        include_metrics: List of metrics to include in hover

    Returns:
        Hover template string
    """
    template_parts = [
        "<b>%{customdata[0]}</b><br>",  # Model name
        "Family: %{customdata[1]}<br>",  # Model family
    ]

    if include_metrics:
        for i, metric in enumerate(include_metrics):
            display_name = METRIC_INFO.get(metric, {}).get("display_name", metric)
            template_parts.append(f"{display_name}: %{{customdata[{i + 2}]}}<br>")

    template_parts.append("<extra></extra>")  # Hide the trace box

    return "".join(template_parts)
