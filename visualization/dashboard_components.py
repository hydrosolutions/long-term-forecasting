"""
Reusable UI components for the model evaluation dashboard.

This module provides consistent UI elements that can be used across different tabs
and sections of the dashboard.
"""

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc
from typing import List, Dict, Optional, Any
from plotting_utils import METRIC_INFO, MONTH_NAMES


def create_model_selector(model_options: List[str], default_models: List[str] = None) -> dcc.Dropdown:
    """
    Create a multi-select dropdown for model selection.
    
    Args:
        model_options: List of available model names
        default_models: List of models to select by default
        
    Returns:
        Dash dropdown component
    """
    if default_models is None:
        default_models = model_options[:3] if len(model_options) > 3 else model_options
    
    return dcc.Dropdown(
        id='model-selector',
        options=[{'label': model, 'value': model} for model in model_options],
        value=default_models,
        multi=True,
        placeholder="Select models to compare...",
        style={'width': '100%'}
    )


def create_metric_selector(metric_options: List[str] = None, default_metric: str = 'nse') -> dcc.Dropdown:
    """
    Create a dropdown for metric selection.
    
    Args:
        metric_options: List of available metrics (uses METRIC_INFO keys if None)
        default_metric: Default metric to select
        
    Returns:
        Dash dropdown component
    """
    if metric_options is None:
        metric_options = list(METRIC_INFO.keys())
    
    options = []
    for metric in metric_options:
        if metric in METRIC_INFO:
            label = METRIC_INFO[metric]['display_name']
        else:
            label = metric.upper()
        options.append({'label': label, 'value': metric})
    
    return dcc.Dropdown(
        id='metric-selector',
        options=options,
        value=default_metric,
        clearable=False,
        style={'width': '100%'}
    )


def create_basin_selector(basin_options: List[str], default_basin: str = None) -> dcc.Dropdown:
    """
    Create a dropdown for basin selection.
    
    Args:
        basin_options: List of available basin codes
        default_basin: Default basin to select
        
    Returns:
        Dash dropdown component
    """
    if default_basin is None and basin_options:
        default_basin = basin_options[0]
    
    return dcc.Dropdown(
        id='basin-selector',
        options=[{'label': f'Basin {code}', 'value': code} for code in basin_options],
        value=default_basin,
        clearable=False,
        style={'width': '100%'}
    )


def create_month_selector(month_options: List[int], default_month: int = None, multi: bool = False) -> dcc.Dropdown:
    """
    Create a dropdown for month selection.
    
    Args:
        month_options: List of available months
        default_month: Default month to select
        multi: Whether to allow multiple selection
        
    Returns:
        Dash dropdown component
    """
    options = []
    for month in month_options:
        label = MONTH_NAMES.get(month, f"Month {month}")
        options.append({'label': label, 'value': month})
    
    if default_month is None and month_options:
        default_month = month_options[0] if not multi else month_options
    
    return dcc.Dropdown(
        id='month-selector',
        options=options,
        value=default_month,
        multi=multi,
        clearable=False,
        style={'width': '100%'}
    )


def create_date_range_picker(start_date: str = None, end_date: str = None) -> dcc.DatePickerRange:
    """
    Create a date range picker for time series filtering.
    
    Args:
        start_date: Default start date
        end_date: Default end date
        
    Returns:
        Dash date range picker component
    """
    return dcc.DatePickerRange(
        id='date-range-picker',
        start_date=start_date,
        end_date=end_date,
        display_format='YYYY-MM-DD',
        style={'width': '100%'}
    )


def create_evaluation_level_filter() -> dcc.RadioItems:
    """
    Create radio buttons for filtering evaluation level.
    
    Returns:
        Dash radio items component
    """
    return dcc.RadioItems(
        id='evaluation-level-filter',
        options=[
            {'label': 'All Levels', 'value': 'all'},
            {'label': 'Overall', 'value': 'overall'},
            {'label': 'Per Basin', 'value': 'per_code'},
            {'label': 'Per Month', 'value': 'per_month'},
            {'label': 'Per Basin & Month', 'value': 'per_code_month'}
        ],
        value='all',
        inline=True,
        style={'marginBottom': '10px'}
    )


def create_loading_wrapper(component_id: str, children: Any) -> dcc.Loading:
    """
    Wrap a component in a loading indicator.
    
    Args:
        component_id: ID for the loading component
        children: Component(s) to wrap
        
    Returns:
        Dash loading component
    """
    return dcc.Loading(
        id=f"loading-{component_id}",
        type="default",
        children=children,
        style={'minHeight': '400px'}
    )


def create_info_card(title: str, content: str, color: str = "primary") -> dbc.Card:
    """
    Create an info card component.
    
    Args:
        title: Card title
        content: Card content
        color: Bootstrap color theme
        
    Returns:
        Dash bootstrap card component
    """
    return dbc.Card(
        [
            dbc.CardHeader(html.H5(title)),
            dbc.CardBody(html.P(content))
        ],
        color=color,
        outline=True,
        style={'marginBottom': '10px'}
    )


def create_control_panel(controls: List[Dict[str, Any]]) -> html.Div:
    """
    Create a control panel with labeled controls.
    
    Args:
        controls: List of dicts with 'label' and 'control' keys
        
    Returns:
        Dash html.Div containing the control panel
    """
    control_elements = []
    
    for control_dict in controls:
        control_elements.extend([
            html.Label(control_dict['label'], style={'fontWeight': 'bold', 'marginTop': '10px'}),
            control_dict['control'],
        ])
    
    return html.Div(
        control_elements,
        style={
            'padding': '15px',
            'backgroundColor': '#f8f9fa',
            'borderRadius': '5px',
            'marginBottom': '20px'
        }
    )


def create_metric_table_columns(metrics: List[str]) -> List[Dict[str, Any]]:
    """
    Create column definitions for the metrics data table.
    
    Args:
        metrics: List of metric names to include
        
    Returns:
        List of column definitions for dash_table
    """
    columns = [
        {"name": "Model", "id": "model", "type": "text"},
        {"name": "Family", "id": "model_family", "type": "text"},
        {"name": "Basin", "id": "code", "type": "text"},
        {"name": "Month", "id": "month", "type": "numeric"},
    ]
    
    for metric in metrics:
        display_name = METRIC_INFO.get(metric, {}).get('display_name', metric.upper())
        columns.append({
            "name": display_name,
            "id": metric,
            "type": "numeric",
            "format": {"specifier": METRIC_INFO.get(metric, {}).get('format', '.3f')}
        })
    
    return columns


def create_export_button() -> html.Button:
    """
    Create an export button for downloading data.
    
    Returns:
        Dash html.Button component
    """
    return html.Button(
        "Export to CSV",
        id="export-button",
        n_clicks=0,
        style={
            'marginTop': '10px',
            'padding': '10px 20px',
            'backgroundColor': '#28a745',
            'color': 'white',
            'border': 'none',
            'borderRadius': '5px',
            'cursor': 'pointer'
        }
    )


def create_header() -> html.Div:
    """
    Create the dashboard header.
    
    Returns:
        Dash html.Div containing the header
    """
    return html.Div(
        [
            html.H1(
                "Model Evaluation Dashboard",
                style={'textAlign': 'center', 'marginBottom': '20px'}
            ),
            html.Hr()
        ]
    )


def create_tab_layout(tab_id: str, controls: List[Dict[str, Any]], graph_id: str, 
                     additional_components: List[Any] = None) -> html.Div:
    """
    Create a standard tab layout with controls and graph.
    
    Args:
        tab_id: Unique identifier for the tab
        controls: List of control definitions
        graph_id: ID for the main graph component
        additional_components: Additional components to include
        
    Returns:
        Dash html.Div containing the tab layout
    """
    layout_components = [
        create_control_panel(controls),
        create_loading_wrapper(graph_id, dcc.Graph(id=graph_id))
    ]
    
    if additional_components:
        layout_components.extend(additional_components)
    
    return html.Div(
        layout_components,
        id=f"{tab_id}-layout",
        style={'padding': '20px'}
    )