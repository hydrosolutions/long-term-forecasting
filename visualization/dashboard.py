# -*- coding: utf-8 -*-
"""
Enhanced Model Evaluation Dashboard for Monthly Discharge Forecasting.

This dashboard provides interactive visualizations for comparing model performance
across different metrics, basins, and time periods.
"""

import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64

# Import custom modules
from data_handlers import MetricsDataHandler, PredictionDataHandler
from plotting_utils import (
    get_color_mapping, apply_default_layout, format_metric_value,
    create_performance_annotation, highlight_best_performers,
    METRIC_INFO, MONTH_NAMES
)
from dashboard_components import (
    create_model_selector, create_metric_selector, create_basin_selector,
    create_month_selector, create_date_range_picker, create_evaluation_level_filter,
    create_loading_wrapper, create_header, create_control_panel,
    create_metric_table_columns, create_export_button
)

# Initialize data handlers
metrics_handler = MetricsDataHandler()
prediction_handler = PredictionDataHandler()

# Initialize Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define app layout
app.layout = html.Div([
    create_header(),
    
    # Global controls
    html.Div([
        html.Label("Select Models:", style={'fontWeight': 'bold'}),
        create_model_selector(metrics_handler.available_models),
    ], style={'padding': '20px', 'backgroundColor': '#f0f0f0'}),
    
    # Tabs
    dcc.Tabs(id="main-tabs", value='tab-1', children=[
        dcc.Tab(label='Performance by Month', value='tab-1'),
        dcc.Tab(label='Time Series Comparison', value='tab-2'),
        dcc.Tab(label='Model Comparison by Basin', value='tab-3'),
        dcc.Tab(label='Monthly Performance Heatmap', value='tab-4'),
        dcc.Tab(label='Data Table', value='tab-5'),
    ]),
    
    # Tab content
    html.Div(id='tab-content'),
    
    # Hidden div to store intermediate values
    html.Div(id='intermediate-value', style={'display': 'none'}),
    
    # Download component
    dcc.Download(id="download-dataframe-csv"),
])


# Tab 1: Performance by Month and Code
def create_tab1_layout():
    return html.Div([
        create_control_panel([
            {'label': 'Select Metric:', 'control': create_metric_selector()},
            {'label': 'Filter by Basin (optional):', 
             'control': dcc.Dropdown(
                 id='tab1-basin-filter',
                 options=[{'label': 'All Basins', 'value': 'all'}] + 
                         [{'label': f'Basin {code}', 'value': code} for code in metrics_handler.available_codes],
                 value='all',
                 clearable=False
             )}
        ]),
        create_loading_wrapper('tab1-graph', dcc.Graph(id='tab1-graph'))
    ])


# Tab 2: Observed vs Predicted Time Series
def create_tab2_layout():
    return html.Div([
        create_control_panel([
            {'label': 'Select Basin:', 'control': create_basin_selector(metrics_handler.available_codes)},
            {'label': 'Date Range (optional):', 'control': create_date_range_picker()}
        ]),
        create_loading_wrapper('tab2-graph', dcc.Graph(id='tab2-graph')),
        html.Div(id='tab2-metrics-display', style={'marginTop': '20px'})
    ])


# Tab 3: Model Comparison by Basin
def create_tab3_layout():
    return html.Div([
        create_control_panel([
            {'label': 'Select Metric:', 'control': create_metric_selector()},
            {'label': 'Sort Basins By:', 
             'control': dcc.RadioItems(
                 id='tab3-sort-by',
                 options=[
                     {'label': 'Basin Code', 'value': 'code'},
                     {'label': 'Median Performance', 'value': 'median'}
                 ],
                 value='median',
                 inline=True
             )}
        ]),
        create_loading_wrapper('tab3-graph', dcc.Graph(id='tab3-graph'))
    ])


# Tab 4: Monthly Performance Analysis
def create_tab4_layout():
    return html.Div([
        create_control_panel([
            {'label': 'Select Metric:', 'control': create_metric_selector()},
            {'label': 'Visualization Type:', 
             'control': dcc.RadioItems(
                 id='tab4-viz-type',
                 options=[
                     {'label': 'Heatmap', 'value': 'heatmap'},
                     {'label': 'Grouped Bar Chart', 'value': 'bar'}
                 ],
                 value='heatmap',
                 inline=True
             )}
        ]),
        create_loading_wrapper('tab4-graph', dcc.Graph(id='tab4-graph'))
    ])


# Tab 5: Interactive Data Table
def create_tab5_layout():
    return html.Div([
        create_control_panel([
            {'label': 'Evaluation Level Filter:', 'control': create_evaluation_level_filter()},
            {'label': 'Show Metrics:', 
             'control': dcc.Checklist(
                 id='tab5-metric-filter',
                 options=[{'label': METRIC_INFO[m]['display_name'], 'value': m} 
                         for m in metrics_handler.available_metrics],
                 value=metrics_handler.available_metrics[:5],  # Show first 5 by default
                 inline=True
             )}
        ]),
        dash_table.DataTable(
            id='tab5-table',
            columns=[],
            data=[],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            page_action="native",
            page_current=0,
            page_size=20,
            style_cell={
                'textAlign': 'left',
                'padding': '10px',
                'whiteSpace': 'normal',
                'height': 'auto',
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            export_format='csv',
        ),
        create_export_button()
    ])


# Callback to render tab content
@app.callback(Output('tab-content', 'children'),
              Input('main-tabs', 'value'))
def render_tab_content(active_tab):
    if active_tab == 'tab-1':
        return create_tab1_layout()
    elif active_tab == 'tab-2':
        return create_tab2_layout()
    elif active_tab == 'tab-3':
        return create_tab3_layout()
    elif active_tab == 'tab-4':
        return create_tab4_layout()
    elif active_tab == 'tab-5':
        return create_tab5_layout()


# Tab 1 callback: Performance by Month
@app.callback(
    Output('tab1-graph', 'figure'),
    [Input('model-selector', 'value'),
     Input('metric-selector', 'value'),
     Input('tab1-basin-filter', 'value')]
)
def update_tab1_graph(selected_models, selected_metric, basin_filter):
    if not selected_models:
        return go.Figure().add_annotation(text="Please select at least one model")
    
    # Filter data
    if basin_filter == 'all':
        df = metrics_handler.get_filtered_data(
            models=selected_models,
            evaluation_level='per_month'
        )
    else:
        df = metrics_handler.get_filtered_data(
            models=selected_models,
            codes=[basin_filter],
            evaluation_level='per_code_month'
        )
    
    # Remove 'all months' data
    df = df[df['month'] > 0]
    
    # Create boxplot
    color_map = get_color_mapping(df)
    
    fig = go.Figure()
    
    for model in selected_models:
        model_df = df[df['model'] == model]
        if not model_df.empty:
            # Group by month and calculate statistics
            for month in sorted(model_df['month'].unique()):
                month_data = model_df[model_df['month'] == month][selected_metric].dropna()
                if len(month_data) > 0:
                    fig.add_trace(go.Box(
                        y=month_data,
                        x=[MONTH_NAMES[month]] * len(month_data),
                        name=model,
                        marker_color=color_map[model],
                        boxmean='sd',
                        showlegend=(month == sorted(model_df['month'].unique())[0])
                    ))
    
    title = f"{METRIC_INFO[selected_metric]['display_name']} by Month"
    if basin_filter != 'all':
        title += f" (Basin {basin_filter})"
    
    fig.update_layout(
        title=title,
        xaxis_title="Month",
        yaxis_title=METRIC_INFO[selected_metric]['display_name'],
        boxmode='group'
    )
    
    return apply_default_layout(fig)


# Tab 2 callback: Time Series Comparison
@app.callback(
    [Output('tab2-graph', 'figure'),
     Output('tab2-metrics-display', 'children')],
    [Input('model-selector', 'value'),
     Input('basin-selector', 'value'),
     Input('date-range-picker', 'start_date'),
     Input('date-range-picker', 'end_date')]
)
def update_tab2_graph(selected_models, selected_basin, start_date, end_date):
    if not selected_models or not selected_basin:
        return go.Figure().add_annotation(text="Please select models and a basin"), ""
    
    fig = go.Figure()
    metrics_cards = []
    
    # Add observed data (from first model's predictions)
    first_model_data = prediction_handler.get_observed_vs_predicted(
        selected_models[0], selected_basin, start_date, end_date
    )
    
    if not first_model_data.empty:
        fig.add_trace(go.Scatter(
            x=first_model_data['date'],
            y=first_model_data['Q_obs'],
            mode='lines',
            name='Observed',
            line=dict(color='black', width=2)
        ))
    
    # Add predicted data for each model
    color_map = get_color_mapping(metrics_handler.df[metrics_handler.df['model'].isin(selected_models)])
    
    for model in selected_models:
        pred_data = prediction_handler.get_observed_vs_predicted(
            model, selected_basin, start_date, end_date
        )
        
        if not pred_data.empty:
            fig.add_trace(go.Scatter(
                x=pred_data['date'],
                y=pred_data['Q_pred'],
                mode='lines',
                name=f'{model} (Predicted)',
                line=dict(color=color_map.get(model, '#808080'), dash='dash')
            ))
            
            # Calculate metrics for this model-basin combination
            metrics_df = metrics_handler.get_filtered_data(
                models=[model],
                codes=[selected_basin],
                evaluation_level='per_code'
            )
            
            if not metrics_df.empty:
                metrics_dict = metrics_df.iloc[0].to_dict()
                metric_text = create_performance_annotation({
                    k: v for k, v in metrics_dict.items() 
                    if k in ['nse', 'rmse', 'pbias', 'kge']
                })
                
                metrics_cards.append(
                    html.Div([
                        html.H5(model, style={'color': color_map.get(model, '#808080')}),
                        html.Pre(metric_text)
                    ], style={'display': 'inline-block', 'margin': '10px', 
                             'padding': '10px', 'border': '1px solid #ddd',
                             'borderRadius': '5px'})
                )
    
    fig.update_layout(
        title=f"Observed vs Predicted Discharge - Basin {selected_basin}",
        xaxis_title="Date",
        yaxis_title="Discharge (m3/s)",
        hovermode='x unified'
    )
    
    metrics_display = html.Div(metrics_cards)
    
    return apply_default_layout(fig), metrics_display


# Tab 3 callback: Model Comparison by Basin
@app.callback(
    Output('tab3-graph', 'figure'),
    [Input('model-selector', 'value'),
     Input('metric-selector', 'value'),
     Input('tab3-sort-by', 'value')]
)
def update_tab3_graph(selected_models, selected_metric, sort_by):
    if not selected_models:
        return go.Figure().add_annotation(text="Please select at least one model")
    
    # Get per-basin data
    df = metrics_handler.get_filtered_data(
        models=selected_models,
        evaluation_level='per_code'
    )
    
    if df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    # Sort basins
    if sort_by == 'median':
        # Calculate median performance per basin
        basin_medians = df.groupby('code')[selected_metric].median().sort_values(
            ascending=not METRIC_INFO[selected_metric]['higher_is_better']
        )
        basin_order = basin_medians.index.tolist()
    else:
        basin_order = sorted(df['code'].unique())
    
    # Create subplots
    from plotly.subplots import make_subplots
    n_basins = len(basin_order)
    n_cols = 4
    n_rows = (n_basins + n_cols - 1) // n_cols
    
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f"Basin {code}" for code in basin_order],
        vertical_spacing=0.15,
        horizontal_spacing=0.05
    )
    
    color_map = get_color_mapping(df)
    
    for idx, basin in enumerate(basin_order):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        basin_df = df[df['code'] == basin]
        
        for model in selected_models:
            model_data = basin_df[basin_df['model'] == model]
            if not model_data.empty and selected_metric in model_data.columns:
                value = model_data[selected_metric].iloc[0]
                if not pd.isna(value):
                    fig.add_trace(
                        go.Bar(
                            x=[model],
                            y=[value],
                            name=model,
                            marker_color=color_map.get(model, '#808080'),
                            showlegend=(idx == 0),
                            text=format_metric_value(value, selected_metric),
                            textposition='auto'
                        ),
                        row=row, col=col
                    )
    
    fig.update_layout(
        title=f"{METRIC_INFO[selected_metric]['display_name']} Comparison by Basin",
        height=200 * n_rows,
        showlegend=True
    )
    
    # Update axes
    fig.update_xaxes(tickangle=-45)
    fig.update_yaxes(title_text=METRIC_INFO[selected_metric]['display_name'], row=n_rows, col=1)
    
    return fig


# Tab 4 callback: Monthly Performance Analysis
@app.callback(
    Output('tab4-graph', 'figure'),
    [Input('model-selector', 'value'),
     Input('metric-selector', 'value'),
     Input('tab4-viz-type', 'value')]
)
def update_tab4_graph(selected_models, selected_metric, viz_type):
    if not selected_models:
        return go.Figure().add_annotation(text="Please select at least one model")
    
    # Get per-month data
    df = metrics_handler.get_filtered_data(
        models=selected_models,
        evaluation_level='per_month'
    )
    
    # Remove 'all months' data
    df = df[df['month'] > 0]
    
    if df.empty:
        return go.Figure().add_annotation(text="No monthly data available")
    
    # Prepare data for visualization
    pivot_df = df.pivot(index='model', columns='month', values=selected_metric)
    
    # Rename columns to month names
    pivot_df.columns = [MONTH_NAMES[month] for month in pivot_df.columns]
    
    if viz_type == 'heatmap':
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            colorscale='RdBu_r' if METRIC_INFO[selected_metric]['higher_is_better'] else 'RdBu',
            text=[[format_metric_value(val, selected_metric) for val in row] 
                  for row in pivot_df.values],
            texttemplate='%{text}',
            textfont={"size": 10},
            colorbar=dict(title=METRIC_INFO[selected_metric]['display_name'])
        ))
        
        fig.update_layout(
            title=f"Monthly {METRIC_INFO[selected_metric]['display_name']} Heatmap",
            xaxis_title="Month",
            yaxis_title="Model"
        )
    else:
        # Create grouped bar chart
        fig = go.Figure()
        color_map = get_color_mapping(df)
        
        for model in selected_models:
            model_data = df[df['model'] == model].sort_values('month')
            if not model_data.empty:
                fig.add_trace(go.Bar(
                    x=[MONTH_NAMES[m] for m in model_data['month']],
                    y=model_data[selected_metric],
                    name=model,
                    marker_color=color_map.get(model, '#808080'),
                    text=[format_metric_value(v, selected_metric) 
                          for v in model_data[selected_metric]],
                    textposition='auto'
                ))
        
        fig.update_layout(
            title=f"Monthly {METRIC_INFO[selected_metric]['display_name']} Comparison",
            xaxis_title="Month",
            yaxis_title=METRIC_INFO[selected_metric]['display_name'],
            barmode='group'
        )
    
    return apply_default_layout(fig)


# Tab 5 callbacks: Data Table
@app.callback(
    [Output('tab5-table', 'columns'),
     Output('tab5-table', 'data')],
    [Input('model-selector', 'value'),
     Input('evaluation-level-filter', 'value'),
     Input('tab5-metric-filter', 'value')]
)
def update_tab5_table(selected_models, eval_level, selected_metrics):
    if not selected_models or not selected_metrics:
        return [], []
    
    # Get filtered data
    if eval_level == 'all':
        df = metrics_handler.get_filtered_data(models=selected_models)
    else:
        df = metrics_handler.get_filtered_data(
            models=selected_models,
            evaluation_level=eval_level
        )
    
    # Convert month numbers to names
    df['month_name'] = df['month'].apply(lambda x: MONTH_NAMES.get(x, str(x)))
    
    # Prepare columns
    columns = [
        {"name": "Model", "id": "model", "type": "text"},
        {"name": "Family", "id": "model_family", "type": "text"},
        {"name": "Basin", "id": "code", "type": "text"},
        {"name": "Month", "id": "month_name", "type": "text"},
    ]
    
    for metric in selected_metrics:
        if metric in df.columns:
            columns.append({
                "name": METRIC_INFO[metric]['display_name'],
                "id": metric,
                "type": "numeric",
                "format": {"specifier": METRIC_INFO[metric]['format']}
            })
    
    # Prepare data
    data = df.to_dict('records')
    
    return columns, data


# Export callback
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("export-button", "n_clicks"),
    State("tab5-table", "data"),
    prevent_initial_call=True,
)
def export_data(n_clicks, table_data):
    if n_clicks > 0 and table_data:
        df = pd.DataFrame(table_data)
        return dcc.send_data_frame(df.to_csv, "model_evaluation_data.csv", index=False)


# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=8050)