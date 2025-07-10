# Model Evaluation Dashboard

An interactive dashboard for visualizing monthly discharge forecasting model performance metrics and predictions.

## Overview

This dashboard provides comprehensive visualizations for comparing model performance across different metrics, basins, and time periods. It includes 5 interactive tabs:

1. **Performance by Month and Code** - Boxplots showing metric distribution across months
2. **Observed vs Predicted Time Series** - Line plots comparing observations with predictions
3. **Model Comparison by Basin** - Performance comparison across different basins
4. **Monthly Performance Analysis** - Heatmaps and bar charts showing seasonal patterns
5. **Interactive Data Table** - Filterable/sortable table with export functionality

## Prerequisites

The dashboard requires:
- Metrics data at `../monthly_forecasting_results/evaluation/metrics.csv`
- Prediction files in `../monthly_forecasting_results/` directory structure
- Python 3.11+ with uv installed

## Installation

The required dependencies (Dash, Plotly) are already specified in the project's `pyproject.toml`.

If you haven't already, sync the dependencies:
```bash
uv sync
```

## Running the Dashboard

From the project root directory (`monthly_forecasting/`), run:

```bash
uv run python visualization/dashboard.py
```

The dashboard will start and be available at: http://127.0.0.1:8050/

## Usage

1. **Select Models**: Use the multi-select dropdown at the top to choose which models to compare
2. **Navigate Tabs**: Click on different tabs to access various visualizations
3. **Filter Data**: Each tab has specific filtering options (metrics, basins, months)
4. **Export Data**: Use the Data Table tab to export filtered data as CSV

## Features

- **Consistent Color Coding**: Model families are color-coded consistently across all visualizations
  - BaseCase models: Blue shades
  - SCA_Based models: Green shades
  - SnowMapper_Based models: Orange shades
- **Interactive Plots**: All visualizations support zooming, panning, and hover information
- **Performance Optimized**: Data is cached for faster loading when switching between views
- **Responsive Design**: Dashboard adapts to different screen sizes

## Troubleshooting

If the dashboard fails to start:
1. Ensure you're in the correct directory (`monthly_forecasting/`)
2. Check that the metrics.csv file exists at the expected location
3. Verify all dependencies are installed: `uv pip list | grep -E "(dash|plotly)"`
4. Check for any error messages in the terminal

## Development

To modify the dashboard:
- `dashboard.py` - Main application and callback logic
- `data_handlers.py` - Data loading and processing
- `plotting_utils.py` - Styling and color management
- `dashboard_components.py` - Reusable UI components