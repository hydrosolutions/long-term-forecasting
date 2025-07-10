# Enhanced Model Evaluation Dashboard

## Objective
Implement a comprehensive interactive dashboard for visualizing model performance metrics and predictions with multiple visualization tabs, consistent color coding, and interactive data exploration capabilities.

## Context
- GitHub Issue: #18
- Existing dashboard skeleton: `old_files/model_dashboard.py`
- Metrics data: `../monthly_forecasting_results/evaluation/metrics.csv`
- Prediction loader: `evaluation/prediction_loader.py`

## Plan

### Phase 1: Project Setup and Data Handlers
- [ ] Create `visualization/` directory structure
- [ ] Implement `MetricsDataHandler` class
  - Load metrics.csv with proper data types
  - Handle different aggregation levels (overall, per_code)
  - Provide filtering methods by model, metric, code, month
- [ ] Implement `PredictionDataHandler` class
  - Integrate with existing prediction_loader.py
  - Cache loaded predictions for performance
  - Handle ensemble members appropriately

### Phase 2: Color Management and Utilities
- [ ] Create color scheme dictionary for model families
  - BaseCase: blue shades
  - SCA_Based: green shades  
  - SnowMapper_Based: orange shades
- [ ] Implement plotting utilities
  - Consistent styling functions
  - Legend management
  - Axis formatting helpers

### Phase 3: Dashboard Layout and Controls
- [ ] Main dashboard structure with Dash
  - Header with title
  - Model selection dropdown (multi-select)
  - Metric selection dropdown
  - Tab container
- [ ] Implement responsive layout
- [ ] Add loading indicators

### Phase 4: Tab Implementation

#### Tab 1: Performance by Month and Code
- [ ] Boxplot visualization
  - X-axis: months (with proper labels)
  - Y-axis: selected metric
  - Color by model (using family colors)
- [ ] Basin code filter dropdown
- [ ] Option to show all codes or specific code

#### Tab 2: Observed vs Predicted Time Series  
- [ ] Load predictions for selected models and basin
- [ ] Line plot with:
  - Q_obs as solid line
  - Q_pred as dashed lines (one per model)
  - Consistent colors per model family
- [ ] Display metrics (NSE, RMSE, etc.) in plot annotation
- [ ] Date range selector

#### Tab 3: Model Comparison by Basin
- [ ] Boxplot or violin plot
  - One subplot per basin
  - Models on x-axis
  - Selected metric on y-axis
- [ ] Sort basins by median performance
- [ ] Highlight best performing model per basin

#### Tab 4: Monthly Performance Analysis
- [ ] Heatmap visualization
  - Rows: Models
  - Columns: Months
  - Values: Selected metric
- [ ] Alternative: Grouped bar chart
- [ ] Highlight seasonal patterns

#### Tab 5: Interactive Data Table
- [ ] DataTable component with:
  - All metrics as columns
  - Model, Family, Code, Month columns
  - Sorting and filtering capabilities
- [ ] Evaluation procedure filter:
  - Overall
  - Per code
  - Per month
  - Per code and month
- [ ] Export to CSV functionality
- [ ] Column visibility toggles

### Phase 5: Performance and Polish
- [ ] Implement data caching strategy
  - Cache metrics data on first load
  - Cache predictions per model
- [ ] Add error handling and user feedback
- [ ] Optimize plot rendering for large datasets
- [ ] Add help tooltips for controls

## Implementation Notes

### Data Structure Considerations
```python
# Metrics data columns
metrics_cols = ['r2', 'rmse', 'nrmse', 'mae', 'mape', 'nse', 'kge', 'bias', 'pbias']

# Model family mapping
MODEL_FAMILIES = {
    "BaseCase": ["DeviationLR", "LR_Q_T_P", "PerBasinScalingLR", "ShortTermLR", "ShortTerm_Features"],
    "SCA_Based": ["LR_Q_SCA", "LR_Q_T_SCA"],
    "SnowMapper_Based": ["CondenseLR", "LR_Q_SWE", "LR_Q_SWE_T", "LR_Q_T_P_SWE", "LR_Q_dSWEdt_T_P", "LongTermLR", "ShortTermLR", "ShortTerm_Features"]
}

# Color scheme
FAMILY_COLORS = {
    "BaseCase": "#1f77b4",  # Blue
    "SCA_Based": "#2ca02c",  # Green
    "SnowMapper_Based": "#ff7f0e"  # Orange
}
```

### Key Design Decisions
1. Use Dash callbacks for interactivity
2. Implement client-side callbacks where possible for performance
3. Use Plotly's built-in theming for consistent appearance
4. Modular component design for maintainability

## Testing Strategy
1. Unit tests for data handlers
2. Integration tests for dashboard callbacks
3. Visual regression tests for plots
4. Performance benchmarks for large datasets

## Review Points
- Color consistency across all visualizations
- Performance with full dataset
- Error handling for missing data
- Mobile responsiveness
- Export functionality completeness