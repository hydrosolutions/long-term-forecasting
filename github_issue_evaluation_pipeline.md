# GitHub Issue: Implement Comprehensive Evaluation Pipeline for Monthly Discharge Forecasting

## Summary
Create an automated evaluation pipeline that scrapes all model predictions, evaluates them at multiple levels (per-code, per-month, overall), creates ensemble predictions, and enhances the dashboard for better model comparison and visualization.

## Background
Currently, we have multiple model predictions stored in `../monthly_forecasting_results/model_type*/class_*/predictions.csv`. The existing evaluation code in `old_files/evaluate_hindcast.py` needs to be refactored to use the metric functions from `eval_scr` module. The dashboard needs enhancement to show predictions vs observations.

## Requirements

### 1. Evaluation Pipeline
- Automatically discover and load all prediction files
- Evaluate models on:
  - Per-basin (code) level
  - Per-month level
  - Overall aggregated level
- Use standardized metrics from `eval_scr.metric_functions`

### 2. Ensemble Creation
- Create ensemble predictions for each model family:
  - BaseCase (LR_Q_T_P, DeviationLR, etc.)
  - SCA_Based (LR_Q_SCA, LR_Q_T_SCA)
  - SnowMapper_Based (LR_Q_SWE, LR_Q_SWE_T, etc.)
- Create a global ensemble using all model families
- Handle existing ensemble columns (Q_model_name format)

### 3. Output Requirements
- Save evaluation metrics to `../monthly_forecasting_results/evaluation/metrics.csv`
- Include metrics: r2, rmse, nrmse, mae, mape, nse, kge, bias, pbias
- Save ensemble predictions and their evaluations
- Generate summary statistics

### 4. Dashboard Enhancement
- Add new tab for "Predictions vs Observations" visualization
- Show time series plots with multiple models
- Include ensemble models in comparisons
- Maintain compatibility with existing visualizations

## Tasks Breakdown

### Task 1: Core Evaluation Module (Priority: High)
**File**: `evaluation/evaluate_models.py`
- [ ] Import metrics from `eval_scr.metric_functions`
- [ ] Implement `calculate_metrics()` function
- [ ] Implement `evaluate_per_code()` function
- [ ] Implement `evaluate_per_month()` function
- [ ] Implement `evaluate_overall()` function
- [ ] Add probability of exceedance calculations for quantile predictions

### Task 2: Prediction Loader (Priority: High)
**File**: `evaluation/prediction_loader.py`
- [ ] Implement `scan_prediction_files()` to discover all predictions
- [ ] Implement `load_predictions()` with column standardization
- [ ] Handle different naming conventions (Q_pred, Q_mean, Q50, Q_model_name)
- [ ] Create model family mapping
- [ ] Add data validation and quality checks

### Task 3: Ensemble Builder (Priority: High)
**File**: `evaluation/ensemble_builder.py`
- [ ] Implement `create_family_ensemble()` for model families
- [ ] Implement `create_global_ensemble()` across all families
- [ ] Handle missing models gracefully
- [ ] Save ensemble predictions in standardized format

### Task 4: Main Pipeline Script (Priority: High)
**File**: `evaluate_pipeline.py`
- [ ] Create main orchestration script
- [ ] Implement full workflow: load → ensemble → evaluate → save
- [ ] Add logging and progress tracking
- [ ] Handle errors gracefully with detailed reporting

### Task 5: Dashboard Enhancement (Priority: Medium)
**File**: Update `model_dashboard.py`
- [ ] Add Tab 4: "Predictions vs Observations"
- [ ] Implement time series visualization
- [ ] Add model family filtering
- [ ] Include ensemble models in dropdowns
- [ ] Add date range selection

### Task 6: Testing (Priority: Medium)
- [ ] Unit tests for evaluation functions
- [ ] Unit tests for prediction loader
- [ ] Unit tests for ensemble builder
- [ ] Integration test for full pipeline
- [ ] Test dashboard with new features

### Task 7: Documentation (Priority: Low)
- [ ] Document evaluation pipeline usage
- [ ] Document metric definitions
- [ ] Create example usage scripts
- [ ] Update README

## Technical Specifications

### Metric Functions to Use
```python
from eval_scr import metric_functions
# Available: r2_score, rmse, mae, nse, kge, bias
```

### Expected Output Structure
```
../monthly_forecasting_results/
└── evaluation/
    ├── metrics.csv              # All model evaluations
    ├── metrics_summary.json     # Aggregated statistics
    ├── ensemble_predictions.csv # Ensemble model predictions
    └── model_family_metrics.csv # Family-level comparisons
```

### Model Families
- **BaseCase**: LR_Q_T_P, DeviationLR, PerBasinScalingLR, ShortTermLR, ShortTerm_Features
- **SCA_Based**: LR_Q_SCA, LR_Q_T_SCA
- **SnowMapper_Based**: LR_Q_SWE, LR_Q_SWE_T, LR_Q_T_P_SWE, LR_Q_dSWEdt_T_P, CondenseLR, LongTermLR, ShortTermLR

## Acceptance Criteria
1. ✅ All available model predictions are automatically discovered and evaluated
2. ✅ Metrics are calculated using `eval_scr.metric_functions`
3. ✅ Ensemble predictions are created for each model family and globally
4. ✅ Evaluation results are saved in the correct format for dashboard
5. ✅ Dashboard shows predictions vs observations for all models
6. ✅ Pipeline runs without manual intervention
7. ✅ All tests pass

## Dependencies
- Existing `eval_scr` module for metric functions
- Pandas for data manipulation
- Current prediction file structure
- Dash/Plotly for dashboard updates

## Notes
- The evaluation should handle missing models gracefully
- Consider memory efficiency for large datasets
- Maintain backward compatibility with existing dashboard features
- Log all warnings and errors for debugging