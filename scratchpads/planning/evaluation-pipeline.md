# Comprehensive Evaluation Pipeline for Monthly Discharge Forecasting

## Objective
Create an automated evaluation pipeline that:
1. Scrapes all available predictions from model results
2. Evaluates models on per-code and per-month levels
3. Creates ensemble predictions for each model family and a global ensemble
4. Saves evaluation metrics for dashboard consumption
5. Enhances dashboard to show predictions vs observations

## Context
- Current predictions are stored in: `../lt_forecasting_results/model_type*/class_*/predictions.csv`
- Existing evaluation logic in `old_files/evaluate_hindcast.py` needs refactoring
- Metric functions should use `eval_scr.metric_functions` module
- Dashboard (`model_dashboard.py`) needs enhancement for prediction visualization

## Architecture Overview

```
lt_forecasting_results/
├── BaseCase/
│   ├── LR_Q_T_P/predictions.csv
│   ├── DeviationLR/predictions.csv
│   └── ...
├── SCA_Based/
│   ├── LR_Q_SCA/predictions.csv
│   └── ...
├── SnowMapper_Based/
│   ├── LR_Q_SWE/predictions.csv
│   └── ...
└── evaluation/
    ├── metrics.csv
    ├── metrics_summary.json
    ├── ensemble_predictions.csv
    └── model_family_metrics.csv
```

## Implementation Plan

### Phase 1: Core Evaluation Module Refactoring

#### Task 1.1: Create New Evaluation Module
- [ ] Create `evaluation/evaluate_models.py` as the main evaluation module
- [ ] Import metric functions from `eval_scr.metric_functions`
- [ ] Implement core evaluation functions:
  - `calculate_metrics()` - compute all metrics for a given prediction series
  - `evaluate_per_code()` - evaluate metrics for each basin
  - `evaluate_per_month()` - evaluate metrics for each month
  - `evaluate_overall()` - compute aggregated metrics

#### Task 1.2: Implement Metric Calculations
- [ ] Use these metrics from `eval_scr.metric_functions`:
  - R² Score (`r2_score`)
  - RMSE (`rmse`) 
  - MAE (`mae`)
  - NSE (`nse`)
  - KGE (`kge`)
  - Bias (`bias`)
- [ ] Calculate derived metrics:
  - NRMSE (normalized RMSE)
  - MAPE (Mean Absolute Percentage Error)
  - PBIAS (Percent Bias)
- [ ] Implement probability of exceedance for quantile predictions

### Phase 2: Prediction Scraping and Loading

#### Task 2.1: Create Prediction Loader
- [ ] Create `evaluation/prediction_loader.py`
- [ ] Implement `scan_prediction_files()` to discover all available predictions
- [ ] Implement `load_predictions()` to read and standardize prediction data
- [ ] Handle different column naming conventions (Q_pred, Q_mean, Q50, Q_model_name) normally the prediction for single models (LR) are saved with Q_modelname, for enseble predictions there are ensemble member (Q_xgb, Q_lgbm, Q_catboost) and the average Q_model_name
- [ ] Create mapping of model families: BaseCase, SCA_Based, SnowMapper_Based

#### Task 2.2: Data Validation
- [ ] Validate date formats and station codes
- [ ] Check for missing observations or predictions
- [ ] Use only the same station for all models
- [ ] create a global variable which states for which day of the month the evaluation should be performed (default=end) -> filtering out only the last day of each month
- [ ] Log warnings for data quality issues

### Phase 3: Ensemble Creation

#### Task 3.1: Model Family Ensembles
- [ ] Create `evaluation/ensemble_builder.py`
- [ ] Implement `create_family_ensemble()` for each model family
- [ ] Calculate mean predictions across models in each family
- [ ] Handle existing ensemble predictions (check for Q_model_name columns)
- [ ] Save family ensemble predictions

#### Task 3.2: Global Ensemble
- [ ] Implement `create_global_ensemble()` using all model families
- [ ] Weight ensembles equally or implement weighted averaging
- [ ] Include confidence intervals if quantile predictions available
- [ ] Save global ensemble predictions

### Phase 4: Evaluation Pipeline

#### Task 4.1: Main Evaluation Script
- [ ] Create `evaluate_pipeline.py` as the main entry point
- [ ] Implement workflow:
  1. Load all predictions
  2. Create model family ensembles
  3. Create global ensemble
  4. Evaluate all models (individual + ensembles)
  5. Save results to evaluation directory

#### Task 4.2: Output Generation
- [ ] Generate `metrics.csv` with columns:
  - model, code, month, r2, rmse, nrmse, mae, mape, nse, kge, bias, pbias
- [ ] Generate `ensemble_predictions.csv` with all ensemble results
- [ ] Generate `model_family_metrics.csv` for family-level comparisons

### Phase 5: Dashboard Enhancement

#### Task 5.1: Add Prediction Visualization Tab
- [ ] Modify `model_dashboard.py` to add Tab 4: "Predictions vs Observations"
- [ ] Implement time series plot showing:
  - Observations (Q_obs) as solid line
  - Multiple model predictions as different colored lines
  - Confidence intervals if available
- [ ] Add filters for:
  - Station code selection
  - Model family selection
  - Date range selection

#### Task 5.2: Enhance Existing Visualizations
- [ ] Add ensemble models to model selection dropdown
- [ ] Add KGE metric to available metrics
- [ ] Implement model family comparison view
- [ ] Add export functionality for plots

### Phase 6: Testing and Documentation

#### Task 6.1: Unit Tests
- [ ] Create tests for evaluation functions
- [ ] Create tests for ensemble creation
- [ ] Create tests for prediction loading
- [ ] Ensure all edge cases are covered

#### Task 6.2: Integration Tests
- [ ] Test full pipeline with sample data
- [ ] Verify output file formats
- [ ] Test dashboard with new metrics

#### Task 6.3: Documentation
- [ ] Document evaluation pipeline usage
- [ ] Document metric definitions
- [ ] Create example usage scripts
- [ ] Update README with evaluation instructions

## Technical Details

### Ensemble Calculation
```python
# For model family ensemble
family_predictions = []
for model in family_models:
    pred_col = f'Q_{model}' if f'Q_{model}' in df.columns else 'Q_pred'
    family_predictions.append(df[pred_col])
ensemble_pred = pd.concat(family_predictions, axis=1).mean(axis=1)
```

### Metric Calculation Pattern
```python
from eval_scr import metric_functions

def calculate_metrics(obs, pred):
    return {
        'r2': metric_functions.r2_score(obs, pred),
        'rmse': metric_functions.rmse(obs, pred),
        'nrmse': metric_functions.rmse(obs, pred) / obs.mean(),
        'mae': metric_functions.mae(obs, pred),
        'mape': (metric_functions.mae(obs, pred) / obs.mean()) * 100,
        'nse': metric_functions.nse(obs, pred),
        'kge': metric_functions.kge(obs, pred),
        'bias': metric_functions.bias(obs, pred),
        'pbias': (metric_functions.bias(obs, pred) / obs.mean()) * 100
    }
```

### Directory Structure for New Files
```
evaluation/
├── __init__.py
├── evaluate_models.py      # Core evaluation functions
├── prediction_loader.py    # Loading and standardizing predictions
├── ensemble_builder.py     # Creating ensemble predictions
├── evaluate_pipeline.py    # Main orchestration script
└── tests/
    ├── test_evaluate_models.py
    ├── test_prediction_loader.py
    └── test_ensemble_builder.py
```

## Success Criteria
1. Automated evaluation of all available models
2. Consistent metric calculations using eval_scr module
3. Working ensemble predictions at family and global levels
4. Enhanced dashboard with prediction visualization
5. All results saved in standardized format for dashboard consumption
6. Comprehensive test coverage

## Future Enhancements
- Weighted ensemble methods
- Skill score comparisons
- Seasonal evaluation breakdowns
- Export functionality for evaluation reports
- Real-time evaluation triggers