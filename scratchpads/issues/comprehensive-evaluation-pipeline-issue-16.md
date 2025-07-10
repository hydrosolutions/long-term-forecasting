# Comprehensive Evaluation Pipeline for Monthly Discharge Forecasting - Issue 16

## Objective
Create an automated evaluation pipeline that:
1. Scrapes all available predictions from model results
2. Evaluates models on per-code and per-month levels
3. Creates ensemble predictions for each model family and a global ensemble
4. Saves evaluation metrics for dashboard consumption
5. Enhances dashboard to show predictions vs observations

## Context
- **GitHub Issue**: https://github.com/hydrosolutions/monthly_forecasting/issues/16
- **Current predictions stored in**: `../monthly_forecasting_results/model_type*/class_*/predictions.csv`
- **Model families identified**: BaseCase, SCA_Based, SnowMapper_Based
- **Existing evaluation logic**: `old_files/evaluate_hindcast.py` (needs refactoring)
- **Metrics module**: `eval_scr/metric_functions.py` (comprehensive metric functions available)
- **Dashboard**: `old_files/model_dashboard.py` (needs enhancement)

## Analysis Summary

### Available Metrics in eval_scr/metric_functions.py:
- **Core metrics**: R2, RMSE, MAE, NSE, KGE, Bias
- **Convenience wrappers**: `r2_score()`, `rmse()`, `mae()`, `nse()`, `bias()`, `kge()`
- **Advanced metrics**: CRPS, Quantile Loss, forecast accuracy
- **Utility functions**: Robust error handling, NaN management

### Current Data Structure:
```
monthly_forecasting_results/
├── BaseCase/
│   ├── DeviationLR/
│   ├── LR_Q_T_P/
│   ├── PerBasinScalingLR/
│   ├── ShortTermLR/
│   └── ShortTerm_Features/
├── SCA_Based/
│   ├── LR_Q_SCA/
│   └── LR_Q_T_SCA/
└── SnowMapper_Based/
    ├── CondenseLR/
    ├── LR_Q_SWE/
    ├── LR_Q_SWE_T/
    ├── LR_Q_T_P_SWE/
    ├── LR_Q_dSWEdt_T_P/
    ├── LongTermLR/
    ├── ShortTermLR/
    └── ShortTerm_Features/
```

### Column Naming Conventions Observed:
- Single models: `Q_modelname` (e.g., `Q_LR_Q_T_P`)
- Ensemble predictions: `Q_model_name` (mean), `Q_xgb`, `Q_lgbm`, `Q_catboost` (members)
- Alternative formats: `Q_pred`, `Q_mean`, `Q50`

## Implementation Plan

### Phase 1: Core Evaluation Module ✅
- [x] Create `evaluation/` directory structure
- [x] Implement `evaluation/evaluate_models.py` with core evaluation functions
- [x] Import and use `eval_scr.metric_functions` consistently
- [x] Implement per-code, per-month, and overall evaluation functions

### Phase 2: Prediction Loading System ✅
- [x] Create `evaluation/prediction_loader.py`
- [x] Implement `scan_prediction_files()` to discover all predictions
- [x] Implement `load_predictions()` to standardize prediction data
- [x] Handle different column naming conventions
- [x] Map models to families (BaseCase, SCA_Based, SnowMapper_Based)

### Phase 3: Ensemble Creation ✅
- [x] Create `evaluation/ensemble_builder.py`
- [x] Implement family ensemble creation
- [x] Implement global ensemble creation
- [x] Handle existing ensemble predictions appropriately

### Phase 4: Pipeline Orchestration ✅
- [x] Create `evaluation/evaluate_pipeline.py` as main entry point
- [x] Implement complete workflow from loading to output generation
- [x] Generate standardized outputs for dashboard consumption

### Phase 5: Dashboard Enhancement 🔄
- [ ] Enhance `model_dashboard.py` with prediction visualization
- [ ] Add Tab 4: "Predictions vs Observations"
- [ ] Implement model family comparison views
- [ ] Add ensemble models to existing visualizations

### Phase 6: Testing & Validation 🔄
- [ ] Create comprehensive test suite
- [ ] Validate output formats and metrics
- [ ] Test edge cases and error handling
- [ ] Integration testing with sample data

## Technical Implementation Details

### Key Design Decisions:
1. **Modular Architecture**: Separate concerns into distinct modules
2. **Consistent Metrics**: Use `eval_scr.metric_functions` throughout
3. **Robust Error Handling**: Handle missing data, NaN values, and edge cases
4. **Standardized Outputs**: Generate consistent formats for dashboard consumption
5. **Flexible Configuration**: Support different evaluation parameters

### Evaluation Configuration:
- **Target date**: Last day of each month (configurable)
- **Minimum samples**: Configurable per evaluation
- **Common basins**: Use intersection of all model basins
- **Metrics**: R2, RMSE, NRMSE, MAE, MAPE, NSE, KGE, Bias, PBIAS

### Output Structure:
```
monthly_forecasting_results/evaluation/
├── metrics.csv              # Comprehensive metrics by model/code/month
├── metrics_summary.json     # Aggregated statistics
├── ensemble_predictions.csv # All ensemble predictions
├── model_family_metrics.csv # Family-level comparisons
└── evaluation_metadata.json # Pipeline configuration and runtime info
```

## Testing Strategy

### Unit Tests:
- `test_prediction_loader.py`: Data loading and validation
- `test_evaluate_models.py`: Metric calculation accuracy
- `test_ensemble_builder.py`: Ensemble creation logic

### Integration Tests:
- Full pipeline execution with sample data
- Output format validation
- Dashboard compatibility testing

### Edge Case Testing:
- Missing data scenarios
- Single model families
- Empty prediction files
- Inconsistent date formats

## Next Steps

1. **Complete Implementation**: Finish remaining modules
2. **Testing**: Create comprehensive test suite
3. **Dashboard Enhancement**: Add prediction visualization
4. **Documentation**: Update README and create usage examples
5. **Validation**: Test with real data and verify outputs

## Success Criteria

- [x] Automated evaluation of all available models
- [x] Consistent metric calculations using eval_scr module
- [x] Working ensemble predictions at family and global levels
- [ ] Enhanced dashboard with prediction visualization
- [x] All results saved in standardized format
- [ ] Comprehensive test coverage (>80%)
- [ ] Full integration with existing workflow

## Future Enhancements

- Weighted ensemble methods based on historical performance
- Skill score comparisons relative to climatology
- Seasonal evaluation breakdowns
- Real-time evaluation triggers
- Export functionality for evaluation reports