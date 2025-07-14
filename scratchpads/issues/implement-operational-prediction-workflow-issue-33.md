# Implement Operational Prediction Workflow for Monthly Discharge Forecasting - Issue 33

## Objective
Enhance the existing `scripts/run_operational_prediction.py` script to implement a complete operational prediction workflow for monthly discharge forecasting. The script currently has data loading functionality but needs the core prediction workflow.

## Context
- **GitHub Issue**: https://github.com/hydrosolutions/monthly_forecasting/issues/33
- **Current State**: Data loading functions implemented (`load_discharge()`, `load_forcing()`, `load_snowmapper()`, `load_static_data()`, `create_data_frame()`)
- **Missing**: Core prediction workflow, configuration loading, model initialization, performance evaluation
- **Reference Implementation**: `scripts/calibrate_hindcast.py` for patterns and structure

## Plan

### Phase 1: Core Infrastructure
- [ ] **Task 1.1**: Implement `load_operational_configs()` function
  - Load configurations for all models in `MODELS_OPERATIONAL` dictionary
  - Use pattern from `calibrate_hindcast.py:load_configuration()`
  - Handle configuration files: `general_config.json`, `model_config.json`, `feature_config.json`, `data_config.json`

- [ ] **Task 1.2**: Implement `shift_data_to_current_year()` function
  - Add +1 year to data dates to mock current year (data is not up to date)
  - Handle date shifting for all data frames consistently

- [ ] **Task 1.3**: Enhance `create_data_frame()` function
  - Add dynamic data loading based on model's data_config
  - Support proper snow HRU selection for different models

### Phase 2: Model Workflow
- [ ] **Task 2.1**: Implement `create_model_instance()` function
  - Use pattern from `calibrate_hindcast.py:create_model()`
  - Support both LinearRegressionModel and SciRegressor model types
  - Handle model initialization with appropriate configurations

- [ ] **Task 2.2**: Implement `run_operational_prediction()` function
  - Main prediction workflow that iterates through model families
  - Process `BaseCase` and `SnowMapper_Based` model families
  - Execute predictions for each model in the family

### Phase 3: Performance Evaluation
- [ ] **Task 3.1**: Implement `calculate_average_discharge()` function
  - Calculate average discharge for each basin from `valid_from` to `valid_to` period
  - Handle observed vs predicted discharge comparison

- [ ] **Task 3.2**: Implement `evaluate_predictions()` function
  - Calculate overall R² score across all predictions
  - Identify predictions with error > 30%
  - Generate per-basin performance metrics

### Phase 4: Performance Monitoring & Output
- [ ] **Task 4.1**: Implement `time_prediction_process()` function
  - Overall timing for complete forecasting process
  - Individual model prediction timing
  - Resource monitoring and logging

- [ ] **Task 4.2**: Implement `generate_outputs()` function
  - Create `operational_predictions.csv` with all model predictions
  - Create `performance_metrics.csv` with evaluation metrics
  - Create `timing_report.json` with detailed timing statistics
  - Create `quality_report.txt` with prediction quality summary

### Phase 5: Integration & Testing
- [ ] **Task 5.1**: Integrate all components into `run_operational()` function
  - Orchestrate the complete workflow
  - Add comprehensive error handling
  - Implement logging throughout the process

- [ ] **Task 5.2**: Write comprehensive tests
  - Test individual functions with mock data
  - Test complete workflow end-to-end
  - Test error handling and edge cases

## Implementation Notes

### Key Reference Functions
- `calibrate_hindcast.py:load_configuration()` - Configuration loading pattern
- `calibrate_hindcast.py:create_model()` - Model initialization pattern
- `calibrate_hindcast.py:load_data()` - Data loading pattern

### Configuration Structure
Based on `example_config/DUMMY_MODEL/`:
- `general_config.json`: Model type, features, parameters
- `model_config.json`: Model-specific configuration
- `feature_config.json`: Feature engineering settings
- `data_config.json`: Data loading configuration
- `data_paths.json`: File paths configuration

### Model Definitions
```python
MODELS_OPERATIONAL = {
    'BaseCase': [
        ('LR', 'LR_Q_T_P'),
        ('SciRegressor', 'ShortTerm_Features'),
        ('SciRegressor', 'NormBased')
    ],
    'SnowMapper_Based': [
        ('LR', 'LR_Q_dSWEdt_T_P'),
        ('LR', 'LR_Q_SWE_T'),
        ('LR', 'LR_Q_T_P_SWE'),
        ('LR', 'LR_Q_SWE'),
        ('SciRegressor', 'NormBased'),
        ('SciRegressor', 'ShortTermLR')
    ]
}
```

### Expected Outputs
- Console: Progress indicators, timing stats, performance metrics, error warnings
- Files: `operational_predictions.csv`, `performance_metrics.csv`, `timing_report.json`, `quality_report.txt`

## Testing Strategy

### Unit Tests
- Test configuration loading for all model types
- Test data date shifting functionality
- Test model initialization and prediction workflow
- Test performance evaluation calculations
- Test output file generation

### Integration Tests
- Test complete workflow from start to finish
- Test error handling with malformed configurations
- Test performance with different model combinations
- Test output consistency and format

### Performance Tests
- Verify timing implementation accuracy
- Test memory usage monitoring
- Validate R² calculation correctness
- Check error flagging for poor predictions

## Review Points
- Configuration loading robustness across all model types
- Data date shifting consistency across all data sources
- Model initialization error handling
- Performance evaluation accuracy
- Output file format and completeness
- Logging comprehensiveness
- Error handling preventing script crashes