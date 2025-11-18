# Implement Historical Performance-Weighted Meta-Learning Framework

## Objective
Complete the meta-learning framework that weights base model predictions by historical performance. The system should calculate performance metrics for each basin-period combination and use global performance as fallback when insufficient data is available.

## Context
**Issue**: [GitHub Issue #41](https://github.com/hydrosolutions/lt_forecasting/issues/41)  
**Previous Work**: PR #40 implemented basic meta-learning framework structure but left implementations incomplete  
**Current State**: Skeleton implementation exists but has critical bugs and missing functionality

## Current Implementation Analysis

### Existing Files
- `lt_forecasting/forecast_models/meta_learners/base_meta_learner.py` - Complete abstract base class
- `lt_forecasting/forecast_models/meta_learners/historical_meta_learner.py` - Incomplete implementation with bugs
- Evaluation infrastructure in `dev_tools/eval_scr/metric_functions.py` and `dev_tools/evaluation/`

### Critical Issues Found
1. **Naming Conflict**: Both classes in `historical_meta_learner.py` are named `BaseMetaLearner`
2. **Missing `__init__.py`**: Meta-learners directory not importable
3. **Incomplete Methods**: All core methods are stubs with pseudo-code
4. **Missing Dependencies**: Referenced `meta_utils.py` doesn't exist
5. **No Tests**: Missing test files despite references in `__pycache__`

## Implementation Plan

### Phase 1: Foundation Fixes (Critical)
- [ ] Fix class declaration bug in `HistoricalMetaLearner` (rename from `BaseMetaLearner`)
- [ ] Create `__init__.py` files in meta_learners directory
- [ ] Create `meta_utils.py` with utility functions
- [ ] Move metric functions from `dev_tools/eval_scr/metric_functions.py` to `lt_forecasting/scr/`
- [ ] Add normalized MSE metric to evaluation suite
- [ ] Ensure `get_periods()` utility is accessible in meta-learner

### Phase 2: Core Meta-Learning Logic  
- [ ] Complete `__preprocess_data__` method
  - Load and merge base predictors with observed data
  - Create temporal periods using `get_periods()`
  - Prepare data for historical performance calculation
- [ ] Implement `__calculate_historical_performance__` method
  - Calculate performance metrics for each model per basin-period
  - Handle insufficient data cases with global performance fallback
  - Return structured DataFrame with performance metrics
- [ ] Implement `__get_weights__` method
  - Convert performance metrics to weights using softmax/normalized approach
  - Handle metric inversion for error-based metrics
  - Return weights DataFrame for ensemble creation
- [ ] Implement `__create_ensemble__` method
  - Apply performance-based weights to base model predictions
  - Handle edge cases (missing models, zero weights)

### Phase 3: Training Pipeline
- [ ] Complete `__loocv__` method
  - Implement Leave-One-Out Cross-Validation by year
  - Calculate historical performance on training data
  - Generate ensemble predictions for validation year
- [ ] Complete `calibrate_model_and_hindcast` method
  - Run full LOOCV training pipeline
  - Save performance weights as artifacts
  - Return hindcast predictions

### Phase 4: Operational Prediction
- [ ] Complete `predict_operational` method
  - Load or calculate historical performance weights
  - Apply weights to current base model predictions
  - Generate operational ensemble forecasts

### Phase 5: Model Persistence
- [ ] Implement `save_model` and `load_model` methods
  - Save/load performance weights and metadata
  - Handle model versioning and validation
- [ ] Implement `tune_hyperparameters` method
  - Optimize weighting strategy parameters
  - Tune minimum sample thresholds

### Phase 6: Testing and Integration
- [ ] Create unit tests for individual methods
- [ ] Create integration tests for full workflow
- [ ] Create functionality tests with real data
- [ ] Add configuration examples
- [ ] Update documentation

## Implementation Notes

### Key Technical Decisions
- **Metric Integration**: Use existing evaluation infrastructure from `dev_tools/eval_scr/`
- **Period Handling**: Leverage existing `get_periods()` utility for temporal grouping
- **Performance Calculation**: Per basin-period with global fallback for insufficient data
- **Weighting Strategy**: Softmax normalization with metric inversion for error-based metrics

### Architecture Considerations
- Maintain compatibility with existing `BaseForecastModel` interface
- Follow established logging patterns using `lt_forecasting.log_config`
- Use configuration-driven approach consistent with other models
- Preserve evaluation pipeline integration

### Edge Cases to Handle
- Missing base model predictions
- Insufficient historical data for performance calculation
- Zero or negative weights
- Missing basins or periods in training data
- Model prediction failures

## Testing Strategy

### Unit Tests
- Test individual method implementations with mock data
- Test edge cases (missing data, zero weights)
- Test metric calculations and weight transformations

### Integration Tests
- Test full LOOCV workflow with real data
- Test operational prediction pipeline
- Test model persistence (save/load)

### Functionality Tests
- Validate meta-learning performance improvements over base models
- Test with different basin-period combinations
- Validate ensemble creation accuracy

## Expected Outcome
A production-ready meta-learning framework that intelligently weights base model predictions based on historical performance, with proper fallback mechanisms, comprehensive testing coverage, and seamless integration with existing codebase architecture.

## Next Steps
1. Start with Phase 1 fixes to establish working foundation
2. Create new branch for implementation
3. Implement and test each phase incrementally
4. Run full test suite after each phase
5. Create PR for review once complete