# Comprehensive Test Coverage for Core Modules - Issue #6

## Objective
Write comprehensive test coverage for the core utility modules and regression classes to ensure reliability and maintainability.

## Context
- **Issue**: https://github.com/hydrosolutions/monthly_forecasting/issues/6
- **Current Status**: Some tests exist but coverage is incomplete
- **Target**: Achieve at least 80% code coverage for sci_utils, data_utils, and regression classes

## Current Test Coverage Analysis

### Existing Tests
1. **test_data_utils.py** (recently created) - covers normalization functions:
   - `apply_inverse_normalization_global`
   - `apply_inverse_normalization_per_basin`
   - `apply_inverse_normalization_different_variable`
   - `apply_inverse_normalization_missing_column`
   - `apply_inverse_normalization_missing_scaler_key`
   - `normalization_inverse_consistency`
   - `per_basin_normalization_different_scales`

2. **test_sciregressor.py** - covers SciRegressor class
3. **test_linear_regression.py** - covers LinearRegression class  
4. **test_feature_processing.py** - covers feature processing

### Missing Test Coverage

#### sci_utils.py - NO TESTS CURRENTLY
Functions needing tests:
- `get_model()` - Model factory function
- `fit_model()` - Model training function
- `get_feature_importance()` - Feature importance extraction
- `optimize_hyperparams()` - Hyperparameter optimization
- `_objective_xgb()` - XGBoost optimization objective
- `_objective_lgbm()` - LightGBM optimization objective
- `_objective_catboost()` - CatBoost optimization objective
- `_objective_mlp()` - MLP optimization objective

#### data_utils.py - PARTIAL COVERAGE
Functions needing tests:
- `get_position_name()` - Date position naming
- `discharge_m3_to_mm()` - Unit conversion (appears empty)
- `discharge_mm_to_m3()` - Unit conversion (appears empty)
- `create_target()` - Target variable creation
- `glacier_mapper_features()` - Glacier feature mapping
- `get_elevation_bands_per_percentile()` - Elevation band calculations
- `calculate_percentile_snow_bands()` - Snow band calculations
- `get_normalization_params()` - Normalization parameter calculation
- `apply_normalization()` - Apply normalization
- `normalize_features()` - Feature normalization wrapper
- `get_normalization_params_per_basin()` - Per-basin normalization params
- `apply_normalization_per_basin()` - Apply per-basin normalization
- `normalize_features_per_basin()` - Per-basin normalization wrapper
- `get_long_term_mean_per_basin()` - Long-term mean calculation
- `apply_long_term_mean()` - Apply long-term mean
- `apply_long_term_mean_scaling()` - Long-term mean scaling
- `calculate_elevation_band_areas()` - Elevation band area calculations
- `average_by_elevation_band()` - Elevation band averaging
- `create_lag_features()` - Lag feature creation
- `agg_with_min_obs()` - Aggregation with minimum observations
- `create_monthly_df()` - Monthly DataFrame creation

#### Regression Classes - PARTIAL COVERAGE
1. **LinearRegressionModel** - Basic tests exist but may need enhancement
2. **SciRegressor** - Tests exist but may need enhancement
3. **BaseForecastModel** - Abstract base class needs interface testing

## Implementation Plan

### Phase 1: Create test_sci_utils.py
- [ ] Test `get_model()` function with all supported model types
- [ ] Test `fit_model()` function with different model types
- [ ] Test `get_feature_importance()` for all model types
- [ ] Test `optimize_hyperparams()` with mock data
- [ ] Test objective functions with validation data

### Phase 2: Expand test_data_utils.py
- [ ] Test position naming functions
- [ ] Test unit conversion functions
- [ ] Test target creation with various parameters
- [ ] Test glacier feature mapping
- [ ] Test elevation band functions
- [ ] Test snow band calculations
- [ ] Test normalization functions not currently covered
- [ ] Test long-term mean functions
- [ ] Test aggregation and monthly DataFrame functions

### Phase 3: Enhance Regression Class Tests
- [ ] Enhance LinearRegressionModel tests
- [ ] Enhance SciRegressor tests if needed
- [ ] Test BaseForecastModel interface compliance

### Phase 4: Integration and Performance Tests
- [ ] Add integration tests for complete workflows
- [ ] Add performance tests for computationally intensive functions
- [ ] Add edge case and error handling tests

## Testing Strategy

### Test Structure
```python
# Each test file should follow this pattern:
class TestFunctionGroup:
    \"\"\"Test group for related functions.\"\"\"
    
    def test_function_normal_case(self):
        \"\"\"Test normal operation.\"\"\"
        pass
    
    def test_function_edge_cases(self):
        \"\"\"Test edge cases.\"\"\"
        pass
    
    def test_function_error_handling(self):
        \"\"\"Test error conditions.\"\"\"
        pass
```

### Test Data Strategy
- Use minimal, focused test data
- Create fixtures for commonly used test data
- Mock external dependencies (file I/O, network calls)
- Use parametrized tests for testing multiple scenarios

### Mock Strategy
- Mock external dependencies (sklearn models, file systems)
- Mock time-dependent functions for consistent testing
- Mock random number generation for reproducible tests

## Acceptance Criteria
- [ ] All public functions have corresponding tests
- [ ] Tests cover normal cases, edge cases, and error conditions
- [ ] Tests are well-documented with clear descriptions
- [ ] All tests pass in CI/CD pipeline
- [ ] Code coverage report shows ≥80% for target modules
- [ ] Tests follow pytest framework conventions
- [ ] Tests use appropriate mocking for external dependencies

## Files to Create/Modify
1. `tests/test_sci_utils.py` - NEW
2. `tests/test_data_utils.py` - EXPAND
3. `tests/test_linear_regression.py` - ENHANCE (if needed)
4. `tests/test_base_class.py` - NEW (for BaseForecastModel)
5. Update existing test files as needed

## Technical Implementation Notes

### pytest Features to Use
- `pytest.mark.parametrize` for testing multiple scenarios
- `pytest.fixture` for test data setup
- `pytest.raises` for exception testing
- `pytest.approx` for floating point comparisons

### Mock Strategies
- Use `unittest.mock` for mocking external dependencies
- Mock sklearn models for testing model-agnostic functions
- Mock file I/O operations
- Mock optuna optimization for faster tests

### Test Organization
- Group related tests in classes
- Use descriptive test names
- Include docstrings explaining what each test validates
- Follow AAA pattern (Arrange, Act, Assert)

## Review Points
1. Test coverage completeness
2. Edge case handling
3. Error condition testing
4. Performance test appropriateness
5. Mock usage correctness
6. Test documentation quality
7. Integration with existing test suite