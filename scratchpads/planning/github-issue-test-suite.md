# GitHub Issue: Improve Test Suite Efficiency and Coverage for Monthly Forecasting Models

## Title
Improve test suite efficiency and coverage for monthly forecasting models with comprehensive preprocessing tests

## Labels
- `enhancement`
- `testing`
- `performance`
- `high-priority`

## Description

### Problem Statement
The current test suite for monthly forecasting models has several critical issues that impact development velocity and CI/CD reliability:

1. **Excessive Data Generation**: Tests generate 20+ years of synthetic data for multiple basins, creating datasets with 100,000+ records
2. **Slow Hyperparameter Optimization**: Optuna trials run 50-100+ iterations, taking 10+ minutes per test
3. **Incomplete Coverage**: Different preprocessing methods (normalization strategies) are not systematically tested
4. **Over-Mocking**: Some tests mock core functionality instead of actually executing model code
5. **Resource Intensive**: Tests consume excessive memory and CPU, causing timeouts in CI/CD pipelines

### Current Impact
- Test suite takes 15-30 minutes to run completely
- Tests occasionally fail due to timeouts
- Difficult to run tests locally during development
- Missing coverage for preprocessing edge cases
- CI/CD pipeline bottleneck

## Proposed Solution

### 1. Efficient Mock Data Generation
Create a fast, minimal mock data generator that produces:
- **3 basins** (16936, 16940, 16942) instead of 10+
- **5 years** of daily data (2018-2023) instead of 20+
- **~5,500 records total** instead of 100,000+
- **Generation time: <100ms** instead of several seconds

### 2. Preprocessing Test Matrix
Implement comprehensive tests for all preprocessing × model combinations:

| Preprocessing Method | LinearRegression | XGBoost | LightGBM | CatBoost |
|---------------------|------------------|---------|----------|----------|
| No Normalization    | ✓                | ✓       | ✓        | ✓        |
| Global Normalization| ✓                | ✓       | ✓        | ✓        |
| Per-Basin Normal.   | ✓                | ✓       | ✓        | ✓        |
| Long-Term Mean      | ✓                | ✓       | ✓        | ✓        |

### 3. Optimized Hyperparameter Testing
- Limit Optuna to **1 trial** for testing (vs 50-100)
- Use minimal model configurations (10-20 estimators)
- Test parameter ranges and optimization process, not convergence
- Target: <10 seconds per hyperparameter test

### 4. Test Structure Reorganization
```
tests/
├── mock_data.py              # Efficient data generation
├── test_config.py            # Centralized test configuration
├── test_models_unit.py       # Fast unit tests
├── test_models_integration.py # End-to-end workflow tests
├── test_hyperparameter_optimization.py # Optuna tests
├── test_preprocessing.py     # Preprocessing-specific tests
└── test_utils.py            # Shared utilities
```

## Acceptance Criteria

### Performance Requirements
- [ ] Complete test suite runs in **<2 minutes**
- [ ] Mock data generation takes **<100ms**
- [ ] Individual unit tests complete in **<500ms**
- [ ] Integration tests complete in **<5 seconds**
- [ ] Hyperparameter tests complete in **<10 seconds**
- [ ] Memory usage stays **<500MB**

### Coverage Requirements
- [ ] All 4 preprocessing methods tested
- [ ] All 4 model types tested
- [ ] Edge cases covered (missing data, extremes, etc.)
- [ ] Each preprocessing × model combination validated
- [ ] Actual model code executed (minimal mocking)

### Quality Requirements
- [ ] Tests are deterministic (same results every run)
- [ ] Clear test names and documentation
- [ ] Proper pytest fixtures for resource sharing
- [ ] Good error messages on failure
- [ ] Easy to add new preprocessing methods or models

## Implementation Plan

### Phase 1: Foundation (Priority: High)
1. Create `mock_data.py` with efficient data generator
2. Create `test_config.py` with centralized configuration
3. Set up pytest fixtures in `conftest.py`

### Phase 2: Core Tests (Priority: High)
1. Implement unit tests for model components
2. Create preprocessing-specific test suite
3. Add hyperparameter optimization tests (1 trial each)

### Phase 3: Integration (Priority: Medium)
1. End-to-end workflow tests
2. Cross-validation tests
3. Edge case scenarios

### Phase 4: Polish (Priority: Low)
1. Performance optimization
2. Documentation updates
3. CI/CD configuration

## Technical Details

### Mock Data Specifications
```python
# Example mock data configuration
MOCK_DATA_CONFIG = {
    'basins': [16936, 16940, 16942],  # 3 basins
    'date_range': ('2018-01-01', '2022-12-31'),  # 5 years
    'features': {
        'temperature': 'seasonal_pattern + noise',
        'precipitation': 'exponential + seasonal',
        'discharge': 'f(temperature, precipitation, lag)'
    },
    'missing_data': 0.05,  # 5% missing values
    'seed': 42  # Deterministic generation
}
```

### Test Configuration Example
```python
# Minimal hyperparameter ranges for testing
HYPERPARAM_TEST_CONFIG = {
    'xgb': {
        'n_estimators': (10, 20),
        'max_depth': (2, 4),
        'n_trials': 1
    },
    'lgbm': {
        'n_estimators': (10, 20),
        'num_leaves': (10, 20),
        'n_trials': 1
    }
}
```

### Preprocessing Test Example
```python
@pytest.mark.parametrize("preprocessing", ["none", "global", "per_basin", "long_term_mean"])
@pytest.mark.parametrize("model_type", ["linear", "xgb", "lgbm", "catboost"])
def test_preprocessing_model_combination(mock_data, preprocessing, model_type):
    """Test each preprocessing method with each model type"""
    # Create model with preprocessing
    # Train on mock data
    # Verify predictions
    # Check inverse transformation
    assert execution_time < 5.0  # seconds
```

## Benefits

### Immediate Benefits
- **10x faster test execution** (2 min vs 20+ min)
- **Reliable CI/CD** (no more timeouts)
- **Better coverage** (all preprocessing methods)
- **Easier debugging** (faster feedback loop)

### Long-term Benefits
- **Faster development** (quick local testing)
- **Higher confidence** (comprehensive coverage)
- **Easier maintenance** (clear structure)
- **Scalability** (easy to add new tests)

## Related Issues
- #[Previous issue number] - Test timeouts in CI/CD
- #[Previous issue number] - Missing preprocessing test coverage

## Additional Context
This improvement is critical for the upcoming model deployment phase where we need reliable, fast testing for continuous integration and deployment pipelines. The current test suite is a bottleneck for development velocity.

## Checklist for Implementation
- [ ] Review and approve test plan
- [ ] Create mock data generator
- [ ] Implement preprocessing tests
- [ ] Optimize hyperparameter tests
- [ ] Update CI/CD configuration
- [ ] Document new test structure
- [ ] Migrate existing tests
- [ ] Verify performance targets met

---

**Priority**: High  
**Estimated Effort**: 2-3 weeks  
**Impact**: High - Affects all developers and CI/CD pipeline