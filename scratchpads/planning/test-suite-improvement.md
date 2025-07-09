# Test Suite Improvement Plan for Monthly Forecasting Models

## Objective
Create a comprehensive, efficient test suite that validates all model functionality while running quickly enough for CI/CD pipelines. The test suite should cover all preprocessing methods, model types, and edge cases while using minimal computational resources.

## Context
The current test suite has several issues:
- Generates excessive synthetic data (20+ years for multiple basins)
- Hyperparameter optimization runs too many trials (50-100+)
- Tests are slow and sometimes timeout
- Some tests only mock functionality without actually running code
- Inconsistent test data across different test files

## Plan

### 1. Mock Data Generation Strategy

#### Efficient Mock Data Generator (`tests/mock_data.py`)
```python
class MockDataGenerator:
    """Generate small, consistent datasets for testing"""
    
    BASINS = [16936, 16940, 16942]  # 3 basins only
    START_DATE = "2018-01-01"
    END_DATE = "2022-12-31"  # 5 years
    
    @staticmethod
    def generate_timeseries_data(
        basins=None,
        start_date=None,
        end_date=None,
        include_missing=False,
        preprocessing_optimized=None
    ):
        """
        Generate synthetic hydrological time series
        - Temperature: Seasonal pattern + noise
        - Precipitation: Exponential distribution + seasonal
        - Discharge: Complex function of T, P with lag effects
        """
        
    @staticmethod
    def generate_static_data(basins=None):
        """
        Generate static basin characteristics
        - Area, elevation, slope, forest cover
        - Consistent across all tests
        """
```

### 2. Preprocessing-Specific Test Cases

#### 2.1 No Normalization Tests
- **Purpose**: Baseline tests with raw data
- **Test scenarios**:
  - Model training with original scale data
  - Predictions in original units
  - Feature importance interpretation
- **Expected behavior**: 
  - Models should handle different scales
  - Tree-based models less affected than linear models

#### 2.2 Global Normalization Tests
- **Purpose**: Test standard scaling across all data
- **Test scenarios**:
  - Consistent scaling across all basins
  - Inverse transformation accuracy
  - Handling of new data outside training range
- **Expected behavior**:
  - Mean ~0, std ~1 for all features
  - Accurate inverse transformation

#### 2.3 Per-Basin Normalization Tests
- **Purpose**: Test basin-specific scaling
- **Test scenarios**:
  - Different scalers per basin
  - Cross-basin prediction handling
  - Missing basin handling
- **Expected behavior**:
  - Each basin normalized independently
  - Proper scaler selection during prediction

#### 2.4 Long-Term Mean Scaling Tests
- **Purpose**: Test temporal normalization
- **Test scenarios**:
  - Monthly/seasonal scaling factors
  - Handling of incomplete years
  - Future date predictions
- **Expected behavior**:
  - Seasonal patterns preserved
  - Robust to missing periods

### 3. Model-Specific Test Coverage

#### 3.1 LinearRegression Tests
```python
def test_linear_regression_all_preprocessing():
    """Test LinearRegression with each preprocessing method"""
    for preprocessing in ['none', 'global', 'per_basin', 'long_term_mean']:
        # Test initialization
        # Test training
        # Test prediction
        # Verify outputs
```

#### 3.2 XGBoost Tests
```python
def test_xgboost_hyperparameter_optimization():
    """Test XGBoost with 1-trial Optuna optimization"""
    config = {
        'n_trials': 1,  # Single trial only
        'n_estimators_range': (10, 20),  # Small range
        'max_depth_range': (2, 4)
    }
    # Run optimization
    # Verify parameters selected
    # Test model with optimized params
```

#### 3.3 LightGBM Tests
- Similar structure to XGBoost
- Test categorical feature handling
- Verify memory efficiency

#### 3.4 CatBoost Tests
- Test categorical feature encoding
- Verify GPU compatibility (if available)
- Test built-in cross-validation

### 4. Test Organization

#### 4.1 Unit Tests (`test_models_unit.py`)
- **Model Initialization**
  - Config validation
  - Default parameter setting
  - Error handling for invalid inputs

- **Feature Extraction**
  - Window calculations
  - Lag features
  - Static feature merging
  - Temporal encoding

- **Data Preprocessing**
  - Missing value handling
  - Outlier detection
  - Feature scaling
  - Train/test splitting

#### 4.2 Integration Tests (`test_models_integration.py`)
- **End-to-End Workflows**
  ```python
  def test_complete_workflow_with_preprocessing(preprocessing_method):
      """Test complete pipeline for each preprocessing method"""
      # 1. Generate mock data
      # 2. Initialize model
      # 3. Extract features
      # 4. Apply preprocessing
      # 5. Train model
      # 6. Make predictions
      # 7. Evaluate metrics
      # 8. Verify inverse transformation
  ```

- **Cross-Validation Tests**
  - Leave-one-year-out validation
  - Time series split validation
  - Basin-wise validation

#### 4.3 Hyperparameter Tests (`test_hyperparameter_optimization.py`)
```python
class TestHyperparameterOptimization:
    """Test hyperparameter tuning with minimal trials"""
    
    @pytest.fixture
    def optuna_config(self):
        return {
            'n_trials': 1,
            'timeout': 60,  # 1 minute max
            'n_jobs': 1  # Single thread for determinism
        }
    
    def test_xgboost_optimization(self, mock_data, optuna_config):
        """Test XGBoost hyperparameter optimization"""
        # Should complete in <10 seconds
        # Should return valid parameters
        # Should improve over baseline
    
    def test_optimization_with_each_preprocessing(self):
        """Test optimization works with all preprocessing methods"""
        for method in PREPROCESSING_METHODS:
            # Run optimization
            # Verify parameters are valid
            # Check model can be trained
```

### 5. Performance Benchmarks

#### Target Performance Metrics
- **Data Generation**: <100ms for full dataset
- **Unit Tests**: <500ms per test
- **Integration Tests**: <5s per test
- **Hyperparameter Tests**: <10s per model
- **Total Suite Runtime**: <2 minutes

#### Memory Usage Targets
- **Mock Data**: <50MB in memory
- **Model Training**: <200MB peak
- **Prediction**: <100MB

### 6. Edge Case Testing

#### 6.1 Missing Data Scenarios
```python
def test_missing_data_handling():
    """Test various missing data patterns"""
    scenarios = [
        'random_missing',  # 10% random missing
        'systematic_missing',  # Every 7th day missing
        'block_missing',  # Entire month missing
        'basin_missing'  # One basin completely missing
    ]
```

#### 6.2 Extreme Value Testing
- Very high discharge events
- Zero precipitation periods
- Temperature extremes
- Outlier detection and handling

#### 6.3 Data Quality Issues
- Duplicate timestamps
- Inconsistent basin codes
- Misaligned time series
- Different sampling frequencies

### 7. Test Configuration Management

#### Test Configuration File (`test_config.py`)
```python
TEST_CONFIG = {
    'data': {
        'n_basins': 3,
        'n_years': 5,
        'basins': [16936, 16940, 16942],
        'start_date': '2018-01-01',
        'end_date': '2022-12-31'
    },
    'models': {
        'xgb': {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        },
        'lgbm': {
            'n_estimators': 10,
            'num_leaves': 15,
            'learning_rate': 0.1
        },
        'catboost': {
            'iterations': 10,
            'depth': 3,
            'learning_rate': 0.1
        }
    },
    'optimization': {
        'n_trials': 1,
        'timeout': 60,
        'n_jobs': 1
    },
    'preprocessing': {
        'methods': ['none', 'global', 'per_basin', 'long_term_mean'],
        'test_all': True
    }
}
```

### 8. Pytest Fixtures and Utilities

#### Shared Fixtures (`conftest.py`)
```python
@pytest.fixture(scope="session")
def mock_data():
    """Session-wide mock data generation"""
    return MockDataGenerator.generate_timeseries_data()

@pytest.fixture(scope="session")
def static_data():
    """Session-wide static data"""
    return MockDataGenerator.generate_static_data()

@pytest.fixture
def test_model(request):
    """Parameterized model fixture"""
    model_type = request.param
    return create_test_model(model_type)

@pytest.fixture(params=PREPROCESSING_METHODS)
def preprocessing_method(request):
    """Parameterized preprocessing method"""
    return request.param
```

### 9. Implementation Timeline

#### Phase 1: Foundation (Week 1)
- [ ] Create mock data generator
- [ ] Set up test configuration
- [ ] Create test utilities
- [ ] Update pytest configuration

#### Phase 2: Unit Tests (Week 2)
- [ ] Model initialization tests
- [ ] Feature extraction tests
- [ ] Preprocessing tests
- [ ] Prediction method tests

#### Phase 3: Integration Tests (Week 3)
- [ ] End-to-end workflow tests
- [ ] Cross-validation tests
- [ ] Preprocessing-specific tests
- [ ] Edge case tests

#### Phase 4: Optimization & Polish (Week 4)
- [ ] Hyperparameter optimization tests
- [ ] Performance optimization
- [ ] Documentation
- [ ] CI/CD integration

### 10. Success Metrics

#### Quantitative Metrics
- Test execution time: <2 minutes
- Test coverage: >90%
- Memory usage: <500MB peak
- All preprocessing methods tested
- All model types tested

#### Qualitative Metrics
- Clear test organization
- Easy to add new tests
- Deterministic results
- Good error messages
- Fast feedback loop

## Implementation Notes

### Key Principles
1. **Real Testing**: Actually run model code, don't just mock
2. **Fast Execution**: Small data, few iterations
3. **Comprehensive Coverage**: All preprocessing × all models
4. **Maintainable**: Clear structure, good documentation
5. **Deterministic**: Same results every run

### Common Pitfalls to Avoid
- Over-mocking: Test real functionality
- Large datasets: Keep data minimal
- Slow operations: Limit iterations
- Flaky tests: Ensure determinism
- Poor isolation: Each test independent

## Review Points
- Are all preprocessing methods adequately tested?
- Do tests run quickly enough for CI/CD?
- Is the test data realistic enough?
- Are edge cases covered?
- Is the code maintainable?

## Next Steps
1. Review and refine this plan
2. Create GitHub issue with summary
3. Begin implementation with mock data generator
4. Iterate based on feedback