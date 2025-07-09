# Tests Directory

## Purpose
Comprehensive test suite ensuring reliability and correctness of the monthly forecasting system. Implements configuration-driven testing with extensive coverage across all components.

## Contents
- `test_base_class.py`: Tests for abstract forecast model base class
- `test_data_utils.py`: Data utility function tests
- `test_feature_processing.py`: Feature extraction and processing tests
- `test_linear_regression.py`: Linear regression model tests
- `test_sci_utils.py`: Scientific utility function tests
- `test_sciregressor.py`: Tree-based model tests
- `comprehensive_test_configs.py`: Test configuration definitions
- `comprehensive_test_utils.py`: Shared test utilities
- `LOGGING_GUIDE.md`: Test logging best practices
- `README_LINEAR_REGRESSION_TESTS.md`: Linear regression test documentation
- `README_SCIREGRESSOR_TESTS.md`: SciRegressor test documentation
- `logs/`: Test execution logs

## Test Architecture

### Configuration-Driven Testing
Tests use `comprehensive_test_configs.py` to define:
- Model configurations
- Preprocessing methods
- Feature engineering parameters
- Performance thresholds
- Data generation parameters

### Test Categories

#### Unit Tests
- Individual function testing
- Edge case handling
- Error condition verification
- Input validation

#### Integration Tests
- End-to-end workflows
- Model training pipelines
- Data processing chains
- Cross-component interactions

#### Performance Tests
- Model accuracy validation
- Processing time constraints
- Memory usage monitoring
- Scalability verification

## Important Test Classes

### test_base_class.py
- `TestAbstractForecastModel`: Verifies interface compliance
- Tests inheritance and method implementation
- Validates common functionality

### test_linear_regression.py
- `TestLinearRegressionModel`: Comprehensive linear model tests
- Multi-basin handling
- Feature selection validation
- Performance metric verification

### test_sciregressor.py
- `TestSciRegressor`: Tree-based model testing
- Algorithm-specific tests (XGBoost, Random Forest, CatBoost)
- Feature importance validation
- Preprocessing method testing

### test_data_utils.py
- Data transformation tests
- Preprocessing validation
- Missing data handling
- Scaling and normalization

### test_feature_processing.py
- Feature extraction validation
- Lag feature generation
- Rolling statistics accuracy
- Temporal feature creation

## Test Utilities

### comprehensive_test_utils.py
- `generate_comprehensive_test_data()`: Creates realistic test datasets
- `create_test_artifacts()`: Generates model artifacts
- Performance assertion helpers
- Data validation utilities

## Running Tests

```bash
# Run all tests
python -m pytest -ra

# Run specific test file
python -m pytest tests/test_linear_regression.py -v

# Run with coverage
python -m pytest --cov=. --cov-report=html

# Run specific test class
python -m pytest tests/test_sciregressor.py::TestSciRegressor -v
```

## Test Data Generation
Tests use synthetic data that mimics real hydrological patterns:
- Seasonal variations
- Trend components
- Random noise
- Missing data scenarios
- Multiple basin configurations

## Performance Benchmarks
Each model type has expected performance thresholds:
- R² > 0.7 for well-specified models
- RMSE within acceptable ranges
- NSE > 0.5 for time series predictions
- Training time constraints

## Integration Points
- CI/CD pipeline integration
- Pre-commit hook compatibility
- Coverage reporting
- Performance tracking
- Artifact generation validation