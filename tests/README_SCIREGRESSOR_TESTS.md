# SciRegressor Model Test Suite

This directory contains comprehensive tests for the SciRegressor forecasting model, including calibration/hindcasting, operational forecasting, and hyperparameter tuning functionality.

## Files

- `test_sciregressor.py` - Main test suite with comprehensive tests (64 total tests)
- `comprehensive_test_configs.py` - Configuration constants for comprehensive testing
- `comprehensive_test_utils.py` - Utility functions for comprehensive testing
- `run_sciregressor_tests.py` - Simple runner script (no pytest required)

## Test Coverage

### Comprehensive Testing (64 Total Tests)

This test suite provides comprehensive coverage for **3 models** × **4 preprocessing methods** × **4 workflow components** + **12 complete workflow tests** + **4 multi-model integration tests** = **64 total tests**.

#### Models Tested
- **XGBoost** (`xgb`) - Gradient boosting trees
- **LightGBM** (`lgbm`) - Light gradient boosting machine  
- **CatBoost** (`catboost`) - Categorical boosting

#### Preprocessing Methods Tested
- **No Normalization** - Raw data without scaling
- **Global Normalization** - Standard scaling across all basins
- **Per-Basin Normalization** - Basin-specific scaling
- **Long-term Mean Scaling** - Seasonal normalization based on historical means

#### Workflow Components Tested
1. **Hyperparameter Tuning** - Optuna-based optimization (12 tests)
2. **Calibration** - Leave-One-Year-Out cross-validation (12 tests)
3. **Hindcast** - Historical validation (12 tests)
4. **Operational Prediction** - Real-time forecasting (12 tests)
5. **Complete Workflow** - End-to-end pipeline testing (12 tests)
6. **Multi-Model Integration** - Cross-model ensemble testing (4 tests)

## Features Tested

### 1. Model Initialization
- Proper loading of configurations with ensemble model support
- Feature extraction from synthetic data
- Model instance creation and validation
- Multiple model types (XGBoost, LightGBM, CatBoost)

### 2. Feature Processing
- Feature extraction based on configuration
- Window-based aggregations (mean, sum)
- Temporal feature engineering (cyclical encoding)
- Static feature integration
- Categorical feature handling

### 3. Calibration and Hindcasting
- Leave-One-Year-Out cross-validation
- Global model training across basins
- Ensemble prediction generation
- Performance validation

### 4. Operational Forecasting
- Real-time prediction capability
- Model loading and application
- Date validation and formatting
- Multi-basin ensemble forecasting

### 5. Hyperparameter Tuning
- Parameter optimization functionality
- Cross-validation based tuning
- Model selection and validation

### 6. Model Persistence
- Model saving functionality
- Model loading capability
- Configuration and artifact persistence

## Test Data

The tests use enhanced synthetic hydro-meteorological data with complex relationships:

### Time Series Data
- **Period**: 2000-present (daily data)
- **Basins**: 3 synthetic basins (codes: 16936, 16940, 16942)
- **Variables**:
  - `discharge`: Complex streamflow with multiple influences
  - `T`: Temperature with trends and elevation effects
  - `P`: Precipitation with extremes and seasonal patterns
- **Enhanced Patterns**: 
  - Delayed precipitation effects
  - Temperature-dependent snowmelt
  - Extreme weather events
  - Inter-annual variability
  - Basin-specific scaling

### Static Data
- Basin area, elevation, slope, forest cover, coordinates
- Enhanced characteristics for ensemble modeling
- Basin-specific parameter ranges

## Configuration

### General Configuration
```json
{
    "prediction_horizon": 30,
    "offset": 30,
    "num_features": 5,
    "base_features": ["discharge", "P", "T"],
    "models": ["xgboost", "random_forest"],
    "target": "target",
    "missing_value_handling": "drop",
    "normalization_type": "standard",
    "use_temporal_features": true,
    "use_static_features": true,
    "cat_features": ["code_str"],
    "static_features": ["area", "elevation"]
}
```

### Feature Configuration
```json
{
    "discharge": [
        {
            "operation": "mean",
            "windows": [15, 30],
            "lags": {}
        }
    ],
    "P": [
        {
            "operation": "sum",
            "windows": [30],
            "lags": {}
        }
    ],
    "T": [
        {
            "operation": "mean",
            "windows": [15],
            "lags": {}
        }
    ]
}
```

### Model Configuration
```json
{
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1
    },
    "random_forest": {
        "n_estimators": 50,
        "max_depth": 10
    }
}
```

## Running the Tests

### Option 1: Comprehensive Tests (Recommended)
```bash
# From the tests directory
python test_sciregressor.py --comprehensive

# Run with only failure details shown
python test_sciregressor.py --comprehensive --only-failures

# Run with verbose logging
python test_sciregressor.py --comprehensive --verbose
```

### Option 2: Original Basic Tests
```bash
# From the tests directory
python test_sciregressor.py

# From the monthly_forecasting directory
python run_sciregressor_tests.py
```

### Option 3: With pytest (if available)

#### Run All Comprehensive Tests (64 tests)
```bash
# From the monthly_forecasting directory
pytest tests/test_sciregressor.py -v

# Run specific comprehensive test
pytest tests/test_sciregressor.py::test_xgb_hyperparameter_tuning_no_normalization -v

# Run all XGBoost tests
pytest tests/test_sciregressor.py -k "xgb" -v

# Run all global normalization tests
pytest tests/test_sciregressor.py -k "global_normalization" -v

# Run all hyperparameter tuning tests
pytest tests/test_sciregressor.py -k "hyperparameter_tuning" -v

# With coverage
pytest tests/test_sciregressor.py --cov=forecast_models.SciRegressor
```

#### Run Original Basic Tests (6 tests)
```bash
pytest tests/test_sciregressor.py::test_sciregressor_initialization -v
pytest tests/test_sciregressor.py::test_sciregressor_calibration_hindcast -v
```

### Command Line Options

```bash
# Comprehensive testing mode
python tests/test_sciregressor.py --comprehensive

# Verbose logging (DEBUG level)
python tests/test_sciregressor.py --verbose

# Only show detailed logs for failed tests
python tests/test_sciregressor.py --only-failures

# Custom log file
python tests/test_sciregressor.py --log-file debug_sciregressor.log

# Combined options for comprehensive testing
python tests/test_sciregressor.py --comprehensive --verbose --only-failures --log-file detailed_test.log
```

## Expected Output

### Comprehensive Test Output (64 tests)
The comprehensive test suite will output:
1. **Setup Information**: Data generation details and test environment setup
2. **Test Progress**: Individual test results with ✓/✗ indicators for all 64 tests
3. **Model Performance**: Validation for each model and preprocessing combination
4. **Summary**: Overall test results and success/failure count

### Sample Comprehensive Output
```
============================================================
STARTING COMPREHENSIVE SCIREGRESSOR TESTS
============================================================

[1/64] xgb_hyperparameter_tuning_no_normalization...
✓ xgb_hyperparameter_tuning_no_normalization passed (0.34s)

[2/64] xgb_hyperparameter_tuning_global_normalization...
✓ xgb_hyperparameter_tuning_global_normalization passed (0.30s)

[3/64] xgb_hyperparameter_tuning_per_basin_normalization...
✓ xgb_hyperparameter_tuning_per_basin_normalization passed (0.38s)

[4/64] xgb_hyperparameter_tuning_long_term_mean_scaling...
✓ xgb_hyperparameter_tuning_long_term_mean_scaling passed (0.38s)

[5/64] lgbm_hyperparameter_tuning_no_normalization...
✓ lgbm_hyperparameter_tuning_no_normalization passed (0.35s)

... (continues for all 64 tests)

[64/64] multi_model_ensemble_long_term_mean_scaling...
✓ multi_model_ensemble_long_term_mean_scaling passed (0.42s)

============================================================
COMPREHENSIVE TEST SUMMARY
============================================================
Total tests run: 64
Tests passed: 64
Tests failed: 0
🎉 ALL COMPREHENSIVE TESTS PASSED!
```

### Basic Test Output (6 tests)
The basic test suite will output:
```
======================================================================
STARTING SCIREGRESSOR MODEL TESTS
======================================================================

[1/6] Model Initialization...
✓ Model Initialization passed

[2/6] Feature Extraction...
✓ Feature Extraction passed

[3/6] Configuration Loading...
✓ Configuration Loading passed

[4/6] Calibration and Hindcast...
✓ Calibration and Hindcast passed
  Generated 1000 predictions for 3 basins

[5/6] Operational Forecast...
✓ Operational Forecast passed
  Generated forecasts for 3 basins

[6/6] Hyperparameter Tuning...
✓ Hyperparameter Tuning passed
  Tuning result: True, Message: Hyperparameters tuned successfully

======================================================================
TEST SUMMARY
======================================================================
Tests passed: 6/6
🎉 ALL TESTS PASSED!
```

## Mock Dependencies

The tests include comprehensive mocking for external dependencies that might not be available in the test environment:

- **GeoPandas operations**: HRU shapefile processing
- **External data utilities**: Glacier mapper features
- **Model training**: XGBoost, Random Forest implementations
- **Feature processing**: Complex artifacts and transformations

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the correct directory
2. **Missing Dependencies**: The tests mock external dependencies automatically
3. **Memory Issues**: Enhanced synthetic data generation uses reasonable memory
4. **Path Issues**: Tests create temporary directories and clean up automatically

### Dependencies
- pandas, numpy, scikit-learn
- Standard Python libraries (logging, datetime, pathlib, etc.)
- Mock utilities for external dependencies

### Test Data Issues
If you encounter issues:
- Check that the date ranges are valid
- Verify basin codes are properly formatted
- Ensure all required columns are present
- Review the enhanced synthetic data generation logic

## Customization

### Modifying Test Parameters
You can customize the test by modifying the constants at the top of `test_sciregressor.py`:

- `GENERAL_CONFIG`: Adjust models, features, processing options
- `FEATURE_CONFIG`: Change feature extraction parameters
- `MODEL_CONFIG`: Modify model-specific hyperparameters
- Basin codes, date ranges, complexity in `TestDataGenerator`

### Adding New Tests
To add new tests:
1. Add a new method to the `SciRegressorTester` class
2. Add the method to the `test_suite` list in `run_all_tests()`
3. Create a corresponding pytest function if using pytest

## Comprehensive Test Structure

### Test Organization
The comprehensive test suite is organized into logical phases:

1. **Phase 1: Individual Component Tests (48 tests)**
   - Tests each workflow component separately
   - Validates each model type with each preprocessing method
   - Ensures individual components work correctly in isolation

2. **Phase 2: Complete Workflow Tests (12 tests)**
   - Tests end-to-end workflows: tuning → calibration → operational prediction
   - Validates complete model lifecycle
   - Ensures workflow components integrate properly

3. **Phase 3: Multi-Model Integration Tests (4 tests)**
   - Tests ensemble functionality across different models
   - Validates cross-model compatibility
   - Ensures model combinations work correctly

### Test Naming Convention
All comprehensive tests follow a consistent naming pattern:
- `test_{model_type}_{component}_{preprocessing_method}()`
- Examples: `test_xgb_hyperparameter_tuning_no_normalization()`
- Makes it easy to identify which combination is being tested

### Test Execution Features
- **Timeout Protection**: Each test has appropriate timeout limits
- **Error Handling**: Comprehensive error reporting with detailed logs
- **Environment Isolation**: Each test runs in a clean, isolated environment
- **Resource Cleanup**: Automatic cleanup of temporary files and directories
- **Progress Tracking**: Real-time progress indicators during execution

## SciRegressor-Specific Features

### Ensemble Modeling
- Tests multiple model types simultaneously
- Validates ensemble prediction generation
- Checks model-specific configuration handling

### Global Training Approach
- Tests training across all basins and periods
- Validates feature standardization and scaling
- Checks categorical feature encoding

### Advanced Feature Processing
- Tests temporal feature engineering
- Validates static feature integration
- Checks complex preprocessing pipelines

### Hyperparameter Optimization
- Tests parameter tuning workflows
- Validates cross-validation approaches
- Checks model selection procedures

## Integration

These tests can be integrated into CI/CD pipelines:
- Return exit codes (0 for success, 1 for failure)
- Generate output suitable for automated parsing
- Work with mocked dependencies for isolated testing
- Create temporary files that are automatically cleaned up
