# SciRegressor Model Test Suite

This directory contains comprehensive tests for the SciRegressor forecasting model, including calibration/hindcasting, operational forecasting, and hyperparameter tuning functionality.

## Files

- `test_sciregressor.py` - Main test suite with comprehensive tests
- `run_sciregressor_tests.py` - Simple runner script (no pytest required)

## Features Tested

### 1. Model Initialization
- Proper loading of configurations with ensemble model support
- Feature extraction from synthetic data
- Model instance creation and validation
- Multiple model types (XGBoost, Random Forest, etc.)

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

### Option 1: Simple Runner (Recommended)
```bash
# From the monthly_forecasting directory
python run_sciregressor_tests.py
```

### Option 2: Direct Execution
```bash
# From the tests directory
cd tests
python test_sciregressor.py
```

### Option 3: With pytest (if available)
```bash
# From the monthly_forecasting directory
pytest tests/test_sciregressor.py -v

# Run specific test
pytest tests/test_sciregressor.py::test_sciregressor_calibration_hindcast -v

# With coverage
pytest tests/test_sciregressor.py --cov=forecast_models.SciRegressor
```

### Command Line Options

```bash
# Verbose logging (DEBUG level)
python tests/test_sciregressor.py --verbose

# Only show detailed logs for failed tests
python tests/test_sciregressor.py --only-failures

# Custom log file
python tests/test_sciregressor.py --log-file debug_sciregressor.log

# Combined options
python tests/test_sciregressor.py --verbose --only-failures --log-file detailed_test.log
```

## Expected Output

The test suite will output:
1. **Setup Information**: Data generation details and test environment setup
2. **Test Progress**: Individual test results with ✓/✗ indicators
3. **Model Performance**: Metrics and statistics from calibration
4. **Summary**: Overall test results and success/failure count

### Sample Output
```
======================================================================
STARTING SCIREGRESSOR MODEL TESTS
======================================================================

[1/8] Model Initialization...
✓ Model Initialization passed

[2/8] Feature Extraction...
✓ Feature Extraction passed

[3/8] Configuration Loading...
✓ Configuration Loading passed

[4/8] Calibration and Hindcast...
✓ Calibration and Hindcast passed
  Generated 1000 predictions for 3 basins

[5/8] Operational Forecast...
✓ Operational Forecast passed
  Generated forecasts for 3 basins

[6/8] Hyperparameter Tuning...
✓ Hyperparameter Tuning passed
  Tuning result: True, Message: Hyperparameters tuned successfully

[7/8] Model Persistence...
✓ Model Persistence passed

[8/8] End-to-End Workflow...
✓ End-to-End Workflow passed
  Results saved to: /tmp/sciregressor_test_xyz/results

======================================================================
TEST SUMMARY
======================================================================
Tests passed: 8/8
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
