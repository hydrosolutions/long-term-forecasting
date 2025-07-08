# Linear Regression Model Test Suite

This directory contains comprehensive tests for the Linear Regression forecasting model, including both calibration/hindcasting and operational forecasting functionality.

## Files

- `test_linear_regression.py` - Main test suite with comprehensive tests
- `run_linear_regression_tests.py` - Simple runner script (no pytest required)

## Features Tested

### 1. Model Initialization
- Proper loading of configurations
- Feature extraction from synthetic data
- Model instance creation and validation

### 2. Feature Processing
- Feature extraction based on configuration
- Window-based aggregations (mean, sum)
- Lag feature creation
- Target variable generation

### 3. Calibration and Hindcasting
- Leave-One-Year-Out cross-validation
- Model training for multiple basins
- Prediction generation
- Performance validation

### 4. Operational Forecasting
- Real-time prediction capability
- Period-based forecasting
- Date validation and formatting
- Multi-basin forecasting

### 5. Metrics Calculation
- R² (coefficient of determination)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- Bias calculation
- NSE (Nash-Sutcliffe Efficiency) if available

### 6. Model Persistence
- Model saving functionality
- Model loading capability
- Configuration persistence

## Test Data

The tests use synthetic hydro-meteorological data with realistic characteristics:

### Time Series Data
- **Period**: 2000-2020 (daily data)
- **Basins**: 3 synthetic basins (codes: 16936, 16940, 16942)
- **Variables**:
  - `discharge`: Streamflow with seasonal patterns and precipitation/temperature effects
  - `T`: Temperature with realistic seasonal cycles (-10°C to 20°C)
  - `P`: Precipitation with seasonal variability and stochastic components
- **Patterns**: 
  - Spring snowmelt peaks
  - Temperature-driven discharge
  - Precipitation-runoff relationships
  - Basin-specific scaling factors

### Static Data
- Basin area, elevation, slope, forest cover
- Realistic parameter ranges
- Basin-specific characteristics

## Configuration

### General Configuration
```json
{
    "prediction_horizon": 30,
    "offset": 30,
    "num_features": 3,
    "base_features": ["discharge", "P", "T"],
    "snow_vars": [],
    "forecast_days": ["end"],
    "filter_years": null,
    "model_name": "TestLinearRegression"
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
    "lr_type": "linear"
}
```

## Running the Tests

### Option 1: Simple Runner (Recommended)
```bash
# From the monthly_forecasting directory
python run_linear_regression_tests.py
```

### Option 2: Direct Execution
```bash
# From the tests directory
cd tests
python test_linear_regression.py
```

### Option 3: With pytest (if available)
```bash
# From the monthly_forecasting directory
pytest tests/test_linear_regression.py -v

# Run specific test
pytest tests/test_linear_regression.py::test_lr_calibration_hindcast -v

# With coverage
pytest tests/test_linear_regression.py --cov=forecast_models.LINEAR_REGRESSION
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
STARTING LINEAR REGRESSION MODEL TESTS
======================================================================

[1/6] Running test_model_initialization...
✓ Model initialization test passed

[2/6] Running test_feature_extraction...
✓ Feature extraction test passed

[3/6] Running test_configuration_loading...
✓ Configuration loading test passed

[4/6] Running test_model_persistence...
✓ Model persistence test skipped (no fitted models)

[5/6] Running test_operational_forecast...
✓ Operational forecast test passed
  Generated forecasts for 3 basins
  Forecast range: 45.23 to 123.67

[6/6] Running test_end_to_end_workflow...
✓ End-to-end workflow test passed
  Results saved to: /tmp/lr_test_xyz/results

======================================================================
TEST SUMMARY
======================================================================
Tests passed: 6/6
🎉 ALL TESTS PASSED!
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running the tests from the correct directory
2. **Missing Dependencies**: Ensure all required packages are installed
3. **Memory Issues**: The synthetic data generation uses reasonable amounts of memory
4. **Path Issues**: Tests create temporary directories and clean up automatically

### Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib (for plotting, if used)
- Standard Python libraries (logging, datetime, pathlib, etc.)

### Test Data Issues
If you encounter issues with the synthetic data:
- Check that the date ranges are valid
- Verify basin codes are properly formatted
- Ensure all required columns are present

## Customization

### Modifying Test Parameters
You can customize the test by modifying the constants at the top of `test_linear_regression.py`:

- `GENERAL_CONFIG`: Adjust prediction horizon, offset, features
- `FEATURE_CONFIG`: Change feature extraction parameters
- `MODEL_CONFIG`: Modify model-specific settings
- Basin codes, date ranges, noise levels in `TestDataGenerator`

### Adding New Tests
To add new tests:
1. Add a new method to the `LinearRegressionTester` class
2. Add the method to the `tests` list in `run_all_tests()`
3. Create a corresponding pytest function if using pytest

## Integration

These tests can be integrated into CI/CD pipelines:
- Return exit codes (0 for success, 1 for failure)
- Generate output suitable for automated parsing
- Create temporary files that are automatically cleaned up
- Work without external dependencies beyond the model requirements
