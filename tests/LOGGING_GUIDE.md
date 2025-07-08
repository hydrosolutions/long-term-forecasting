# How to See Logger Outputs for Linear Regression Tests

## Different Ways to Run Tests with Logging

### 1. Basic Run (Shows INFO level and above)
```bash
python tests/test_linear_regression.py
```

### 2. Verbose Mode (Shows DEBUG level logs)
```bash
python tests/test_linear_regression.py --verbose
```

### 3. Only Show Detailed Logs for Failed Tests
```bash
python tests/test_linear_regression.py --only-failures
```

### 4. Custom Log File
```bash
python tests/test_linear_regression.py --log-file my_custom_test.log
```

### 5. Combined Options
```bash
python tests/test_linear_regression.py --verbose --only-failures --log-file debug_test.log
```

### 6. Using the Runner Script
```bash
python run_linear_regression_tests.py --verbose
python run_linear_regression_tests.py --only-failures
```

## Log Outputs Explained

### Console Output
- **Basic mode**: Shows test progress and results only
- **Verbose mode**: Shows detailed DEBUG logs from all modules
- **Only-failures mode**: Only shows detailed logs when tests fail

### Log File
- Always contains DEBUG level logs regardless of console setting
- Includes full tracebacks for exceptions
- Persists between runs (overwrites previous log)

### What You'll See for Failed Tests

When a test fails, you'll see:

1. **Console**: 
   ```
   ✗ Model Initialization FAILED: AttributeError: 'LinearRegressionModel' object has no attribute 'fitted_models'
   Check log file for detailed traceback
   ```

2. **Log File**:
   ```
   2025-07-03 14:30:15,123 - __main__ - ERROR - ✗ Model Initialization failed: AttributeError: 'LinearRegressionModel' object has no attribute 'fitted_models'
   2025-07-03 14:30:15,123 - __main__ - ERROR - Exception type: AttributeError
   2025-07-03 14:30:15,124 - __main__ - ERROR - Exception details:
   Traceback (most recent call last):
     File "/path/to/test_linear_regression.py", line 345, in test_model_initialization
       assert hasattr(model, 'fitted_models'), "Model should have fitted_models attribute"
   AttributeError: 'LinearRegressionModel' object has no attribute 'fitted_models'
   ```

## Quick Test for Specific Failures

If you want to quickly test and see only failure details:

```bash
# This will run quietly and only show detailed output for failures
python tests/test_linear_regression.py --only-failures --log-file quick_debug.log

# Then check the log file for full details
cat quick_debug.log
```

## Understanding Log Levels

- **DEBUG**: Very detailed information, including internal model operations
- **INFO**: General information about test progress and results
- **WARNING**: Potential issues that don't cause test failure
- **ERROR**: Test failures and exceptions

## Module-Specific Logging

The test automatically configures logging for different modules:

- `forecast_models.LINEAR_REGRESSION`: The main model being tested
- `scr.FeatureExtractor`: Feature processing details
- `eval_scr.metric_functions`: Metrics calculation details

In verbose mode, all modules show DEBUG level. In normal mode, only the main test shows DEBUG while others show INFO or WARNING level.
