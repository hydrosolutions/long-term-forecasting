# eval_scr - Evaluation Scripts Module

This module provides evaluation utilities and metric functions for hydrological forecasting models in the monthly forecasting system.

## Module Structure

### Files

#### `__init__.py`
Empty initialization file that makes this directory a Python package.

#### `eval_helper.py`
Contains helper functions for model evaluation and visualization.

**Key Functions:**
- `get_r2_rmse(df_predictions)`: Calculates R² and RMSE metrics for predictions
  - Computes overall R² and RMSE
  - Calculates per-station R² and normalized RMSE (NRMSE)
  - Generates box plots for R² and NRMSE distributions
  - Saves visualization to output directory
  - Note: References `PATH_CONFIG` which needs to be defined in the calling context

**Dependencies:**
- pandas, numpy, matplotlib, seaborn, geopandas
- sklearn (for metrics and preprocessing)
- tqdm, datetime, joblib
- logging

#### `metric_functions.py`
Comprehensive collection of metric functions for evaluating hydrological forecasts.

**Core Metric Functions:**

1. **Forecast Accuracy Metrics:**
   - `sdivsigma_nse()`: Calculates forecast efficacy (s/σ) and Nash-Sutcliffe Efficiency (NSE)
   - `calc_accuracy()`: Simple accuracy calculation based on threshold
   - `forecast_accuracy_hydromet()`: Hydromet-specific accuracy calculation with delta tolerance

2. **Standard Performance Metrics:**
   - `calculate_NSE()`: Nash-Sutcliffe Efficiency with robust error handling
   - `calculate_RMSE()`: Root Mean Square Error (with optional normalization)
   - `calculate_MAE()`: Mean Absolute Error (with optional normalization)
   - `calculate_R2()`: R-squared coefficient of determination

3. **Advanced Metrics:**
   - `calculate_QuantileLoss()`: Pinball loss for quantile forecasts
   - `calculate_crps_from_quantiles()`: Continuous Ranked Probability Score from quantile forecasts
   - `calculate_mean_CRPS()`: Mean CRPS for ensemble forecasts
   - `kge()`: Kling-Gupta Efficiency

4. **Convenience Wrappers:**
   - `r2_score()`: Wrapper for R² calculation
   - `rmse()`: Wrapper for RMSE calculation
   - `mae()`: Wrapper for MAE calculation
   - `bias()`: Calculate mean error (bias)
   - `nse()`: Wrapper for NSE calculation

5. **Temporal Aggregation Functions:**
   - `calculate_metrics_pentad()`: Calculate metrics for pentad (5-day) periods
   - `calculate_metrics_decad()`: Calculate metrics for decad (10-day) periods

**Key Features:**
- Robust handling of NaN values and edge cases
- Numerical stability checks
- Extensive error handling and logging
- Support for both deterministic and probabilistic forecasts
- Normalized and non-normalized versions of metrics

## Usage

### Import Examples

```python
# Import specific functions
from eval_scr import eval_helper, metric_functions
from eval_scr.metric_functions import r2_score, rmse, mae, nse, kge

# Calculate metrics
r2 = metric_functions.calculate_R2(observed, simulated)
nse_value = metric_functions.calculate_NSE(observed, simulated)
rmse_value = metric_functions.calculate_RMSE(observed, simulated, normalize=True)
```

### Example: Evaluating Model Predictions

```python
import pandas as pd
from eval_scr import eval_helper

# Assuming df_predictions has columns: 'Q_obs', 'Q_pred', 'code'
eval_helper.get_r2_rmse(df_predictions)
# This will generate visualization and print metrics
```

### Example: Calculating Forecast Accuracy

```python
from eval_scr.metric_functions import forecast_accuracy_hydromet

# Calculate accuracy with tolerance
results = forecast_accuracy_hydromet(
    data=df,
    observed_col='observed',
    simulated_col='predicted',
    delta_col='tolerance'
)
accuracy = results['accuracy']
```

## Dependencies

This module is used by:
- `calibrate_hindcast.py`: For model calibration and evaluation
- `tune_hyperparams.py`: For hyperparameter tuning evaluation
- `evaluation/evaluate_models.py`: For comprehensive model evaluation
- Test files in `tests/` directory

## Integration Notes

- The module expects data in pandas DataFrame format
- Most functions handle NaN values gracefully
- Metric functions return numpy.nan for invalid calculations
- Logging is configured at INFO level by default
- Functions are designed to work with hydrological time series data

## Performance Considerations

- Functions use numpy arrays internally for efficient computation
- Vectorized operations are preferred over loops
- Memory-efficient handling of large datasets
- Proper numerical stability checks to avoid overflow/underflow