# Evaluation Scripts Directory

## Purpose
Contains evaluation utilities for assessing model performance through various metrics and visualization tools.

## Contents
- `__init__.py`: Package initialization
- `eval_helper.py`: Helper functions for model evaluation workflows
- `metric_functions.py`: Implementation of performance metrics

## Important Functions

### metric_functions.py
- `rsquared()`: Calculates R-squared (coefficient of determination)
- `rmse()`: Computes Root Mean Square Error
- `nse()`: Calculates Nash-Sutcliffe Efficiency
- `bias()`: Computes prediction bias
- `calculate_metrics()`: Wrapper function to compute all metrics

### eval_helper.py
- Evaluation workflow orchestration
- Result aggregation functions
- Performance visualization utilities

## Key Features
- Support for multiple evaluation metrics
- Handles both time series and cross-sectional evaluation
- Robust error handling for edge cases
- Consistent metric calculation across model types

## Usage
```python
from eval_scr.metric_functions import calculate_metrics

metrics = calculate_metrics(observed, predicted)
```

## Integration Points
- Used by all model classes during evaluation
- Called in test scripts for performance assessment
- Integrated with calibration workflows
- Results stored in model artifacts