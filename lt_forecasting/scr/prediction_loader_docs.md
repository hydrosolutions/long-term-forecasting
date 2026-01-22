# Prediction Loader Documentation

## Overview

The `prediction_loader` module provides utilities for loading and processing model predictions from various sources (filesystem, databases, APIs). This enables database compatibility and better separation of concerns.

## Expected Input Formats

### For SciRegressor Models

SciRegressor expects predictions in the following format:

#### CSV File Format
```csv
date,code,Q_model1,Q_model2,...
2024-01-01,1,100.5,95.3
2024-01-01,2,150.2,145.8
2024-01-02,1,105.1,98.7
```

**Required Columns:**
- `date`: Date in YYYY-MM-DD format (or any pandas-parseable date format)
- `code`: Integer basin/station code
- `Q_{model_name}`: Prediction columns with Q_ prefix followed by model name

**File Structure (for filesystem loading):**
```
predictions/
├── model1/
│   └── predictions.csv
├── model2/
│   └── predictions.csv
└── model3/
    └── predictions.csv
```

The model name is extracted from the parent directory name.

#### DataFrame Format
```python
import pandas as pd

predictions = pd.DataFrame({
    'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
    'code': [1, 1],
    'Q_model1': [100.5, 105.1],
    'Q_model2': [95.3, 98.7]
})
```

**Expected Units:**
- Predictions should be in **m³/s** (cubic meters per second)
- The `apply_area_conversion()` function converts to mm/month using: `value * area_km2 / 86.4`

#### Usage with SciRegressor

```python
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    apply_area_conversion
)
from lt_forecasting.forecast_models.SciRegressor import SciRegressor

# Method 1: Load from filesystem
paths = [
    "/data/predictions/model1/predictions.csv",
    "/data/predictions/model2/predictions.csv",
]
base_preds, model_names = load_predictions_from_filesystem(
    paths,
    join_type="inner"  # Only keep common (date, code) pairs
)

# Apply area conversion (required for SciRegressor)
base_preds = apply_area_conversion(base_preds, static_data, model_names)

# Create model with external predictions
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,      # Pass external predictions
    base_model_names=model_names      # Pass column names
)

# Method 2: Load from database
import sqlalchemy as sa

engine = sa.create_engine("postgresql://...")
query = "SELECT date, code, Q_model1, Q_model2 FROM predictions"
df = pd.read_sql(query, engine)

base_preds, model_names = load_predictions_from_dataframe(
    df,
    model_names=['model1', 'model2']
)
base_preds = apply_area_conversion(base_preds, static_data, model_names)

model = SciRegressor(..., base_predictors=base_preds, base_model_names=model_names)
```

---

### For BaseMetaLearner Models

BaseMetaLearner (including HistoricalMetaLearner and UncertaintyMixtureModel) expects predictions in a similar format but with different naming conventions.

#### CSV File Format
```csv
date,code,Q_model1,Q_model2_member1,Q_model2_member2,...
2024-01-01,1,100.5,95.3,96.1
2024-01-01,2,150.2,145.8,146.5
2024-01-02,1,105.1,98.7,99.2
```

**Required Columns:**
- `date`: Date in YYYY-MM-DD format
- `code`: Integer basin/station code
- `Q_{model_name}` or `Q_{model_name}_{member}`: Prediction columns

**Special Handling:**
- If a CSV contains multiple Q_ columns, they're treated as ensemble members
- Ensemble mean column (Q_{model_name}) is optional
- Column naming: `{model_name}_{sub_model}` for ensemble members

**File Structure:**
```
predictions/
├── model1/
│   └── predictions.csv          # Contains Q_model1
├── ensemble_model/
│   └── predictions.csv          # Contains Q_ensemble_model, Q_ensemble_model_member1, Q_ensemble_model_member2
└── model3/
    └── predictions.csv
```

#### DataFrame Format
```python
predictions = pd.DataFrame({
    'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
    'code': [1, 1],
    'Q_model1': [100.5, 105.1],
    'Q_ensemble_member1': [95.3, 98.7],
    'Q_ensemble_member2': [96.1, 99.2]
})
```

**Column Naming Convention:**
- External predictions use `Q_{model_name}` format
- Internally, BaseMetaLearner strips the `Q_` prefix
- Final column names: `model1`, `ensemble_member1`, `ensemble_member2`

**Expected Units:**
- BaseMetaLearner does NOT apply area conversion by default
- Predictions should already be in the target unit system
- If needed, apply conversions before passing to the model

#### Usage with BaseMetaLearner

```python
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    handle_duplicate_predictions
)
from lt_forecasting.forecast_models.meta_learners.base_meta_learner import BaseMetaLearner

# Load from filesystem
paths = [
    "/data/predictions/model1/predictions.csv",
    "/data/predictions/ensemble/predictions.csv",
]
base_preds, model_names = load_predictions_from_filesystem(
    paths,
    join_type="left"  # Keep all dates from first model
)

# Handle any duplicates
base_preds = handle_duplicate_predictions(base_preds, pred_cols=model_names)

# Create model with external predictions
model = BaseMetaLearner(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,      # Pass external predictions
    base_model_names=model_names      # Pass column names (with Q_ prefix)
)

# Note: BaseMetaLearner will automatically strip Q_ prefix internally
```

---

## Key Differences Between SciRegressor and BaseMetaLearner

| Aspect | SciRegressor | BaseMetaLearner |
|--------|--------------|-----------------|
| **Join Type** | `inner` (intersection) | `left` (keep all dates from main data) |
| **Area Conversion** | Yes, required | No, assumes already converted |
| **Column Naming** | Uses `Q_{model}` as-is | Strips `Q_` prefix internally |
| **Ensemble Handling** | Single prediction per model | Supports ensemble members |
| **Duplicate Handling** | Not built-in | Built-in averaging |

---

## Complete Examples

### Example 1: SciRegressor with Filesystem Predictions

```python
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    apply_area_conversion
)
from lt_forecasting.forecast_models.SciRegressor import SciRegressor

# Configuration
path_config = {
    "path_to_lr_predictors": [
        "/data/predictions/linear_regression/predictions.csv",
        "/data/predictions/xgboost/predictions.csv",
        "/data/predictions/lstm/predictions.csv",
    ]
}

# Load predictions
base_preds, model_names = load_predictions_from_filesystem(
    path_config["path_to_lr_predictors"],
    join_type="inner"
)
# model_names will be: ['Q_linear_regression', 'Q_xgboost', 'Q_lstm']

# Apply area-based unit conversion
base_preds = apply_area_conversion(
    predictions=base_preds,
    static_data=static_data,  # Must contain 'code' and 'area_km2' columns
    pred_cols=model_names
)

# Create model
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,
    base_model_names=model_names
)

# Calibrate and hindcast
hindcast_df = model.calibrate_model_and_hindcast(
    train_years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    test_years=[2018, 2019, 2020]
)
```

### Example 2: BaseMetaLearner with Database Predictions

```python
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_dataframe,
    handle_duplicate_predictions
)
from lt_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner
)
import sqlalchemy as sa

# Load from database
engine = sa.create_engine("postgresql://user:pass@localhost/forecasts")
query = """
    SELECT
        date,
        code,
        prediction_lr as Q_linear_regression,
        prediction_xgb as Q_xgboost,
        prediction_lstm as Q_lstm
    FROM model_predictions
    WHERE date >= '2010-01-01'
"""
df = pd.read_sql(query, engine)

# Convert to standard format
base_preds, model_names = load_predictions_from_dataframe(
    df=df,
    model_names=['linear_regression', 'xgboost', 'lstm']
)
# model_names will be: ['Q_linear_regression', 'Q_xgboost', 'Q_lstm']

# Handle any duplicate (date, code) pairs
base_preds = handle_duplicate_predictions(base_preds, pred_cols=model_names)

# Create meta-learner
model = HistoricalMetaLearner(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,
    base_model_names=model_names
)

# Train meta-learner
hindcast_df = model.calibrate_model_and_hindcast(
    train_years=[2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017],
    test_years=[2018, 2019, 2020]
)
```

### Example 3: Hybrid Approach (Cache Predictions)

```python
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    apply_area_conversion
)

# Load predictions once
base_preds, model_names = load_predictions_from_filesystem(
    path_config["path_to_lr_predictors"],
    join_type="inner"
)
base_preds = apply_area_conversion(base_preds, static_data, model_names)

# Reuse across multiple model instances
model1 = SciRegressor(..., base_predictors=base_preds, base_model_names=model_names)
model2 = SciRegressor(..., base_predictors=base_preds, base_model_names=model_names)
model3 = SciRegressor(..., base_predictors=base_preds, base_model_names=model_names)

# Benefits: Load predictions only once, saving I/O time
```

---

## Data Validation

### Expected Data Types

```python
# Predictions DataFrame
predictions = pd.DataFrame({
    'date': pd.Timestamp or datetime,    # Will be converted to datetime64
    'code': int,                         # Basin/station identifier
    'Q_model1': float,                   # Prediction values
    'Q_model2': float,
    # ... more prediction columns
})

# Static Data DataFrame (required for area conversion)
static_data = pd.DataFrame({
    'code': int,                         # Must match prediction codes
    'area_km2': float,                   # Basin area in km²
    # ... other static features
})
```

### Common Issues and Solutions

#### Issue 1: Missing Codes in Static Data
```python
# Problem: Predictions contain codes not in static_data
# Solution: apply_area_conversion() will skip those codes with a warning

base_preds = apply_area_conversion(base_preds, static_data, model_names)
# Warning: Code 123 not found in static data. Skipping conversion for this basin.
```

#### Issue 2: Duplicate (date, code) Pairs
```python
# Problem: Multiple predictions for same date and code
# Solution: Use handle_duplicate_predictions() to average them

base_preds = handle_duplicate_predictions(base_preds, pred_cols=model_names)
# Info: Found 5 duplicate (date, code) pairs. Averaging prediction values.
```

#### Issue 3: Missing Prediction Columns
```python
# Problem: CSV doesn't have Q_{model_name} column
# Solution: load_predictions_from_filesystem() will skip that file with a warning

base_preds, model_names = load_predictions_from_filesystem(paths)
# Warning: Prediction column 'Q_model1' not found in model1. Skipping this model.
```

#### Issue 4: Inconsistent Date Formats
```python
# Problem: Dates in different formats
# Solution: Pandas will attempt to parse automatically

# These all work:
dates = ['2024-01-01', '01/01/2024', '2024-1-1', '20240101']
df['date'] = pd.to_datetime(df['date'])  # Handles most formats
```

---

## Advanced Usage

### Custom Unit Conversions

```python
from lt_forecasting.scr.prediction_loader import apply_area_conversion

# Custom conversion for different units
def custom_conversion(predictions, static_data, pred_cols, conversion_factor=86.4):
    """Apply custom unit conversion."""
    result = predictions.copy()

    for code in result["code"].unique():
        if code in static_data["code"].values:
            area = static_data[static_data["code"] == code]["area_km2"].values[0]
            result.loc[result["code"] == code, pred_cols] = (
                result.loc[result["code"] == code, pred_cols] * area / conversion_factor
            )

    return result

# Use custom conversion
base_preds = custom_conversion(base_preds, static_data, model_names, conversion_factor=100.0)
```

### Loading from APIs

```python
import requests
from lt_forecasting.scr.prediction_loader import load_predictions_from_dataframe

# Fetch from REST API
response = requests.get("https://api.example.com/predictions")
data = response.json()

# Convert JSON to DataFrame
df = pd.DataFrame(data['predictions'])

# Load into standard format
base_preds, model_names = load_predictions_from_dataframe(
    df,
    model_names=['model1', 'model2']
)
```

### Filtering Predictions by Date Range

```python
# Load all predictions
base_preds, model_names = load_predictions_from_filesystem(paths)

# Filter to specific date range
mask = (base_preds['date'] >= '2020-01-01') & (base_preds['date'] <= '2020-12-31')
base_preds = base_preds[mask].reset_index(drop=True)

# Use filtered predictions
model = SciRegressor(..., base_predictors=base_preds, base_model_names=model_names)
```

---

## Migration Guide

### From Old Pattern to New Pattern

#### Old Pattern (Deprecated)
```python
# Predictions loaded internally from path_config
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config  # Contains path_to_lr_predictors
)
# DeprecationWarning: Loading predictions internally is deprecated.
```

#### New Pattern (Recommended)
```python
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    apply_area_conversion
)

# Load predictions externally
base_preds, model_names = load_predictions_from_filesystem(
    path_config["path_to_lr_predictors"]
)
base_preds = apply_area_conversion(base_preds, static_data, model_names)

# Pass to model
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,      # NEW
    base_model_names=model_names      # NEW
)
```

---

## API Reference

See docstrings in `lt_forecasting/scr/prediction_loader.py` for detailed API documentation:

```python
from lt_forecasting.scr import prediction_loader

help(prediction_loader.load_predictions_from_filesystem)
help(prediction_loader.load_predictions_from_dataframe)
help(prediction_loader.apply_area_conversion)
help(prediction_loader.handle_duplicate_predictions)
help(prediction_loader.standardize_prediction_columns)
```
