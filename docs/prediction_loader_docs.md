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

**RECOMMENDED APPROACH**: Use `prediction_utils.load_base_predictions_for_model()` which handles prefix stripping automatically:

```python
from lt_forecasting.scr.prediction_utils import load_base_predictions_for_model
from lt_forecasting.forecast_models.SciRegressor import SciRegressor

# Automatically handles area conversion and prefix stripping
base_preds, model_names = load_base_predictions_for_model(
    model_type="sciregressor",
    path_config={"path_to_lr_predictors": ["/data/model1/", "/data/model2/"]},
    static_data=static_data
)

model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,
    base_model_names=model_names  # Already stripped of Q_ prefix
)
```

**MANUAL APPROACH** (if you need more control):

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
# model_names will be ['Q_model1', 'Q_model2'] at this point

# Strip Q_ prefix for SciRegressor (it adds prefix internally)
model_names = [name.replace("Q_", "") for name in model_names]
# model_names is now ['model1', 'model2']

# Apply area conversion (required for SciRegressor)
# Note: Still use the Q_-prefixed column names for conversion
base_preds = apply_area_conversion(base_preds, static_data, ['Q_model1', 'Q_model2'])

# Create model with external predictions
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,      # Pass external predictions
    base_model_names=model_names      # Pass model names WITHOUT Q_ prefix
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

## Input Format Specification

This section provides a comprehensive reference for all input formats and conventions used in the prediction loader system.

### CSV File Format

When using `load_predictions_from_filesystem()`, CSV files must follow this structure:

**Required Columns:**
- `date`: Date in ISO format (YYYY-MM-DD) or any pandas-parseable format
- `code`: Integer basin/station code
- `Q_{model_name}`: Float prediction values (one or more columns)

**Example:**
```csv
date,code,Q_model1,Q_model2
2024-01-01,1,100.5,95.3
2024-01-01,2,150.2,145.8
2024-01-02,1,105.1,98.7
2024-01-02,2,152.8,147.2
```

**Notes:**
- Date format is flexible (e.g., "2024-01-01", "01/01/2024", "20240101" all work)
- Code must be an integer matching codes in static_data
- Q_ prefix is mandatory in the CSV file
- Column names after Q_ must match the folder name containing the CSV

### DataFrame Format

When using `load_predictions_from_dataframe()`, input columns can have either format:

**With Q_ Prefix (Recommended):**
```python
df = pd.DataFrame({
    'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
    'code': [1, 1],
    'Q_model1': [100.5, 105.1],
    'Q_model2': [95.3, 98.7]
})
```

**Without Q_ Prefix (Also Accepted):**
```python
df = pd.DataFrame({
    'date': pd.to_datetime(['2024-01-01', '2024-01-02']),
    'code': [1, 1],
    'model1': [100.5, 105.1],
    'model2': [95.3, 98.7]
})
```

**Important:** The output will always have the Q_ prefix, regardless of input format.

### Column Naming Convention Reference

The following table shows how column names are transformed at different stages:

| Context | Input Format | Output Format | Example |
|---------|-------------|---------------|---------|
| File loading | Q_{folder_name} | Q_{model_name} | Folder: `lstm/` → Column: `Q_lstm` |
| DataFrame loading (with prefix) | Q_{model_name} | Q_{model_name} | `Q_xgboost` → `Q_xgboost` |
| DataFrame loading (without prefix) | {model_name} | Q_{model_name} | `random_forest` → `Q_random_forest` |
| **prediction_loader output** | - | **Q_{model_name}** | **Always returns with Q_ prefix** |
| **SciRegressor input** | **{model_name}** | - | **Expects WITHOUT Q_ prefix** |
| SciRegressor internal | {model_name} | Q_{model_name} | `lstm` → `Q_lstm` (adds prefix) |
| **BaseMetaLearner input** | **Q_{model_name} or {model_name}** | - | **Accepts both formats** |
| BaseMetaLearner internal | Q_{model_name} | {model_name} | `Q_lstm` → `lstm` (strips prefix) |

**Key Points:**
- File loading: Model name is extracted from the parent folder name
- DataFrame loading: Q_ prefix is added if not present
- **prediction_loader output**: Always returns column names WITH Q_ prefix
- **SciRegressor input**: Expects `base_model_names` WITHOUT Q_ prefix (model adds prefix internally)
- **BaseMetaLearner input**: Accepts either format (strips prefix internally if present)
- **Recommendation**: Use `prediction_utils.load_base_predictions_for_model()` which handles prefix stripping automatically

### Understanding the Q_ Prefix Inconsistency

**The Problem:**

Different model types have different expectations for the `base_model_names` parameter:

1. **prediction_loader output**: Always returns column names WITH `Q_` prefix
   ```python
   predictions, model_names = load_predictions_from_filesystem(paths)
   # model_names = ['Q_model1', 'Q_model2']
   ```

2. **SciRegressor expects**: Model names WITHOUT `Q_` prefix (adds it internally)
   ```python
   # CORRECT for SciRegressor:
   model = SciRegressor(..., base_model_names=['model1', 'model2'])

   # INCORRECT for SciRegressor:
   model = SciRegressor(..., base_model_names=['Q_model1', 'Q_model2'])
   # This would result in Q_Q_model1, Q_Q_model2
   ```

3. **BaseMetaLearner accepts**: Either format (handles both gracefully)
   ```python
   # Both work for BaseMetaLearner:
   model = BaseMetaLearner(..., base_model_names=['Q_model1', 'Q_model2'])
   model = BaseMetaLearner(..., base_model_names=['model1', 'model2'])
   ```

**The Solution:**

Use `prediction_utils.load_base_predictions_for_model()` which automatically handles the prefix stripping:

```python
from lt_forecasting.scr.prediction_utils import load_base_predictions_for_model

# Automatically strips Q_ prefix for SciRegressor
preds, model_names = load_base_predictions_for_model(
    model_type="sciregressor",
    path_config=path_config,
    static_data=static_data
)
# model_names will be ['model1', 'model2'] (no Q_ prefix)

# Keeps Q_ prefix for BaseMetaLearner
preds, model_names = load_base_predictions_for_model(
    model_type="UncertaintyMixtureModel",
    path_config=path_config,
    static_data=static_data
)
# model_names will be ['Q_model1', 'Q_model2'] (with Q_ prefix)
```

**Manual Handling (if not using prediction_utils):**

```python
from lt_forecasting.scr.prediction_loader import load_predictions_from_filesystem

# Load predictions (always returns with Q_ prefix)
predictions, model_names = load_predictions_from_filesystem(paths)

# For SciRegressor: Strip Q_ prefix
if model_type == "sciregressor":
    model_names = [name.replace("Q_", "") for name in model_names]

# For BaseMetaLearner: Use as-is (or strip, both work)
# No changes needed
```

### Unit Handling Reference

Different models have different unit requirements:

| Model | Expected Input Unit | Processing Applied | Output Unit |
|-------|--------------------|--------------------|-------------|
| SciRegressor | m³/s (cubic meters/second) | Converts to mm/day using area_km2 | mm/day |
| BaseMetaLearner | Any (model-dependent) | No conversion applied | Same as input |
| HistoricalMetaLearner | Any (model-dependent) | No conversion applied | Same as input |
| UncertaintyMixtureModel | Any (model-dependent) | No conversion applied | Same as input |

**Conversion Formula (SciRegressor only):**
```python
# Applied by apply_area_conversion()
value_mm_day = value_m3_s * area_km2 / 86.4
```

**Important Notes:**
- SciRegressor **requires** `apply_area_conversion()` to be called before passing predictions
- BaseMetaLearner models assume predictions are already in the correct units
- If your base models output in different units, convert them before loading

### Migration Example: Old vs New Pattern

This example shows how to migrate from the deprecated internal loading to the new external loading pattern.

#### Old Pattern (Deprecated)
```python
# Predictions loaded automatically from path_config
from lt_forecasting.forecast_models.SciRegressor import SciRegressor

path_config = {
    "path_to_lr_predictors": [
        "/data/predictions/model1/predictions.csv",
        "/data/predictions/model2/predictions.csv"
    ]
}

model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config  # Predictions loaded internally
)
# DeprecationWarning: Loading predictions from path_config is deprecated
```

**Problems with old pattern:**
- No flexibility in data sources (filesystem only)
- No control over loading process
- No ability to cache/reuse predictions
- Tight coupling between model and data loading
- No database or API support

#### New Pattern (Recommended)
```python
# Predictions loaded explicitly before model creation
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    apply_area_conversion
)
from lt_forecasting.forecast_models.SciRegressor import SciRegressor

path_config = {
    "path_to_lr_predictors": [
        "/data/predictions/model1/predictions.csv",
        "/data/predictions/model2/predictions.csv"
    ]
}

# Step 1: Load predictions explicitly
base_preds, model_names = load_predictions_from_filesystem(
    path_config["path_to_lr_predictors"],
    join_type="inner"
)

# Step 2: Apply necessary conversions
base_preds = apply_area_conversion(
    predictions=base_preds,
    static_data=static_data,
    pred_cols=model_names
)

# Step 3: Pass predictions to model
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_preds,      # NEW: External predictions
    base_model_names=model_names      # NEW: Column names
)
```

**Benefits of new pattern:**
- Flexibility: Load from filesystem, database, API, or any pandas DataFrame
- Control: Inspect and transform predictions before passing to model
- Reusability: Load once, use with multiple models
- Testability: Easy to mock prediction sources for testing
- Separation of concerns: Data loading separated from model logic

#### Alternative Sources (New Pattern Only)

**From Database:**
```python
import sqlalchemy as sa
from lt_forecasting.scr.prediction_loader import load_predictions_from_dataframe

engine = sa.create_engine("postgresql://user:pass@localhost/db")
df = pd.read_sql("SELECT date, code, Q_model1, Q_model2 FROM predictions", engine)

base_preds, model_names = load_predictions_from_dataframe(
    df, model_names=['model1', 'model2']
)
base_preds = apply_area_conversion(base_preds, static_data, model_names)
```

**From API:**
```python
import requests
from lt_forecasting.scr.prediction_loader import load_predictions_from_dataframe

response = requests.get("https://api.example.com/predictions")
df = pd.DataFrame(response.json()['data'])

base_preds, model_names = load_predictions_from_dataframe(
    df, model_names=['model1', 'model2']
)
base_preds = apply_area_conversion(base_preds, static_data, model_names)
```

**From In-Memory Processing:**
```python
from lt_forecasting.scr.prediction_loader import load_predictions_from_dataframe

# Generate predictions programmatically
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=100),
    'code': [1] * 100,
    'Q_model1': np.random.randn(100) * 50 + 100,
    'Q_model2': np.random.randn(100) * 45 + 95
})

base_preds, model_names = load_predictions_from_dataframe(
    df, model_names=['model1', 'model2']
)
base_preds = apply_area_conversion(base_preds, static_data, model_names)
```

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
