# Refactor: Externalize Prediction Loading Logic for Database Compatibility

**Issue**: [#58](https://github.com/hydrosolutions/monthly_forecasting/issues/58)
**Date**: 2026-01-22

## Objective

Refactor prediction loading logic out of model classes (SciRegressor, BaseMetaLearner, UncertaintyMixtureModel) into a centralized module to enable database compatibility and improve testability.

## Context

### Current Problems

1. **SciRegressor** (`__load_lr_predictors__()` - lines 308-368)
   - Loads from `path_config["path_to_lr_predictors"]`
   - Assumes folder structure for model naming
   - Performs area-based unit conversion (area_km2 / 86.4) internally
   - Returns DataFrame with predictions and column names

2. **BaseMetaLearner** (`__load_base_predictors__()` - lines 75-170)
   - Loads from `path_config["path_to_base_predictors"]`
   - Complex ensemble member naming logic
   - Handles deduplication of (date, code) pairs
   - Returns DataFrame with predictions and model names

### Key Differences Between Loaders

**SciRegressor Loader:**
- Expects column format: `Q_{model_name}`
- Simpler naming: uses folder name as model name
- Performs unit conversion: `prediction * area / 86.4`
- Merges on inner join
- Returns: (DataFrame, List[column_names])

**BaseMetaLearner Loader:**
- Expects columns: multiple `Q_*` columns (excluding `Q_obs`)
- Complex naming: handles ensemble members with `{model}_{sub_model}` format
- Handles duplicates by averaging
- Merges on left join
- Returns: (DataFrame, List[model_names])

## Implementation Plan

### Phase 1: Create Centralized Prediction Loader Module

**File**: `lt_forecasting/scr/prediction_loader.py`

**Functions to Implement:**

1. `load_predictions_from_filesystem(paths: List[str], join_type: str = "inner") -> Tuple[pd.DataFrame, List[str]]`
   - Load predictions from file paths
   - Extract model names from folder structure
   - Merge all predictions
   - Return unified DataFrame + model names

2. `load_predictions_from_dataframe(df: pd.DataFrame, model_names: List[str]) -> Tuple[pd.DataFrame, List[str]]`
   - Accept pre-loaded predictions (e.g., from database)
   - Validate structure
   - Return in standard format

3. `apply_area_conversion(predictions: pd.DataFrame, static_data: pd.DataFrame, pred_cols: List[str]) -> pd.DataFrame`
   - Extract unit conversion logic from SciRegressor
   - Apply: `prediction * area / 86.4`
   - Keep this reusable for both loaders

4. `handle_duplicate_predictions(df: pd.DataFrame) -> pd.DataFrame`
   - Extract deduplication logic from BaseMetaLearner
   - Average predictions for duplicate (date, code) pairs

5. `standardize_prediction_columns(df: pd.DataFrame) -> pd.DataFrame`
   - Ensure consistent column naming
   - Handle Q_ prefix conventions

### Phase 2: Update Model Classes

#### 2.1 Update SciRegressor

**Changes to `__init__`:**
```python
def __init__(
    self,
    data: pd.DataFrame,
    static_data: pd.DataFrame,
    general_config: dict,
    model_config: dict,
    feature_config: dict,
    path_config: dict,
    base_predictors: Optional[pd.DataFrame] = None,  # NEW
    base_model_names: Optional[List[str]] = None,    # NEW
):
```

**Logic:**
- If `base_predictors` and `base_model_names` provided: use them
- Else: fall back to `__load_lr_predictors__()` with deprecation warning
- Keep unit conversion logic in model for now (can extract later)

**Deprecation Strategy:**
- Add warning: "Loading predictions internally is deprecated. Pass base_predictors parameter."
- Keep old method working for backward compatibility

#### 2.2 Update BaseMetaLearner

**Changes to `__init__`:**
- Already has `base_predictors` and `base_model_names` parameters
- Update logic to prefer provided predictions
- Add deprecation warning to `__load_base_predictors__()`

**Key Change:**
```python
if self.base_predictors is not None and self.base_model_names is not None:
    # Use provided predictions
    pass
else:
    # Fall back to file loading with deprecation warning
    warnings.warn(
        "Loading predictions internally is deprecated. "
        "Use prediction_loader module and pass base_predictors parameter.",
        DeprecationWarning,
        stacklevel=2
    )
    self.base_predictors, self.base_model_names = self.__load_base_predictors__()
```

### Phase 3: Update Training Scripts

**Scripts to Update:**
1. `scripts/calibrate_hindcast.py`
2. `scripts/tune_hyperparams.py`
3. `scripts/run_operational_prediction.py`

**Pattern:**
```python
# Import prediction loader
from lt_forecasting.scr.prediction_loader import (
    load_predictions_from_filesystem,
    apply_area_conversion
)

# Load predictions before creating model
if "path_to_lr_predictors" in path_config:
    base_preds, model_names = load_predictions_from_filesystem(
        path_config["path_to_lr_predictors"],
        join_type="inner"
    )
    # Apply conversions if needed for SciRegressor
    base_preds = apply_area_conversion(base_preds, static_data, model_names)
else:
    base_preds, model_names = None, None

# Pass to model
model = SciRegressor(
    data, static_data, general_config, model_config,
    feature_config, path_config,
    base_predictors=base_preds,
    base_model_names=model_names
)
```

### Phase 4: Testing Strategy

#### 4.1 Unit Tests for Prediction Loader

**File**: `tests/unit/test_prediction_loader.py`

Tests:
- `test_load_predictions_from_filesystem_single_model`
- `test_load_predictions_from_filesystem_multiple_models`
- `test_load_predictions_from_dataframe`
- `test_apply_area_conversion`
- `test_handle_duplicate_predictions`
- `test_standardize_prediction_columns`
- `test_invalid_paths_raise_error`
- `test_missing_columns_handled_gracefully`

#### 4.2 Functionality Tests

**File**: `tests/functionality/test_model_prediction_loading.py`

Tests:
- `test_sciregressor_with_external_predictions`
- `test_sciregressor_backward_compatibility`
- `test_basemetalearner_with_external_predictions`
- `test_basemetalearner_backward_compatibility`
- `test_deprecation_warnings_raised`

#### 4.3 Integration Tests

**File**: `tests/integration/test_end_to_end_workflow.py`

Tests:
- `test_calibrate_hindcast_with_new_loader`
- `test_operational_prediction_with_new_loader`
- `test_results_match_old_implementation`

### Phase 5: Implementation Steps

1. ✅ Create scratchpad with plan
2. Create new branch: `feature/issue-58-refactor-prediction-loading`
3. Implement `prediction_loader.py` module
4. Write unit tests for prediction loader
5. Update SciRegressor class
6. Update BaseMetaLearner class
7. Update training scripts
8. Run full test suite
9. Fix any failing tests
10. Create PR

## Technical Details

### Unit Conversion Formula

From SciRegressor.__load_lr_predictors__():
```python
# Convert from m³/s to mm/month (or similar unit)
prediction_in_new_unit = prediction_m3s * area_km2 / 86.4
```

### Column Naming Conventions

**SciRegressor:**
- Input: `Q_{model_name}` in CSV
- Output: Same column names

**BaseMetaLearner:**
- Input: Multiple `Q_*` columns
- Output: Either `{model_name}` or `{model_name}_{sub_model}`

### Merge Strategies

**SciRegressor:** Uses `how="inner"` - only keeps common (date, code) pairs
**BaseMetaLearner:** Uses `how="left"` - keeps all dates from main data

## Backward Compatibility

### Guaranteed Compatible:
- Existing scripts using old pattern will continue working
- Deprecation warnings guide users to new pattern
- All existing tests should pass

### Migration Path:
1. Old way works with warning
2. New way is documented and tested
3. Future version can remove old methods

## Expected Benefits

1. **Database Ready**: Can load predictions from any source
2. **Testability**: Easy to mock predictions without files
3. **Maintainability**: Single source of truth for loading logic
4. **Performance**: Cache predictions across model runs
5. **Flexibility**: Support different storage backends

## Breaking Changes

None if users:
- Use model constructors normally
- Don't subclass and override loading methods
- Don't directly call `__load_lr_predictors__()` or `__load_base_predictors__()`

## Notes

- Keep unit conversion logic for now (can extract later if needed)
- Preserve exact numerical behavior for backward compatibility
- Add comprehensive logging for debugging
- Document new usage pattern in docstrings
