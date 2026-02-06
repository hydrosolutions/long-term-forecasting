# Finalize Prediction Loading Refactoring - Issue #59

**Date**: 2026-01-23
**Branch**: `feature/issue-58-refactor-prediction-loading`

## Current Status Summary

### What's Been Completed
1. **prediction_loader module** (`lt_forecasting/scr/prediction_loader.py`) - 373 lines
   - `load_predictions_from_filesystem()` - Load from CSV files
   - `load_predictions_from_dataframe()` - Load from DataFrames (DB-ready)
   - `apply_area_conversion()` - Unit conversion (m³/s → mm/day)
   - `handle_duplicate_predictions()` - Deduplication by averaging
   - `standardize_prediction_columns()` - Column naming standardization

2. **SciRegressor updated** - Accepts `base_predictors` and `base_model_names`
3. **BaseMetaLearner updated** - Accepts `base_predictors` and `base_model_names`
4. **24 unit tests** for prediction_loader (all passing)
5. **Documentation** - `prediction_loader_docs.md` (527 lines)

### What Needs Work
1. **HistoricalMetaLearner** - Doesn't accept `base_predictors` parameter
2. **UncertaintyMixtureModel** - Has parameters but doesn't pass them to parent
3. **Scripts** - `calibrate_hindcast.py` and `tune_hyperparams.py` not updated
4. **Input structure documentation** - Clear specification needed
5. **Column naming verification** - Ensure backward compatibility

---

## Remaining Tasks

### Task 1: Fix Model Class Inheritance Issues

#### 1.1 Update HistoricalMetaLearner
**File**: `lt_forecasting/forecast_models/meta_learners/historical_meta_learner.py`

**Current** (line 44-69):
```python
class HistoricalMetaLearner(BaseMetaLearner):
    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
    ) -> None:
        super().__init__(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )
```

**Required** (add parameters and pass to parent):
```python
from typing import Optional, List

class HistoricalMetaLearner(BaseMetaLearner):
    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
        base_predictors: Optional[pd.DataFrame] = None,
        base_model_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=base_predictors,
            base_model_names=base_model_names,
        )
```

#### 1.2 Fix UncertaintyMixtureModel
**File**: `lt_forecasting/forecast_models/deep_models/uncertainty_mixture.py`

**Current Bug** (line 80-89): Has parameters but doesn't pass to parent
```python
super().__init__(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
)

self.base_predictors = base_predictors  # Stored separately (BUG)
self.base_model_names = base_model_names
```

**Fix**: Pass to parent class
```python
super().__init__(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config,
    base_predictors=base_predictors,
    base_model_names=base_model_names,
)
# Remove duplicate storage (handled by parent)
```

---

### Task 2: Define Clear Input Structure

#### 2.1 Expected Input Format for prediction_loader

**For `load_predictions_from_filesystem()`:**
```
Input: List of paths to directories containing predictions.csv
       OR direct paths to CSV files

Expected CSV structure:
| date       | code | Q_{model_name} |
|------------|------|----------------|
| 2024-01-01 | 1    | 100.5          |
| 2024-01-01 | 2    | 150.2          |
| 2024-02-01 | 1    | 95.3           |

Where:
- date: ISO format date string (YYYY-MM-DD)
- code: Integer basin code
- Q_{model_name}: Float prediction value (column name must match folder name)
```

**For `load_predictions_from_dataframe()`:**
```
Input DataFrame:
| date       | code | model1 | model2 | ... |
|------------|------|--------|--------|-----|
| 2024-01-01 | 1    | 100.5  | 95.3   | ... |

OR with Q_ prefix:
| date       | code | Q_model1 | Q_model2 | ... |
|------------|------|----------|----------|-----|

Where:
- date: datetime or string (will be converted)
- code: Integer basin code
- model columns: Float values (Q_ prefix optional)
```

#### 2.2 Column Naming Convention

| Context | Input Format | Output Format | Notes |
|---------|-------------|---------------|-------|
| File loading | `Q_{folder_name}` in CSV | `Q_{model_name}` | Folder name becomes model name |
| DataFrame loading | `model_name` or `Q_model_name` | `Q_model_name` | Prefix added if missing |
| SciRegressor internal | `Q_{model_name}` | `Q_{model_name}` | Keeps prefix |
| BaseMetaLearner internal | `Q_{model_name}` | `{model_name}` | **Strips prefix** |

#### 2.3 Unit Handling

| Model | Expected Input Unit | Processing | Output |
|-------|---------------------|------------|--------|
| SciRegressor | m³/s | Converts to mm/day via `* area / 86.4` | mm/day internally |
| BaseMetaLearner | Any consistent unit | No conversion | Same as input |
| HistoricalMetaLearner | Any consistent unit | No conversion | Same as input |

---

### Task 3: Update Scripts for Integration

#### 3.1 Create Helper Function for Script Integration
**Add to scripts or create utility module:**

```python
def load_base_predictions_for_model(
    model_type: str,
    path_config: Dict[str, Any],
    static_data: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Load base predictions based on model type.

    Args:
        model_type: Type of model (sciregressor, historical_meta_learner, etc.)
        path_config: Configuration with paths
        static_data: Static data for area conversion

    Returns:
        Tuple of (base_predictors DataFrame, model_names list)
        Returns (None, None) if no paths configured
    """
    from lt_forecasting.scr.prediction_loader import (
        load_predictions_from_filesystem,
        apply_area_conversion,
    )

    if model_type == "sciregressor":
        path_key = "path_to_lr_predictors"
        needs_conversion = True
    elif model_type in ["historical_meta_learner", "UncertaintyMixtureModel"]:
        path_key = "path_to_base_predictors"
        needs_conversion = False
    else:
        return None, None

    paths = path_config.get(path_key)
    if not paths:
        return None, None

    # Load predictions
    base_preds, model_names = load_predictions_from_filesystem(
        paths if isinstance(paths, list) else [paths]
    )

    # Apply conversion if needed
    if needs_conversion:
        base_preds = apply_area_conversion(base_preds, static_data, model_names)

    return base_preds, model_names
```

#### 3.2 Update `create_model()` in calibrate_hindcast.py

```python
def create_model(
    model_name: str,
    configs: Dict[str, Any],
    data: pd.DataFrame,
    static_data: pd.DataFrame,
):
    general_config = configs["general_config"]
    model_config = configs["model_config"]
    feature_config = configs["feature_config"]
    path_config = configs["path_config"]

    if "model_name" not in general_config:
        general_config["model_name"] = model_name

    model_type = general_config.get("model_type", "linear_regression")

    # Load base predictions externally
    base_preds, base_model_names = load_base_predictions_for_model(
        model_type, path_config, static_data
    )

    # Create model instance with external predictions
    if model_type == "linear_regression":
        model = LinearRegressionModel(...)  # No base predictions needed
    elif model_type == "sciregressor":
        model = SciRegressor(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=base_preds,
            base_model_names=base_model_names,
        )
    elif model_type == "historical_meta_learner":
        model = HistoricalMetaLearner(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=base_preds,
            base_model_names=base_model_names,
        )
    elif model_type == "UncertaintyMixtureModel":
        model = UncertaintyMixtureModel(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            base_predictors=base_preds,
            base_model_names=base_model_names,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model
```

#### 3.3 Apply Same Pattern to tune_hyperparams.py

Same updates as calibrate_hindcast.py

---

### Task 4: Backward Compatibility Verification

#### 4.1 Column Naming Tests to Add

```python
def test_column_naming_backward_compatibility():
    """Verify column names work with existing code patterns."""
    # Test 1: SciRegressor expects Q_prefix columns
    preds = pd.DataFrame({
        'date': ['2024-01-01'],
        'code': [1],
        'Q_model1': [100.0]
    })
    # Should work with SciRegressor

    # Test 2: BaseMetaLearner strips Q_prefix internally
    preds_with_prefix = pd.DataFrame({
        'date': ['2024-01-01'],
        'code': [1],
        'Q_model1': [100.0]
    })
    # After loading, internal columns should be 'model1' not 'Q_model1'

    # Test 3: DataFrame loading with/without prefix
    df_no_prefix = pd.DataFrame({
        'date': ['2024-01-01'],
        'code': [1],
        'model1': [100.0]
    })
    df_with_prefix = pd.DataFrame({
        'date': ['2024-01-01'],
        'code': [1],
        'Q_model1': [100.0]
    })
    # Both should produce same result
```

#### 4.2 Verify Deprecation Warnings

```python
def test_deprecation_warnings_shown():
    """Verify deprecation warnings when using old pattern."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Create model without external predictions
        model = SciRegressor(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
            # No base_predictors - should trigger warning
        )

        # Check deprecation warning was raised
        assert any(issubclass(warning.category, DeprecationWarning) for warning in w)
```

---

### Task 5: Documentation Updates

#### 5.1 Add Input Structure to prediction_loader_docs.md

Update documentation to include:
1. Clear CSV file format specification
2. Column naming requirements per model type
3. Unit expectations and conversion rules
4. Migration examples from old to new pattern

#### 5.2 Update Docstrings

Ensure all model `__init__` methods document:
- When to use `base_predictors` parameter
- Expected DataFrame format
- When unit conversion is applied

---

## Implementation Checklist

- [ ] **Task 1.1**: Update HistoricalMetaLearner to accept base_predictors
- [ ] **Task 1.2**: Fix UncertaintyMixtureModel to pass base_predictors to parent
- [ ] **Task 2**: Document clear input structure (in this scratchpad - DONE)
- [ ] **Task 3.1**: Create helper function for loading predictions
- [ ] **Task 3.2**: Update calibrate_hindcast.py create_model()
- [ ] **Task 3.3**: Update tune_hyperparams.py create_model()
- [ ] **Task 4.1**: Add column naming compatibility tests
- [ ] **Task 4.2**: Add deprecation warning tests
- [ ] **Task 5**: Update documentation
- [ ] Run full test suite: `uv run pytest -v`
- [ ] Run ruff format: `uv run ruff format`
- [ ] Verify no deprecation warnings in updated scripts

## Testing Strategy

After each task:
1. Run unit tests: `uv run pytest tests/unit/ -v`
2. Run full test suite before final commit
3. Manually test with a simple config to verify workflow

## Notes

- Keep processing (mm/day conversion) inside models - only loading is externalized
- BaseMetaLearner strips Q_ prefix internally (historical behavior to preserve)
- SciRegressor keeps Q_ prefix throughout
- All existing tests (191) should continue passing
