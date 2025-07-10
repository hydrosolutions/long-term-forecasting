# Enhancement: Improve Long-Term Mean Scaling with Day-of-Year Granularity - Issue #22

## Objective
Enhance the long-term mean scaling functionality to use day-of-year instead of month for more accurate temporal representation, implement selective feature scaling based on variable patterns, and support relative scaling for target variables.

## Context
- GitHub Issue: [#22](https://github.com/hydrosolutions/monthly_forecasting/issues/22)
- Related planning doc: `scratchpads/planning/long_term_mean_scaling_adaptation`
- Previous PR #9 fixed length mismatch in long-term mean scaling

## Current Implementation Analysis
The current implementation uses month-based grouping:
- `get_long_term_mean_per_basin()` groups by `["code", "month"]` (line 506)
- `apply_long_term_mean_scaling()` merges on `["code", "month"]` (line 584)
- `apply_inverse_long_term_mean_scaling()` merges on `["code", "month"]` (line 647)
- FeatureProcessingArtifacts stores long_term_means but doesn't track which features used relative scaling

## Plan

### Step 1: Update Long-Term Mean Calculation for Day-of-Year
- [ ] Modify `get_long_term_mean_per_basin()` in `scr/data_utils.py`
  - Change from `df["month"] = df["date"].dt.month` to `df["day_of_year"] = df["date"].dt.dayofyear`
  - Update groupby columns from `["code", "month"]` to `["code", "day_of_year"]`
  - Update docstring to reflect day-of-year grouping

### Step 2: Implement Selective Feature Scaling
- [ ] Create `get_relative_scaling_features()` function in `scr/data_utils.py`
  - Takes list of features and relative_scaling_vars as input
  - Returns features that match pattern "{var}_" for any var in relative_scaling_vars
- [ ] Modify `apply_long_term_mean_scaling()` to support selective scaling
  - Add `relative_scaling_vars` parameter
  - Split features into relative and per-basin groups
  - Apply long-term mean scaling only to relative features
  - Apply per-basin scaling to other features
  - Update merge key from `["code", "month"]` to `["code", "day_of_year"]`

### Step 3: Update Inverse Scaling Functions
- [ ] Update `apply_inverse_long_term_mean_scaling()` in `scr/data_utils.py`
  - Change from month to day_of_year for merging
  - Add logic to handle mixed scaling types
  - Ensure target gets rescaled with correct statistics if use_relative_target

### Step 4: Enhance FeatureProcessingArtifacts
- [ ] Add new attributes to `FeatureProcessingArtifacts` class:
  - `self.relative_features: List[str]` - features using relative scaling
  - `self.relative_scaling_vars: List[str]` - config patterns
  - `self.use_relative_target: bool` - target scaling flag
  - `self.per_basin_features: List[str]` - features using per-basin scaling
- [ ] Update `_normalization_training()` to implement feature separation
- [ ] Update `post_process_predictions()` for mixed scaling types
- [ ] Update save/load methods to persist new attributes

### Step 5: Configuration Support
- [ ] Add `relative_scaling_vars` to general_config (default: empty list)
- [ ] Add `use_relative_target` to experiment configs (default: False)
- [ ] Update config documentation

### Step 6: Write Comprehensive Tests
- [ ] Test day-of-year grouping (365/366 groups, leap years)
- [ ] Test selective feature scaling logic
- [ ] Test relative target scaling
- [ ] Test inverse transformations for mixed scaling
- [ ] Test artifact save/load with new attributes
- [ ] Test backward compatibility

### Step 7: Documentation
- [ ] Update all function docstrings
- [ ] Add usage examples in docstrings
- [ ] Document new configuration parameters

## Implementation Notes

### Day-of-Year Handling
```python
# Instead of:
df["month"] = df["date"].dt.month
groupby_cols = ["code", "month"]

# Use:
df["day_of_year"] = df["date"].dt.dayofyear
groupby_cols = ["code", "day_of_year"]
```

### Feature Selection Logic
```python
def get_relative_scaling_features(features, relative_scaling_vars):
    """Identify features that should use relative scaling based on patterns."""
    relative_features = []
    for var in relative_scaling_vars:
        pattern = f"{var}_"
        relative_features.extend([f for f in features if pattern in f])
    return list(set(relative_features))  # Remove duplicates
```

### Mixed Scaling Application
```python
# In apply_long_term_mean_scaling:
relative_features = get_relative_scaling_features(features, relative_scaling_vars)
per_basin_features = [f for f in features if f not in relative_features]

# Apply long-term mean scaling to relative features
if relative_features:
    df = apply_long_term_mean_scaling(df, long_term_mean, relative_features)

# Apply per-basin scaling to other features
if per_basin_features:
    df = apply_per_basin_scaling(df, scaler, per_basin_features)
```

## Testing Strategy
1. Unit tests for each new/modified function
2. Integration tests for full pipeline with mixed scaling
3. Regression tests to ensure existing functionality unaffected
4. Performance tests to compare with month-based approach

## Review Points
- Proper handling of leap years (day 366)
- Memory usage with 365 vs 12 time points
- Backward compatibility for existing models
- Clear separation of scaling logic for different feature types