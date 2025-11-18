# Length Mismatch Error in apply_long_term_mean_scaling Function - Issue #8

## Objective
Fix the length mismatch error in the `apply_long_term_mean_scaling` function that occurs when trying to assign column names to the long-term mean DataFrame.

## Context
- Issue: https://github.com/hydrosolutions/lt_forecasting/issues/8
- Error: `ValueError: Length mismatch: Expected axis has 42 elements, new values have 35 elements`
- Location: `scr/data_utils.py:529`
- This error blocks the entire hyperparameter tuning process

## Root Cause Analysis

The issue occurs in the `apply_long_term_mean_scaling` function when trying to flatten MultiIndex columns created by `get_long_term_mean_per_basin`.

### The Problem
1. `get_long_term_mean_per_basin` creates a DataFrame using `grouped[features].agg(['mean'])`
2. This creates a MultiIndex column structure where each feature becomes `(feature_name, 'mean')`
3. The function attempts to flatten this structure by assigning new column names:
   ```python
   ltm.columns = ['code', 'month'] + [f'{feat}_mean' for feat in features]
   ```
4. But this assumes the MultiIndex structure can be flattened with exactly 2 index columns + feature columns
5. The actual structure is more complex, leading to the mismatch

### Current Structure After agg(['mean'])
- Index columns: `code`, `month` (reset_index makes these regular columns)
- Feature columns: `(feature1, 'mean')`, `(feature2, 'mean')`, etc.

### Expected vs Actual Columns
- Expected: 35 columns (2 index + 33 features)
- Actual: 42 columns (suggesting MultiIndex structure is preserved)

## Plan

### Step 1: Fix the Column Flattening Logic
- [ ] Modify `apply_long_term_mean_scaling` to properly handle MultiIndex columns
- [ ] Use `ltm.columns.droplevel(1)` to remove the 'mean' level from MultiIndex
- [ ] Ensure the resulting columns match the expected format

### Step 2: Write Test to Reproduce the Issue
- [ ] Create a test that reproduces the length mismatch error
- [ ] Test with a realistic feature set (33 features)
- [ ] Ensure test fails with current code and passes with the fix

### Step 3: Implement the Fix
- [ ] Update the MultiIndex handling logic in `apply_long_term_mean_scaling`
- [ ] Ensure both MultiIndex and regular column cases are handled correctly
- [ ] Maintain backward compatibility

### Step 4: Verify the Fix
- [ ] Run the new test to ensure it passes
- [ ] Run the full test suite to ensure no regressions
- [ ] Test the hyperparameter tuning process manually if needed

## Implementation Notes

### Current Problematic Code
```python
if isinstance(ltm.columns, pd.MultiIndex):
    # after agg(['mean']), cols look like (feature, 'mean')
    ltm.columns = [
        'code', 'month'
    ] + [f'{feat}_mean' for feat in features]
```

### Proposed Fix
```python
if isinstance(ltm.columns, pd.MultiIndex):
    # after agg(['mean']), cols look like (feature, 'mean')
    # First, let's flatten the MultiIndex properly
    ltm.columns = ltm.columns.droplevel(1)  # Remove 'mean' level
    # Now rename the feature columns
    rename_map = {feat: f'{feat}_mean' for feat in features}
    ltm = ltm.rename(columns=rename_map)
```

### Alternative Approach
If the structure is more complex, we could also:
1. Inspect the actual column structure at runtime
2. Use `ltm.columns.get_level_values(0)` to get the feature names
3. Dynamically create the appropriate column names

## Testing Strategy

### Unit Test
Create a test that:
1. Creates a DataFrame with realistic feature count (33 features)
2. Calls `get_long_term_mean_per_basin` to create MultiIndex structure
3. Calls `apply_long_term_mean_scaling` and ensures it doesn't fail
4. Verifies the resulting column names are correct

### Integration Test
Run the hyperparameter tuning process to ensure it completes successfully.

## Review Points
1. The fix correctly handles the MultiIndex structure
2. Both MultiIndex and regular column cases work correctly
3. The resulting DataFrame has the expected column names
4. No regressions in existing functionality
5. The fix is robust and handles edge cases

## Files to Modify
1. `scr/data_utils.py` - Fix the `apply_long_term_mean_scaling` function
2. `tests/test_data_utils.py` - Add test for the fix