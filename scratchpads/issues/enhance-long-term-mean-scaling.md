# Enhancement: Improve Long-Term Mean Scaling with Day-of-Year Granularity and Selective Feature Scaling

## Issue Title
Enhance long-term mean scaling: day-of-year granularity, selective feature scaling, and relative target support

## Summary
Currently, the long-term mean scaling functionality groups data by basin and month. This issue proposes enhancing the scaling system to:
1. Use day-of-year instead of month for more accurate temporal representation
2. Apply selective scaling based on variable name patterns
3. Support relative scaling for the target variable
4. Update inverse transformations and artifact handling

## Background
The current implementation in `data_utils.py` and `FeatureProcessingArtifacts.py` calculates long-term means per basin and month. This approach:
- Loses temporal granularity by only considering 12 time points per year
- Applies the same scaling approach to all features regardless of their nature
- Doesn't support relative scaling for target variables

## Detailed Requirements

### 1. Day-of-Year Averaging
- **Current**: Groups by `["code", "month"]`
- **Required**: Group by `["code", "day_of_year"]` for 365/366 data points per year
- **Benefit**: More accurate representation of seasonal variations

### 2. Selective Feature Scaling
- **New Config Parameter**: `relative_scaling_vars` (list) in general_config
  - Example: `["SWE", "T", "discharge"]`
- **Behavior**:
  - Features containing `"{var}_"` pattern use relative scaling (long-term mean)
  - Other features use per-basin scaling
  - Static features maintain current scaling approach

### 3. Relative Target Scaling
- **New Config Parameter**: `use_relative_target` (boolean) in experiment_config
- **Behavior**: When True, apply relative scaling to target variable using day-of-year means

### 4. Enhanced Artifact Management
- Track which features used relative scaling vs per-basin scaling
- Store this information in FeatureProcessingArtifacts for proper inverse transformation

## Implementation Plan

### Step 1: Update Long-Term Mean Calculation
**File**: `scr/data_utils.py`
- [ ] Modify `get_long_term_mean_per_basin()`:
  - Add `day_of_year` column: `df["day_of_year"] = df["date"].dt.dayofyear`
  - Change groupby to `["code", "day_of_year"]`
  - Update docstring

### Step 2: Implement Selective Feature Scaling
**File**: `scr/data_utils.py`
- [ ] Create `get_relative_scaling_features()` function:
  ```python
  def get_relative_scaling_features(features, relative_scaling_vars):
      """Identify features that should use relative scaling based on patterns."""
      # Return list of features matching any pattern in relative_scaling_vars
  ```
- [ ] Modify `apply_long_term_mean_scaling()`:
  - Add `relative_scaling_vars` parameter
  - Split features into relative vs per-basin groups
  - Apply appropriate scaling to each group
  - Merge on `["code", "day_of_year"]` instead of `["code", "month"]`

### Step 3: Update Inverse Scaling Functions
**File**: `scr/data_utils.py`
- [ ] Update `apply_inverse_long_term_mean_scaling()`:
  - Add `day_of_year` column for merging
  - Handle mixed scaling types (some features relative, others per-basin)
  - Update merge keys to `["code", "day_of_year"]`

### Step 4: Enhance FeatureProcessingArtifacts
**File**: `scr/FeatureProcessingArtifacts.py`
- [ ] Add new attributes:
  - `self.relative_features: List[str]` - features using relative scaling
  - `self.relative_scaling_vars: List[str]` - config patterns
  - `self.use_relative_target: bool` - target scaling flag
- [ ] Update `_normalization_training()`:
  - Implement feature separation logic
  - Calculate appropriate scalers for each feature group
  - Store relative features list
- [ ] Update `post_process_predictions()`:
  - Check if feature was relatively scaled before inverse transform
  - Apply correct inverse transformation based on feature type
- [ ] Update save/load methods:
  - Include new attributes in all formats (joblib, pickle, hybrid)
  - Ensure backward compatibility

### Step 5: Configuration Support
**File**: `configs/general_config.py` (or equivalent)
- [ ] Add `relative_scaling_vars` parameter with default value
- [ ] Add documentation for new parameters

**File**: Experiment configs
- [ ] Add `use_relative_target` boolean parameter

### Step 6: Write Comprehensive Tests
**File**: `tests/test_data_utils.py`
- [ ] Test day-of-year grouping:
  - Verify 365/366 groups per basin
  - Check leap year handling
- [ ] Test selective feature scaling:
  - Features matching patterns use relative scaling
  - Non-matching features use per-basin scaling
  - Static features unchanged
- [ ] Test relative target scaling:
  - Target scaled when flag is True
  - Target not scaled when flag is False
- [ ] Test inverse transformations:
  - Correct inverse for relatively scaled features
  - Correct inverse for per-basin scaled features

**File**: `tests/test_feature_processing.py`
- [ ] Test artifact save/load with new attributes
- [ ] Test post_process_predictions with mixed scaling
- [ ] Test backward compatibility
- [ ] Integration test with full pipeline

### Step 7: Update Documentation
- [ ] Update docstrings for all modified functions
- [ ] Add examples showing new functionality
- [ ] Document configuration parameters

## Acceptance Criteria
- [ ] Day-of-year grouping produces 365/366 groups per basin
- [ ] Features are correctly identified for relative vs per-basin scaling
- [ ] Target scaling respects the `use_relative_target` flag
- [ ] Inverse transformations correctly restore original scale
- [ ] All existing tests pass
- [ ] New tests provide >90% coverage of new functionality
- [ ] Artifacts correctly save/load new attributes
- [ ] Performance remains comparable to current implementation

## Technical Considerations
- Ensure backward compatibility for existing models
- Handle leap years correctly (day 366)
- Consider memory usage with 365 vs 12 time points
- Validate that day_of_year merging handles date boundaries correctly

## Dependencies
- No external dependencies required
- Uses existing pandas/numpy functionality

## Risk Assessment
- **Medium Risk**: Changing from month to day_of_year affects all downstream processing
- **Mitigation**: Comprehensive testing and option to fallback to month-based if needed

## Estimated Effort
- Implementation: 2-3 days
- Testing: 1-2 days
- Documentation: 0.5 days
- Total: ~5 days

## References
- Current implementation: `scr/data_utils.py` (lines 496-666)
- FeatureProcessingArtifacts: `scr/FeatureProcessingArtifacts.py`
- Planning document: `scratchpads/planning/long_term_mean_scaling_adaptation`