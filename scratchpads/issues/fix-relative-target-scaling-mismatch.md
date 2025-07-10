# Fix Relative Target Scaling Mismatch

## Issue Description
User reported getting good R² values (>0.9) on the model fit but poor values on the original scale after inverse transformation. This indicated that the forward and inverse transformations were not symmetric.

## Root Cause
The issue was a mismatch between how the target was scaled during training and how it was inverse-scaled during prediction:

1. When `use_relative_target` is True, the target is added to `relative_features` in the artifacts
2. BUT in `apply_long_term_mean_scaling`, the function uses pattern matching (`get_relative_scaling_features`) to determine which features to scale relatively
3. If the target column name (e.g., "target" or "discharge") doesn't match any patterns in `relative_scaling_vars` (e.g., ["SWE", "T"]), it won't actually be scaled relatively
4. During inverse transformation, the code sees `use_relative_target=True` and tries to inverse-scale relatively, but the target was never scaled that way!

This mismatch caused the predictions to be incorrectly inverse-transformed, leading to poor results on the original scale.

## Solution
Added an `explicit_relative_features` parameter to `apply_long_term_mean_scaling` that overrides the pattern-based feature selection. This ensures that when `use_relative_target` is True, the target is actually scaled with the relative method regardless of its name.

### Code Changes

1. **Modified `apply_long_term_mean_scaling` in data_utils.py**:
   - Added `explicit_relative_features` parameter
   - When provided, uses this list directly instead of pattern matching

2. **Updated `_normalization_training` in FeatureProcessingArtifacts.py**:
   - Passes `explicit_relative_features=relative_features` to ensure target is scaled correctly

3. **Updated `_apply_normalization` in FeatureProcessingArtifacts.py**:
   - Also passes explicit relative features for consistency during test data processing

4. **Improved `apply_inverse_long_term_mean_scaling` logic**:
   - Better determination of whether to use relative or per-basin inverse scaling
   - Checks if the variable was actually in relative_features list

## Test Cases
Created comprehensive tests to verify:
1. Target is scaled correctly even when its name doesn't match patterns
2. Mixed scaling (some features relative, others per-basin) works correctly
3. Forward and inverse transformations are symmetric

## Verification
The fix ensures that:
- When `use_relative_target=True`, the target is always scaled with the relative method
- The inverse transformation correctly identifies how the target was scaled
- Good R² values on scaled data translate to good results on original scale