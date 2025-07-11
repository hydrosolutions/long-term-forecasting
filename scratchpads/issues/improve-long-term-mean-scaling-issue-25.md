# Improve Long-Term Mean Scaling with Period Granularity and Selective Feature Scaling

## Objective
Enhance the long-term mean scaling functionality to use period-based temporal grouping (36 groups instead of 12), implement standardization formula, add selective feature scaling, and fix inverse transformation bugs that cause R2 degradation.

## Context
- GitHub Issue: #25
- Previous attempts: PR #23 and #24 (both closed due to issues)
- Critical bug: R2=0.94 in transformed space but only 0.3 in original space
- Current implementation uses month-based grouping (12 groups per basin)
- The `get_periods()` function exists but isn't integrated

## Key Problems to Solve
1. **Temporal Granularity**: Need 36 period groups instead of 12 months
2. **Scaling Formula**: Change from `x/mean` to `(x-mean)/std`
3. **Selective Scaling**: Only scale features matching certain patterns
4. **Target Scaling**: Option to use relative scaling for target variable
5. **Inverse Transform Bug**: Fix the R2 degradation issue

## Plan

### Phase 1: Update Period-Based Statistics
- [ ] Examine current `get_long_term_mean_per_basin()` implementation
- [ ] Update to use periods instead of months
- [ ] Calculate both mean and std in aggregation
- [ ] Handle edge cases (zero std)
- [ ] Create tests for period-based grouping

### Phase 2: Implement Standardization Formula
- [ ] Update `apply_long_term_mean_scaling()` for new formula
- [ ] Ensure proper MultiIndex column handling (learning from issue #8)
- [ ] Test standardization properties (mean=0, std=1)

### Phase 3: Selective Feature Scaling
- [ ] Implement `get_relative_scaling_features()` for pattern matching
- [ ] Update scaling functions to accept feature list parameter
- [ ] Test pattern matching logic

### Phase 4: Fix Inverse Transformation
- [ ] Create dedicated `apply_inverse_long_term_mean_scaling_predictions()`
- [ ] Update existing inverse function for period support
- [ ] Ensure target statistics are used correctly
- [ ] Validate R2 consistency between spaces

### Phase 5: Update FeatureProcessingArtifacts
- [ ] Add new attributes for tracking scaling methods
- [ ] Update normalization methods for selective scaling
- [ ] Ensure backward compatibility
- [ ] Update save/load methods

### Phase 6: Configuration and Integration
- [ ] Add config parameters
- [ ] Update hyperparameter tuning compatibility
- [ ] Ensure date/code columns preserved throughout

### Phase 7: Testing and Validation
- [ ] Unit tests for each component
- [ ] Integration test for full pipeline
- [ ] Performance validation (R2 consistency)
- [ ] Edge case testing

## Implementation Notes

### Critical Areas to Watch
1. **MultiIndex Handling**: Previous issue #8 showed problems with column flattening
2. **Date/Code Preservation**: Essential for merging operations
3. **Inverse Transform**: Must use correct statistics for target variable
4. **Backward Compatibility**: Existing models should still work

### Key Functions to Modify
1. `get_long_term_mean_per_basin()` - Add period support
2. `apply_long_term_mean_scaling()` - New formula & selective scaling
3. `apply_inverse_long_term_mean_scaling()` - Period support
4. NEW: `apply_inverse_long_term_mean_scaling_predictions()` - Target-specific
5. NEW: `get_relative_scaling_features()` - Pattern matching

### Testing Strategy
1. Test period calculation (36 unique periods)
2. Test scaling formula (mean=0, std=1)
3. Test pattern matching
4. Test full transform → model → inverse pipeline
5. Validate R2 consistency (< 5% difference)

## Review Points
- Period merging logic correctness
- Inverse transformation using correct statistics
- Backward compatibility maintained
- Performance impact of 3x more groups
- Edge case handling (missing periods, zero std)

## Success Criteria
- [ ] 36 period groups per basin per year
- [ ] Standardization formula correctly applied
- [ ] Selective scaling based on patterns works
- [ ] R2 scores consistent between spaces (< 5% difference)
- [ ] All existing tests pass
- [ ] Backward compatible with existing models