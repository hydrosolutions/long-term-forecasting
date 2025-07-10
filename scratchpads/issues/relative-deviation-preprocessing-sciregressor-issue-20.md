# Feature: Add Relative Deviation Preprocessing for SciRegressor - Issue #20

## Objective
Implement relative deviation preprocessing that calculates long-term means for each day of the year and basin, then creates relative features (variable_t_i / long_term_mean_t_i) to normalize seasonal patterns across different basins.

## Context
- Issue: https://github.com/hydrosolutions/monthly_forecasting/issues/20
- Building on existing long-term mean scaling functionality in data_utils.py
- Current system uses monthly aggregation, new system needs daily aggregation (day of year)
- This will help normalize seasonal patterns across different basins with different SWE-discharge relationships

## Prior Art Analysis
- Existing functions: `get_long_term_mean_per_basin()` and `apply_long_term_mean_scaling()` in data_utils.py
- Previous issue #8 fixed MultiIndex column handling in these functions
- Current implementation uses monthly aggregation, new system needs day-of-year aggregation
- FeatureProcessingArtifacts.py already integrates long-term mean scaling in training pipeline

## Plan

### Phase 1: Core Preprocessing Functions (data_utils.py)
- [ ] Implement `calculate_long_term_means()` for day-of-year aggregation
  - Calculate means for each day of the year (1-366) and basin
  - Handle leap years appropriately
  - Return DataFrame with columns: [basin_code, day_of_year, variable_name, long_term_mean]
- [ ] Implement `apply_relative_scaling()` for relative feature creation
  - Transform variables to relative scale: var_t_i / long_term_mean_t_i
  - Handle division by zero (replace 0 with 1)
  - Create new columns with suffix "_rel_norm"
- [ ] Implement `inverse_relative_scaling()` for predictions
  - Transform back to original scale: var_rel_norm * long_term_mean_t_i
  - Handle missing normalization data gracefully

### Phase 2: Configuration & Model Integration
- [ ] Add `relative_scaling_vars: list[str]` parameter to general_config
  - Support pattern matching for variable names
  - Enable relative scaling for both input features and target variables
- [ ] Integrate norm_df saving in SciRegressor.save_model()
  - Save normalization DataFrame with model artifacts
  - Ensure proper serialization/deserialization
- [ ] Integrate norm_df loading in SciRegressor.load_model()
  - Load normalization DataFrame with model artifacts
  - Handle missing norm_df gracefully for backward compatibility

### Phase 3: Training Pipeline Integration
- [ ] Apply preprocessing in training pipeline before feature engineering
- [ ] Handle target variable specially:
  - Apply same normalization to target variable
  - Preserve original scale data for validation
  - Create target_rel_norm variable while keeping original target
- [ ] Update FeatureProcessingArtifacts to support relative scaling
  - Add norm_df to artifacts
  - Apply relative scaling in process_training_data()
  - Apply inverse scaling in post_process_predictions()

### Phase 4: Testing & Validation
- [ ] Unit tests for all new utility functions
  - Test calculate_long_term_means() with various date ranges
  - Test apply_relative_scaling() with edge cases
  - Test inverse_relative_scaling() for round-trip accuracy
- [ ] Integration tests with SciRegressor pipeline
  - Test saving/loading model with norm_df
  - Test end-to-end training with relative scaling
  - Test predictions with inverse transformation
- [ ] Edge case testing
  - New basins without historical data
  - Missing data handling
  - Leap year handling
  - Division by zero scenarios

### Phase 5: Documentation & Cleanup
- [ ] Update configuration documentation
- [ ] Add usage examples
- [ ] Ensure backward compatibility
- [ ] Code review and optimization

## Implementation Notes

### Data Structure for norm_df
```python
# DataFrame with columns:
# - basin_code: str
# - day_of_year: int (1-366)
# - variable_name: str
# - long_term_mean: float
```

### Configuration Example
```python
general_config = {
    "relative_scaling_vars": ["SWE", "T", "discharge"],
    # ... other config
}
```

### Key Design Decisions
1. **Day-of-Year vs Monthly**: Using day-of-year (1-366) for more granular seasonal patterns
2. **Leap Year Handling**: Day 366 will be used for Feb 29, regular years map to day 365
3. **Variable Naming**: New variables get "_rel_norm" suffix
4. **Target Handling**: Target variable gets special treatment to preserve original scale
5. **Zero Division**: Replace 0 with 1 to avoid division by zero
6. **Backward Compatibility**: New features are optional and don't break existing functionality

### Integration Points
1. **SciRegressor.__init__()**: Add relative_scaling_vars parameter
2. **SciRegressor.save_model()**: Save norm_df with model artifacts
3. **SciRegressor.load_model()**: Load norm_df with model artifacts
4. **FeatureProcessingArtifacts**: Apply relative scaling before feature engineering
5. **Training Pipeline**: Apply relative scaling during data preprocessing

## Testing Strategy

### Unit Tests
- Test each new function in isolation
- Test edge cases (zero values, missing data, leap years)
- Test round-trip accuracy (apply -> inverse should return original)

### Integration Tests
- Test full training pipeline with relative scaling enabled
- Test model saving/loading with norm_df
- Test predictions with inverse transformation

### Performance Tests
- Ensure no significant performance regression
- Test with realistic dataset sizes

## Review Points
1. Correct handling of day-of-year calculations including leap years
2. Proper integration with existing long-term mean scaling system
3. Backward compatibility maintained
4. Target variable handling preserves original scale for validation
5. Error handling for edge cases (missing data, zero division)
6. Performance impact is minimal
7. Configuration system is intuitive and well-documented

## Files to Create/Modify
1. `scr/data_utils.py` - Add new relative scaling functions
2. `forecast_models/SciRegressor.py` - Add configuration and norm_df handling
3. `scr/FeatureProcessingArtifacts.py` - Integrate relative scaling in pipeline
4. `tests/test_data_utils.py` - Add comprehensive tests for new functions
5. `tests/test_sciregressor.py` - Add integration tests for SciRegressor

## Acceptance Criteria
- [ ] Configuration parameter `relative_scaling_vars` is functional
- [ ] Long-term means are calculated correctly for day-of-year and basin
- [ ] Relative scaling is applied during training and prediction
- [ ] Model artifacts include norm_df for persistence
- [ ] Inverse transformation works correctly for predictions
- [ ] Original data is preserved when transforming targets
- [ ] All existing functionality remains unchanged
- [ ] Comprehensive tests cover all new functionality
- [ ] Performance regression is minimal
- [ ] Documentation is complete and accurate