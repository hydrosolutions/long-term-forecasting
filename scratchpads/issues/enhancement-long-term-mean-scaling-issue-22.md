# Enhancement: Improve Long-Term Mean Scaling with Day-of-Year Granularity - Issue #22

## Objective
Enhance the long-term mean scaling functionality to use period-based (day-of-year) grouping instead of monthly grouping, implement selective feature scaling based on variable patterns, support relative scaling for target variables, and update artifact handling accordingly.

## Context
- GitHub Issue: [#22](https://github.com/hydrosolutions/monthly_forecasting/issues/22)
- Current implementation groups by basin and month (12 time points per year)
- Need to improve temporal granularity using period format (month-day or month-end)
- Period format reference: LINEAR_REGRESSION.py lines 83-103

## Plan
- [ ] Step 1: Create get_periods function in data_utils.py
- [ ] Step 2: Update get_long_term_mean_per_basin to use period instead of month
- [ ] Step 3: Implement get_relative_scaling_features function for selective scaling
- [ ] Step 4: Update apply_long_term_mean_scaling for selective feature scaling
- [ ] Step 5: Update apply_inverse_long_term_mean_scaling for period and selective scaling
- [ ] Step 6: Update FeatureProcessingArtifacts to track scaling metadata
- [ ] Step 7: Add configuration support for new parameters
- [ ] Step 8: Write comprehensive tests for all changes
- [ ] Step 9: Test integration with existing workflow and hyperparameter tuning
- [ ] Step 10: Update documentation

## Implementation Notes

### Period Format
- Format: `<month>-<day>` or `<month>-end`
- Example: "3-15" for March 15th, "2-end" for last day of February
- Implementation pattern from LINEAR_REGRESSION.py:
```python
period_suffix = np.where(
    data["date"].dt.day == data["date"].dt.days_in_month,
    "end",
    data["date"].dt.day.astype(str),
)
period = data["month"].astype(str) + "-" + period_suffix
```

### Selective Scaling Logic
- Config parameter: `relative_scaling_vars` (list) in general_config
- If feature name contains pattern `"{var}_"`, use relative scaling
- Otherwise, use per-basin scaling
- Static features maintain current approach

### Relative Target Scaling
- Config parameter: `use_relative_target` (boolean) in experiment_config
- When True, apply relative scaling to target variable
- Predictions represent deviation from norm (e.g., 1.2 = 20% above normal)

### Artifact Updates
- New attributes: `relative_features`, `relative_scaling_vars`, `use_relative_target`
- Track which features used which scaling method
- Ensure proper save/load functionality

## Testing Strategy
1. Unit tests for get_periods function
2. Test period-based grouping produces correct number of groups
3. Test selective feature identification logic
4. Test scaling and inverse scaling with mixed approaches
5. Test artifact serialization with new attributes
6. Integration test with full training pipeline
7. Test backward compatibility

## Review Points
- Period boundary handling (end-of-month logic)
- Proper handling of missing data in period grouping
- Correct inverse transformation for mixed scaling
- Configuration parameter validation
- Performance impact of increased temporal granularity