# Long-Term Mean Scaling Configuration

## New Configuration Options

When using `normalization_type: "long_term_mean"`, the following new configuration options are available:

### relative_scaling_vars (in experiment_config)
- **Type**: `list` of strings
- **Default**: `None` (all features use long-term mean scaling)
- **Description**: List of variable patterns for selective feature scaling. Features containing `"{var}_"` pattern will use relative scaling (long-term mean), while other features will use per-basin scaling.
- **Example**: 
  ```json
  {
    "normalization_type": "long_term_mean",
    "relative_scaling_vars": ["SWE", "T", "discharge"]
  }
  ```
  This will apply relative scaling to features like "SWE_mean", "T_max", "discharge_lag1", etc.

### use_relative_target (in experiment_config)
- **Type**: `boolean`
- **Default**: `False`
- **Description**: When `True`, applies relative scaling (long-term mean) to the target variable. When `False`, the target uses per-basin scaling.
- **Example**:
  ```json
  {
    "normalization_type": "long_term_mean",
    "use_relative_target": true
  }
  ```

## Complete Example Configuration

```json
{
  "normalization_type": "long_term_mean",
  "handle_na": "long_term_mean",
  "relative_scaling_vars": ["SWE", "T", "P"],
  "use_relative_target": false
}
```

In this example:
- Features containing "SWE_", "T_", or "P_" will use relative scaling (day-of-year long-term mean)
- Other features will use per-basin scaling
- The target variable will use per-basin scaling

## Technical Details

### Day-of-Year Grouping
The long-term mean is now calculated per basin and day-of-year (365/366 groups) instead of per month (12 groups), providing more accurate temporal representation.

### Feature Processing Artifacts
The FeatureProcessingArtifacts class now stores:
- `relative_features`: List of features using relative scaling
- `per_basin_features`: List of features using per-basin scaling
- `relative_scaling_vars`: Configuration patterns used
- `use_relative_target`: Whether target uses relative scaling
- `per_basin_scaler`: Scaler for per-basin features

### Backward Compatibility
If `relative_scaling_vars` is not specified, all features will use long-term mean scaling as before, ensuring backward compatibility with existing configurations.