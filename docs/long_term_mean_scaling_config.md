# Long-Term Mean Scaling Configuration Guide

This guide explains how to configure the enhanced long-term mean scaling functionality introduced in Issue #22.

## Overview

The enhanced long-term mean scaling provides:
- Period-based (day-of-year) grouping instead of monthly grouping
- Selective feature scaling based on variable patterns
- Optional relative scaling for target variables

## Configuration Parameters

### In general_config.json

Add the following parameters to your `general_config.json` file:

```json
{
    "model_name": "YourModelName",
    "model_type": "sciregressor",
    // ... other existing parameters ...
    
    // New parameters for selective scaling:
    "relative_scaling_vars": ["SWE", "T", "discharge"],
    "use_relative_target": false,
    
    // Required for long-term mean scaling:
    "normalize": true,
    "normalization_type": "long_term_mean"
}
```

### Parameter Details

#### `relative_scaling_vars` (list, optional)
- **Purpose**: Specifies which variable patterns should use relative scaling (long-term mean)
- **Type**: List of strings
- **Default**: `null` (all features use relative scaling when using long_term_mean normalization)
- **Example**: `["SWE", "T", "discharge"]`
- **Behavior**: Features containing `"{var}_"` pattern will use relative scaling
  - Example: With `["T"]`, features like `T_mean`, `T_max`, `T_min` will use relative scaling
  - Other features will use per-basin scaling

#### `use_relative_target` (boolean, optional)
- **Purpose**: Whether to apply relative scaling to the target variable
- **Type**: Boolean
- **Default**: `false`
- **Example**: `true`
- **Benefit**: When `true`, predictions represent deviation from norm (e.g., 1.2 = 20% above normal)

## Period-Based Grouping

The system now automatically uses period-based grouping instead of monthly grouping:
- **Format**: `<month>-<day>` or `<month>-end`
- **Examples**: 
  - "3-15" for March 15th
  - "2-end" for last day of February
- **Benefit**: Provides more temporal granularity (up to 365 groups per basin vs 12)

## Example Configurations

### Full Relative Scaling (Default Behavior)
```json
{
    "normalize": true,
    "normalization_type": "long_term_mean"
    // No relative_scaling_vars specified - all features use relative scaling
}
```

### Selective Scaling
```json
{
    "normalize": true,
    "normalization_type": "long_term_mean",
    "relative_scaling_vars": ["P", "T", "SWE"],
    "use_relative_target": false
}
```

### Relative Target Scaling
```json
{
    "normalize": true,
    "normalization_type": "long_term_mean",
    "relative_scaling_vars": ["discharge", "P", "T"],
    "use_relative_target": true
}
```

## Integration with Existing Workflow

1. The configuration is automatically loaded when using:
   - `tune_hyperparams.py`
   - `calibrate_hindcast.py`
   - Direct model initialization

2. The scaling metadata is stored in `FeatureProcessingArtifacts` and persisted across:
   - Training/validation splits
   - Model saving/loading
   - Prediction post-processing

3. Inverse scaling is automatically applied during prediction post-processing

## Backward Compatibility

### For Existing Code
- The `apply_long_term_mean_scaling` function now returns a tuple `(DataFrame, metadata)` by default
- To maintain compatibility with existing code, use `return_metadata=False` to get only the DataFrame
- Example: `df_scaled = apply_long_term_mean_scaling(df, ltm, features, return_metadata=False)`

### For Existing Models
- Models trained with month-based grouping will continue to work
- The system automatically detects whether to use 'month' or 'period' columns
- Existing models without the new configuration parameters will use default values:
  - `relative_scaling_vars = None` (all features use relative scaling)
  - `use_relative_target = False`

### Period vs Month Grouping
- New models automatically use period-based grouping (day-of-year)
- Existing models using month-based grouping remain compatible
- The system detects the grouping type from the long_term_mean DataFrame structure