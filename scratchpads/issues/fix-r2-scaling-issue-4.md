# Fix R2 Calculation Issues with Scaling - Issue #4

## Objective
Fix the performance degradation when scaling is enabled in SciRegressor. The issue is that R2 is calculated on scaled values during hyperparameter tuning, which can give misleading results when the scaling significantly changes the variance structure.

## Context
- Issue: https://github.com/hydrosolutions/monthly_forecasting/issues/4
- When scaling is enabled, both features and targets are scaled together (which is correct for discharge forecasting)
- The problem is that R2 is calculated on scaled values, not original values
- This leads to poor R2 scores during hyperparameter tuning, causing suboptimal model selection

## Root Cause Analysis
1. In `sci_utils.py`, the objective functions (e.g., `_objective_xgb`) calculate R2 on scaled y_val and predictions
2. The scaling happens in `process_training_data()` which correctly scales both features and target
3. There's already a `post_process_predictions()` function that can inverse transform predictions
4. However, this inverse transformation is not used during hyperparameter optimization

## Additional Issues Found
1. The `apply_inverse_normalization_per_basin` function in `data_utils.py` is incorrectly implemented - it applies normalization instead of inverse normalization (line 398)
2. There's no global version of `apply_inverse_normalization` function

## Plan

### Step 1: Fix inverse normalization functions in data_utils.py
- [ ] Fix `apply_inverse_normalization_per_basin` to correctly inverse transform (multiply by std and add mean)
- [ ] Add `apply_inverse_normalization` function for global normalization
- [ ] Ensure both functions handle the case where the variable to scale is different from the variable used for scaling

### Step 2: Modify objective functions to calculate R2 on original scale
- [ ] Update `_objective_xgb`, `_objective_lgbm`, `_objective_catboost`, and `_objective_mlp` functions
- [ ] Each function needs to:
  1. Get predictions in scaled space
  2. Create a temporary DataFrame with predictions and necessary columns (code for per-basin)
  3. Use `post_process_predictions` to inverse transform
  4. Calculate R2 on original scale

### Step 3: Pass necessary artifacts to objective functions
- [ ] Modify `optimize_hyperparams` to accept artifacts parameter
- [ ] Pass artifacts to the objective function via trial.set_user_attr or closure
- [ ] Update `tune_hyperparameters` in SciRegressor to pass artifacts

### Step 4: Handle edge cases
- [ ] Ensure the solution works for all normalization types: 'global', 'per_basin', 'long_term_mean'
- [ ] Handle the case when normalization is disabled
- [ ] Ensure backward compatibility

## Implementation Notes

### 1. Fix inverse normalization in data_utils.py
```python
def apply_inverse_normalization(df: pd.DataFrame, 
                              scaler: dict, 
                              var_to_scale: str,
                              var_used_for_scaling: str) -> pd.DataFrame:
    """Apply inverse normalization (denormalization) for global scaling."""
    df = df.copy()
    if var_used_for_scaling in scaler:
        mean_val, std_val = scaler[var_used_for_scaling]
        df[var_to_scale] = df[var_to_scale] * std_val + mean_val
    return df

# Fix the per_basin version
def apply_inverse_normalization_per_basin(df: pd.DataFrame, 
                                        scaler: dict, 
                                        var_to_scale: str, 
                                        var_used_for_scaling: str):
    df = df.copy()
    for code in df.code.unique():
        if code in scaler and var_used_for_scaling in scaler[code]:
            basin_mask = df['code'] == code
            mean_val, std_val = scaler[code][var_used_for_scaling]
            # Fix: multiply by std and add mean (inverse of normalization)
            df.loc[basin_mask, var_to_scale] = (
                df.loc[basin_mask, var_to_scale] * std_val + mean_val
            )
    return df
```

### 2. Update objective functions pattern
```python
def _objective_xgb(trial, X_train, y_train, X_val, y_val, artifacts, experiment_config, target):
    # ... existing parameter setup ...
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Get predictions in scaled space
    y_pred_scaled = model.predict(X_val)
    
    # If normalization is enabled, inverse transform for R2 calculation
    if experiment_config.get('normalize', False) and artifacts is not None:
        # Create temporary DataFrame for post-processing
        df_temp = pd.DataFrame({
            'prediction': y_pred_scaled,
            target: y_val.values if hasattr(y_val, 'values') else y_val
        })
        
        # Add code column if needed for per-basin normalization
        if experiment_config.get('normalization_type') == 'per_basin':
            # Need to pass basin codes - this needs to be handled
            pass
        
        # Apply inverse transformation
        df_temp = post_process_predictions(
            df_predictions=df_temp,
            artifacts=artifacts,
            experiment_config=experiment_config,
            prediction_column='prediction',
            target=target
        )
        
        # Calculate R2 on original scale
        return r2_score(df_temp[target], df_temp['prediction'])
    else:
        # No normalization, calculate R2 directly
        return r2_score(y_val, y_pred_scaled)
```

## Testing Strategy

1. **Unit Tests for Inverse Normalization**
   - Test that inverse normalization correctly reverses normalization
   - Test all normalization types: global, per_basin, long_term_mean
   - Test edge cases (empty data, missing keys, etc.)

2. **Integration Tests for Hyperparameter Tuning**
   - Compare R2 scores with and without the fix
   - Ensure that model selection is consistent
   - Test that the fix doesn't break existing functionality

3. **Regression Tests**
   - Run existing tests to ensure no breaking changes
   - Verify that models can still be saved and loaded

## Review Points
1. The implementation correctly handles all normalization types
2. R2 calculation reflects true model performance
3. No performance regression in hyperparameter tuning
4. Backward compatibility is maintained
5. Tests adequately cover the changes

## Files to Modify
1. `scr/data_utils.py` - Fix and add inverse normalization functions
2. `scr/sci_utils.py` - Update objective functions to use inverse transformation
3. `forecast_models/SciRegressor.py` - Pass artifacts to optimize_hyperparams
4. `scr/FeatureProcessingArtifacts.py` - Update post_process_predictions if needed
5. `tests/test_data_utils.py` - Add tests for inverse normalization
6. `tests/test_sciregressor.py` - Add integration tests for the fix