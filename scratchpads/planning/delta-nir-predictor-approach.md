# Delta NIR Predictor Approach - Implementation Plan

## Objective
Evaluate whether using residual NIR (delta NIR) as a predictor improves monthly discharge forecasting compared to using raw NIR values.

## Context
The idea is to:
1. Model the "expected" NIR based on easily available base features (Q, P, T)
2. Calculate delta NIR = observed NIR - predicted NIR
3. Use delta NIR as a predictor for discharge target
4. Hypothesis: delta NIR captures glacier-specific information not explained by basic meteorological/hydrological variables

## Does This Make Sense?

### ✅ Potential Benefits

1. **Removes Redundant Information**:
   - Base features (Q, P, T) already correlate with NIR
   - Delta NIR isolates the "unique" signal from NIR that's NOT explained by Q, P, T
   - Reduces multicollinearity in the final model

2. **Focuses on Anomalies**:
   - Delta NIR represents departures from expected glacier behavior
   - Could capture glacier-specific processes (e.g., unusual melt, debris cover effects)
   - More interpretable: "NIR is higher/lower than expected given current conditions"

3. **Physically Meaningful**:
   - Expected NIR from Q, P, T represents normal conditions
   - Residuals might indicate:
     - Unexpected glacier melt
     - Snow cover anomalies on glaciers
     - Debris exposure changes
     - Climate-driven deviations

### ⚠️ Potential Issues

1. **Information Loss**:
   - If the Q→P→T→NIR relationship is weak (low R²), delta NIR ≈ raw NIR
   - If the relationship is strong (high R²), you might remove useful predictive signal
   - Need to check R² from your predictability analysis

2. **Timing/Causality Concerns**:
   - Q, P, T at time t used to predict NIR at time t
   - But Q at time t already contains information about discharge!
   - This could create circular logic if not careful with temporal structure

3. **Two-Stage Modeling Errors**:
   - Errors in Stage 1 (predicting NIR) propagate to Stage 2
   - Prediction intervals don't account for Stage 1 uncertainty
   - More complex to tune/optimize

4. **Overfitting Risk**:
   - Training both models on the same data could lead to overfitting
   - Need proper cross-validation strategy

## Alternative Interpretation: Regularization Approach

This is conceptually similar to:
- **Orthogonalization**: Removing shared variance between predictors
- **Partial regression**: Using residuals as features
- **Feature engineering**: Creating a "de-trended" NIR feature

## Implementation Plan

### Phase 1: Exploratory Analysis (EDA - Already Mostly Done)

**Tasks:**
- [x] Calculate correlation between NIR and base features (Q, P, T)
- [x] Calculate partial correlation (NIR vs target, controlling for Q, P, T)
- [x] Calculate VIF to assess multicollinearity
- [x] Predict NIR from Q, P, T and examine R² distribution
- [ ] **NEW**: Analyze when R² is high vs low (by month, glacier fraction)

**Key Questions:**
- Is NIR highly predictable from Q, P, T in all basins/months?
- Does the predictability vary by glacier fraction?
- What does high/low R² tell us about the potential value of delta NIR?

### Phase 2: Feature Engineering Implementation

**Task 2.1: Create Delta NIR Features**

```python
def create_delta_nir_features(
    data: pd.DataFrame,
    nir_col: str,
    base_predictor_cols: list[str],
    *,
    cv_strategy: str = "basin-wise",  # or "time-series"
) -> pd.DataFrame:
    """
    Create delta NIR features using cross-validated predictions.

    Args:
        data: DataFrame with all features
        nir_col: Name of NIR column (e.g., 'NIR_roll_last_value_30')
        base_predictor_cols: Base features to predict NIR from
        cv_strategy: How to split data for predictions

    Returns:
        DataFrame with added columns: 'NIR_predicted', 'NIR_delta'
    """
    # Use cross-validation to avoid overfitting
    # For each fold, train on other basins/periods, predict on held-out
    # This ensures delta NIR is not contaminated by overfitting
    pass
```

**Implementation Details:**
- Use GroupKFold (by basin code) or TimeSeriesSplit for CV
- Fit LinearRegression(Q, P, T) → NIR for each fold
- Predict NIR for held-out data
- Calculate delta = observed - predicted
- Store both predicted and delta for analysis

**Task 2.2: Temporal Alignment Check**

```python
# Check temporal structure to avoid leakage
# For discharge forecasting at t+30:
# - Use Q, P, T available at forecast time t
# - Use NIR available at forecast time t (10-day lag data)
# - Ensure no future information leaks into base model
```

### Phase 3: Comparative Model Evaluation

**Task 3.1: Train Baseline Models**

Train models on the same train/test splits:

1. **Baseline 1**: Q, P, T only (no NIR)
2. **Baseline 2**: Q, P, T + raw NIR
3. **Proposed**: Q, P, T + delta NIR
4. **Extended**: Q, P, T + raw NIR + delta NIR (to see if both add value)

**Task 3.2: Evaluation Strategy**

```python
# Use existing evaluation framework
# Key metrics:
# - NSE (Nash-Sutcliffe Efficiency)
# - R²
# - RMSE
# - Performance by glacier fraction quartiles
# - Performance by month
# - Performance by basin
```

**Task 3.3: Statistical Comparison**

```python
# Compare models using:
# - Paired t-tests on basin-level performance
# - Wilcoxon signed-rank test (non-parametric)
# - Confidence intervals for metric differences
# - Stratified analysis by glacier fraction
```

### Phase 4: Interpretation & Analysis

**Task 4.1: Feature Importance Analysis**

```python
# For models that include delta NIR:
# - Examine coefficients (if linear)
# - Feature importance (if ensemble)
# - Partial dependence plots
# - SHAP values for complex models
```

**Task 4.2: When Does Delta NIR Help?**

Analyze:
- Performance gain by month (seasonal patterns?)
- Performance gain by glacier fraction (glacier-dominated basins?)
- Performance gain by elevation (altitude effects?)
- Correlation between Stage 1 R² and Stage 2 improvement

**Task 4.3: Physical Interpretation**

```python
# Examine cases where delta NIR is large:
# - What physical conditions lead to large residuals?
# - Do large positive/negative deltas correspond to
#   unusual hydrological events?
# - Can we interpret the delta in terms of glacier processes?
```

### Phase 5: Production Implementation (If Promising)

**Task 5.1: Pipeline Integration**

```python
class DeltaNIRPredictor:
    """
    Two-stage predictor that creates delta NIR features.
    """
    def __init__(self, base_model, final_model):
        self.stage1_model = base_model  # Q,P,T → NIR
        self.stage2_model = final_model  # Q,P,T,delta_NIR → target

    def fit(self, X, y):
        # Fit stage 1: predict NIR from base features
        # Calculate residuals
        # Fit stage 2: predict target with delta NIR
        pass

    def predict(self, X):
        # Stage 1: predict NIR from base features
        # Calculate delta NIR
        # Stage 2: predict target using all features
        pass
```

**Task 5.2: Uncertainty Quantification**

```python
# Account for two-stage prediction uncertainty:
# - Bootstrap both stages
# - Propagate Stage 1 uncertainty to Stage 2
# - Generate prediction intervals
```

## Alternative Approaches to Consider

### A1: Regularization Instead of Delta
- Use Lasso/Ridge regression with all features
- Let the model determine optimal weighting
- Simpler and statistically cleaner

### A2: Interaction Terms
- Use NIR × glacier_fraction interaction
- Captures that NIR matters more in glaciated basins
- Easier to interpret than residuals

### A3: Multi-Task Learning
- Jointly predict NIR and discharge target
- Share representations between tasks
- More sophisticated but potentially more powerful

### A4: Principal Components
- Apply PCA to [Q, P, T, NIR]
- Use orthogonal components as predictors
- Removes multicollinearity systematically

## Recommendation

### Next Steps (In Order):

1. **Complete Phase 1**:
   - Analyze when NIR is predictable from Q, P, T (by month/glacier fraction)
   - If R² is consistently very high (>0.8) or very low (<0.2) everywhere,
     delta NIR approach may not be valuable

2. **Quick Prototype (Phase 2-3)**:
   - Implement delta NIR feature engineering with CV
   - Train Baseline 1, 2, and Proposed models
   - Quick comparison on validation set
   - If no improvement or worse: STOP and use raw NIR
   - If marginal improvement: proceed to full evaluation

3. **Full Evaluation (Phase 3-4)** - Only if prototype shows promise:
   - Comprehensive model comparison
   - Statistical testing
   - Interpretation analysis

4. **Production (Phase 5)** - Only if clear benefit:
   - Integrate into forecasting pipeline
   - Add uncertainty quantification
   - Document approach for operational use

## Expected Outcomes

### If R² (Q,P,T→NIR) is LOW (< 0.3):
- Delta NIR ≈ raw NIR
- Little benefit from delta approach
- **Recommendation**: Use raw NIR directly

### If R² (Q,P,T→NIR) is MODERATE (0.3-0.7):
- Delta NIR captures "orthogonal" information
- Likely sweet spot for this approach
- **Recommendation**: Compare both approaches

### If R² (Q,P,T→NIR) is HIGH (> 0.7):
- NIR mostly redundant with Q, P, T
- Delta NIR might remove useful signal
- **Recommendation**: Consider if NIR adds value at all

## Timeline Estimate

- Phase 1 (Complete EDA): 2-3 hours
- Phase 2 (Implement Delta NIR): 3-4 hours
- Phase 3 (Model Comparison): 4-6 hours
- Phase 4 (Interpretation): 3-4 hours
- Phase 5 (Production): 6-8 hours (if needed)

**Total**: 18-25 hours (assuming promising results justify continuation)

## Success Criteria

The delta NIR approach is worth pursuing if:
1. ✅ Improves NSE by at least 0.02 on validation set
2. ✅ Shows consistent improvement across majority of basins
3. ✅ Provides interpretable physical insights
4. ✅ Doesn't significantly increase computational cost
5. ✅ Robust to different train/test splits

If ≥ 3 criteria met: proceed to production
If < 3 criteria met: abandon and use simpler approach
