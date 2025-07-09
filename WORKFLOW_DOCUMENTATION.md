# Monthly Forecasting System - Overall Workflow Documentation

## System Overview
The monthly forecasting system is a modular machine learning framework for hydrological discharge prediction. It supports multiple model types, preprocessing methods, and evaluation strategies.

## End-to-End Workflow

### 1. Data Preparation Phase
**Location**: `scr/data_loading.py`, `scr/data_utils.py`

1. **Data Loading**
   - Load training and test datasets from CSV/Parquet files
   - Parse dates and handle time series indexing
   - Validate data integrity and completeness

2. **Initial Preprocessing**
   - Handle missing values through interpolation
   - Apply date/time feature extraction
   - Normalize basin identifiers

### 2. Feature Engineering Phase
**Location**: `scr/FeatureExtractor.py`, `scr/FeatureProcessingArtifacts.py`

1. **Temporal Features**
   - Extract month, season, year components
   - Create cyclical encodings for temporal patterns
   
2. **Lag Features**
   - Generate lagged values (1-12 months)
   - Create multi-step lag combinations
   
3. **Rolling Statistics**
   - Calculate rolling means (3, 6, 12 months)
   - Compute rolling std, min, max
   - Generate seasonal decomposition features

4. **Preprocessing Methods**
   - `none`: Raw data without scaling
   - `standardize`: Zero mean, unit variance
   - `min_max`: Scale to [0, 1] range
   - `monthly_bias`: Remove monthly systematic bias
   - `long_term_mean`: Scale by historical averages

### 3. Model Training Phase
**Location**: `forecast_models/`

#### Linear Regression Workflow
1. **Basin-Specific Training**
   - Train separate model per basin
   - Apply feature selection (optional)
   - Store coefficients and intercepts

2. **Model Persistence**
   - Save models as joblib files
   - Store preprocessing parameters
   - Track feature configurations

#### Tree-Based Model Workflow (SciRegressor)
1. **Global Model Training**
   - Train on all basins simultaneously
   - Apply selected preprocessing method
   - Use algorithm-specific hyperparameters

2. **Supported Algorithms**
   - XGBoost: Gradient boosting
   - Random Forest: Ensemble of decision trees
   - CatBoost: Categorical feature support

3. **Feature Importance**
   - Extract importance scores
   - Rank features by contribution
   - Save importance reports

### 4. Model Evaluation Phase
**Location**: `eval_scr/`

1. **Metric Calculation**
   - R-squared (R²): Variance explained
   - RMSE: Root mean square error
   - NSE: Nash-Sutcliffe efficiency
   - Bias: Systematic error

2. **Performance Visualization**
   - Time series plots
   - Scatter plots (predicted vs observed)
   - Residual analysis
   - Basin-specific performance

### 5. Model Deployment Phase

1. **Model Selection**
   - Compare performance metrics
   - Select best preprocessing method
   - Choose optimal hyperparameters

2. **Prediction Pipeline**
   - Load saved model and artifacts
   - Apply same preprocessing pipeline
   - Generate forecasts
   - Post-process predictions

## Execution Scripts

### tune_and_calibrate_script.sh
```bash
#!/bin/bash
# Main execution pipeline
1. Hyperparameter tuning (tune_hyperparams.py)
2. Model calibration (calibrate_hindcast.py)
3. Performance evaluation
4. Artifact generation
```

### calibration_script.sh
```bash
#!/bin/bash
# Focused calibration workflow
1. Load optimal hyperparameters
2. Train on full dataset
3. Generate hindcasts
4. Save production models
```

## Data Flow Diagram

```
Raw Data → Data Loading → Feature Engineering → Model Training → Evaluation
    ↓           ↓               ↓                    ↓              ↓
CSV/Parquet  Cleaned     Enhanced Features    Trained Models   Metrics
             Data        & Artifacts          & Predictions     & Plots
```

## Configuration Management

### Model Configuration
- Stored in JSON files
- Includes hyperparameters
- Preprocessing specifications
- Feature engineering settings

### Experiment Tracking
- Each run generates unique artifacts
- Timestamps for reproducibility
- Complete configuration logging
- Performance metric history

## Testing Workflow

1. **Unit Tests**
   - Test individual components
   - Validate data transformations
   - Check metric calculations

2. **Integration Tests**
   - End-to-end pipeline validation
   - Cross-component compatibility
   - Performance benchmarking

3. **Test Execution**
   ```bash
   python -m pytest -ra
   ```

## Best Practices

1. **Data Handling**
   - Always validate input data
   - Log preprocessing steps
   - Maintain data lineage

2. **Model Development**
   - Start with simple baselines
   - Incrementally add complexity
   - Document assumptions

3. **Performance Optimization**
   - Profile bottlenecks
   - Parallelize where possible
   - Cache intermediate results

4. **Reproducibility**
   - Set random seeds
   - Log all configurations
   - Version control artifacts

## Troubleshooting Guide

### Common Issues
1. **Memory Errors**
   - Reduce batch sizes
   - Use data generators
   - Optimize feature storage

2. **Convergence Problems**
   - Adjust learning rates
   - Increase iterations
   - Check data quality

3. **Poor Performance**
   - Review feature engineering
   - Try different preprocessing
   - Validate data splits

## Future Enhancements
- Deep learning models integration
- Real-time prediction API
- Automated model selection
- Uncertainty quantification
- Multi-step ahead forecasting