# Monthly Discharge Forecasting Code Refactoring Prompt

## Context and Background

You are working on a modular monthly discharge forecasting system that uses different types of machine learning models (Linear Regression and Tree-based models like XGBoost, LightGBM, CatBoost). The goal is to create a clean, modular architecture where models are grouped by feature sets, enabling efficient ensemble predictions.

## Current Architecture Overview

The system follows a modular approach with:
- **Base Class**: `BaseForecastModel` (abstract class defining the interface)
- **Model Classes**: 
  - `LinearRegressionModel` (partially implemented)
  - `SciRegressor` (skeleton implementation for tree-based models)
- **Feature Processing**: `FeatureExtractor` for creating time series features
- **Evaluation**: Helper functions for metrics and validation
- **Configuration-driven**: JSON configs for features, models, and paths

## Key Design Principles

1. **Feature Set Grouping**: Models with the same feature set should be grouped together
2. **Ensemble Approach**: Multiple models (XGB, LGBM, CatBoost) should be fitted simultaneously for each feature set
3. **Leave-One-Out Cross-Validation**: Yearly LOOCV with last 3 years as final test set
4. **Modular Configuration**: JSON-based configuration for flexibility
5. **Consistent Interface**: All model classes implement the same abstract methods

## Task Distribution for Multiple Agents

### Agent 1: Linear Regression Implementation
**Objective**: Complete the `LinearRegressionModel` class implementation

**Specific Tasks**:
1. **Complete the `calibrate_model_and_hindcast()` method**:
   - Implement yearly leave-one-out cross-validation (LOOCV)
   - Reserve last n years as final test set (there is the configurable option to leave some years for a final evaluation (List (years (int))))
   - For each period (day_of_the_month - month), select top N features based on correlation
   - Return predictions DataFrame with columns: ['date', 'model', 'code', 'Q_pred']
   - Take an example of the predict_operational(), method and take a look at the old files (linear_regression.py) - the days of the month where a forecast is produced should be also configurable in a config file


**Reference Files**:
- `old_files/linear_regression.py` (contains the original implementation logic)
- `old_files/linreg_all_basins.py` (shows feature configuration patterns)
- Current `forecast_models/LINEAR_REGRESSION.py` (partial implementation)

### Agent 2: SciRegressor Implementation  
**Objective**: Complete the `SciRegressor` class for tree-based models (XGB, LGBM, CatBoost)

**Specific Tasks**:
1. **Implement `calibrate_model_and_hindcast()` method**:
   - Support multiple models specified in `self.models` list
   - Implement yearly LOOCV for each model simultaneously
   - Use consistent feature engineering across all models
   - Return ensemble predictions with individual model predictions and mean

2. **Complete `predict_operational()` method**:
   - Load all trained models for the feature set
   - Generate predictions from all models
   - Return ensemble results with uncertainty metrics

3. **Add feature selection and preprocessing**:
   - Implement feature selection suitable for tree-based models
   - Handle scaling and preprocessing consistently
   - Support various feature types (discharge, precipitation, temperature, snow data)

4. **Model persistence**:
   - Save/load multiple models simultaneously
   - Store preprocessing pipelines and feature selections


**Reference Files**:
- `scr/tree_utils.py` (contains tree-based model utilities)
- 'old_files/ML_MONTHLY_FORECAST.py' was the old wrapper for those models - the same functionallites should now just be wrapped in a class 
  - should also have the option to add linear regressions as an input
  - the same preprocessing steps should be applied (tree_utils.py file)
- Current `forecast_models/SciRegressor.py` (skeleton implementation)


### Agent 3: Hyperparameter Tuning Implementation
**Objective**: Create a comprehensive hyperparameter tuning system

**Specific Tasks**:
1. **Implement `tune_hyperparameters()` method for both model classes**:
   - Use Optuna for optimization (as referenced in tree_utils.py)
   - Use the same parameter space as already implemented
   - Use time series cross-validation for hyperparameter selection (use the last n=3 (configurable) years to validate)
   - Save optimal parameters to model configuration files

2. **Create model-specific tuning strategies**:
   - Handle multiple models in SciRegressor simultaneously

3. **Results persistence**:
   - Save hyperparameters to `hyperparams_config.json` in model directories
   - update the model_config.json file with the newly found hyperparameters
   - Create tuning reports and validation curves
   - Integrate with existing configuration system

4. **Create a standalone tuning script**:
   - `tune_hyperparams.py` that can tune parameters for any model configuration
   - Command-line interface for operational use


**Reference Files**:
- `scr/tree_utils.py` (contains Optuna usage patterns)
- Existing configuration structure in model directories

### Agent 4: Evaluation and Calibration System
**Objective**: Create the evaluation framework and calibration pipeline

**Specific Tasks**:
1. **Implement comprehensive evaluation script**:
   - `calibrate_hindcast.py` that orchestrates the entire calibration process
   - Load data using existing `data_loading.py` utilities
   - Run LOOCV evaluation for all configured models
   - Generate standardized metrics and visualizations

2. **Enhance evaluation metrics**:
   - Extend `eval_scr/metric_functions.py` with additional metrics
   - Create evaluation reports with statistical significance testing

3. **Results management**:
   - Save predictions to `monthly_forecasting_results/` directory structure
   - Generate `predictions.csv` and `metrics.csv` for each model
   - Create summary reports across all models

4. **Visualization and reporting**:
   - Generate performance plots and model comparison charts
   - Create ensemble performance analysis
   - Statistical testing for model comparisons

**Reference Files**:
- `eval_scr/eval_helper.py` (existing evaluation utilities)
- `eval_scr/metric_functions.py` (current metrics implementation)
- README.md (output format specifications)

### Agent 5: Documentation and running
**Objectives**: Write documentation and clean up the existing code

**Specific Tasks**
1. **Write a documentation of the workflow in the description file**
   - How are the folders expected to be set up
   - functionaliteis of the forecasting class
   - what can be configured
   - short description of the files
2. **Add comments**
   - add comments to existing code
   - clean existing code, remove unuesed imports etc. 
3. **Adapt the tree_utils.py file**
   - this file is referenced across many models, update it in a way that it is compatible with the new forecastign classeas and the workflows.
4. **Bash script**
    - write a bash script which can be used to run the models by first tune the hyperparamers and the calibrate them 

## Technical Requirements

### Data Flow
1. **Input**: Time series data with discharge, precipitation, temperature, snow variables
2. **Feature Engineering**: Rolling windows, lags, seasonal features via FeatureExtractor
3. **Model Training**: LOOCV with hyperparameter optimization
4. **Output**: Standardized predictions and metrics

### Configuration Structure
```
monthly_forecasting_models/
├── config/
│   ├── data_paths.json
│   └── base_learner_paths.json
└── [ModelName]_[FeatureSet]/
    ├── feature_config.json
    ├── data_config.json
    ├── general_config.json
    ├── hyperparams_config.json
    └── scalers/
```

### Expected Output Format
- **predictions.csv**: `date | Q_model1 | Q_model2 | Q_model3 | Q_mean | valid_from | valid_to`
- **metrics.csv**: Performance metrics per model and basin

## Implementation Guidelines

1. **Maintain backward compatibility** with existing configuration formats
2. **Use existing utilities** from `scr/` directory (data_loading, data_utils, FeatureExtractor)
3. **Follow logging patterns** established in the codebase
4. **Handle missing data** and edge cases robustly
5. **Optimize for computational efficiency** when fitting multiple models
6. **Ensure reproducibility** through proper random seeding

## Validation Criteria

Each agent should ensure their implementation:
- ✅ Passes basic functionality tests
- ✅ Produces expected output formats
- ✅ Handles configuration variations correctly
- ✅ Integrates seamlessly with other components
- ✅ Maintains performance standards from legacy implementation
- ✅ Includes appropriate error handling and logging

## Coordination Between Agents

- **Agent 1 & 2**: Ensure consistent interface implementation and data flow
- **Agent 3**: Provide tuning capabilities that work with both model types
- **Agent 4**: Create evaluation framework that works with all model outputs
- **All Agents**: Use shared configuration and data loading utilities

This modular approach will create a robust, maintainable forecasting system that can easily accommodate new models and feature sets while maintaining high prediction accuracy through ensemble methods.
