# Monthly Discharge Forecasting System Documentation

## Overview

This system provides a modular framework for monthly discharge forecasting using machine learning models. It supports both linear regression models (fitted per basin-period) and tree-based ensemble models (fitted globally across all basins).

## Architecture

### Core Components

1. **Base Class**: `BaseForecastModel` - Abstract interface defining common methods
2. **Model Classes**:
   - `LinearRegressionModel` - Per-basin linear regression with feature selection
   - `SciRegressor` - Global tree-based ensemble models (XGBoost, LightGBM, CatBoost)
3. **Utilities**:
   - `FeatureExtractor` - Time series feature engineering
   - `tree_utils` - Tree-based model utilities and hyperparameter optimization
   - `eval_scr` - Evaluation metrics and helper functions

### Key Design Principles

- **Modular Architecture**: Clean separation between model types and utilities
- **Configuration-Driven**: JSON-based configuration for flexibility
- **Feature Set Grouping**: Models with same features can be ensembled
- **Global vs Local Fitting**: Tree models train globally, linear models per basin
- **Leave-One-Out Cross-Validation**: Yearly LOOCV with reserved test years

## Directory Structure

```
monthly_forecasting/
├── forecast_models/              # Core model implementations
│   ├── base_class.py            # Abstract base class
│   ├── LINEAR_REGRESSION.py     # Linear regression implementation
│   └── SciRegressor.py          # Tree-based ensemble implementation
├── scr/                         # Utilities and data processing
│   ├── FeatureExtractor.py      # Feature engineering
│   ├── data_loading.py          # Data loading utilities
│   ├── data_utils.py            # Data manipulation
│   └── tree_utils.py            # Tree model utilities
├── eval_scr/                    # Evaluation utilities
│   ├── eval_helper.py           # Evaluation helpers
│   └── metric_functions.py      # Metrics calculation
├── old_files/                   # Legacy implementations (reference)
├── monthly_forecasting_models/  # Model configurations
│   └── [ModelName_FeatureSet]/  # Individual model directories
├── monthly_forecasting_results/ # Output results
│   └── [ModelName]/             # Results per model
├── tune_hyperparams.py          # Hyperparameter tuning script
├── calibrate_hindcast.py        # Calibration and evaluation script
└── run_model_workflow.sh        # Complete workflow automation
```

## Configuration Structure

Each model requires a configuration directory with the following files:

### Model Configuration Directory
```
ModelName_FeatureSet/
├── general_config.json          # General model settings
├── model_config.json            # Model-specific parameters
├── feature_config.json          # Feature engineering settings
├── data_config.json             # Data loading configuration
└── hyperparams_config.json      # Optimized hyperparameters (generated)
```

### Configuration Files

#### `general_config.json`
```json
{
    "model_name": "XGBoost_AllFeatures",
    "model_type": "sciregressor",
    "models": ["xgboost", "lightgbm", "catboost"],
    "prediction_horizon": 30,
    "offset": 30,
    "num_features": 20,
    "base_features": ["discharge", "precip", "temp"],
    "snow_vars": ["SWE", "SCA"],
    "static_features": ["area_km2", "elevation_mean", "glacier_fraction"],
    "forecast_days": [5, 10, 15, 20, 25, "end"],
    "test_years": 3,
    "hyperparam_tuning_trials": 100,
    "hyperparam_tuning_years": 3
}
```

#### `model_config.json`
```json
{
    "missing_value_handling": "drop",
    "normalization_type": "standard",
    "normalize_per_basin": false,
    "use_pca": false,
    "use_lr_predictors": false,
    "use_temporal_features": true,
    "use_static_features": true,
    "cat_features": ["code_str"],
    "use_basin_encoding": true,
    "xgboost_params": {
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "max_depth": 6
    },
    "lightgbm_params": {
        "n_estimators": 1000,
        "learning_rate": 0.1,
        "num_leaves": 31
    }
}
```

#### `feature_config.json`
```json
{
    "rolling_windows": [7, 14, 30, 60],
    "lag_features": [1, 7, 14, 30],
    "seasonal_features": true,
    "trend_features": true,
    "snow_elevation_bands": [1000, 2000, 3000, 4000],
    "precipitation_aggregation": ["mean", "sum", "max"],
    "temperature_features": ["mean", "min", "max"]
}
```

## Model Types

### 1. LinearRegressionModel

**Approach**: Per-basin, per-period training
- **Training**: Separate model for each basin-period combination
- **Feature Selection**: Correlation-based selection per period
- **Cross-Validation**: Leave-One-Year-Out per basin
- **Hyperparameters**: Alpha for Ridge/Lasso regression

**Use Cases**:
- Interpretable models
- Basin-specific relationships
- Simple baseline models

### 2. SciRegressor (Tree-Based Ensemble)

**Approach**: Global training across all basins
- **Training**: Single model per period trained on all basins
- **Feature Engineering**: Basin identity encoded as features (dummy variables + static characteristics)
- **Cross-Validation**: Global Leave-One-Year-Out
- **Ensemble**: Multiple models (XGBoost, LightGBM, CatBoost) averaged

**Use Cases**:
- Complex non-linear relationships
- Cross-basin learning
- High-performance forecasting

## Workflow

### 1. Data Preparation
- Load time series data (discharge, precipitation, temperature, snow)
- Load static basin characteristics
- Apply feature engineering (rolling windows, lags, seasonal features)

### 2. Model Training
**Linear Regression**:
- For each period and basin:
  - Select features by correlation
  - Train regularized linear model
  - Validate using LOOCV

**Tree-Based Models**:
- For each period:
  - Combine all basin data
  - Create basin dummy variables
  - Train ensemble of tree models globally
  - Validate using global LOOCV

### 3. Evaluation
- Generate hindcast predictions
- Calculate metrics (R², RMSE, MAE, NSE, KGE, Bias)
- Create performance reports
- Save results in standardized format

### 4. Hyperparameter Optimization
- Use Optuna for optimization
- Time series cross-validation
- Model-specific parameter spaces
- Save best parameters to configuration

## Usage Examples

### Command Line Interface

#### Hyperparameter Tuning
```bash
python tune_hyperparams.py \
    --config_dir monthly_forecasting_models/XGBoost_AllFeatures \
    --model_name XGBoost_AllFeatures \
    --trials 100
```

#### Model Calibration
```bash
python calibrate_hindcast.py \
    --config_dir monthly_forecasting_models/XGBoost_AllFeatures \
    --model_name XGBoost_AllFeatures \
    --output_dir results/xgb_calibration/
```

#### Complete Workflow
```bash
./run_model_workflow.sh \
    --config_dir monthly_forecasting_models/XGBoost_AllFeatures \
    --model_name XGBoost_AllFeatures \
    --tune_hyperparams \
    --trials 200
```

### Python API

```python
from forecast_models.SciRegressor import SciRegressor
import pandas as pd
import json

# Load configuration
with open('monthly_forecasting_models/XGBoost_AllFeatures/general_config.json') as f:
    general_config = json.load(f)

# Load data
data = pd.read_csv('data/hydro_data.csv')
static_data = pd.read_csv('data/static_data.csv')

# Create model
model = SciRegressor(
    data=data,
    static_data=static_data,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config
)

# Run calibration
hindcast_df = model.calibrate_model_and_hindcast(data)

# Make operational forecast
forecast_df = model.predict_operational()
```

## Output Format

### Predictions (`predictions.csv`)
```csv
date,model,code,Q_pred
2020-03-15,XGBoost_AllFeatures,15101,45.67
2020-03-15,XGBoost_AllFeatures_xgboost,15101,44.32
2020-03-15,XGBoost_AllFeatures_lightgbm,15101,46.89
```

### Metrics (`metrics.csv`)
```csv
model,code,n_predictions,r2,rmse,mae,bias,kge,nse,mean_obs,mean_pred
XGBoost_AllFeatures,15101,120,0.78,12.34,8.91,2.1,0.82,0.76,45.6,47.7
```

### Operational Forecast
```csv
forecast_date,model_name,code,valid_from,valid_to,Q,Q_05,Q_10,Q_50,Q_90,Q_95
2024-07-01,XGBoost_AllFeatures,15101,2024-07-02,2024-08-01,48.5,35.2,38.7,48.5,58.3,62.1
```

## Best Practices

### Model Development
1. **Start Simple**: Begin with LinearRegressionModel for baseline
2. **Feature Engineering**: Use domain knowledge for feature selection
3. **Cross-Validation**: Always use proper time series CV
4. **Ensemble Models**: Combine multiple algorithms for robustness

### Configuration Management
1. **Version Control**: Track configuration changes
2. **Naming Convention**: Use descriptive model names
3. **Documentation**: Document feature engineering choices
4. **Validation**: Test configurations before large runs

### Operational Deployment
1. **Model Persistence**: Save trained models for reuse
2. **Error Handling**: Implement robust error handling
3. **Monitoring**: Track model performance over time
4. **Updates**: Regular retraining with new data

## Troubleshooting

### Common Issues

1. **Feature Selection Fails**
   - Check for sufficient data points
   - Verify feature naming consistency
   - Ensure target variable is present

2. **Memory Issues with Global Models**
   - Reduce number of features
   - Use feature selection
   - Consider per-period training

3. **Poor Model Performance**
   - Check data quality
   - Verify feature engineering
   - Tune hyperparameters
   - Consider ensemble methods

4. **Configuration Errors**
   - Validate JSON syntax
   - Check file paths
   - Verify required fields

### Debugging Tips

1. **Enable Debug Logging**
   ```bash
   python script.py --log_level DEBUG
   ```

2. **Check Intermediate Outputs**
   - Inspect feature extraction results
   - Validate data preprocessing
   - Monitor cross-validation scores

3. **Use Subset for Testing**
   - Test with fewer basins
   - Use shorter time periods
   - Reduce hyperparameter trials

## Performance Optimization

### Computational Efficiency
1. **Parallel Processing**: Use multiprocessing for independent tasks
2. **Feature Caching**: Cache expensive feature calculations
3. **Early Stopping**: Use in hyperparameter optimization
4. **Memory Management**: Clear large objects when not needed

### Model Accuracy
1. **Feature Engineering**: Domain-specific features often outperform generic ones
2. **Ensemble Methods**: Combine multiple models for better performance
3. **Regular Updates**: Retrain models with recent data
4. **Hyperparameter Tuning**: Invest time in proper optimization

## Extension Guidelines

### Adding New Model Types
1. Inherit from `BaseForecastModel`
2. Implement all abstract methods
3. Follow existing naming conventions
4. Add configuration validation
5. Update documentation

### Adding New Features
1. Extend `FeatureExtractor` class
2. Update configuration schemas
3. Add feature validation
4. Test with existing models
5. Document new features

### Adding New Metrics
1. Add functions to `metric_functions.py`
2. Follow existing error handling patterns
3. Include proper documentation
4. Add unit tests
5. Update calibration scripts

This system provides a robust, scalable framework for monthly discharge forecasting that can be easily extended and maintained.