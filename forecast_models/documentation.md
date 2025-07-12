# Forecast Models Module

This module contains implementations of various forecasting models for monthly discharge prediction in hydrology applications. All models inherit from a common base class and follow a consistent interface for training, prediction, and model management.

## Overview

The forecast_models module provides:
- A base class abstraction for all forecast models
- Linear regression-based models with feature selection
- Ensemble machine learning models with global fitting capabilities
- Consistent interfaces for operational forecasting and hindcasting

## Directory Structure

```
forecast_models/
├── __init__.py              # Module initialization (currently empty)
├── base_class.py            # Abstract base class for all forecast models
├── LINEAR_REGRESSION.py     # Linear regression with dynamic feature selection
├── SciRegressor.py          # Ensemble ML models (XGBoost, LightGBM, CatBoost)
└── README.md                # This file
```

## Models

### 1. BaseForecastModel (`base_class.py`)

Abstract base class that defines the interface for all forecast models.

**Key Methods:**
- `predict_operational()`: Make operational forecasts for current conditions
- `calibrate_model_and_hindcast()`: Train model and generate historical predictions
- `tune_hyperparameters()`: Optimize model hyperparameters
- `save_model()` / `load_model()`: Model persistence

**Required Inputs:**
- `data`: Time series DataFrame with columns ['date', 'code', 'discharge', ...]
- `static_data`: Basin characteristics DataFrame
- `general_config`: General configuration dictionary
- `model_config`: Model-specific parameters
- `feature_config`: Feature engineering settings
- `path_config`: File paths for saving/loading

### 2. LinearRegressionModel (`LINEAR_REGRESSION.py`)

A linear regression model with dynamic feature selection based on correlation analysis.

**Key Features:**
- **Dynamic Feature Selection**: Selects the most correlated features for each basin and forecast period
- **Period-Specific Models**: Trains separate models for different days of the month (5th, 10th, 15th, 20th, 25th, end)
- **Leave-One-Year-Out Cross-Validation**: Evaluates model performance using LOOCV
- **Support for Ridge/Lasso**: Can use regularized regression variants

**Model Types:**
- Standard Linear Regression
- Ridge Regression (with alpha parameter)
- Lasso Regression (with alpha parameter)

**Feature Selection Process:**
1. Extracts features using StreamflowFeatureExtractor
2. Calculates correlation with target variable
3. Selects top N features (configurable) with diversity across variable types
4. Ensures minimum data requirements are met

**Operational Workflow:**
1. Determines current forecast period based on date
2. Filters data for the specific period
3. Selects best features for each basin
4. Trains model on historical data
5. Makes predictions for current conditions

### 3. SciRegressor (`SciRegressor.py`)

An ensemble regressor supporting multiple machine learning algorithms with global fitting approach.

**Key Features:**
- **Global Model Training**: Single model trained on all basins simultaneously
- **Ensemble Predictions**: Combines multiple ML algorithms for robust forecasts
- **Advanced Preprocessing**: Handles missing data, scaling, and feature engineering
- **Feature Processing Artifacts**: Saves preprocessing steps for consistent application
- **Support for Categorical Features**: Can handle basin-specific categorical variables

**Supported Models:**
- XGBoost
- LightGBM
- CatBoost
- Other scikit-learn compatible regressors

**Data Processing Pipeline:**
1. **Glacier Mapper Features**: Integrates glacier-related features if configured
2. **Discharge Transformation**: Converts from m³/s to mm/day for normalization
3. **Snow Data Processing**: Calculates percentile-based elevation bands
4. **Feature Engineering**: Creates temporal features (cyclical encoding)
5. **Static Feature Integration**: Merges basin characteristics
6. **Categorical Encoding**: One-hot encodes categorical variables

**Advanced Capabilities:**
- **Hyperparameter Tuning**: Uses Optuna for automated optimization
- **Feature Importance**: Tracks and saves feature importance scores
- **Early Stopping**: Prevents overfitting during training
- **Multiple Validation Strategies**: Supports various cross-validation approaches

## Dependencies

The forecast_models module depends on several components from the `scr/` folder:

### From `scr/`:
- **FeatureExtractor**: Creates lagged features, rolling statistics, and other time series features
- **FeatureProcessingArtifacts**: Handles preprocessing artifacts (scalers, imputers, feature lists)
- **data_utils**: Utility functions for data manipulation (glacier features, snow band calculations)
- **sci_utils**: Machine learning utilities (model creation, fitting, hyperparameter optimization)

### External Dependencies:
- scikit-learn: Linear models, preprocessing, metrics
- pandas/numpy: Data manipulation
- xgboost/lightgbm/catboost: Ensemble models (for SciRegressor)
- optuna: Hyperparameter optimization (for SciRegressor)
- geopandas: Spatial data processing (for elevation bands)

## Configuration

### General Configuration Keys:
- `model_name`: Name identifier for the model
- `prediction_horizon`: Number of days to forecast ahead
- `offset`: Days offset from current date
- `num_features`: Number of features to select (LinearRegression)
- `base_features`: List of base variable names to consider
- `snow_vars`: Snow-related variables (e.g., 'SWE')
- `forecast_days`: Days of month to generate forecasts
- `filter_years`: Years to exclude from training

### Model-Specific Configuration:

**LinearRegressionModel:**
```python
model_config = {
    "lr_type": "linear",  # or "ridge", "lasso"
    "alpha": 1.0  # For ridge/lasso regularization
}
```

**SciRegressor:**
```python
model_config = {
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        # ... other XGBoost parameters
    },
    "lightgbm": {
        # LightGBM parameters
    }
}
```

## Usage Example

```python
from forecast_models.LINEAR_REGRESSION import LinearRegressionModel
from forecast_models.SciRegressor import SciRegressor

# Initialize model
model = LinearRegressionModel(
    data=time_series_df,
    static_data=basin_characteristics_df,
    general_config=general_config,
    model_config=model_config,
    feature_config=feature_config,
    path_config=path_config
)

# Calibrate and generate hindcasts
hindcast_df = model.calibrate_model_and_hindcast()

# Make operational forecast
forecast_df = model.predict_operational()

# Save trained model
model.save_model()
```

## Model Selection Guidelines

### Use LinearRegressionModel when:
- Interpretability is important
- Limited training data available
- Simple relationships between features and target
- Need period-specific models

### Use SciRegressor when:
- Large amounts of training data available
- Complex non-linear relationships expected
- Multiple basins with shared patterns
- Ensemble predictions desired for robustness

## Output Format

Both models produce consistent output formats:

**Hindcast Output:**
```
columns: ['date', 'code', 'Q_<model_name>', 'Q_obs', ...]
```

**Operational Forecast Output:**
```
columns: ['forecast_date', 'code', 'valid_from', 'valid_to', 'Q_<model_name>', ...]
```

## Notes

1. The LinearRegressionModel fits a new model for each prediction, while SciRegressor saves and reuses trained models
2. Both models handle missing data but with different strategies
3. Feature engineering is crucial for model performance - configure feature_config carefully
4. Models automatically handle unit conversions between m³/s and mm/day
5. Ensemble predictions (SciRegressor) typically provide more robust forecasts but require more computational resources