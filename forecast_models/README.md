# Forecast Models Directory

## Purpose
Core model implementations for monthly discharge forecasting, including abstract base class and concrete implementations for different model types.

## Contents
- `base_class.py`: Abstract base class defining model interface
- `LINEAR_REGRESSION.py`: Per-basin linear regression implementation
- `SciRegressor.py`: Global tree-based ensemble model implementation

## Important Classes and Functions

### base_class.py - AbstractForecastModel
- `fit()`: Abstract method for model training
- `predict()`: Abstract method for making predictions
- `evaluate()`: Common evaluation method using eval_scr metrics
- `save_model()`: Model persistence functionality
- `load_model()`: Model loading functionality

### LINEAR_REGRESSION.py - LinearRegressionModel
- `fit()`: Trains separate linear regression for each basin
- `predict()`: Makes predictions using basin-specific models
- Supports feature selection and preprocessing
- Handles missing data through interpolation

### SciRegressor.py - SciRegressor
- `fit()`: Trains global tree-based models (XGBoost, Random Forest, CatBoost)
- `predict()`: Makes predictions using ensemble methods
- `extract_and_store_features()`: Advanced feature engineering
- Supports multiple preprocessing methods
- Feature importance analysis

## Key Features
- Consistent interface across model types
- Built-in cross-validation support
- Automated feature selection
- Comprehensive logging
- Model artifact management

## Usage
```python
# Linear Regression
from forecast_models.LINEAR_REGRESSION import LinearRegressionModel
model = LinearRegressionModel(config)
model.fit(train_data)
predictions = model.predict(test_data)

# Tree-based Models
from forecast_models.SciRegressor import SciRegressor
model = SciRegressor(algorithm='xgb', preprocessing_method='monthly_bias')
model.fit(features, targets)
```

## Integration Points
- Used by calibration scripts
- Integrated with hyperparameter tuning
- Connected to evaluation pipeline
- Artifact storage in tests_output/