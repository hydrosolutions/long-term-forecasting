# SCR (Source) Module - Core Utilities for Monthly Discharge Forecasting

This directory contains the core utilities and modules that power the monthly discharge forecasting system. These modules handle data loading, feature engineering, data preprocessing, and machine learning utilities.

## Module Overview

### 1. `__init__.py`
Defines global constants for column names used throughout the project:
- `AREA_KM2_COL`: Area in square kilometers
- `GLACIER_FRACTION_COL`: Glacier fraction
- `H_MIN_COL`: Minimum elevation in meters
- `H_MAX_COL`: Maximum elevation in meters

### 2. `data_loading.py`
Handles all data loading operations for the forecasting system.

**Key Functions:**
- `load_data()`: Main function to load discharge, forcing, static, and snow data
- `load_snow_data()`: Loads snow data (SWE, HS, ROF) from CSV files
- `time_shift_sla_data()`: Adjusts SLA data timestamps from decadal to daily basis
- `transform_data_gateway_data()`: Transforms raw data gateway format into usable format
- `reindex_dataframe()`: Creates complete date ranges for all basin codes
- `load_base_learners()`: Loads predictions from base learner models for ensemble building

**Usage Example:**
```python
from scr import data_loading as dl

hydro_ca, static = dl.load_data(
    path_discharge="discharge.csv",
    path_forcing="forcing.csv",
    path_static_data="static.csv",
    path_to_sca="sca.csv",
    path_to_swe="swe_folder/",
    path_to_hs="hs_folder/",
    path_to_rof="rof_folder/",
    HRU_SWE="HRU_",
    HRU_HS="HRU_",
    HRU_ROF="HRU_"
)
```

### 3. `data_utils.py`
Provides data manipulation and preprocessing utilities.

**Key Functions:**
- **Normalization:**
  - `get_normalization_params()`: Calculate global normalization parameters
  - `apply_normalization()`: Apply global normalization
  - `get_normalization_params_per_basin()`: Calculate per-basin normalization
  - `apply_normalization_per_basin()`: Apply per-basin normalization
  - `apply_inverse_normalization()`: Reverse global normalization
  - `apply_inverse_normalization_per_basin()`: Reverse per-basin normalization

- **Long-term Mean Scaling:**
  - `get_long_term_mean_per_basin()`: Calculate period-based statistics (36 periods/year)
  - `apply_long_term_mean()`: Fill missing values with long-term means
  - `apply_long_term_mean_scaling()`: Apply relative scaling using long-term stats
  - `apply_inverse_long_term_mean_scaling()`: Reverse long-term mean scaling
  - `apply_inverse_long_term_mean_scaling_predictions()`: Special inverse scaling for predictions

- **Elevation Band Processing:**
  - `calculate_percentile_snow_bands()`: Aggregate snow data by elevation percentiles
  - `get_elevation_bands_per_percentile()`: Assign elevation bands to percentiles
  - `calculate_elevation_band_areas()`: Calculate relative areas for elevation bands

- **GlacierMapper Features:**
  - `glacier_mapper_features()`: Process glacier-related features (SLA, FSC, melt potential)
  
- **Helper Functions:**
  - `get_periods()`: Create 36 annual periods (10th, 20th, end of each month)
  - `get_relative_scaling_features()`: Identify features for relative scaling

### 4. `FeatureExtractor.py`
Creates time series features for machine learning models.

**Main Class: `StreamflowFeatureExtractor`**
- Creates rolling window features with various operations
- Supports multiple window sizes and lag configurations
- Generates time-based features (cyclical encodings)

**Supported Operations:**
- Basic statistics: mean, sum, min, max, std
- Advanced metrics: slope, peak-to-peak, median absolute deviation
- Time-based: time distance from peak, last value occurrence
- Signal processing: mean difference

**Usage Example:**
```python
from scr.FeatureExtractor import StreamflowFeatureExtractor

feature_configs = {
    "discharge": [
        {
            "operation": "mean",
            "windows": [7, 30, 90],
            "lags": {7: [1, -1], 30: [1], 90: []}
        }
    ]
}

extractor = StreamflowFeatureExtractor(
    feature_configs=feature_configs,
    prediction_horizon=30,
    offset=30
)

features_df = extractor.create_all_features(data_df)
```

### 5. `FeatureProcessingArtifacts.py`
Manages preprocessing artifacts for consistent train/test processing.

**Main Class: `FeatureProcessingArtifacts`**
- Stores all preprocessing parameters (scalers, imputers, feature selections)
- Supports multiple save formats (joblib, pickle, hybrid)
- Ensures reproducible preprocessing across train/test splits

**Key Functions:**
- `process_training_data()`: Process training data and create artifacts
- `process_test_data()`: Apply artifacts to test data
- `post_process_predictions()`: Reverse normalization on predictions
- `save_artifacts_for_production()`: Save artifacts with versioning
- `load_artifacts_for_production()`: Load versioned artifacts

**Artifact Components:**
- Imputation parameters
- Normalization scalers (global/per-basin)
- Long-term statistics for relative scaling
- Feature selection results
- Static feature scalers

### 6. `sci_utils.py`
Machine learning utilities for model training and optimization.

**Key Functions:**
- `get_model()`: Create model instances (XGBoost, LightGBM, CatBoost, RF, MLP)
- `fit_model()`: Train models with validation split
- `get_feature_importance()`: Extract feature importance from trained models
- `optimize_hyperparams()`: Hyperparameter optimization using Optuna

**Supported Models:**
- XGBoost (`xgb`)
- LightGBM (`lgbm`)
- CatBoost (`catboost`)
- Random Forest (`rf`)
- Gradient Boosting (`gradient_boosting`)
- Neural Network (`mlp`)

**Usage Example:**
```python
from scr import sci_utils

# Create and train model
model = sci_utils.get_model('xgb', params={'n_estimators': 100})
fitted_model = sci_utils.fit_model(model, X_train, y_train)

# Optimize hyperparameters
best_params = sci_utils.optimize_hyperparams(
    X_train, y_train, X_val, y_val,
    model_type='xgb',
    n_trials=100
)
```

## Integration with Main Pipeline

These utilities are used throughout the forecasting system:
- **Model Training**: SciRegressor and LINEAR_REGRESSION models use these utilities
- **Data Pipeline**: tune_hyperparams.py and calibrate_hindcast.py rely on data_loading
- **Evaluation**: The evaluation pipeline uses artifacts for consistent preprocessing
- **Testing**: Comprehensive test suite validates all utility functions

## Best Practices

1. **Always use artifacts**: When preprocessing data, create and save artifacts from training data
2. **Handle missing values carefully**: Use appropriate strategies (drop, impute, or long-term mean)
3. **Consider normalization type**: Choose between global, per-basin, or relative scaling based on your data
4. **Feature engineering**: Use StreamflowFeatureExtractor for consistent feature creation
5. **Model selection**: Use sci_utils for standardized model creation and optimization

## Dependencies

- pandas, numpy: Data manipulation
- scikit-learn: Machine learning utilities
- xgboost, lightgbm, catboost: Gradient boosting models
- optuna: Hyperparameter optimization
- geopandas: Spatial data processing
- scipy: Statistical functions