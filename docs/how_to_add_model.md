# How to Add a New Forecast Model

This guide explains how to implement a custom forecast model that integrates seamlessly with the existing forecasting system.

## Overview

All forecast models must inherit from `BaseForecastModel` and implement a set of required abstract methods. The base class provides a standardized interface for:

- Model calibration and hindcasting (cross-validation)
- Operational predictions
- Hyperparameter tuning
- Model persistence (save/load)

## Quick Start

### 1. Create Model File

Create a new Python file in `lt_forecasting/forecast_models/`:

```bash
touch lt_forecasting/forecast_models/MY_CUSTOM_MODEL.py
```

### 2. Basic Template

```python
import pandas as pd
import datetime
from typing import Dict, Any
import logging

from lt_forecasting.forecast_models.base_class import BaseForecastModel
from lt_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MyCustomModel(BaseForecastModel):
    """
    Custom forecast model implementation.

    Description: [Describe what this model does and when to use it]
    """

    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
    ) -> None:
        """
        Initialize the custom model.

        Args:
            data: Time-series data (discharge, weather, snow, etc.)
            static_data: Basin characteristics (area, elevation, etc.)
            general_config: General experiment settings
            model_config: Model-specific hyperparameters
            feature_config: Feature engineering configuration
            path_config: File paths for input/output
        """
        super().__init__(
            data=data,
            static_data=static_data,
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            path_config=path_config,
        )

        # Initialize model-specific attributes
        self.model = None
        self.is_fitted = False

        logger.info(f"Initialized {self.name}")

    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate model and generate hindcast predictions.

        This method should:
        1. Split data into train/test sets
        2. Perform cross-validation (e.g., Leave-One-Year-Out)
        3. Train final model on all available data
        4. Save model artifacts
        5. Return hindcast predictions

        Returns:
            pd.DataFrame: Hindcast predictions with columns:
                - date: Forecast date
                - code: Basin identifier
                - model: Model name
                - Q_pred: Predicted discharge
                - (Optional) Q_05, Q_10, Q_50, Q_90, Q_95: Quantiles
        """
        logger.info(f"Starting calibration for {self.name}")

        # TODO: Implement your calibration logic

        # Example structure:
        # 1. Feature engineering
        # 2. Cross-validation
        # 3. Final training
        # 4. Save model

        hindcast = pd.DataFrame()  # Replace with actual predictions
        return hindcast

    def predict_operational(
        self, today: datetime.datetime = None
    ) -> pd.DataFrame:
        """
        Generate operational forecast.

        This method should:
        1. Load trained model
        2. Extract features from recent data
        3. Make predictions for future period
        4. Return formatted predictions

        Args:
            today: Reference date for forecast (default: current datetime)

        Returns:
            pd.DataFrame: Operational predictions with same format as hindcast
        """
        if today is None:
            today = datetime.datetime.now()

        logger.info(f"Generating operational forecast for {today}")

        # TODO: Implement operational prediction logic

        forecast = pd.DataFrame()  # Replace with actual predictions
        return forecast

    def tune_hyperparameters(self) -> None:
        """
        Tune model hyperparameters using Optuna.

        This method should:
        1. Define hyperparameter search space
        2. Create Optuna objective function
        3. Run optimization
        4. Save best parameters to config

        Returns:
            None (saves best parameters to model_config.json)
        """
        logger.info(f"Starting hyperparameter tuning for {self.name}")

        # TODO: Implement hyperparameter tuning
        # See SciRegressor.tune_hyperparameters() for reference

        logger.info("Hyperparameter tuning not implemented")

    def save_model(self) -> None:
        """
        Save trained model to disk.

        This method should save:
        1. Model weights/parameters
        2. Preprocessing artifacts (scalers, imputers, etc.)
        3. Metadata (training date, performance metrics)

        Model files saved to: path_config['model_path']
        """
        model_path = self.path_config.get("model_path", "./models/")
        logger.info(f"Saving model to {model_path}")

        # TODO: Implement model saving logic

    def load_model(self) -> None:
        """
        Load trained model from disk.

        This method should load:
        1. Model weights/parameters
        2. Preprocessing artifacts
        3. Metadata

        Model files loaded from: path_config['model_path']
        """
        model_path = self.path_config.get("model_path", "./models/")
        logger.info(f"Loading model from {model_path}")

        # TODO: Implement model loading logic

        self.is_fitted = True
```

## Required Methods

### 1. `__init__()`

**Purpose**: Initialize the model with configuration and data.

**Key Tasks**:
- Call `super().__init__()` to initialize base class
- Set model-specific attributes
- Initialize model objects (but don't train yet)

**Example**:
```python
def __init__(self, data, static_data, general_config, model_config,
             feature_config, path_config):
    super().__init__(data, static_data, general_config, model_config,
                     feature_config, path_config)

    # Model-specific initialization
    self.model_type = model_config.get("model_type", "default")
    self.model = None
    self.artifacts = None
```

### 2. `calibrate_model_and_hindcast()`

**Purpose**: Train the model and generate hindcast (validation) predictions.

**Key Tasks**:
1. **Feature Engineering**: Extract features from raw data
2. **Data Splitting**: Separate train/validation/test sets
3. **Cross-Validation**: Perform Leave-One-Year-Out (or similar)
4. **Final Training**: Train on all available data
5. **Model Persistence**: Save trained model and artifacts
6. **Return Hindcast**: DataFrame with predictions

**Example**:
```python
def calibrate_model_and_hindcast(self) -> pd.DataFrame:
    # 1. Feature engineering
    from lt_forecasting.scr.FeatureExtractor import StreamflowFeatureExtractor

    extractor = StreamflowFeatureExtractor(
        feature_configs=self.feature_config,
        prediction_horizon=30,
        offset=30
    )

    features_df = extractor.create_all_features(self.data)
    target = extractor.create_target(self.data)

    # 2. Cross-validation
    years = features_df['year'].unique()
    test_years = years[-3:]  # Last 3 years for testing
    train_years = years[:-3]

    hindcast_predictions = []

    for year in train_years:
        # Leave-one-year-out
        train_mask = (features_df['year'] != year) & \
                     (features_df['year'].isin(train_years))
        val_mask = features_df['year'] == year

        X_train = features_df[train_mask]
        y_train = target[train_mask]
        X_val = features_df[val_mask]

        # Train model
        model = self._train_model(X_train, y_train)

        # Predict
        y_pred = model.predict(X_val)

        # Store predictions
        hindcast_predictions.append({
            'year': year,
            'predictions': y_pred
        })

    # 3. Final training on all data
    X_all = features_df[features_df['year'].isin(train_years)]
    y_all = target[features_df['year'].isin(train_years)]

    self.model = self._train_model(X_all, y_all)

    # 4. Save model
    self.save_model()

    # 5. Format hindcast
    hindcast_df = self._format_predictions(hindcast_predictions)

    return hindcast_df
```

### 3. `predict_operational()`

**Purpose**: Generate operational forecast for future period.

**Key Tasks**:
1. **Load Model**: Ensure trained model is available
2. **Data Validation**: Check for sufficient recent data
3. **Feature Extraction**: Process latest observations
4. **Prediction**: Generate forecast
5. **Format Output**: Return standardized DataFrame

**Example**:
```python
def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
    if today is None:
        today = datetime.datetime.now()

    # 1. Load model if not already loaded
    if not self.is_fitted:
        self.load_model()

    # 2. Filter recent data
    lookback_days = self.general_config.get('lookback_days', 90)
    start_date = today - datetime.timedelta(days=lookback_days)

    recent_data = self.data[
        (self.data['date'] >= start_date) &
        (self.data['date'] <= today)
    ]

    # 3. Extract features
    extractor = StreamflowFeatureExtractor(
        feature_configs=self.feature_config,
        prediction_horizon=30,
        offset=30
    )

    features_df = extractor.create_all_features(recent_data)

    # Get last row (most recent forecast date)
    X_operational = features_df.iloc[[-1]]

    # 4. Make prediction
    y_pred = self.model.predict(X_operational)

    # 5. Format output
    forecast_date = today + datetime.timedelta(days=30)

    forecast_df = pd.DataFrame({
        'date': [forecast_date],
        'code': [X_operational['basin_id'].values[0]],
        'model': [self.name],
        'Q_pred': y_pred
    })

    return forecast_df
```

### 4. `tune_hyperparameters()`

**Purpose**: Optimize model hyperparameters using Optuna.

**Key Tasks**:
1. **Define Search Space**: Specify hyperparameter ranges
2. **Create Objective**: Function to minimize/maximize
3. **Run Optimization**: Execute Optuna study
4. **Save Results**: Update model_config.json with best parameters

**Example**:
```python
def tune_hyperparameters(self) -> None:
    import optuna

    # 1. Define objective function
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 0.001, 0.1, log=True)
        max_depth = trial.suggest_int('max_depth', 3, 10)
        n_estimators = trial.suggest_int('n_estimators', 100, 1000)

        # Create model with suggested parameters
        model = MyModel(
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators
        )

        # Cross-validation
        scores = []
        for year in train_years:
            # Train/validation split
            X_train, y_train, X_val, y_val = split_by_year(year)

            # Train and evaluate
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = compute_metric(y_val, y_pred)  # e.g., NSE, RMSE
            scores.append(score)

        return np.mean(scores)

    # 2. Run optimization
    study = optuna.create_study(
        direction='maximize',  # or 'minimize'
        study_name=f'{self.name}_tuning'
    )

    study.optimize(objective, n_trials=100, timeout=3600)

    # 3. Save best parameters
    best_params = study.best_params
    logger.info(f"Best parameters: {best_params}")

    # Update config file
    config_path = os.path.join(
        self.path_config['config_path'],
        'model_config.json'
    )

    with open(config_path, 'w') as f:
        json.dump(best_params, f, indent=4)
```

### 5. `save_model()` and `load_model()`

**Purpose**: Persist and load trained models.

**Key Tasks** (save):
- Save model weights/parameters
- Save preprocessing artifacts
- Save metadata (training date, metrics, etc.)

**Key Tasks** (load):
- Load model weights/parameters
- Load preprocessing artifacts
- Validate model integrity

**Example**:
```python
def save_model(self) -> None:
    import joblib

    model_path = self.path_config.get('model_path', './models/')
    os.makedirs(model_path, exist_ok=True)

    # Save model
    model_file = os.path.join(model_path, f'{self.name}_model.joblib')
    joblib.dump(self.model, model_file)

    # Save preprocessing artifacts (if used)
    if self.artifacts is not None:
        from lt_forecasting.scr.FeatureProcessingArtifacts import \
            save_artifacts_for_production

        save_artifacts_for_production(
            artifacts=self.artifacts,
            path=model_path,
            format='hybrid'
        )

    # Save metadata
    metadata = {
        'model_name': self.name,
        'training_date': datetime.datetime.now().isoformat(),
        'feature_config_hash': hash(str(self.feature_config)),
    }

    metadata_file = os.path.join(model_path, 'metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"Model saved to {model_path}")

def load_model(self) -> None:
    import joblib

    model_path = self.path_config.get('model_path', './models/')

    # Load model
    model_file = os.path.join(model_path, f'{self.name}_model.joblib')
    self.model = joblib.load(model_file)

    # Load preprocessing artifacts (if used)
    from lt_forecasting.scr.FeatureProcessingArtifacts import \
        load_artifacts_for_production

    try:
        self.artifacts = load_artifacts_for_production(model_path)
    except FileNotFoundError:
        logger.warning("No preprocessing artifacts found")
        self.artifacts = None

    self.is_fitted = True
    logger.info(f"Model loaded from {model_path}")
```

## Configuration Structure

Each model requires a configuration directory with JSON files:

```
example_config/MY_CUSTOM_MODEL/
├── data_paths.json           # Input data file paths
├── experiment_config.json    # Experiment setup
├── feature_config.json       # Feature engineering parameters
├── general_config.json       # Model settings
└── model_config.json         # Algorithm-specific hyperparameters
```

### Example: `general_config.json`

```json
{
    "model_name": "MY_CUSTOM_MODEL",
    "model_class": "MyCustomModel",
    "module_path": "lt_forecasting.forecast_models.MY_CUSTOM_MODEL",
    "prediction_horizon": 30,
    "offset": 30,
    "test_years": [2020, 2021, 2022],
    "lookback_days": 90
}
```

### Example: `model_config.json`

```json
{
    "model_type": "gradient_boosting",
    "learning_rate": 0.01,
    "max_depth": 6,
    "n_estimators": 500,
    "early_stopping_rounds": 50,
    "regularization": {
        "l1": 0.0,
        "l2": 1.0
    }
}
```

## Integration with System

### 1. Register Model

Update the model factory/registry if your system uses one:

```python
# In lt_forecasting/forecast_models/__init__.py

from lt_forecasting.forecast_models.MY_CUSTOM_MODEL import MyCustomModel

MODEL_REGISTRY = {
    'LINEAR_REGRESSION': LinearRegressionModel,
    'SciRegressor': SciRegressor,
    'UncertaintyMixtureModel': UncertaintyMixtureModel,
    'MY_CUSTOM_MODEL': MyCustomModel,  # Add your model
}
```

### 2. Update Scripts

The model should work with existing scripts:

```bash
# Hyperparameter tuning
uv run python scripts/tune_hyperparams.py --config_path example_config/MY_CUSTOM_MODEL

# Calibration and hindcast
uv run python scripts/calibrate_hindcast.py --config_path example_config/MY_CUSTOM_MODEL

# Operational forecast
uv run python scripts/operational_forecast.py --config_path example_config/MY_CUSTOM_MODEL
```

## Best Practices

### 1. Logging

Use structured logging throughout:

```python
logger.info(f"Starting training with {len(X_train)} samples")
logger.warning(f"Missing data detected: {missing_count} records")
logger.error(f"Model training failed: {error_message}")
```

### 2. Error Handling

Implement robust error handling:

```python
try:
    self.model.fit(X_train, y_train)
except Exception as e:
    logger.error(f"Training failed: {e}")
    raise RuntimeError(f"Failed to train {self.name}: {e}")
```

### 3. Data Validation

Validate inputs before processing:

```python
def _validate_data(self, df: pd.DataFrame):
    """Validate input data."""
    required_cols = ['date', 'code', 'Q', 'P', 'T']
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.isnull().all().any():
        logger.warning("Some columns are entirely NaN")
```

### 4. Testing

Write unit tests for your model:

```python
# tests/unit/test_my_custom_model.py

import pytest
from lt_forecasting.forecast_models.MY_CUSTOM_MODEL import MyCustomModel


def test_model_initialization():
    """Test model can be initialized."""
    model = MyCustomModel(
        data=test_data,
        static_data=test_static,
        general_config=test_general_config,
        model_config=test_model_config,
        feature_config=test_feature_config,
        path_config=test_path_config,
    )

    assert model.name == "MY_CUSTOM_MODEL"
    assert model.is_fitted is False


def test_calibration():
    """Test model calibration."""
    model = MyCustomModel(...)
    hindcast = model.calibrate_model_and_hindcast()

    assert isinstance(hindcast, pd.DataFrame)
    assert 'Q_pred' in hindcast.columns
    assert len(hindcast) > 0


def test_operational_prediction():
    """Test operational forecast."""
    model = MyCustomModel(...)
    model.calibrate_model_and_hindcast()

    forecast = model.predict_operational()

    assert isinstance(forecast, pd.DataFrame)
    assert len(forecast) > 0
```

### 5. Documentation

Document your model class thoroughly:

```python
class MyCustomModel(BaseForecastModel):
    """
    Custom forecast model using [algorithm name].

    This model implements [brief description of approach]. It is particularly
    well-suited for [use case description].

    Key Features:
    - Feature 1
    - Feature 2
    - Feature 3

    Training Strategy:
    - Describe how model is trained
    - Cross-validation approach
    - Any special considerations

    References:
    - Paper or documentation links
    - Algorithm references

    Examples:
        >>> model = MyCustomModel(
        ...     data=discharge_data,
        ...     static_data=basin_features,
        ...     general_config=config1,
        ...     model_config=config2,
        ...     feature_config=config3,
        ...     path_config=config4
        ... )
        >>> hindcast = model.calibrate_model_and_hindcast()
        >>> forecast = model.predict_operational()
    """
```

## Common Patterns

### Pattern 1: Using FeatureExtractor

```python
from lt_forecasting.scr.FeatureExtractor import StreamflowFeatureExtractor

extractor = StreamflowFeatureExtractor(
    feature_configs=self.feature_config,
    prediction_horizon=30,
    offset=30
)

features_df = extractor.create_all_features(self.data)
target = extractor.create_target(self.data)
```

### Pattern 2: Using Preprocessing Artifacts

```python
from lt_forecasting.scr.FeatureProcessingArtifacts import (
    process_training_data,
    process_test_data,
    save_artifacts_for_production,
    load_artifacts_for_production,
)

# Training
artifacts, X_train, y_train = process_training_data(
    df=features_df,
    target_col='Q_target',
    config=self.general_config
)

# Production
artifacts = load_artifacts_for_production(model_path)
X_processed = process_test_data(features_df, artifacts)
```

### Pattern 3: Multi-Basin Handling

```python
# Combine all basins
basins = self.data['code'].unique()
all_data = []

for basin in basins:
    basin_data = self.data[self.data['code'] == basin]
    basin_data['basin_id'] = basin
    all_data.append(basin_data)

combined_data = pd.concat(all_data, ignore_index=True)
```

## Checklist

Before submitting your new model:

- [ ] Inherits from `BaseForecastModel`
- [ ] Implements all required abstract methods
- [ ] Has configuration files in `example_config/`
- [ ] Includes logging statements
- [ ] Handles errors gracefully
- [ ] Saves and loads models correctly
- [ ] Has unit tests
- [ ] Has docstrings for class and methods
- [ ] Works with existing scripts
- [ ] Follows project code style (ruff format)
- [ ] Updated README.md if needed

## References

- **Base Class**: `lt_forecasting/forecast_models/base_class.py`
- **Example Implementations**:
  - `lt_forecasting/forecast_models/LINEAR_REGRESSION.py` (simple approach)
  - `lt_forecasting/forecast_models/SciRegressor.py` (advanced approach)
  - `lt_forecasting/forecast_models/deep_models/uncertainty_mixture.py` (deep learning)
- **Feature Engineering**: `docs/feature_engineering.md`
- **Model Descriptions**: `docs/model_description.md`
