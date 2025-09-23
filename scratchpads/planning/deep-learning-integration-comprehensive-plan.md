# Comprehensive Deep Learning Integration Plan

## Objective
Integrate deep learning models into the monthly discharge forecasting workflow with a clean, maintainable structure that allows easy addition of new models while maintaining compatibility with existing workflows.

## Context
The current codebase has legacy deep learning code in `deep_scr/` that needs to be restructured and integrated into the production workflow. The goal is to create a unified interface for both traditional ML models (SciRegressor) and deep learning models, supporting both standalone forecasting and meta-learning approaches.

## Plan

### 1. Project Structure Reorganization

#### A. Move and restructure deep learning components:
```
monthly_forecasting/
├── forecast_models/
│   ├── deep_models/                    # NEW: Deep learning models directory
│   │   ├── __init__.py
│   │   ├── deep_regressor.py          # Main deep learning forecaster class
│   │   ├── deep_meta_learner.py       # Meta-learning variant  
│   │   ├── architectures/             # Neural network architectures
│   │   │   ├── __init__.py
│   │   │   ├── lstm_models.py         # LSTM variants
│   │   │   ├── cnn_lstm_models.py     # CNN-LSTM models
│   │   │   ├── tide_models.py         # TiDE architecture
│   │   │   ├── tsmixer_models.py      # TSMixer architecture
│   │   │   └── mamba_models.py        # Mamba architecture
│   │   ├── losses/                    # Loss functions
│   │   │   ├── __init__.py
│   │   │   ├── quantile_loss.py
│   │   │   └── asymmetric_laplace_loss.py
│   │   └── utils/                     # Deep learning utilities
│   │       ├── __init__.py
│   │       ├── data_utils.py          # Dataset classes
│   │       ├── callbacks.py           # Training callbacks
│   │       └── lightning_base.py      # Base Lightning module
│   ├── base_class.py                  # Existing base class
│   └── ...                            # Other existing models
```

#### B. Update dependencies in pyproject.toml:
- Add PyTorch, PyTorch Lightning, and other deep learning dependencies

### 2. Core Deep Learning Infrastructure

#### A. Enhanced Dataset Class (`deep_models/utils/data_utils.py`):
- Generic dataset supporting the required input structure:
  - `x_past`: (batch, past_time_steps, past_features)
  - `x_nan_mask`: (batch, past_time_steps, past_features) 
  - `x_future`: (batch, future_time_steps, future_features)
  - `x_now`: (batch, 1, now_features)
  - `x_static`: (batch, static_features)
- Configurable NaN handling (mask or drop)
- Integration with existing FeatureExtractor

#### B. Base Deep Learning Classes:
- `DeepRegressor`: Main class inheriting from `BaseForecastModel`
- `DeepMetaLearner`: Meta-learning variant 
- `LightningForecastBase`: PyTorch Lightning base module

### 3. Model Architecture Implementation

#### A. Implement neural network backbones:
- **LSTM Models**: Basic LSTM, Bidirectional LSTM, Stacked LSTM
- **CNN-LSTM**: Convolutional layers followed by LSTM
- **TiDE**: Time-series Dense Encoder for long-term forecasting
- **TSMixer**: Transformer-based time series model
- **Mamba**: State-space model for sequence modeling

#### B. Specialized Models:
- **AL Uncertainty Model**: Asymmetric Laplace uncertainty estimation
- **DeepForecaster**: Direct forecasting with uncertainty quantification

### 4. Integration with Existing Workflow

#### A. Standardized Interface Implementation:
- `predict_operational()`: Load models, preprocess data, generate predictions
- `calibrate_model_and_hindcast()`: LOOCV training and evaluation
- `tune_hyperparameters()`: Hyperparameter optimization using Optuna
- `save_model()` / `load_model()`: Model persistence

#### B. Data Processing Integration:
- Reuse existing `FeatureExtractor` and `FeatureProcessingArtifacts`
- Adapt preprocessing pipeline for deep learning requirements
- Maintain compatibility with current evaluation pipeline

### 5. Training and Evaluation Framework

#### A. LOOCV Implementation:
- Year-based cross-validation
- Early stopping using validation split
- Consistent with existing SciRegressor approach

#### B. Hyperparameter Tuning:
- Optuna integration for architecture and training hyperparameters
- Tunable parameters: hidden_size, learning_rate, dropout, weight_decay, architecture choice

#### C. Operational Prediction:
- Model loading and inference pipeline
- Handling of missing data and edge cases
- Consistent output format with existing models

### 6. Configuration and Extensibility

#### A. Model Configuration System:
- JSON-based configuration following existing patterns
- Support for different architectures and hyperparameters
- Easy addition of new model types

#### B. Modular Architecture Design:
- Plugin-style architecture registration
- Easy addition of new neural network backbones
- Consistent interface for all deep learning models

### 7. Testing and Validation

#### A. Comprehensive Test Suite:
- Unit tests for data loading and preprocessing
- Integration tests with existing evaluation pipeline
- Functionality tests for each model architecture

#### B. Performance Benchmarking:
- Comparison with existing SciRegressor models
- Evaluation metrics integration with current framework

## Implementation Notes

### Data Structure Requirements
At prediction time t, the system needs to handle:
- **Past time steps**: t-lookback : t
- **Future time steps**: t : t+future_steps  
- **Current data**: t

Input tensors:
- `x_past`: (batch, past_time_steps, past_features) - past discharge, P, T, past predictions
- `x_nan_mask`: (batch, past_time_steps, past_features) - binary mask for missing features
- `x_future`: (batch, future_time_steps, future_vars) - weather forecast, temporal features
- `x_now`: (batch, 1, now_vars) - current predictions/errors from other models
- `x_static`: static basin features

### Key Design Principles
1. **Maintainability**: Clean separation of concerns, modular design
2. **Extensibility**: Easy to add new architectures and model types
3. **Consistency**: Unified interface across all model types
4. **Integration**: Seamless compatibility with existing workflows
5. **Performance**: Efficient training and inference pipelines

## Testing Strategy

### Unit Tests
- Data loading and preprocessing components
- Individual model architectures
- Loss function implementations

### Integration Tests  
- End-to-end training and prediction workflows
- Compatibility with existing evaluation pipeline
- Model saving/loading functionality

### Functionality Tests
- LOOCV cross-validation
- Hyperparameter tuning
- Operational prediction pipeline

## Review Points

1. **Architecture Design**: Ensure the modular structure supports easy extension
2. **Data Pipeline**: Verify compatibility with existing FeatureExtractor
3. **Performance**: Benchmark against existing SciRegressor models
4. **Configuration**: Validate JSON config system works for all model types
5. **Testing**: Comprehensive coverage of all components
6. **Documentation**: Clear examples of how to add new model architectures

This plan provides a comprehensive roadmap for integrating deep learning models while maintaining the existing workflow's strengths and ensuring future extensibility.