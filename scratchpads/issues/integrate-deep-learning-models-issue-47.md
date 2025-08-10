# Integrate Deep Learning Models into Monthly Forecasting Workflow - Issue 47

## Objective
Integrate deep learning models into the monthly discharge forecasting workflow with a clean, maintainable structure that allows easy addition of new models while maintaining compatibility with existing workflows.

## Context
- **GitHub Issue**: https://github.com/hydrosolutions/issues/47
- **Background**: Legacy deep learning code exists in `deep_scr/` that needs restructuring
- **Goal**: Create unified interface for both traditional ML (SciRegressor) and deep learning models
- **Prior Work**: Comprehensive plan already exists in `scratchpads/planning/deep-learning-integration-comprehensive-plan.md`

## Plan

### Phase 1: Infrastructure Setup ✅
- [x] Analyze existing codebase and legacy deep learning components
- [x] Review interface requirements from `BaseForecastModel`
- [x] Document data structure requirements from legacy code
- [ ] Check and update PyTorch/Lightning dependencies in pyproject.toml
- [ ] Create feature branch: `feature/issue-47-deep-learning-integration`

### Phase 2: Core Structure Creation 🔄
- [ ] Create new directory structure under `forecast_models/deep_models/`
  - [ ] `deep_models/__init__.py`
  - [ ] `deep_models/deep_regressor.py` - Main forecaster class
  - [ ] `deep_models/deep_meta_learner.py` - Meta-learning variant
  - [ ] `deep_models/architectures/` - Neural network architectures
  - [ ] `deep_models/losses/` - Loss functions  
  - [ ] `deep_models/utils/` - Training utilities and datasets

### Phase 3: Data Pipeline Implementation
- [ ] Create enhanced dataset classes (`deep_models/utils/data_utils.py`)
  - [ ] Generic dataset supporting multi-input structure
  - [ ] NaN mask handling for missing features
  - [ ] Integration with existing `FeatureExtractor`
  - [ ] Data structure support:
    - `x_past`: (batch, past_time_steps, past_features)
    - `x_nan_mask`: (batch, past_time_steps, past_features) 
    - `x_future`: (batch, future_time_steps, future_vars)
    - `x_now`: (batch, 1, now_vars)
    - `x_static`: static basin features

### Phase 4: Base Model Classes
- [ ] Implement `DeepRegressor` class inheriting from `BaseForecastModel`
  - [ ] `predict_operational()` method
  - [ ] `calibrate_model_and_hindcast()` method
  - [ ] `tune_hyperparameters()` method
  - [ ] `save_model()` / `load_model()` methods
- [ ] Implement `DeepMetaLearner` for meta-learning approaches
- [ ] Create PyTorch Lightning base module (`utils/lightning_base.py`)

### Phase 5: Model Architecture Migration
- [ ] Migrate and restructure existing architectures:
  - [ ] `AL_UncertaintyNet` -> `architectures/uncertainty_models.py`
  - [ ] LSTM variants -> `architectures/lstm_models.py`
  - [ ] CNN-LSTM -> `architectures/cnn_lstm_models.py`
  - [ ] Prepare structure for TiDE, TSMixer, Mamba models
- [ ] Migrate loss functions from `deep_scr/loss_function.py`
- [ ] Migrate training utilities from `deep_scr/callbacks_helper.py`

### Phase 6: Training & Evaluation Logic
- [ ] Implement LOOCV (yearly cross-validation) logic
- [ ] Implement hyperparameter tuning with Optuna integration
- [ ] Create model persistence (save/load) functionality
- [ ] Ensure compatibility with existing evaluation pipeline

### Phase 7: Testing & Integration
- [ ] Create comprehensive test suite:
  - [ ] Unit tests for data loading and preprocessing
  - [ ] Integration tests with existing pipeline
  - [ ] Functionality tests for each architecture
- [ ] Performance benchmarking against SciRegressor
- [ ] End-to-end workflow testing

### Phase 8: Documentation & Cleanup
- [ ] Update configuration examples
- [ ] Create usage documentation
- [ ] Clean up legacy `deep_scr/` directory (after migration)
- [ ] Update main README if needed

## Implementation Notes

### Key Design Principles
1. **Unified Interface**: All deep learning models inherit from `BaseForecastModel`
2. **Modular Architecture**: Easy to add new neural network backbones
3. **Data Compatibility**: Seamless integration with existing `FeatureExtractor`
4. **Configuration-Driven**: JSON-based configuration following existing patterns
5. **Extensible**: Plugin-style architecture for new model types

### Data Flow Requirements
- Past time steps: t-lookback : t
- Future time steps: t : t+future_steps  
- Current data: t
- Static basin features: time-invariant

### Legacy Components to Migrate
- `AL_UncertaintyNet.py` - Asymmetric Laplace uncertainty model
- `data_class.py` - Dataset handling logic
- `meta_base.py` - Base meta-learning class  
- `train_eval.py` - Training/evaluation procedures
- `loss_function.py` - Custom loss functions
- `callbacks_helper.py` - Training callbacks

## Testing Strategy

### Unit Tests
- Data loading and preprocessing components
- Individual model architectures
- Loss function implementations
- Configuration handling

### Integration Tests  
- End-to-end training and prediction workflows
- Compatibility with existing evaluation pipeline
- Model saving/loading functionality
- LOOCV cross-validation

### Functionality Tests
- Hyperparameter tuning pipeline
- Operational prediction workflow
- Multi-model ensemble compatibility

## Success Criteria

1. ✅ Deep learning models follow same interface as `SciRegressor`
2. ✅ Models can be easily configured through JSON
3. ✅ Full integration with existing evaluation pipeline
4. ✅ Comprehensive test coverage (>90%)
5. ✅ Performance comparable to or better than existing models
6. ✅ Easy extensibility for new architectures

## Review Points

1. **Architecture Design**: Ensure modular structure supports easy extension
2. **Data Pipeline**: Verify compatibility with existing `FeatureExtractor`  
3. **Performance**: Benchmark against existing models
4. **Configuration**: Validate JSON config system works for all model types
5. **Testing**: Comprehensive coverage of all components
6. **Documentation**: Clear examples for adding new architectures

## Dependencies

Current requirements to verify/add:
- PyTorch
- PyTorch Lightning
- Additional deep learning libraries as needed

This implementation plan provides a comprehensive roadmap for integrating deep learning models while maintaining existing workflow strengths and ensuring future extensibility.