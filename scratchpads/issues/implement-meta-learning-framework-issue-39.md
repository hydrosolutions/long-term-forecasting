# Meta-Learning Framework Implementation - Issue #39

## Objective
Implement a comprehensive meta-learning system that combines multiple base model predictions through intelligent ensemble weighting and advanced meta-modeling techniques.

## Context
- **GitHub Issue**: https://github.com/hydrosolutions/monthly_forecasting/issues/39
- **Current State**: Basic ensemble averaging exists in `dev_tools/evaluation/ensemble_builder.py`
- **Need**: Production-ready meta-learning framework with intelligent weighting
- **Challenge**: Must avoid dependencies on `dev_tools/` for operational deployment

## Analysis Summary

### Current Infrastructure
- **Base Models**: `BaseForecastModel` abstract class provides interface
- **Tree Models**: `SciRegressor` supports XGBoost, LightGBM, CatBoost
- **Metrics**: `dev_tools/eval_scr/metric_functions.py` (needs production version)
- **LOOCV**: Leave-One-Year-Out cross-validation framework exists
- **Ensemble**: Basic averaging in `dev_tools/evaluation/ensemble_builder.py`

### Issue Requirements Analysis

#### Phase 1: Historical Performance-Based Meta-Learning (HistMeta)
**Priority**: High (significant improvement, moderate complexity)
- Performance-based weighting using historical metrics (RMSE, R2, MAE, NSE)
- Basin-specific weighting (different model weights per basin)
- Temporal weighting (seasonal/monthly performance patterns)
- LOOCV integration for proper cross-validation

#### Phase 2: Advanced Meta-Model Framework
**Priority**: Very High (substantial improvement, higher complexity)
- Input features: base predictions + performance metrics + basin characteristics + temporal context
- Meta-model options: GBT (recommended), MLP, SVR
- Sophisticated feature engineering for meta-learning
- Integration with existing SciRegressor infrastructure

#### Phase 3: Distributional Meta-Learning (Advanced)
**Priority**: High (cutting-edge capability, high complexity)
- Neural network for asymmetric Laplace distribution parameters
- Uncertainty quantification and prediction intervals
- Flood probability estimation capabilities

## Implementation Plan

### Phase 1: Production Infrastructure Setup (Week 1)

#### Task 1.1: Create Production Meta-Learning Module Structure
- [x] Create `monthly_forecasting/forecast_models/meta_learners/` directory
- [x] Implement `__init__.py` for module initialization
- [x] Create `base_meta_learner.py` with abstract base class
- [x] Plan `historical_meta_learner.py` implementation

#### Task 1.2: Create Production Evaluation Utilities
- [x] Create `monthly_forecasting/scr/evaluation_utils.py` (production version of eval_scr)
- [x] Implement core metrics: R2, RMSE, MAE, NSE, KGE, Bias
- [x] Add derived metrics: NRMSE, MAPE, PBIAS
- [x] Ensure robust error handling and NaN management

#### Task 1.3: Create Meta-Learning Support Utilities
- [x] Create `monthly_forecasting/scr/meta_utils.py` for meta-learning utilities
- [x] Create `monthly_forecasting/scr/performance_metrics.py` for performance tracking
- [x] Create `monthly_forecasting/scr/ensemble_utils.py` for production ensemble creation

### Phase 2: Historical Meta-Learning Implementation (Week 2)

#### Task 2.1: Implement HistoricalMetaLearner Class
- [ ] Inherit from `BaseForecastModel`
- [ ] Implement `compute_weights()` method for historical performance weighting
- [ ] Support basin-specific and temporal weighting strategies
- [ ] Integration with LOOCV framework

#### Task 2.2: Performance-Based Weighting System
- [ ] Implement `calculate_historical_performance()` method
- [ ] Support multiple weighting strategies (inverse error, rank-based, exponential)
- [ ] Handle missing performance data gracefully
- [ ] Create weight normalization and validation

#### Task 2.3: Basin-Specific and Temporal Weighting
- [ ] Implement per-basin performance tracking
- [ ] Support seasonal/monthly performance patterns
- [ ] Create temporal weight smoothing algorithms
- [ ] Add basin similarity-based weight transfer

### Phase 3: Advanced Meta-Model Framework (Week 3-4)

#### Task 3.1: Implement AdvancedMetaLearner Class
- [ ] Create `advanced_meta_learner.py` with multiple meta-model support
- [ ] Support GBT, MLP, SVR meta-models
- [ ] Rich feature engineering (base predictions + context)
- [ ] Integration with existing SciRegressor infrastructure

#### Task 3.2: Meta-Feature Engineering
- [ ] Combine base model predictions with performance context
- [ ] Add basin characteristics (static features)
- [ ] Include temporal context (season, trends)
- [ ] Create prediction confidence/uncertainty features

#### Task 3.3: Meta-Model Training and Validation
- [ ] Implement meta-model training pipeline
- [ ] LOOCV validation for meta-models
- [ ] Hyperparameter tuning for meta-models
- [ ] Model selection and ensemble strategies

### Phase 4: Testing and Validation (Week 5)

#### Task 4.1: Comprehensive Testing Suite
- [ ] Unit tests for all meta-learner classes
- [ ] Integration tests with existing model evaluation pipeline
- [ ] Performance benchmarking against simple ensemble baseline
- [ ] LOOCV validation tests

#### Task 4.2: Performance Validation
- [ ] Compare against simple averaging baseline
- [ ] Measure 5-15% RMSE improvement target
- [ ] Analyze per-basin and seasonal performance
- [ ] Test extreme event prediction improvements

## Technical Implementation Details

### Directory Structure (Production-Ready)
```
monthly_forecasting/
├── forecast_models/
│   ├── meta_learners/
│   │   ├── __init__.py
│   │   ├── base_meta_learner.py
│   │   ├── historical_meta_learner.py
│   │   ├── advanced_meta_learner.py
│   │   └── distributional_meta_learner.py (Phase 3)
├── scr/
│   ├── evaluation_utils.py      # Production metrics
│   ├── meta_utils.py            # Meta-learning utilities
│   ├── performance_metrics.py   # Performance tracking
│   └── ensemble_utils.py        # Production ensemble creation
tests/
└── meta_learning/
    ├── test_meta_learners.py
    ├── test_meta_utils.py
    └── test_ensemble_utils.py
```

### Key Design Decisions

#### 1. Production-Ready Architecture
- **NO dependencies on `dev_tools/`**: All functionality recreated in production codebase
- **Self-contained evaluation**: New `evaluation_utils.py` replaces `dev_tools/eval_scr/`
- **Operational readiness**: All modules designed for production deployment

#### 2. Meta-Learning Configuration
```python
meta_learning_config = {
    "enabled": True,
    "type": "historical",  # or "advanced", "distributional"
    "base_models": ["xgb", "lgbm", "catboost"],
    "performance_metric": "rmse",
    "weighting_strategy": "basin_temporal",
    "meta_model_type": "gbt",
    "distributional": False
}
```

#### 3. Integration Points
- **Base Model Integration**: Extend `BaseForecastModel` interface
- **Evaluation Integration**: Use NEW production `evaluation_utils.py`
- **Ensemble Integration**: Build NEW production `ensemble_utils.py`
- **LOOCV Integration**: Leverage existing cross-validation framework

### Expected Performance Improvements

#### Quantitative Targets
- **5-15% RMSE improvement** over simple ensemble averaging
- **Better extreme event prediction** through intelligent weighting
- **Improved seasonal performance** through temporal weighting
- **Basin-specific optimization** through localized weighting

#### Qualitative Benefits
- **Interpretable ensemble behavior** through weight analysis
- **Robust performance** across different hydrological conditions
- **Adaptive weighting** based on model performance patterns
- **Operational deployment ready** with production-grade code

## Testing Strategy

### Unit Tests
- Test performance weight calculation accuracy
- Test meta-feature creation and validation
- Test LOOCV integration correctness
- Test base model integration

### Integration Tests
- Test full meta-learning pipeline
- Test with different base model combinations
- Test operational prediction mode
- Performance benchmarking vs. simple ensemble

### Validation Approach
- Use existing LOOCV framework for validation
- Compare against simple averaging baseline
- Analyze performance improvements per basin/season
- Test robustness across different data conditions

## Success Criteria

### Phase 1 (Historical Meta-Learning)
- [x] Production-ready module structure created
- [x] Evaluation utilities implemented and tested
- [ ] HistoricalMetaLearner class fully implemented
- [ ] Performance-based weighting system working
- [ ] LOOCV integration validated

### Phase 2 (Advanced Meta-Learning)
- [ ] AdvancedMetaLearner class implemented
- [ ] Rich meta-feature engineering working
- [ ] Multiple meta-model types supported
- [ ] Hyperparameter tuning integrated

### Phase 3 (Validation and Testing)
- [ ] Comprehensive test suite passing
- [ ] Performance improvements demonstrated
- [ ] Operational prediction workflow supported
- [ ] Documentation and examples complete

### Overall Success Metrics
- **ZERO dependencies on `dev_tools/` modules**
- **5-15% RMSE improvement over baseline**
- **All existing tests continue to pass**
- **Operational deployment readiness**
- **Comprehensive documentation and testing**

## Next Steps

1. **Complete Phase 1 Implementation**: Finish HistoricalMetaLearner implementation
2. **Validate Performance**: Test against simple ensemble baseline
3. **Implement Phase 2**: Advanced meta-model framework
4. **Comprehensive Testing**: Full test suite and validation
5. **Documentation**: Complete usage examples and documentation

## Review Points

### Technical Review
- Code quality and maintainability
- Integration with existing codebase
- Performance and efficiency considerations
- Error handling and edge cases

### Scientific Review
- Meta-learning algorithm correctness
- Performance improvement validation
- Statistical significance of improvements
- Interpretability of ensemble weights

### Operational Review
- Production deployment readiness
- Operational prediction workflow integration
- Configuration management
- Monitoring and logging capabilities

This scratchpad will be updated as implementation progresses and new insights are gained.