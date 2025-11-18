# Comprehensive SciRegressor Testing - Issue #10

## Objective
Implement extensive testing for the SciRegressor models (XGBoost, LightGBM, CatBoost) with comprehensive coverage of hyperparameter tuning, different preprocessing steps, calibration, hindcast, and operational prediction workflows.

## Context
- **Issue**: https://github.com/hydrosolutions/lt_forecasting/issues/10
- **Current Status**: Basic tests exist (6 pytest functions) but comprehensive coverage is missing
- **Target**: 64 total test functions covering all combinations of models, preprocessing methods, and workflow components

## Current Test Analysis

### Existing Tests (6 functions):
1. `test_sciregressor_initialization` - Basic model initialization
2. `test_sciregressor_feature_extraction` - Feature extraction functionality
3. `test_sciregressor_calibration_hindcast` - Calibration and hindcast workflow
4. `test_sciregressor_operational_forecast` - Operational forecasting
5. `test_sciregressor_hyperparameter_tuning` - Hyperparameter tuning
6. `test_sciregressor_configuration_loading` - Configuration loading

### Missing Coverage:
- **Model Types**: Only basic coverage, need specific tests for XGBoost, LightGBM, CatBoost
- **Preprocessing Methods**: Only standard normalization tested, need:
  - No normalization
  - Global normalization
  - Per-basin normalization
  - Long-term mean scaling
- **Workflow Components**: Limited coverage of individual components
- **Integration Tests**: No end-to-end workflow testing
- **Cross-Model Tests**: No ensemble/comparison testing

## Implementation Plan

### Phase 1: Extend Current Test Framework (Days 1-2)
- [ ] Modify existing `SciRegressorTester` class to support parameterized testing
- [ ] Add preprocessing method configuration support
- [ ] Create test data generators for different normalization scenarios
- [ ] Add model-specific configuration handling

### Phase 2: Individual Component Tests (Days 3-8)
Create 48 individual component test functions:

#### 2.1 Hyperparameter Tuning Tests (12 functions)
- [ ] `test_xgb_hyperparameter_tuning_no_normalization()`
- [ ] `test_xgb_hyperparameter_tuning_global_normalization()`
- [ ] `test_xgb_hyperparameter_tuning_per_basin_normalization()`
- [ ] `test_xgb_hyperparameter_tuning_long_term_mean_scaling()`
- [ ] `test_lgbm_hyperparameter_tuning_no_normalization()`
- [ ] `test_lgbm_hyperparameter_tuning_global_normalization()`
- [ ] `test_lgbm_hyperparameter_tuning_per_basin_normalization()`
- [ ] `test_lgbm_hyperparameter_tuning_long_term_mean_scaling()`
- [ ] `test_catboost_hyperparameter_tuning_no_normalization()`
- [ ] `test_catboost_hyperparameter_tuning_global_normalization()`
- [ ] `test_catboost_hyperparameter_tuning_per_basin_normalization()`
- [ ] `test_catboost_hyperparameter_tuning_long_term_mean_scaling()`

#### 2.2 Calibration Tests (12 functions)
- [ ] `test_xgb_calibration_no_normalization()`
- [ ] `test_xgb_calibration_global_normalization()`
- [ ] `test_xgb_calibration_per_basin_normalization()`
- [ ] `test_xgb_calibration_long_term_mean_scaling()`
- [ ] `test_lgbm_calibration_no_normalization()`
- [ ] `test_lgbm_calibration_global_normalization()`
- [ ] `test_lgbm_calibration_per_basin_normalization()`
- [ ] `test_lgbm_calibration_long_term_mean_scaling()`
- [ ] `test_catboost_calibration_no_normalization()`
- [ ] `test_catboost_calibration_global_normalization()`
- [ ] `test_catboost_calibration_per_basin_normalization()`
- [ ] `test_catboost_calibration_long_term_mean_scaling()`

#### 2.3 Hindcast Tests (12 functions)
- [ ] `test_xgb_hindcast_no_normalization()`
- [ ] `test_xgb_hindcast_global_normalization()`
- [ ] `test_xgb_hindcast_per_basin_normalization()`
- [ ] `test_xgb_hindcast_long_term_mean_scaling()`
- [ ] `test_lgbm_hindcast_no_normalization()`
- [ ] `test_lgbm_hindcast_global_normalization()`
- [ ] `test_lgbm_hindcast_per_basin_normalization()`
- [ ] `test_lgbm_hindcast_long_term_mean_scaling()`
- [ ] `test_catboost_hindcast_no_normalization()`
- [ ] `test_catboost_hindcast_global_normalization()`
- [ ] `test_catboost_hindcast_per_basin_normalization()`
- [ ] `test_catboost_hindcast_long_term_mean_scaling()`

#### 2.4 Operational Prediction Tests (12 functions)
- [ ] `test_xgb_operational_prediction_no_normalization()`
- [ ] `test_xgb_operational_prediction_global_normalization()`
- [ ] `test_xgb_operational_prediction_per_basin_normalization()`
- [ ] `test_xgb_operational_prediction_long_term_mean_scaling()`
- [ ] `test_lgbm_operational_prediction_no_normalization()`
- [ ] `test_lgbm_operational_prediction_global_normalization()`
- [ ] `test_lgbm_operational_prediction_per_basin_normalization()`
- [ ] `test_lgbm_operational_prediction_long_term_mean_scaling()`
- [ ] `test_catboost_operational_prediction_no_normalization()`
- [ ] `test_catboost_operational_prediction_global_normalization()`
- [ ] `test_catboost_operational_prediction_per_basin_normalization()`
- [ ] `test_catboost_operational_prediction_long_term_mean_scaling()`

### Phase 3: End-to-End Workflow Tests (Days 9-10)
Create 12 complete workflow tests:

#### 3.1 Complete Workflow Tests (12 functions)
- [ ] `test_xgb_complete_workflow_no_normalization()`
- [ ] `test_xgb_complete_workflow_global_normalization()`
- [ ] `test_xgb_complete_workflow_per_basin_normalization()`
- [ ] `test_xgb_complete_workflow_long_term_mean_scaling()`
- [ ] `test_lgbm_complete_workflow_no_normalization()`
- [ ] `test_lgbm_complete_workflow_global_normalization()`
- [ ] `test_lgbm_complete_workflow_per_basin_normalization()`
- [ ] `test_lgbm_complete_workflow_long_term_mean_scaling()`
- [ ] `test_catboost_complete_workflow_no_normalization()`
- [ ] `test_catboost_complete_workflow_global_normalization()`
- [ ] `test_catboost_complete_workflow_per_basin_normalization()`
- [ ] `test_catboost_complete_workflow_long_term_mean_scaling()`

### Phase 4: Cross-Model Integration Tests (Days 11-12)
Create 4 cross-model integration tests:

#### 4.1 Multi-Model Integration Tests (4 functions)
- [ ] `test_multi_model_ensemble_no_normalization()`
- [ ] `test_multi_model_ensemble_global_normalization()`
- [ ] `test_multi_model_ensemble_per_basin_normalization()`
- [ ] `test_multi_model_ensemble_long_term_mean_scaling()`

### Phase 5: Testing and Documentation (Days 13-14)
- [ ] Run complete test suite and fix any failures
- [ ] Update documentation
- [ ] Create PR and request review

## Technical Implementation Details

### Test Structure
```python
class ComprehensiveSciRegressorTester:
    """Extended tester for comprehensive SciRegressor testing."""
    
    def __init__(self, model_type: str, preprocessing_method: str):
        self.model_type = model_type
        self.preprocessing_method = preprocessing_method
        self.setup_test_environment()
    
    def setup_preprocessing_config(self):
        """Configure preprocessing based on method."""
        if self.preprocessing_method == "no_normalization":
            self.general_config["normalization_type"] = "none"
        elif self.preprocessing_method == "global_normalization":
            self.general_config["normalization_type"] = "standard"
            self.general_config["normalize_per_basin"] = False
        elif self.preprocessing_method == "per_basin_normalization":
            self.general_config["normalization_type"] = "standard"
            self.general_config["normalize_per_basin"] = True
        elif self.preprocessing_method == "long_term_mean_scaling":
            self.general_config["normalization_type"] = "long_term_mean"
    
    def setup_model_config(self):
        """Configure model based on type."""
        if self.model_type == "xgb":
            self.model_config = {"xgb": XGB_PARAMS}
        elif self.model_type == "lgbm":
            self.model_config = {"lgbm": LGBM_PARAMS}
        elif self.model_type == "catboost":
            self.model_config = {"catboost": CATBOOST_PARAMS}
```

### Preprocessing Method Configurations
```python
PREPROCESSING_CONFIGS = {
    "no_normalization": {
        "normalization_type": "none",
        "normalize_per_basin": False,
        "handle_na": "drop"
    },
    "global_normalization": {
        "normalization_type": "standard",
        "normalize_per_basin": False,
        "handle_na": "mean"
    },
    "per_basin_normalization": {
        "normalization_type": "standard",
        "normalize_per_basin": True,
        "handle_na": "mean"
    },
    "long_term_mean_scaling": {
        "normalization_type": "long_term_mean",
        "normalize_per_basin": False,
        "handle_na": "long_term_mean"
    }
}
```

### Model Configurations
```python
MODEL_CONFIGS = {
    "xgb": {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "objective": "reg:squarederror"
    },
    "lgbm": {
        "n_estimators": 50,
        "max_depth": 4,
        "learning_rate": 0.1,
        "objective": "regression"
    },
    "catboost": {
        "iterations": 50,
        "depth": 4,
        "learning_rate": 0.1,
        "loss_function": "RMSE"
    }
}
```

### Test Data Strategy
- Use consistent synthetic data across all tests
- Create separate test data generators for each preprocessing method
- Ensure realistic hydrology scenarios with proper temporal and spatial patterns
- Mock external dependencies consistently

### Validation Criteria
- **Model Training**: Verify models train without errors
- **Hyperparameter Optimization**: Check Optuna trials complete successfully
- **Calibration**: Validate Leave-One-Year-Out cross-validation
- **Predictions**: Ensure predictions are in valid range and format
- **Performance**: Verify R² scores and other metrics are reasonable
- **Preprocessing**: Validate different normalization methods work correctly

## Files to Modify/Create

### Main Test File
- `tests/test_sciregressor.py` - Extend with comprehensive test functions

### Supporting Files
- `tests/comprehensive_test_configs.py` - NEW - Configuration constants
- `tests/comprehensive_test_utils.py` - NEW - Utility functions
- `tests/README_SCIREGRESSOR_TESTS.md` - UPDATE - Documentation

### Test Organization
```
tests/
├── test_sciregressor.py                 # Main test file (extended)
├── comprehensive_test_configs.py        # Configuration constants
├── comprehensive_test_utils.py          # Utility functions
├── README_SCIREGRESSOR_TESTS.md        # Updated documentation
└── test_results/                       # Test output directory
    ├── model_performance_reports/
    └── test_artifacts/
```

## Success Criteria
- [ ] All 64 test functions implemented and passing
- [ ] 100% coverage for XGBoost, LightGBM, CatBoost models
- [ ] All 4 preprocessing methods tested comprehensively
- [ ] Complete workflow testing from tuning to prediction
- [ ] Cross-model integration testing
- [ ] Comprehensive documentation
- [ ] CI/CD integration ready

## Timeline
- **Days 1-2**: Framework extension
- **Days 3-8**: Individual component tests (48 functions)
- **Days 9-10**: End-to-end workflow tests (12 functions)
- **Days 11-12**: Cross-model integration tests (4 functions)
- **Days 13-14**: Testing and documentation

## Review Points
1. Test coverage completeness across all models and preprocessing methods
2. Proper mocking of external dependencies
3. Consistent test data generation
4. Performance validation criteria
5. Error handling and edge cases
6. Documentation quality and examples
7. CI/CD integration readiness