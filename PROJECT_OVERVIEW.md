# Monthly Forecasting Project - High-Level Overview

## Project Purpose
A machine learning system for predicting monthly river discharge across multiple basins, supporting both traditional statistical models and modern ensemble methods.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      Entry Points                            │
│  tune_and_calibrate_script.sh | calibration_script.sh       │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                   Core Scripts                               │
│     tune_hyperparams.py | calibrate_hindcast.py            │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Model Layer                                  │
│  ┌─────────────────────────┐  ┌──────────────────────────┐ │
│  │   Linear Regression     │  │      SciRegressor        │ │
│  │   (Per-basin models)    │  │  (Global tree models)    │ │
│  └─────────────────────────┘  └──────────────────────────┘ │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 Utility Layer                                │
│  ┌──────────┐ ┌──────────────┐ ┌────────────┐ ┌─────────┐ │
│  │   Data   │ │   Feature    │ │ Evaluation │ │  Tests  │ │
│  │ Loading  │ │ Engineering  │ │  Metrics   │ │  Suite  │ │
│  └──────────┘ └──────────────┘ └────────────┘ └─────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Key Components by Location

### 📁 Root Directory
- **Shell Scripts**: Orchestrate complete workflows
  - `tune_and_calibrate_script.sh`: Full pipeline execution
  - `calibration_script.sh`: Model training focus
- **Python Scripts**: Core functionality
  - `tune_hyperparams.py`: Hyperparameter optimization
  - `calibrate_hindcast.py`: Model training and evaluation
- **Configuration**: `log_config.py` for centralized logging

### 📁 forecast_models/
**What happens here**: Model definition and training logic
- `base_class.py`: Abstract interface all models implement
- `LINEAR_REGRESSION.py`: Traditional statistical approach
- `SciRegressor.py`: Modern ML ensemble methods

### 📁 scr/
**What happens here**: Data processing and feature engineering
- `data_loading.py`: Raw data ingestion
- `data_utils.py`: Preprocessing and transformations
- `FeatureExtractor.py`: Advanced feature creation
- `sci_utils.py`: Scientific computing utilities

### 📁 eval_scr/
**What happens here**: Model performance assessment
- `metric_functions.py`: R², RMSE, NSE calculations
- `eval_helper.py`: Evaluation workflow management

### 📁 tests/
**What happens here**: Quality assurance and validation
- Comprehensive unit and integration tests
- Configuration-driven test scenarios
- Performance benchmarking

### 📁 Output Directories
- `logs/`: Runtime logging information
- `tests_output/`: Test execution artifacts
- `catboost_info/`: CatBoost training logs

### 📁 Documentation
- `docs/`: System documentation
- `scratchpads/`: Development planning
- Individual README.md files per directory

## Workflow Summary

### 1. Data Flow
```
Raw CSV/Parquet → Data Loading → Preprocessing → Feature Engineering → Model Input
```

### 2. Model Training Flow
```
Prepared Data → Model Selection → Hyperparameter Tuning → Training → Validation
```

### 3. Evaluation Flow
```
Predictions → Metric Calculation → Performance Reports → Model Selection
```

## Model Types Comparison

| Aspect | Linear Regression | SciRegressor |
|--------|------------------|--------------|
| Scope | Per-basin models | Global model |
| Complexity | Low | High |
| Interpretability | High | Medium |
| Feature Engineering | Basic | Advanced |
| Training Time | Fast | Slower |
| Flexibility | Limited | High |

## Preprocessing Methods

1. **None**: Raw data (baseline)
2. **Standardize**: Zero mean, unit variance
3. **Min-Max**: Scale to [0,1]
4. **Monthly Bias**: Remove seasonal effects
5. **Long-term Mean**: Historical scaling

## Quick Start Commands

```bash
# Run complete pipeline
./tune_and_calibrate_script.sh

# Run tests
python -m pytest -ra

# Train specific model
python calibrate_hindcast.py --model linear_regression

# Activate environment
source "/Users/sandrohunziker/Documents/sapphire_venv/monthly_forecast/bin/activate"
```

## Key Design Principles

1. **Modularity**: Separate concerns for maintainability
2. **Extensibility**: Easy to add new models/features
3. **Reproducibility**: Configuration-driven execution
4. **Testability**: Comprehensive test coverage
5. **Performance**: Optimized for large datasets

## Technology Stack
- **Language**: Python 3.x
- **ML Libraries**: scikit-learn, XGBoost, CatBoost
- **Data Processing**: pandas, numpy
- **Testing**: pytest
- **Logging**: Python logging module

## Future Roadmap
- Deep learning integration
- Real-time prediction API
- Cloud deployment
- Enhanced visualization
- Uncertainty quantification