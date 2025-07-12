# Monthly Discharge Forecasting

Author: Sandro Hunziker

## Overview

The Monthly Discharge Forecasting System is a comprehensive machine learning pipeline for predicting river discharge at monthly timescales. It employs a modular architecture with multiple model types, advanced feature engineering, and ensemble methods to achieve robust forecasts.

For detailed system architecture and workflows, see:
- [Overview.md](Overview.md) - Complete system architecture with workflow diagrams
- [model_description.md](model_description.md) - Detailed model descriptions and guidelines

## Quick Start

### Installation

```bash
# Clone the repository
git clone [repository-url]
cd monthly_forecasting

# Install dependencies using uv
pip install uv
uv sync
```

### Running a Model

```bash
# Run calibration and hindcast
uv run calibrate_hindcast.py --config_path example_config/DUMMY_MODEL

# Run hyperparameter tuning
uv run tune_hyperparams.py --config_path example_config/DUMMY_MODEL

# Run evaluation pipeline
./run_evaluation_pipeline.sh
```

## General Concept

The system follows a modular approach where different forecasting classes implement similar interfaces but handle different model types:

1. **LINEAR_REGRESSION**: Statistical baseline with period-specific models
2. **SciRegressor**: Advanced ML models (XGBoost, LightGBM, CatBoost)

This modular design enables:
- Efficient ensemble creation (process data once, train multiple models)
- Flexible feature combinations
- Easy addition of new model types

Example ensemble strategy:
- Models: XGBoost, LightGBM, CatBoost
- Feature sets: 
  - F₁ ∈ (Q, P, T)
  - F₂ ∈ (Q, T, P, Snow data)
  - F₃ ∈ (GlacierMapper, Q, T, P)
- Result: 9 different models that can be ensembled



## Project Structure

### Core Components

```
monthly_forecasting/
├── scr/                        # Source code for data processing
│   ├── data_loading.py         # Data ingestion and merging
│   ├── data_utils.py           # Preprocessing and transformations
│   ├── FeatureExtractor.py     # Time-series feature engineering
│   ├── FeatureProcessingArtifacts.py  # Preprocessing state management
│   ├── sci_utils.py            # ML utilities
│   └── tree_utils.py           # Tree model utilities
│
├── forecast_models/            # Model implementations
│   ├── base_class.py          # Abstract base class
│   ├── LINEAR_REGRESSION.py   # Linear regression implementation
│   └── SciRegressor.py        # Tree-based models (XGB, LGBM, CatBoost)
│
├── eval_scr/                  # Evaluation utilities
│   ├── metric_functions.py    # Performance metrics
│   └── eval_helper.py         # Evaluation helpers
│
├── evaluation/                # Evaluation pipeline
│   ├── evaluate_pipeline.py   # Main evaluation orchestrator
│   ├── ensemble_builder.py    # Ensemble creation
│   └── prediction_loader.py   # Load model predictions
│
├── visualization/             # Dashboard and plotting
│   ├── dashboard.py          # Interactive dashboard
│   ├── dashboard_components.py # UI components
│   └── plotting_utils.py     # Plotting functions
│
├── example_config/           # Configuration templates
│   └── DUMMY_MODEL/         # Example configuration set
│
├── tests/                   # Test suite
│   ├── test_*.py           # Unit and integration tests
│   └── comprehensive_test_*.py  # Comprehensive test utilities
│
├── docs/                    # Documentation
│   ├── README.md           # Documentation index
│   └── *.md                # Various documentation files
│
├── scratchpads/            # Development notes and planning
│   ├── issues/            # Issue-specific work
│   └── planning/          # Feature planning
│
└── Main Scripts:
    ├── calibrate_hindcast.py    # Model training script
    ├── tune_hyperparams.py      # Hyperparameter optimization
    ├── test_evaluation_pipeline.py  # Evaluation testing
    └── *.sh                     # Shell scripts for workflows
```

### Configuration Files

Each model experiment requires configuration files in a dedicated directory:
- `data_paths.json` - Input data file paths
- `experiment_config.json` - Experiment setup and basins
- `feature_config.json` - Feature engineering parameters
- `general_config.json` - Model and processing settings
- `model_config.json` - Algorithm-specific hyperparameters



## Methods

### Calibration & Validation

The yearly leave-one-out cross-validation is used on all of the years except the last 3 available years. Those are left out as a final test set. So the predictions.csv set of each model is the prediction on these left out years + the test years. For meta-learning and cascade like models, we assume that those LOO-CV prediction represent how the model behave on unseen data. 

### Base Learner Models

We use a set of periodic Linear Regression Models. Predictos include features based on past discharge, precipitation, temperature and snow information from Snowmapper FSM based on different elevation zones. 
For the tree based models we can create a bunch of possible features also based on discharge, precipitation, temperature and snow information from Snowmapper FSM (lumped). Additionally we can use data from GlacierMapper as a earth observation based data source. Some tree based models can also take the predictions from the linear regressions as an additional input.

### Ensemble and Meta-Model

1. For the naive ensemble we have use just the naive mean of all the base predictos (Ensemble Mean).
2. Use a temporal meta-model - which should detect sharp drifts and changes from single model and ingore those - adjust final prediction based on past forecasts and some observations.
3. Uncertainty net: Uses context and the base-learner predictions to introduce a prediction interval (following the asymetric-laplace distribution). 


## Output format

$\textbf{predictions.csv}$ \
date | Q_model1 | Q_model2 | Q_model3 | Q_mean | valid_from | valid_to

Q_model corresponds to the prediction of a ensmeble member and Q_mean is the average over these models. \

$\textbf{predictions.csv}$  (for the meta model with uncertainty)\
date | Q_05| Q_10 | Q_50 | Q_90 | Q_95 | Q_mean | valid_from | valid_to


## Data Sources

The system integrates multiple data sources:

1. **Discharge Data**: Historical river discharge observations
2. **Forcing Data**: Temperature and precipitation
3. **Snow Data**: Snow water equivalent (SWE), height of snow (HS), runoff (ROF)
4. **Snow Cover Area (SCA)**: Satellite-based snow coverage
5. **Static Basin Characteristics**: Elevation, area, glacier coverage
6. **GlacierMapper**: Snow line altitude (SLA) and glacier features

## Feature Engineering

The system supports extensive feature engineering:

### Time-Series Features
- Rolling window statistics (mean, slope, peak-to-peak)
- Multiple window sizes and lag periods
- Period-based features (36 periods = 3 per month)

### Spatial Features
- Elevation band aggregation (configurable zones)
- Basin-specific characteristics
- Glacier-related features from GlacierMapper

### Normalization Options
- Global normalization
- Per-basin normalization
- Long-term mean scaling (period-based)
- Mixed normalization strategies

## Recent Updates

- **GlacierMapper Integration**: Added support for SLA features
- **Enhanced Elevation Bands**: Configurable number of elevation zones
- **Improved Scaling**: Period-based temporal grouping (36 periods)
- **Dashboard Visualization**: Interactive model evaluation dashboard

## Shell Scripts

- `calibration_script.sh` - Run model calibration
- `tune_and_calibrate_script.sh` - Hyperparameter tuning + calibration
- `run_evaluation_pipeline.sh` - Run evaluation pipeline
- `run_model_workflow.sh` - Complete model workflow

## Development

### Running Tests
```bash
uv run pytest -v
```

### Code Formatting
```bash
uv run ruff format
```

### Contributing
See [scratchpads/README.md](scratchpads/README.md) for development workflow and planning.

## Documentation

- [Overview.md](Overview.md) - System architecture and workflows
- [model_description.md](model_description.md) - Detailed model descriptions
- [docs/](docs/) - Additional documentation
- Component-specific READMEs in each directory

