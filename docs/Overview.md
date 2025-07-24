# Monthly Discharge Forecasting System Overview

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Data Flow](#data-flow)
3. [Core Components](#core-components)
4. [Model Families](#model-families)
5. [Workflow Overview](#workflow-overview)
6. [Recent Developments](#recent-developments)
7. [Integration Guide](#integration-guide)

## System Architecture

The Monthly Discharge Forecasting System is a comprehensive machine learning pipeline designed to predict river discharge at monthly timescales. The system employs a modular architecture that enables:

- Multiple model families with different approaches
- Advanced feature engineering from diverse data sources
- Ensemble methods for robust predictions
- Comprehensive evaluation and visualization

### High-Level Architecture Diagram

```mermaid
graph TB
    subgraph "Data Sources"
        DS1[Discharge Data]
        DS2[Forcing Data<br/>T, P]
        DS3[Snow Data<br/>SWE, HS, ROF]
        DS4[Snow Cover Area]
        DS5[Static Basin Data]
        DS6[GlacierMapper<br/>SLA, FSC]
    end

    subgraph "Data Processing Layer"
        DL[data_loading.py]
        DU[data_utils.py]
        FE[FeatureExtractor.py]
        FPA[FeatureProcessingArtifacts.py]
    end

    subgraph "Model Layer"
        subgraph "Model Families"
            BC[BaseCase Models]
            SCA[SCA_Based Models]
            SM[SnowMapper_Based]
            GM[GlacierMapper_Based]
        end
        
        subgraph "Model Types"
            LR[LinearRegression]
            SR[SciRegressor<br/>XGB, LGBM, CatBoost]
            ML[Meta-Learning<br/>Historical Performance<br/>Weighted Ensemble]
        end
    end

    subgraph "Evaluation & Output"
        EV[Evaluation Pipeline]
        ENS[Ensemble Builder]
        VIS[Dashboard Visualization]
    end

    DS1 & DS2 & DS3 & DS4 & DS5 & DS6 --> DL
    DL --> DU
    DU --> FE
    FE --> FPA
    FPA --> BC & SCA & SM & GM
    BC & SCA & SM & GM --> LR & SR
    LR & SR --> EV
    LR & SR --> ML
    ML --> EV
    EV --> ENS
    ENS --> VIS
```

## Data Flow

### 1. Data Ingestion
The system integrates multiple data sources through `scr/data_loading.py`:

- **Discharge Data**: Historical river discharge observations (m³/s)
- **Forcing Data**: Temperature (°C) and precipitation (mm)
- **Snow Data**: 
  - SWE (Snow Water Equivalent)
  - HS (Height of Snow)
  - ROF (Runoff)
- **Snow Cover Area (SCA)**: Satellite-based snow coverage percentages
- **Static Basin Characteristics**: Elevation range, area, glacier fraction
- **GlacierMapper Data**: Snow line altitude (SLA), fractional snow cover (FSC)

### 2. Feature Engineering Pipeline

```mermaid
graph LR
    subgraph "Raw Data"
        RD[Time Series Data]
        SD[Static Data]
    end

    subgraph "Feature Creation"
        TS[Time Series Features<br/>- Rolling statistics<br/>- Lags<br/>- Slopes]
        SP[Spatial Features<br/>- Elevation bands<br/>- Basin characteristics]
        GM[GlacierMapper Features<br/>- SLA variations<br/>- Melt potential]
    end

    subgraph "Preprocessing"
        NRM[Normalization<br/>- Global<br/>- Per-basin<br/>- Period-based]
        IMP[Imputation<br/>- Missing values<br/>- Long-term means]
    end

    RD --> TS
    RD & SD --> SP
    RD --> GM
    TS & SP & GM --> NRM
    NRM --> IMP
```

### 3. Model Training & Prediction

The system supports two main model types:

#### LinearRegression Models
- Period-specific models (6 periods per month)
- Dynamic feature selection based on correlation
- Leave-one-year-out cross-validation

#### SciRegressor Models
- Global models trained on all basins
- Ensemble methods: XGBoost, LightGBM, CatBoost
- Advanced hyperparameter optimization with Optuna

#### Meta-Learning Models
- Historical performance-weighted ensemble framework
- Intelligently combines predictions from multiple base models
- Learns from historical performance patterns per basin and period
- Provides robust fallback mechanisms for insufficient data

### 4. Evaluation & Ensemble Creation

```mermaid
graph TD
    subgraph "Individual Models"
        M1[Model 1]
        M2[Model 2]
        MN[Model N]
    end

    subgraph "Ensemble Levels"
        FE[Family Ensembles<br/>- BaseCase<br/>- SCA_Based<br/>- SnowMapper_Based<br/>- GlacierMapper_Based]
        GE[Global Ensemble]
    end

    subgraph "Evaluation"
        MET[Metrics Calculation<br/>- R², RMSE, NSE<br/>- KGE, MAE, Bias]
        AGG[Aggregation Levels<br/>- Overall<br/>- Per-basin<br/>- Per-month<br/>- Per-basin-month]
    end

    M1 & M2 & MN --> FE
    FE --> GE
    M1 & M2 & MN & FE & GE --> MET
    MET --> AGG
```

## Core Components

### 1. SCR Module (`scr/`)
Core utilities for data processing and feature engineering:

- **data_loading.py**: Unified data loading interface
- **data_utils.py**: Preprocessing, normalization, elevation band processing
- **FeatureExtractor.py**: Time series feature engineering
- **FeatureProcessingArtifacts.py**: Preprocessing state management
- **sci_utils.py**: Machine learning utilities
- **meta_utils.py**: Utility functions for meta-learning workflows
- **metrics.py**: Evaluation metrics for model performance assessment

### 2. Forecast Models (`forecast_models/`)
Model implementations following a common interface:

- **base_class.py**: Abstract base class defining the interface
- **LINEAR_REGRESSION.py**: Statistical baseline models
- **SciRegressor.py**: Tree-based ensemble models
- **meta_learners/**: Meta-learning framework for intelligent ensemble creation
  - **base_meta_learner.py**: Abstract base class for meta-learning models
  - **historical_meta_learner.py**: Historical performance-weighted meta-learning

### 3. Evaluation (`evaluation/`)
Comprehensive evaluation pipeline:

- **evaluate_pipeline.py**: Main orchestrator
- **prediction_loader.py**: Loads model predictions
- **ensemble_builder.py**: Creates family and global ensembles
- **evaluate_models.py**: Calculates performance metrics

### 4. Visualization (`visualization/`)
Interactive dashboard for model comparison:

- **dashboard.py**: Main Dash application
- **data_handlers.py**: Data management for visualization
- **plotting_utils.py**: Consistent styling and color schemes
- **dashboard_components.py**: Reusable UI components

## Model Families

The system organizes models into families based on their input features:

### 1. BaseCase Models
- Basic features: discharge (Q), temperature (T), precipitation (P)
- Examples: LR_Q_T_P, PerBasinScalingLR, ShortTermLR

### 2. SCA_Based Models
- Incorporate Snow Cover Area data
- Examples: LR_Q_SCA, LR_Q_T_SCA

### 3. SnowMapper_Based Models
- Use detailed snow data (SWE, HS, ROF)
- Support multiple elevation zones
- Examples: Various configurations with elevation band features

### 4. GlacierMapper_Based Models
- Leverage GlacierMapper features (SLA, FSC)
- Include glacier melt potential calculations
- Examples: NormBased, Correction, MiniCorrection

## Workflow Overview

### Complete Model Development Workflow

```mermaid
graph LR
    subgraph "Development"
        HP[Hyperparameter<br/>Tuning]
        CAL[Model<br/>Calibration]
        HC[Hindcast<br/>Generation]
    end

    subgraph "Evaluation"
        EVAL[Run Evaluation<br/>Pipeline]
        ENS[Create<br/>Ensembles]
        RANK[Model<br/>Rankings]
    end

    subgraph "Visualization"
        DASH[Interactive<br/>Dashboard]
        EXP[Export<br/>Results]
    end

    HP --> CAL
    CAL --> HC
    HC --> EVAL
    EVAL --> ENS
    ENS --> RANK
    RANK --> DASH
    DASH --> EXP
```

### Key Scripts

1. **tune_hyperparams.py**: Optimize model hyperparameters
2. **calibrate_hindcast.py**: Train models and generate historical predictions
3. **evaluate_pipeline.py**: Run comprehensive evaluation
4. **dashboard.py**: Launch interactive visualization

### Shell Scripts for Automation

- `tune_and_calibrate_script.sh`: Combined tuning and calibration
- `run_evaluation_pipeline.sh`: Execute evaluation pipeline
- `run_model_workflow.sh`: Complete end-to-end workflow

## Recent Developments

### 1. GlacierMapper Integration
- Added support for snow line altitude (SLA) features
- Integrated fractional snow cover (FSC) data
- Implemented glacier melt potential calculations
- Created specialized GlacierMapper_Based model family

### 2. Enhanced Elevation Zone Support
- Configurable number of elevation bands (not fixed to 5)
- Percentile-based elevation band calculation
- Improved snow data aggregation by elevation

### 3. Period-Based Temporal Grouping
- Implemented 36 annual periods (3 per month)
- Period-specific normalization and scaling
- Improved handling of seasonal patterns

### 4. Dashboard Improvements
- Interactive model comparison across metrics
- Basin-specific performance analysis
- Monthly performance heatmaps
- Export functionality for further analysis

### 5. Meta-Learning Framework
- Implemented historical performance-weighted meta-learning
- Intelligent ensemble creation based on basin-period specific performance
- Softmax weighting strategy with configurable temperature
- Fallback mechanisms for insufficient historical data
- Complete LOOCV training pipeline for robust model training

## Integration Guide

### Adding a New Data Source

1. Update `scr/data_loading.py` to load the new data
2. Add preprocessing in `scr/data_utils.py` if needed
3. Configure features in `FeatureExtractor.py`
4. Update model configurations to use new features

### Adding a New Model Type

1. Create a new class inheriting from `BaseForecastModel`
2. Implement required methods:
   - `calibrate_model_and_hindcast()`
   - `predict_operational()`
   - `save_model()` / `load_model()`
3. Add model to appropriate family in evaluation pipeline
4. Update configuration templates

### Running a Complete Experiment

```bash
# 1. Prepare configuration files
cp -r example_config/DUMMY_MODEL example_config/MY_EXPERIMENT

# 2. Edit configuration files
# Update data_paths.json, general_config.json, etc.

# 3. Run hyperparameter tuning (optional)
uv run tune_hyperparams.py --config_path example_config/MY_EXPERIMENT

# 4. Calibrate and generate hindcasts
uv run calibrate_hindcast.py --config_path example_config/MY_EXPERIMENT

# 5. Run evaluation pipeline
./run_evaluation_pipeline.sh

# 6. Launch dashboard
uv run python visualization/dashboard.py
```

## Best Practices

1. **Feature Engineering**: Use `StreamflowFeatureExtractor` for consistent feature creation
2. **Preprocessing**: Always save and reuse `FeatureProcessingArtifacts`
3. **Model Organization**: Place models in appropriate families for ensemble creation
4. **Evaluation**: Use the standard evaluation pipeline for comparable results
5. **Documentation**: Update model descriptions when adding new approaches

## Future Directions

- Integration of additional Earth observation data sources
- Development of uncertainty quantification methods
- Implementation of online learning capabilities
- Extension to sub-monthly forecast horizons
- Enhanced ensemble weighting strategies