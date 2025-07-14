# Evaluation Module

This module provides comprehensive evaluation capabilities for monthly discharge forecasting models, including individual model evaluation, ensemble creation, and dashboard integration.

## Overview

The evaluation module orchestrates the complete evaluation workflow for discharge forecasting models:
1. Discovers and loads predictions from multiple model families
2. Creates ensemble predictions at family and global levels
3. Evaluates all models (individual + ensembles) using standardized metrics
4. Generates outputs formatted for dashboard consumption

## Module Structure

### Core Files

#### `__init__.py`
- Module initialization and public API definition
- Exports main functions for external use
- Version: 1.0.0

#### `prediction_loader.py`
Handles discovery, loading, and standardization of prediction files from various model families.

**Key Features:**
- Scans results directory for prediction files across model families
- Supports multiple prediction columns (ensemble members)
- Standardizes column names and data formats
- Filters predictions to evaluation dates (end of month or specific day)
- Validates data quality and reports warnings

**Model Family Mappings:**
- BaseCase: LR_Q_T_P, PerBasinScalingLR, ShortTermLR, ShortTerm_Features, NormBased
- SCA_Based: LR_Q_SCA, LR_Q_T_SCA
- SnowMapper_Based: Various models using SWE data
- GlacierMapper_Based: NormBased, Correction, MiniCorrection

**Key Functions:**
- `scan_prediction_files()`: Discovers available prediction files
- `load_predictions()`: Loads and standardizes prediction data
- `load_all_predictions()`: Complete workflow for loading all predictions
- `validate_prediction_data()`: Validates data quality

#### `ensemble_builder.py`
Creates ensemble predictions by combining multiple models using various methods.

**Key Features:**
- Creates family-level ensembles (combining models within same family)
- Creates global ensembles (combining all models or family ensembles)
- Supports multiple ensemble methods: mean, median, weighted mean
- Tracks number of models contributing to each ensemble prediction
- Saves ensemble predictions and metadata

**Key Functions:**
- `create_simple_ensemble()`: Core ensemble creation logic
- `create_family_ensemble()`: Creates ensemble for specific model family
- `create_global_ensemble()`: Creates global ensemble from all models
- `create_all_ensembles()`: Orchestrates all ensemble creation
- `save_ensemble_predictions()`: Saves ensemble results to CSV files

#### `evaluate_models.py`
Provides comprehensive model evaluation using standardized metrics from `eval_scr.metric_functions`.

**Key Features:**
- Calculates multiple performance metrics: R², RMSE, NSE, KGE, MAE, bias, etc.
- Evaluates at multiple aggregation levels:
  - Overall: Across all data
  - Per-code: For each basin
  - Per-month: For each calendar month
  - Per-code-month: For each basin-month combination
- Handles missing data gracefully
- Generates model rankings based on metrics

**Metrics Calculated:**
- R² (coefficient of determination)
- RMSE (root mean square error) and NRMSE (normalized)
- MAE (mean absolute error) and MAPE (percentage)
- NSE (Nash-Sutcliffe efficiency)
- KGE (Kling-Gupta efficiency)
- Bias and percent bias
- Sample count

**Key Functions:**
- `calculate_metrics()`: Computes all metrics for obs/pred pairs
- `evaluate_per_code()`: Evaluates performance by basin
- `evaluate_per_month()`: Evaluates performance by month
- `evaluate_model_comprehensive()`: Complete evaluation at all levels
- `calculate_model_rankings()`: Ranks models by specified metric

#### `evaluate_pipeline.py`
Main orchestration script that runs the complete evaluation workflow.

**Key Features:**
- Implements `EvaluationPipeline` class for workflow management
- Configurable evaluation parameters
- Generates multiple output formats for dashboard
- Comprehensive logging and error handling
- Command-line interface for standalone execution

**Pipeline Steps:**
1. Load all predictions from results directory
2. Create family and global ensembles
3. Evaluate all models and ensembles
4. Generate dashboard-ready outputs

**Output Files Generated:**
- `metrics.csv`: Comprehensive evaluation metrics for all models
- `metrics_summary.json`: Statistical summary of metrics
- `model_family_metrics.csv`: Family-level comparison
- `model_rankings.csv`: Model rankings by different metrics
- `evaluation_metadata.json`: Pipeline configuration and metadata
- Family and global ensemble prediction files

**Key Functions:**
- `run_evaluation_pipeline()`: Convenience function to run complete pipeline
- Command-line interface with configurable parameters

## Dependencies

### External Dependencies
- **eval_scr.metric_functions**: Provides standardized metric calculations
  - Used by `evaluate_models.py` for all performance metrics
  - Ensures consistency with other evaluation scripts

### Input Data
- Expects prediction files in: `../monthly_forecasting_results/{family}/{model}/predictions.csv`
- Prediction files must contain columns: date, code, Q_obs, Q_pred (or Q_*)

## Usage

### Basic Usage

```python
from evaluation import run_evaluation_pipeline

# Run complete evaluation pipeline
success = run_evaluation_pipeline(
    results_dir="../monthly_forecasting_results",
    output_dir="../monthly_forecasting_results/evaluation",
    evaluation_day="end",  # or specific day number
    common_codes_only=True,
    ensemble_method="mean"
)
```

### Command Line Usage

```bash
python evaluate_pipeline.py \
    --results_dir ../monthly_forecasting_results \
    --output_dir ../monthly_forecasting_results/evaluation \
    --evaluation_day end \
    --ensemble_method mean \
    --include_code_month
```

### Advanced Usage

```python
from evaluation import (
    load_all_predictions,
    create_all_ensembles,
    evaluate_multiple_models
)

# Load predictions
predictions, validation = load_all_predictions()

# Create ensembles
family_ensembles, global_ensembles = create_all_ensembles(
    predictions, 
    ensemble_method="mean"
)

# Evaluate all models
results = evaluate_multiple_models(
    predictions,
    include_code_month=True
)
```

## Configuration Options

### Evaluation Day
- `"end"`: Use last day of each month (default)
- Integer (1-31): Use specific day of month

### Ensemble Methods
- `"mean"`: Simple average (default)
- `"median"`: Median value
- `"weighted_mean"`: Weighted average (requires weights)

### Minimum Sample Requirements
- `min_samples_overall`: 10 (default)
- `min_samples_code`: 5
- `min_samples_month`: 3
- `min_samples_code_month`: 2

## Output Format

### Metrics CSV
Contains evaluation metrics for each model at different aggregation levels:
- model_id, family, model_name
- level (overall/per_code/per_month/per_code_month)
- All metric values
- Additional grouping columns (code, month) as applicable

### Model Rankings CSV
Rankings for each metric with columns:
- rank, model_id, family, model_name
- metric value, n_samples
- ranking_metric (which metric was used)

### Ensemble Predictions
Saved with columns:
- date, code, Q_obs, Q_pred
- n_models (number of models in ensemble)

## Error Handling

- Missing prediction files are logged as warnings
- Models with insufficient data are skipped
- Validation warnings are collected and reported
- Pipeline continues despite individual model failures
- Comprehensive logging to file and console

## Testing

Test files are located in the `tests/` subdirectory. Run tests using pytest:

```bash
pytest evaluation/tests/ -v
```