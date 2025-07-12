# Evaluation Module

This directory contains the evaluation pipeline for assessing model performance and building ensemble predictions.

## Components

### Core Files

#### `evaluate_pipeline.py`
Main evaluation orchestrator that:
- Loads model predictions from multiple experiments
- Builds ensemble predictions (family and global)
- Calculates comprehensive performance metrics
- Generates outputs for the visualization dashboard

Key functions:
- `run_pipeline()`: Main entry point for evaluation
- `process_experiment()`: Processes individual model predictions
- `save_results()`: Saves evaluation outputs in dashboard-compatible format

#### `ensemble_builder.py`
Handles ensemble creation with different strategies:
- **Family Ensembles**: Combines models of the same type (e.g., all XGBoost models)
- **Global Ensembles**: Combines all available models
- **Ensemble Methods**: 
  - Mean: Simple average of predictions
  - Median: Robust central tendency
  - Weighted: Performance-based weighting (future enhancement)

Key classes:
- `EnsembleBuilder`: Main class for ensemble construction
- Methods: `build_family_ensemble()`, `build_global_ensemble()`

#### `prediction_loader.py`
Loads and validates model predictions:
- Reads hindcast/forecast outputs from model experiments
- Ensures consistent formatting and time alignment
- Handles missing data and validation checks

Key functions:
- `load_predictions()`: Load predictions from a single experiment
- `validate_predictions()`: Check data integrity and completeness

### Tests Directory

#### `tests/__init__.py`
Test module initialization for the evaluation package.

## Workflow

1. **Model Training**: Individual models generate predictions during calibration
2. **Prediction Loading**: `prediction_loader` reads all model outputs
3. **Ensemble Building**: `ensemble_builder` creates combined predictions
4. **Metric Calculation**: Performance metrics computed for all models and ensembles
5. **Output Generation**: Results formatted for dashboard visualization

## Usage

### Running Evaluation Pipeline

```bash
# Run evaluation for specific experiments
uv run evaluation/evaluate_pipeline.py --experiments exp1,exp2,exp3

# Run with configuration file
uv run evaluation/evaluate_pipeline.py --config evaluation_config.json
```

### Shell Script
```bash
./run_evaluation_pipeline.sh
```

## Output Structure

The evaluation pipeline generates:
- `evaluation_results.json`: Comprehensive metrics for all models
- `ensemble_predictions.csv`: Time series of ensemble forecasts
- `model_rankings.json`: Performance-based model rankings
- `basin_specific_metrics.json`: Per-basin performance breakdown

## Performance Metrics

Calculated metrics include:
- **R²**: Coefficient of determination
- **NSE**: Nash-Sutcliffe Efficiency
- **KGE**: Kling-Gupta Efficiency
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Bias**: Systematic over/under-prediction
- **Correlation**: Pearson correlation coefficient

## Configuration

Evaluation behavior can be configured through:
- Ensemble methods selection
- Metric calculation options
- Output format preferences
- Performance thresholds for model inclusion

## Integration

The evaluation module integrates with:
- **Model Outputs**: Reads predictions from `forecast_models/`
- **Visualization**: Provides data for `visualization/dashboard.py`
- **Metrics**: Uses functions from `eval_scr/metric_functions.py`

## Best Practices

1. Run evaluation after all models complete training
2. Verify prediction files exist before running pipeline
3. Check for temporal alignment across all models
4. Monitor ensemble performance vs individual models
5. Use basin-specific metrics for regional analysis