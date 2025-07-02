#!/bin/bash

# Monthly Forecasting Model Workflow Script
# This script orchestrates the complete workflow for monthly discharge forecasting models:
# 1. Hyperparameter tuning (optional)
# 2. Model calibration and hindcasting
# 3. Evaluation and results generation
#
# Usage: ./run_model_workflow.sh [OPTIONS]
#
# Example:
#   ./run_model_workflow.sh --config_dir monthly_forecasting_models/XGBoost_AllFeatures --model_name XGBoost_AllFeatures --tune_hyperparams

set -e  # Exit on any error

# Default values
CONFIG_DIR=""
MODEL_NAME=""
OUTPUT_DIR=""
TUNE_HYPERPARAMS=false
SKIP_CALIBRATION=false
SKIP_METRICS=false
TRIALS=100
LOG_LEVEL="INFO"
PYTHON_CMD="python"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Monthly Forecasting Model Workflow Script

USAGE:
    $0 --config_dir CONFIG_DIR --model_name MODEL_NAME [OPTIONS]

REQUIRED ARGUMENTS:
    --config_dir DIR          Path to model configuration directory
    --model_name NAME         Name of the model to run

OPTIONAL ARGUMENTS:
    --output_dir DIR          Output directory for results (default: monthly_forecasting_results/MODEL_NAME)
    --tune_hyperparams        Run hyperparameter tuning before calibration
    --skip_calibration        Skip model calibration (only run hyperparameter tuning)
    --skip_metrics            Skip metrics calculation during calibration
    --trials N                Number of hyperparameter tuning trials (default: 100)
    --log_level LEVEL         Logging level: DEBUG, INFO, WARNING, ERROR (default: INFO)
    --python COMMAND          Python command to use (default: python)
    --help                    Show this help message

EXAMPLES:
    # Complete workflow with hyperparameter tuning
    $0 --config_dir monthly_forecasting_models/XGBoost_AllFeatures --model_name XGBoost_AllFeatures --tune_hyperparams

    # Calibration only (no hyperparameter tuning)
    $0 --config_dir monthly_forecasting_models/LinearRegression_BasicFeatures --model_name LinearRegression_BasicFeatures

    # Hyperparameter tuning only
    $0 --config_dir monthly_forecasting_models/LightGBM_AdvancedFeatures --model_name LightGBM_AdvancedFeatures --tune_hyperparams --skip_calibration

    # Quick run with fewer trials
    $0 --config_dir models/test_model --model_name test_model --tune_hyperparams --trials 20

WORKFLOW STEPS:
    1. Validate inputs and check required files
    2. [Optional] Run hyperparameter tuning using Optuna
    3. [Optional] Run model calibration and hindcasting with LOOCV
    4. [Optional] Calculate evaluation metrics and generate reports
    5. Generate summary of results

CONFIGURATION REQUIREMENTS:
    The config directory should contain:
    - general_config.json      (model type, features, periods)
    - model_config.json        (model-specific parameters)
    - feature_config.json      (feature engineering configuration)
    - data_config.json         (data loading configuration)
    - ../config/data_paths.json (data file paths)

OUTPUT STRUCTURE:
    output_dir/
    ├── predictions.csv         (hindcast predictions)
    ├── metrics.csv            (evaluation metrics per basin)
    ├── metrics_summary.json   (aggregated metrics)
    ├── calibration_report.json (complete calibration report)
    └── MODEL_NAME_hyperparams.json (best hyperparameters)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config_dir)
            CONFIG_DIR="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --tune_hyperparams)
            TUNE_HYPERPARAMS=true
            shift
            ;;
        --skip_calibration)
            SKIP_CALIBRATION=true
            shift
            ;;
        --skip_metrics)
            SKIP_METRICS=true
            shift
            ;;
        --trials)
            TRIALS="$2"
            shift 2
            ;;
        --log_level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --python)
            PYTHON_CMD="$2"
            shift 2
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown argument: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$CONFIG_DIR" ]]; then
    print_error "Missing required argument: --config_dir"
    show_usage
    exit 1
fi

if [[ -z "$MODEL_NAME" ]]; then
    print_error "Missing required argument: --model_name"
    show_usage
    exit 1
fi

# Set default output directory
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="monthly_forecasting_results/${MODEL_NAME}"
fi

# Validate inputs
print_info "Validating inputs..."

if [[ ! -d "$CONFIG_DIR" ]]; then
    print_error "Configuration directory not found: $CONFIG_DIR"
    exit 1
fi

# Check for required configuration files
required_configs=("general_config.json" "model_config.json" "feature_config.json" "data_config.json")
for config_file in "${required_configs[@]}"; do
    if [[ ! -f "$CONFIG_DIR/$config_file" ]]; then
        print_error "Required configuration file not found: $CONFIG_DIR/$config_file"
        exit 1
    fi
done

# Check Python availability
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    print_error "Python command not found: $PYTHON_CMD"
    exit 1
fi

# Check if required Python scripts exist
script_dir="$(dirname "$0")"
tune_script="$script_dir/tune_hyperparams.py"
calibrate_script="$script_dir/calibrate_hindcast.py"

if [[ "$TUNE_HYPERPARAMS" == true && ! -f "$tune_script" ]]; then
    print_error "Hyperparameter tuning script not found: $tune_script"
    exit 1
fi

if [[ "$SKIP_CALIBRATION" == false && ! -f "$calibrate_script" ]]; then
    print_error "Calibration script not found: $calibrate_script"
    exit 1
fi

print_success "Input validation completed"

# Create output directory
mkdir -p "$OUTPUT_DIR"
print_info "Output directory: $OUTPUT_DIR"

# Start workflow
print_info "Starting Monthly Forecasting Model Workflow"
print_info "Model: $MODEL_NAME"
print_info "Configuration: $CONFIG_DIR"
print_info "Hyperparameter tuning: $TUNE_HYPERPARAMS"
print_info "Model calibration: $([ "$SKIP_CALIBRATION" == false ] && echo "enabled" || echo "disabled")"

# Step 1: Hyperparameter Tuning (optional)
if [[ "$TUNE_HYPERPARAMS" == true ]]; then
    print_info "Step 1: Running hyperparameter tuning..."
    
    tuning_cmd="$PYTHON_CMD $tune_script \
        --config_dir \"$CONFIG_DIR\" \
        --model_name \"$MODEL_NAME\" \
        --trials $TRIALS \
        --save_config \
        --log_level $LOG_LEVEL"
    
    print_info "Command: $tuning_cmd"
    
    if eval "$tuning_cmd"; then
        print_success "Hyperparameter tuning completed successfully"
    else
        print_error "Hyperparameter tuning failed"
        exit 1
    fi
else
    print_info "Step 1: Skipping hyperparameter tuning"
fi

# Step 2: Model Calibration and Hindcasting (optional)
if [[ "$SKIP_CALIBRATION" == false ]]; then
    print_info "Step 2: Running model calibration and hindcasting..."
    
    calibration_cmd="$PYTHON_CMD $calibrate_script \
        --config_dir \"$CONFIG_DIR\" \
        --model_name \"$MODEL_NAME\" \
        --output_dir \"$OUTPUT_DIR\" \
        --log_level $LOG_LEVEL"
    
    if [[ "$SKIP_METRICS" == true ]]; then
        calibration_cmd="$calibration_cmd --skip_metrics"
    fi
    
    print_info "Command: $calibration_cmd"
    
    if eval "$calibration_cmd"; then
        print_success "Model calibration and hindcasting completed successfully"
    else
        print_error "Model calibration and hindcasting failed"
        exit 1
    fi
else
    print_info "Step 2: Skipping model calibration"
fi

# Step 3: Generate Summary
print_info "Step 3: Generating workflow summary..."

# Create workflow summary
summary_file="$OUTPUT_DIR/workflow_summary.json"
cat > "$summary_file" << EOF
{
    "workflow_timestamp": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "model_name": "$MODEL_NAME",
    "config_dir": "$CONFIG_DIR",
    "output_dir": "$OUTPUT_DIR",
    "workflow_steps": {
        "hyperparameter_tuning": $TUNE_HYPERPARAMS,
        "model_calibration": $([ "$SKIP_CALIBRATION" == false ] && echo "true" || echo "false"),
        "metrics_calculation": $([ "$SKIP_METRICS" == false ] && echo "true" || echo "false")
    },
    "configuration": {
        "trials": $TRIALS,
        "log_level": "$LOG_LEVEL",
        "python_command": "$PYTHON_CMD"
    }
}
EOF

print_success "Workflow summary saved to: $summary_file"

# Print final results
print_success "Monthly Forecasting Model Workflow Completed Successfully!"
print_info "Results location: $OUTPUT_DIR"

# List generated files
print_info "Generated files:"
if [[ -d "$OUTPUT_DIR" ]]; then
    for file in "$OUTPUT_DIR"/*; do
        if [[ -f "$file" ]]; then
            filename=$(basename "$file")
            filesize=$(du -h "$file" | cut -f1)
            print_info "  - $filename ($filesize)"
        fi
    done
fi

# Show quick metrics summary if available
metrics_summary="$OUTPUT_DIR/metrics_summary.json"
if [[ -f "$metrics_summary" ]]; then
    print_info "Quick Results Summary:"
    
    # Extract key metrics using Python (if available)
    if command -v python3 &> /dev/null; then
        python3 -c "
import json
try:
    with open('$metrics_summary', 'r') as f:
        data = json.load(f)
    print('  Basins evaluated: {}'.format(data.get('n_basins', 'N/A')))
    print('  Total predictions: {}'.format(data.get('total_predictions', 'N/A')))
    print('  Mean R²: {:.3f}'.format(data.get('r2_mean', float('nan'))))
    print('  Mean RMSE: {:.3f}'.format(data.get('rmse_mean', float('nan'))))
except Exception as e:
    print('  Could not parse metrics summary')
" 2>/dev/null || print_info "  Check $metrics_summary for detailed results"
    fi
fi

print_info "Workflow completed at: $(date)"
print_success "Done!"

exit 0