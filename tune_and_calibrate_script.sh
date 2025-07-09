#!/bin/bash

# This script runs hyperparameter tuning followed by calibration and hindcasting

set -e  # Exit on any error

# Configuration
CONFIG_DIR="../monthly_forecasting_models/SnowMapper_Based/CondenseLR"  # Path to model configuration directory
MODEL_NAME="CondenseLR"  # Model name
INPUT_FAMILY="SnowMapper_Based"  # Input family for the model
LOG_LEVEL="DEBUG"  # Set to INFO or DEBUG as needed

# Hyperparameter tuning specific settings
TRIALS=100  # Number of Optuna trials (adjust as needed)
TUNING_YEARS=3  # Number of years for hyperparameter validation
SAVE_CONFIG="--save_config"  # Save updated config with best hyperparameters

# Environment configuration - Choose one of the following:
# Option 1: Use conda/mamba environment
# PYTHON_CMD="conda run -n your_env_name python"

# Option 2: Use uv
# PYTHON_CMD="uv run python"

# Option 3: Use specific virtual environment
source "/Users/sandrohunziker/Documents/sapphire_venv/monthly_forecast/bin/activate"

# Option 4: Use system python (default)
#PYTHON_CMD="python3"

# Option 5: Auto-detect uv or fall back to python3
if command -v uv &> /dev/null; then
    PYTHON_CMD="uv run"
else
    PYTHON_CMD="python"
fi

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

print_step() {
    echo -e "${PURPLE}[STEP]${NC} $1"
}

# Function to check if directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        exit 1
    fi
}

# Function to check if Python script exists
check_python_script() {
    if [ ! -f "$1" ]; then
        print_error "Python script not found: $1"
        exit 1
    fi
}

# Function to run hyperparameter tuning
run_hyperparameter_tuning() {
    print_step "Starting hyperparameter tuning..."
    print_status "Model: $MODEL_NAME"
    print_status "Configuration directory: $CONFIG_DIR"

    # Check if tuning script exists
    TUNING_SCRIPT="tune_hyperparams.py"
    check_python_script "$TUNING_SCRIPT"
    
    # Run hyperparameter tuning
    print_status "Running hyperparameter tuning with: $PYTHON_CMD"
    
    $PYTHON_CMD "$TUNING_SCRIPT" \
        --config_dir "$CONFIG_DIR" \
        --model_name "$MODEL_NAME" \
        --trials "$TRIALS" \
        --tuning_years "$TUNING_YEARS" \
        --log_level "$LOG_LEVEL" \
        $SAVE_CONFIG
    
    # Check if tuning was successful
    if [ $? -eq 0 ]; then
        print_success "Hyperparameter tuning completed successfully!"
        return 0
    else
        print_error "Hyperparameter tuning failed!"
        return 1
    fi
}

# Function to run calibration and hindcasting
run_calibration_hindcasting() {
    print_step "Starting calibration and hindcasting..."
    print_status "Model: $MODEL_NAME"
    print_status "Input family: $INPUT_FAMILY"
    print_status "Configuration directory: $CONFIG_DIR"
    
    # Check if calibration script exists
    CALIBRATION_SCRIPT="calibrate_hindcast.py"
    check_python_script "$CALIBRATION_SCRIPT"
    
    # Run the calibration script
    print_status "Running calibration and hindcasting with: $PYTHON_CMD"
    
    $PYTHON_CMD "$CALIBRATION_SCRIPT" \
        --config_dir "$CONFIG_DIR" \
        --model_name "$MODEL_NAME" \
        --input_family "$INPUT_FAMILY" \
        --log_level "$LOG_LEVEL"
    
    # Check if calibration was successful
    if [ $? -eq 0 ]; then
        print_success "Calibration and hindcasting completed successfully!"
        return 0
    else
        print_error "Calibration and hindcasting failed!"
        return 1
    fi
}

# Main execution
main() {
    print_status "Starting complete pipeline for model: $MODEL_NAME"
    print_status "Pipeline: Hyperparameter Tuning → Calibration & Hindcasting"
    echo
    
    # Check if configuration directory exists
    check_directory "$CONFIG_DIR"
    
    # Step 1: Run hyperparameter tuning
    if run_hyperparameter_tuning; then
        echo
        print_success "✓ Step 1/2: Hyperparameter tuning completed successfully!"
        echo
        
        # Step 2: Run calibration and hindcasting with optimized parameters
        if run_calibration_hindcasting; then
            echo
            print_success "✓ Step 2/2: Calibration and hindcasting completed successfully!"
            print_success "🎉 Complete pipeline finished successfully!"
            
            # Show final results
            print_status "Pipeline completed for model: $MODEL_NAME"
            if [ -n "$OUTPUT_DIR" ] && [ -d "$OUTPUT_DIR" ]; then
                print_status "Results saved to: $OUTPUT_DIR"
                print_status "Generated files:"
                ls -la "$OUTPUT_DIR"
            fi
        else
            print_error "❌ Step 2/2 failed: Calibration and hindcasting failed!"
            exit 1
        fi
    else
        print_error "❌ Step 1/2 failed: Hyperparameter tuning failed!"
        print_error "Aborting pipeline - calibration will not run"
        exit 1
    fi
}

# Function to display help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "This script runs hyperparameter tuning followed by calibration and hindcasting."
    echo
    echo "Configuration variables (edit at top of script):"
    echo "  CONFIG_DIR      - Path to model configuration directory"
    echo "  MODEL_NAME      - Name of the model"
    echo "  INPUT_FAMILY    - Input family for the model"
    echo "  TRIALS          - Number of Optuna trials for hyperparameter tuning"
    echo "  TUNING_YEARS    - Number of years for hyperparameter validation"
    echo "  LOG_LEVEL       - Logging level (DEBUG, INFO, WARNING, ERROR)"
    echo
    echo "Options:"
    echo "  -h, --help      Show this help message"
    echo
    echo "Examples:"
    echo "  $0              Run with default configuration"
    echo "  $0 --help       Show this help message"
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    "")
        # No arguments, run main
        ;;
    *)
        print_error "Unknown argument: $1"
        show_help
        exit 1
        ;;
esac

# Run main function
main

print_success "Pipeline script completed successfully! 🚀"