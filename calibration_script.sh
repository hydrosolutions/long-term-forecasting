#!/bin/bash


# This script runs the calibration and hindcasting process 

set -e  # Exit on any error

# Configuration
CONFIG_DIR="../monthly_forecasting_models/BaseCase/LR_Q_T_P"
MODEL_NAME="LR_Q_T_P"  # Model name
INPUT_FAMILY="BaseCase"
LOG_LEVEL="DEBUG"  # Set to INFO or DEBUG as needed

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

# Main execution
main() {
    print_status "Starting calibration for model: $MODEL_NAME"
    print_status "Input family: $INPUT_FAMILY"
    print_status "Configuration directory: $CONFIG_DIR"
    
    # Check if configuration directory exists
    check_directory "$CONFIG_DIR"
    
    # Check if Python script exists
    PYTHON_SCRIPT="calibrate_hindcast.py"
    check_python_script "$PYTHON_SCRIPT"
    
    
    # Run the calibration script
    print_status "Running calibration and hindcasting with: $PYTHON_CMD"
    
    $PYTHON_CMD "$PYTHON_SCRIPT" \
        --config_dir "$CONFIG_DIR" \
        --model_name "$MODEL_NAME" \
        --input_family "$INPUT_FAMILY" \
        --log_level "$LOG_LEVEL"
    
    # Check if calibration was successful
    if [ $? -eq 0 ]; then
        print_success "Calibration completed successfully!"
        print_success "Results saved to: $OUTPUT_DIR"
        
        # List generated files
        if [ -d "$OUTPUT_DIR" ]; then
            print_status "Generated files:"
            ls -la "$OUTPUT_DIR"
        fi
    else
        print_error "Calibration failed!"
        exit 1
    fi
}


# Run main function
main

print_status "Script completed."