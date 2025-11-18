#!/bin/bash

# Monthly Discharge Forecasting - Evaluation Pipeline Runner
# This script provides a convenient interface for running the evaluation pipeline
# with various configuration options.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
RESULTS_DIR="../lt_forecasting_results"
OUTPUT_DIR="../lt_forecasting_results/evaluation"
EVALUATION_DAY="end"
ENSEMBLE_METHOD="mean"
COMMON_CODES_ONLY="true"
INCLUDE_CODE_MONTH="true"
MIN_SAMPLES_OVERALL=10
MIN_SAMPLES_CODE=5
MIN_SAMPLES_MONTH=5
MIN_SAMPLES_CODE_MONTH=5
VERBOSE="false"
DRY_RUN="false"

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

# Function to display help
show_help() {
    cat << EOF
Monthly Discharge Forecasting - Evaluation Pipeline Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help                  Show this help message
    -v, --verbose              Enable verbose output
    -n, --dry-run              Show command that would be executed without running it
    
    --results-dir DIR          Directory containing model results (default: $RESULTS_DIR)
    --output-dir DIR           Directory to save evaluation outputs (default: $OUTPUT_DIR)
    --evaluation-day DAY       Day of month for evaluation: 'end' or integer (default: $EVALUATION_DAY)
    --ensemble-method METHOD   Ensemble method: mean, median, weighted_mean (default: $ENSEMBLE_METHOD)
    --all-codes               Use all basin codes (not just common ones)
    --include-code-month      Include per-code-month evaluation (slower)
    
    --min-overall N           Minimum samples for overall evaluation (default: $MIN_SAMPLES_OVERALL)
    --min-code N              Minimum samples for per-code evaluation (default: $MIN_SAMPLES_CODE)
    --min-month N             Minimum samples for per-month evaluation (default: $MIN_SAMPLES_MONTH)
    --min-code-month N        Minimum samples for per-code-month evaluation (default: $MIN_SAMPLES_CODE_MONTH)

EXAMPLES:
    # Run with default settings
    $0

    # Run with verbose output and include code-month evaluation
    $0 --verbose --include-code-month

    # Run with custom output directory and all basin codes
    $0 --output-dir ./my_evaluation --all-codes

    # Run with median ensemble method and higher minimum samples
    $0 --ensemble-method median --min-overall 20 --min-code 10

    # Dry run to see the command without executing
    $0 --dry-run --verbose

OUTPUTS:
    The pipeline generates the following files in the output directory:
    - metrics.csv: Comprehensive evaluation results
    - model_rankings.csv: Performance rankings by different metrics
    - metrics_summary.json: Statistical summaries and family comparisons
    - model_family_metrics.csv: Family-level performance statistics
    - evaluation_metadata.json: Pipeline configuration and runtime metadata
    - *_ensemble_predictions.csv: Family and global ensemble predictions

REQUIREMENTS:
    - Python environment with uv package manager
    - Access to lt_forecasting_results directory
    - All required Python dependencies installed

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE="true"
            shift
            ;;
        -n|--dry-run)
            DRY_RUN="true"
            shift
            ;;
        --results-dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --evaluation-day)
            EVALUATION_DAY="$2"
            shift 2
            ;;
        --ensemble-method)
            ENSEMBLE_METHOD="$2"
            shift 2
            ;;
        --all-codes)
            COMMON_CODES_ONLY="false"
            shift
            ;;
        --include-code-month)
            INCLUDE_CODE_MONTH="true"
            shift
            ;;
        --min-overall)
            MIN_SAMPLES_OVERALL="$2"
            shift 2
            ;;
        --min-code)
            MIN_SAMPLES_CODE="$2"
            shift 2
            ;;
        --min-month)
            MIN_SAMPLES_MONTH="$2"
            shift 2
            ;;
        --min-code-month)
            MIN_SAMPLES_CODE_MONTH="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ ! "$ENSEMBLE_METHOD" =~ ^(mean|median|weighted_mean)$ ]]; then
    print_error "Invalid ensemble method: $ENSEMBLE_METHOD"
    print_error "Valid options: mean, median, weighted_mean"
    exit 1
fi

if [[ "$EVALUATION_DAY" != "end" ]] && ! [[ "$EVALUATION_DAY" =~ ^[0-9]+$ ]]; then
    print_error "Invalid evaluation day: $EVALUATION_DAY"
    print_error "Must be 'end' or a positive integer"
    exit 1
fi

# Check if results directory exists
if [[ ! -d "$RESULTS_DIR" ]]; then
    print_error "Results directory not found: $RESULTS_DIR"
    print_error "Please ensure the lt_forecasting_results directory exists"
    exit 1
fi

# Build the command
CMD="uv run python -m dev_tools.evaluation.evaluate_pipeline"
CMD="$CMD --results_dir \"$RESULTS_DIR\""
CMD="$CMD --output_dir \"$OUTPUT_DIR\""
CMD="$CMD --evaluation_day \"$EVALUATION_DAY\""
CMD="$CMD --ensemble_method \"$ENSEMBLE_METHOD\""
CMD="$CMD --min_samples_overall $MIN_SAMPLES_OVERALL"
CMD="$CMD --min_samples_code $MIN_SAMPLES_CODE"
CMD="$CMD --min_samples_month $MIN_SAMPLES_MONTH"
CMD="$CMD --min_samples_code_month $MIN_SAMPLES_CODE_MONTH"

if [[ "$COMMON_CODES_ONLY" == "false" ]]; then
    CMD="$CMD --no_common_codes"
fi

if [[ "$INCLUDE_CODE_MONTH" == "true" ]]; then
    CMD="$CMD --include_code_month"
fi

# Display configuration
print_info "Evaluation Pipeline Configuration:"
echo "  Results directory: $RESULTS_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Evaluation day: $EVALUATION_DAY"
echo "  Ensemble method: $ENSEMBLE_METHOD"
echo "  Common codes only: $COMMON_CODES_ONLY"
echo "  Include code-month: $INCLUDE_CODE_MONTH"
echo "  Min samples (overall/code/month/code-month): $MIN_SAMPLES_OVERALL/$MIN_SAMPLES_CODE/$MIN_SAMPLES_MONTH/$MIN_SAMPLES_CODE_MONTH"
echo ""

if [[ "$VERBOSE" == "true" ]]; then
    print_info "Command to execute:"
    echo "  $CMD"
    echo ""
fi

if [[ "$DRY_RUN" == "true" ]]; then
    print_warning "DRY RUN MODE - Command would be executed:"
    echo "$CMD"
    exit 0
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Record start time
START_TIME=$(date)
print_info "Starting evaluation pipeline at $START_TIME"
echo ""

# Execute the command
if eval "$CMD"; then
    END_TIME=$(date)
    print_success "Evaluation pipeline completed successfully!"
    print_success "Started: $START_TIME"
    print_success "Finished: $END_TIME"
    print_success "Output directory: $OUTPUT_DIR"
    echo ""
    
    # Show generated files
    if [[ -d "$OUTPUT_DIR" ]]; then
        print_info "Generated files:"
        ls -la "$OUTPUT_DIR" | grep -E '\.(csv|json)$' | while read -r line; do
            echo "  $line"
        done
    fi
    
    echo ""
    print_info "Next steps:"
    echo "  1. Review evaluation results in $OUTPUT_DIR"
    echo "  2. Check model rankings in model_rankings.csv"
    echo "  3. Examine ensemble predictions in *_ensemble_predictions.csv"
    echo "  4. Use outputs for dashboard integration"
    
else
    END_TIME=$(date)
    print_error "Evaluation pipeline failed!"
    print_error "Started: $START_TIME"
    print_error "Failed: $END_TIME"
    print_error "Check the error messages above for details"
    exit 1
fi