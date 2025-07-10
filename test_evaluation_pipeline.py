#!/usr/bin/env python
"""
Test script for the evaluation pipeline.
"""

import sys
import os
from pathlib import Path

# Add current directory to path so we can import the evaluation modules
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.evaluate_pipeline import run_evaluation_pipeline

if __name__ == "__main__":
    print("Testing Evaluation Pipeline...")
    
    # Test with minimal parameters
    success = run_evaluation_pipeline(
        results_dir="../monthly_forecasting_results",
        output_dir="../monthly_forecasting_results/evaluation",
        evaluation_day='end',
        common_codes_only=True,
        ensemble_method='mean',
        include_code_month=False,
        min_samples_overall=5,  # Lower thresholds for testing
        min_samples_code=3,
        min_samples_month=2,
        min_samples_code_month=1
    )
    
    if success:
        print("✅ Evaluation pipeline completed successfully!")
        print("Check output directory: ../monthly_forecasting_results/evaluation/")
    else:
        print("❌ Evaluation pipeline failed!")
        sys.exit(1)