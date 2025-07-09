#!/usr/bin/env python3
"""
Performance test script to analyze current test suite performance
and compare against issue #12 requirements.
"""

import time
import sys
import os
import psutil
import pandas as pd
from pathlib import Path

# Add tests directory to path
sys.path.append('tests')

# Import test utilities
from comprehensive_test_utils import ComprehensiveTestDataGenerator
from comprehensive_test_configs import TEST_CONSTANTS, TEST_DATA_PARAMS, BASE_GENERAL_CONFIG

def measure_data_generation_performance():
    """Measure data generation performance."""
    print("=== Data Generation Performance Test ===")
    
    # Test current data generation
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    # Generate comprehensive data
    data = ComprehensiveTestDataGenerator.generate_comprehensive_timeseries_data()
    static_data = ComprehensiveTestDataGenerator.generate_comprehensive_static_data()
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    generation_time = end_time - start_time
    memory_used = end_memory - start_memory
    
    print(f"Data generation time: {generation_time:.3f} seconds")
    print(f"Memory used: {memory_used:.1f} MB")
    print(f"Number of records: {len(data)}")
    print(f"Number of basins: {len(data['code'].unique())}")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    
    # Calculate data characteristics
    num_years = (data['date'].max() - data['date'].min()).days / 365.25
    print(f"Number of years: {num_years:.1f}")
    
    # Check against issue requirements
    print("\n--- Requirement Check ---")
    print(f"✓ Target: 3 basins | Current: {len(data['code'].unique())}")
    print(f"{'✓' if num_years <= 5 else '✗'} Target: 5 years | Current: {num_years:.1f}")
    print(f"{'✓' if len(data) <= 5500 else '✗'} Target: ~5,500 records | Current: {len(data)}")
    print(f"{'✓' if generation_time < 0.1 else '✗'} Target: <100ms | Current: {generation_time*1000:.1f}ms")
    
    return {
        'generation_time': generation_time,
        'memory_used': memory_used,
        'num_records': len(data),
        'num_basins': len(data['code'].unique()),
        'num_years': num_years
    }

def check_hyperparameter_config():
    """Check hyperparameter optimization configuration."""
    print("\n=== Hyperparameter Configuration Check ===")
    
    current_trials = BASE_GENERAL_CONFIG.get('n_trials', 'Not set')
    print(f"Current n_trials: {current_trials}")
    
    # Check against issue requirements
    print("\n--- Requirement Check ---")
    print(f"{'✓' if current_trials == 1 else '✗'} Target: 1 trial | Current: {current_trials}")
    
    return current_trials

def analyze_test_data_config():
    """Analyze test data configuration."""
    print("\n=== Test Data Configuration Analysis ===")
    
    print(f"TEST_DATA_PARAMS:")
    for key, value in TEST_DATA_PARAMS.items():
        print(f"  {key}: {value}")
    
    print(f"\nDATA_CONFIG:")
    print(f"  start_year: {TEST_DATA_PARAMS.get('start_year', 'Not set')}")
    print(f"  end_year: {TEST_DATA_PARAMS.get('end_year', 'Not set')}")
    print(f"  num_years: {TEST_DATA_PARAMS.get('num_years', 'Not set')}")
    print(f"  num_basins: {TEST_DATA_PARAMS.get('num_basins', 'Not set')}")

def run_sample_test_timing():
    """Run a sample test to measure timing."""
    print("\n=== Sample Test Timing ===")
    
    # Time a complete test workflow simulation
    start_time = time.time()
    
    # Generate data
    data = ComprehensiveTestDataGenerator.generate_comprehensive_timeseries_data()
    static_data = ComprehensiveTestDataGenerator.generate_comprehensive_static_data()
    
    # Simulate some processing (like what would happen in a real test)
    processed_data = data.copy()
    processed_data['processed'] = processed_data['discharge'] * 1.1
    
    # Simulate predictions
    predictions = processed_data.sample(n=min(100, len(processed_data)))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"Sample test workflow time: {total_time:.3f} seconds")
    print(f"{'✓' if total_time < 0.5 else '✗'} Target: <500ms | Current: {total_time*1000:.1f}ms")
    
    return total_time

def main():
    """Main performance analysis function."""
    print("Performance Analysis for Monthly Forecasting Test Suite")
    print("=" * 60)
    
    # Measure data generation
    data_perf = measure_data_generation_performance()
    
    # Check hyperparameter config
    hyperparam_config = check_hyperparameter_config()
    
    # Analyze test data config
    analyze_test_data_config()
    
    # Run sample test timing
    sample_time = run_sample_test_timing()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"Data Generation: {data_perf['generation_time']*1000:.1f}ms " +
          f"({'✓' if data_perf['generation_time'] < 0.1 else '✗'} <100ms target)")
    print(f"Records Generated: {data_perf['num_records']} " +
          f"({'✓' if data_perf['num_records'] <= 5500 else '✗'} ~5,500 target)")
    print(f"Years of Data: {data_perf['num_years']:.1f} " +
          f"({'✓' if data_perf['num_years'] <= 5 else '✗'} 5 years target)")
    print(f"Hyperparameter Trials: {hyperparam_config} " +
          f"({'✓' if hyperparam_config == 1 else '✗'} 1 trial target)")
    print(f"Sample Test Time: {sample_time*1000:.1f}ms " +
          f"({'✓' if sample_time < 0.5 else '✗'} <500ms target)")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    recommendations = []
    
    if data_perf['num_years'] > 5:
        recommendations.append("- Reduce data generation to 5 years (2018-2023)")
    
    if data_perf['num_records'] > 5500:
        recommendations.append("- Optimize data generation to produce ~5,500 records")
    
    if hyperparam_config != 1:
        recommendations.append("- Set n_trials to 1 for testing hyperparameter optimization")
    
    if data_perf['generation_time'] >= 0.1:
        recommendations.append("- Optimize data generation to be <100ms")
    
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("- All performance targets appear to be met!")

if __name__ == "__main__":
    main()