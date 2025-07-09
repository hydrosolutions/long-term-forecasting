#!/usr/bin/env python3
"""
Verification script to demonstrate SciRegressor test performance improvements.
"""

import time
import sys
sys.path.append('tests')

from comprehensive_test_utils import ComprehensiveTestDataGenerator
from test_sciregressor import LegacyTestDataGenerator

def compare_data_generation_performance():
    """Compare performance of old vs new data generation."""
    print("=== SciRegressor Data Generation Performance Comparison ===")
    
    # Test comprehensive data generation (new optimized version)
    print("\n1. Comprehensive Data Generation (Optimized)")
    start_time = time.time()
    
    comprehensive_data = ComprehensiveTestDataGenerator.generate_comprehensive_timeseries_data()
    comprehensive_static = ComprehensiveTestDataGenerator.generate_comprehensive_static_data()
    
    comprehensive_time = time.time() - start_time
    
    print(f"   Time: {comprehensive_time:.3f} seconds")
    print(f"   Records: {len(comprehensive_data)}")
    print(f"   Date range: {comprehensive_data['date'].min()} to {comprehensive_data['date'].max()}")
    print(f"   Years: {(comprehensive_data['date'].max() - comprehensive_data['date'].min()).days / 365.25:.1f}")
    
    # Test legacy data generation (old version)
    print("\n2. Legacy Data Generation (Updated)")
    start_time = time.time()
    
    legacy_data = LegacyTestDataGenerator.generate_synthetic_timeseries_data()
    legacy_static = LegacyTestDataGenerator.generate_synthetic_static_data()
    
    legacy_time = time.time() - start_time
    
    print(f"   Time: {legacy_time:.3f} seconds")
    print(f"   Records: {len(legacy_data)}")
    print(f"   Date range: {legacy_data['date'].min()} to {legacy_data['date'].max()}")
    print(f"   Years: {(legacy_data['date'].max() - legacy_data['date'].min()).days / 365.25:.1f}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    print(f"Comprehensive generation: {comprehensive_time:.3f}s")
    print(f"Legacy generation: {legacy_time:.3f}s")
    print(f"Performance improvement: {legacy_time/comprehensive_time:.1f}x faster" if comprehensive_time < legacy_time else "Similar performance")
    
    # Data characteristics comparison
    print("\n=== Data Characteristics ===")
    print(f"Comprehensive: {len(comprehensive_data)} records, {(comprehensive_data['date'].max() - comprehensive_data['date'].min()).days / 365.25:.1f} years")
    print(f"Legacy: {len(legacy_data)} records, {(legacy_data['date'].max() - legacy_data['date'].min()).days / 365.25:.1f} years")
    
    # Check if both use optimal settings
    print("\n=== Optimization Check ===")
    comp_years = (comprehensive_data['date'].max() - comprehensive_data['date'].min()).days / 365.25
    legacy_years = (legacy_data['date'].max() - legacy_data['date'].min()).days / 365.25
    
    print(f"✓ Comprehensive uses 5 years: {comp_years:.1f} years" if abs(comp_years - 5) < 0.1 else f"✗ Comprehensive uses {comp_years:.1f} years")
    print(f"✓ Legacy uses 5 years: {legacy_years:.1f} years" if abs(legacy_years - 5) < 0.1 else f"✗ Legacy uses {legacy_years:.1f} years")
    print(f"✓ Comprehensive ~5,500 records: {len(comprehensive_data)}" if abs(len(comprehensive_data) - 5500) < 1000 else f"✗ Comprehensive {len(comprehensive_data)} records")
    print(f"✓ Legacy ~5,500 records: {len(legacy_data)}" if abs(len(legacy_data) - 5500) < 1000 else f"✗ Legacy {len(legacy_data)} records")

if __name__ == "__main__":
    compare_data_generation_performance()