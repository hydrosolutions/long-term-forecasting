"""
Tests for data_utils module, focusing on normalization and inverse normalization functions.
"""

import pytest
import pandas as pd
import numpy as np
from scr import data_utils as du


class TestNormalizationFunctions:
    """Test normalization and inverse normalization functions."""
    
    def test_apply_inverse_normalization_global(self):
        """Test global inverse normalization correctly reverses normalization."""
        # Create test data
        df = pd.DataFrame({
            'feature1': [10.0, 20.0, 30.0, 40.0, 50.0],
            'feature2': [5.0, 10.0, 15.0, 20.0, 25.0],
            'target': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        
        # Calculate normalization parameters
        features = ['feature1', 'feature2']
        target = 'target'
        scaler = du.get_normalization_params(df, features, target)
        
        # Apply normalization
        df_normalized = du.apply_normalization(df.copy(), scaler, features + [target])
        
        # Apply inverse normalization
        df_restored = du.apply_inverse_normalization(
            df_normalized.copy(), 
            scaler, 
            var_to_scale='target',
            var_used_for_scaling='target'
        )
        
        # Check that we get back the original values
        np.testing.assert_array_almost_equal(
            df['target'].values, 
            df_restored['target'].values,
            decimal=10
        )
    
    def test_apply_inverse_normalization_per_basin(self):
        """Test per-basin inverse normalization correctly reverses normalization."""
        # Create test data with multiple basins
        df = pd.DataFrame({
            'code': [1, 1, 1, 2, 2, 2],
            'feature1': [10.0, 20.0, 30.0, 100.0, 200.0, 300.0],
            'target': [5.0, 10.0, 15.0, 50.0, 100.0, 150.0]
        })
        
        # Calculate per-basin normalization parameters
        features = ['feature1']
        target = 'target'
        scaler = du.get_normalization_params_per_basin(df, features, target)
        
        # Apply normalization
        df_normalized = du.apply_normalization_per_basin(
            df.copy(), scaler, features + [target]
        )
        
        # Apply inverse normalization
        df_restored = du.apply_inverse_normalization_per_basin(
            df_normalized.copy(),
            scaler,
            var_to_scale='target',
            var_used_for_scaling='target'
        )
        
        # Check that we get back the original values
        np.testing.assert_array_almost_equal(
            df['target'].values,
            df_restored['target'].values,
            decimal=10
        )
    
    def test_apply_inverse_normalization_different_variable(self):
        """Test inverse normalization when variable to scale differs from scaling variable."""
        # Create test data
        df = pd.DataFrame({
            'prediction': [0.5, 1.0, 1.5, 2.0, 2.5],  # Already normalized predictions
            'target': [100.0, 200.0, 300.0, 400.0, 500.0]  # Original scale target
        })
        
        # Create scaler with known parameters
        scaler = {
            'target': (300.0, 158.11388300841898)  # mean=300, std≈158.11
        }
        
        # Apply inverse normalization to predictions using target's scaling
        df_restored = du.apply_inverse_normalization(
            df.copy(),
            scaler,
            var_to_scale='prediction',
            var_used_for_scaling='target'
        )
        
        # Calculate expected values
        mean, std = scaler['target']
        expected = df['prediction'].values * std + mean
        
        # Check results
        np.testing.assert_array_almost_equal(
            expected,
            df_restored['prediction'].values,
            decimal=10
        )
    
    def test_apply_inverse_normalization_missing_column(self):
        """Test inverse normalization handles missing columns gracefully."""
        df = pd.DataFrame({
            'feature1': [1.0, 2.0, 3.0]
        })
        
        scaler = {
            'target': (100.0, 50.0)
        }
        
        # Should return unchanged when column doesn't exist
        df_result = du.apply_inverse_normalization(
            df.copy(),
            scaler,
            var_to_scale='missing_column',
            var_used_for_scaling='target'
        )
        
        pd.testing.assert_frame_equal(df, df_result)
    
    def test_apply_inverse_normalization_missing_scaler_key(self):
        """Test inverse normalization handles missing scaler keys gracefully."""
        df = pd.DataFrame({
            'prediction': [1.0, 2.0, 3.0]
        })
        
        scaler = {
            'other_variable': (100.0, 50.0)
        }
        
        # Should return unchanged when scaler key doesn't exist
        df_result = du.apply_inverse_normalization(
            df.copy(),
            scaler,
            var_to_scale='prediction',
            var_used_for_scaling='missing_key'
        )
        
        pd.testing.assert_frame_equal(df, df_result)
    
    def test_normalization_inverse_consistency(self):
        """Test that normalization followed by inverse gives original values."""
        # Create test data
        np.random.seed(42)
        df = pd.DataFrame({
            'feature1': np.random.randn(100) * 10 + 50,
            'feature2': np.random.randn(100) * 5 + 20,
            'target': np.random.randn(100) * 15 + 100
        })
        
        features = ['feature1', 'feature2']
        target = 'target'
        
        # Get normalization parameters
        scaler = du.get_normalization_params(df, features, target)
        
        # Apply normalization
        df_normalized = du.apply_normalization(df.copy(), scaler, features + [target])
        
        # Apply inverse normalization to all columns
        df_restored = df_normalized.copy()
        for col in features + [target]:
            df_restored = du.apply_inverse_normalization(
                df_restored,
                scaler,
                var_to_scale=col,
                var_used_for_scaling=col
            )
        
        # Check all columns are restored
        for col in features + [target]:
            np.testing.assert_array_almost_equal(
                df[col].values,
                df_restored[col].values,
                decimal=10
            )
    
    def test_per_basin_normalization_different_scales(self):
        """Test per-basin normalization with basins having very different scales."""
        # Create test data with different scales per basin
        df = pd.DataFrame({
            'code': [1] * 5 + [2] * 5,
            'discharge': [10, 20, 30, 40, 50, 1000, 2000, 3000, 4000, 5000],
            'target': [15, 25, 35, 45, 55, 1500, 2500, 3500, 4500, 5500]
        })
        
        features = ['discharge']
        target = 'target'
        
        # Get per-basin normalization parameters
        scaler = du.get_normalization_params_per_basin(df, features, target)
        
        # Apply normalization
        df_normalized = du.apply_normalization_per_basin(
            df.copy(), scaler, features + [target]
        )
        
        # Check that each basin is normalized to mean≈0, std≈1
        for code in df['code'].unique():
            basin_data = df_normalized[df_normalized['code'] == code]
            assert abs(basin_data['discharge'].mean()) < 1e-10
            assert abs(basin_data['discharge'].std() - 1.0) < 1e-10
            assert abs(basin_data['target'].mean()) < 1e-10
            assert abs(basin_data['target'].std() - 1.0) < 1e-10
        
        # Apply inverse normalization
        df_restored = df_normalized.copy()
        for col in features + [target]:
            df_restored = du.apply_inverse_normalization_per_basin(
                df_restored,
                scaler,
                var_to_scale=col,
                var_used_for_scaling=col
            )
        
        # Check restoration (values should match, but dtypes may differ due to float conversion)
        # Check each column separately to handle different dtypes
        for col in ['code'] + features + [target]:
            np.testing.assert_array_almost_equal(
                df[col].values,
                df_restored[col].values,
                decimal=10
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])