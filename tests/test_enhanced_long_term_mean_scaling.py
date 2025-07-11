"""
Tests for enhanced long-term mean scaling functionality with period-based grouping and selective scaling.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scr import data_utils as du


class TestPeriodFunctions:
    """Test period-based grouping functions."""
    
    def test_get_periods(self):
        """Test get_periods function creates correct period format."""
        # Create test data with various dates
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
        df = pd.DataFrame({'date': dates})
        
        # Apply get_periods
        df_with_periods = du.get_periods(df)
        
        # Check that period column was added
        assert 'period' in df_with_periods.columns
        
        # Check specific dates
        # January 15th should be "1-15"
        jan_15_idx = df_with_periods[df_with_periods['date'] == '2020-01-15'].index[0]
        assert df_with_periods.loc[jan_15_idx, 'period'] == '1-15'
        
        # Last day of February (leap year) should be "2-end"
        feb_29_idx = df_with_periods[df_with_periods['date'] == '2020-02-29'].index[0]
        assert df_with_periods.loc[feb_29_idx, 'period'] == '2-end'
        
        # Last day of April should be "4-end"
        apr_30_idx = df_with_periods[df_with_periods['date'] == '2020-04-30'].index[0]
        assert df_with_periods.loc[apr_30_idx, 'period'] == '4-end'
        
        # Check that period_suffix column was removed
        assert 'period_suffix' not in df_with_periods.columns
    
    def test_get_periods_non_leap_year(self):
        """Test get_periods handles non-leap years correctly."""
        # Create test data for non-leap year
        dates = pd.date_range(start='2021-02-01', end='2021-03-01', freq='D')
        df = pd.DataFrame({'date': dates})
        
        df_with_periods = du.get_periods(df)
        
        # February 28th should be "2-end" in non-leap year
        feb_28_idx = df_with_periods[df_with_periods['date'] == '2021-02-28'].index[0]
        assert df_with_periods.loc[feb_28_idx, 'period'] == '2-end'


class TestPeriodBasedLongTermMean:
    """Test period-based long-term mean calculations."""
    
    def test_get_long_term_mean_per_basin_with_periods(self):
        """Test long-term mean calculation uses periods instead of months."""
        # Create test data with multiple years
        dates = []
        codes = []
        discharge_values = []
        temp_values = []
        
        for year in [2019, 2020, 2021]:
            for month in [1, 2, 3]:
                for day in [5, 15, 25, 28 if month == 2 else 30]:
                    if month == 2 and day == 30:
                        continue
                    for basin in ['A', 'B']:
                        dates.append(pd.Timestamp(year, month, day))
                        codes.append(basin)
                        # Create predictable values
                        base_value = 100 * month + day
                        discharge_values.append(base_value * (1.1 if basin == 'B' else 1.0))
                        temp_values.append(20 + month + day/10)
        
        df = pd.DataFrame({
            'date': dates,
            'code': codes,
            'discharge': discharge_values,
            'temperature': temp_values
        })
        
        # Calculate long-term mean
        features = ['discharge', 'temperature']
        ltm = du.get_long_term_mean_per_basin(df, features)
        
        # Check that period column exists in result
        assert 'period' in ltm.columns
        
        # Check that we have the expected number of unique periods
        # Note: February has both "2-28" and "2-end" which are the same in non-leap years
        # So we expect: (4 days * 2 months + 5 days for Feb) * 2 basins = 26 rows
        assert len(ltm) == 26
        
        # Check specific period values
        # Basin A, January 5th
        mask = (ltm['code'] == 'A') & (ltm['period'] == '1-5')
        assert mask.sum() == 1
        
        # Basin B, February end
        mask = (ltm['code'] == 'B') & (ltm['period'] == '2-end')
        assert mask.sum() == 1


class TestSelectiveFeatureScaling:
    """Test selective feature scaling based on variable patterns."""
    
    def test_get_relative_scaling_features(self):
        """Test identification of features for relative vs per-basin scaling."""
        features = [
            'SWE_mean', 'SWE_max', 'SWE_min',
            'T_mean', 'T_max', 
            'P_sum',
            'discharge_lag1',
            'elevation', 'area',
            'NDVI_mean'
        ]
        
        relative_scaling_vars = ['SWE', 'T', 'discharge']
        
        relative_features, per_basin_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )
        
        # Check relative features
        expected_relative = ['SWE_mean', 'SWE_max', 'SWE_min', 'T_mean', 'T_max', 'discharge_lag1']
        assert set(relative_features) == set(expected_relative)
        
        # Check per-basin features
        expected_per_basin = ['P_sum', 'elevation', 'area', 'NDVI_mean']
        assert set(per_basin_features) == set(expected_per_basin)
    
    def test_get_relative_scaling_features_empty_list(self):
        """Test with empty relative_scaling_vars list."""
        features = ['SWE_mean', 'T_max', 'P_sum']
        relative_scaling_vars = []
        
        relative_features, per_basin_features = du.get_relative_scaling_features(
            features, relative_scaling_vars
        )
        
        # All features should be per-basin
        assert relative_features == []
        assert per_basin_features == features


class TestEnhancedLongTermMeanScaling:
    """Test enhanced long-term mean scaling with selective features."""
    
    def test_apply_long_term_mean_scaling_with_selective_features(self):
        """Test scaling with selective feature handling."""
        # Create test data
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='MS')
        df = pd.DataFrame({
            'date': list(dates) + list(dates),
            'code': ['A'] * len(dates) + ['B'] * len(dates),
            'SWE_mean': np.random.rand(len(dates) * 2) * 100,
            'T_mean': np.random.rand(len(dates) * 2) * 30,
            'P_sum': np.random.rand(len(dates) * 2) * 50,
            'elevation': [1000] * len(dates) + [2000] * len(dates),
            'target': np.random.rand(len(dates) * 2) * 200
        })
        
        features = ['SWE_mean', 'T_mean', 'P_sum', 'elevation']
        
        # First calculate long-term means
        ltm = du.get_long_term_mean_per_basin(df, features + ['target'])
        
        # Apply scaling with selective features
        df_scaled, metadata = du.apply_long_term_mean_scaling(
            df, ltm, features,
            relative_scaling_vars=['SWE', 'T'],
            use_relative_target=True
        )
        
        # Check metadata
        assert 'relative_features' in metadata
        assert 'per_basin_features' in metadata
        assert set(metadata['relative_features']) == {'SWE_mean', 'T_mean', 'target'}
        assert set(metadata['per_basin_features']) == {'P_sum', 'elevation'}
        
        # Check that target was scaled
        assert 'target' in df_scaled.columns
        
        # Verify scaling was applied (values should be around 1.0 for relative scaling)
        assert df_scaled['SWE_mean'].mean() < 10  # Should be close to 1
        assert df_scaled['T_mean'].mean() < 10
        
        # Per-basin features should remain unchanged
        np.testing.assert_array_equal(df['P_sum'].values, df_scaled['P_sum'].values)
        np.testing.assert_array_equal(df['elevation'].values, df_scaled['elevation'].values)
    
    def test_apply_inverse_scaling_with_metadata(self):
        """Test inverse scaling using metadata."""
        # Create simple test data
        dates = pd.date_range(start='2020-01-01', end='2020-03-01', freq='MS')
        df = pd.DataFrame({
            'date': dates,
            'code': ['A', 'A', 'A'],
            'discharge': [100, 120, 110],
            'target': [100, 120, 110]
        })
        
        # Calculate long-term means
        ltm = du.get_long_term_mean_per_basin(df, ['target'])
        
        # Apply scaling
        df_scaled, metadata = du.apply_long_term_mean_scaling(
            df, ltm, ['target'],
            use_relative_target=True
        )
        
        # Create prediction data (copy scaled target to prediction)
        df_pred = df_scaled.copy()
        df_pred['prediction'] = df_pred['target']
        
        # Apply inverse scaling
        df_restored = du.apply_inverse_long_term_mean_scaling(
            df_pred,
            ltm,
            var_to_scale='prediction',
            var_used_for_scaling='target',
            scaling_metadata=metadata
        )
        
        # Check that predictions are restored to original scale
        np.testing.assert_array_almost_equal(
            df['target'].values,
            df_restored['prediction'].values,
            decimal=10
        )


class TestIntegrationScenarios:
    """Test complete integration scenarios."""
    
    def test_full_pipeline_with_period_scaling(self):
        """Test complete pipeline from raw data to predictions."""
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range(start='2018-01-01', end='2020-12-31', freq='D')
        
        data = []
        for date in dates:
            for basin in ['Basin_A', 'Basin_B']:
                # Create seasonal patterns
                day_of_year = date.dayofyear
                temp = 15 + 10 * np.sin(2 * np.pi * day_of_year / 365) + np.random.randn()
                swe = max(0, 50 * np.cos(2 * np.pi * day_of_year / 365) + 10 * np.random.randn())
                discharge = 50 + 30 * np.sin(2 * np.pi * (day_of_year - 30) / 365) + 5 * np.random.randn()
                
                data.append({
                    'date': date,
                    'code': basin,
                    'T_mean': temp,
                    'SWE_mean': swe,
                    'discharge': discharge,
                    'target': discharge  # For simplicity
                })
        
        df = pd.DataFrame(data)
        
        # Split data
        train_mask = df['date'] < '2020-01-01'
        df_train = df[train_mask].copy()
        df_test = df[~train_mask].copy()
        
        features = ['T_mean', 'SWE_mean']
        
        # Training: Calculate long-term means and apply scaling
        ltm = du.get_long_term_mean_per_basin(df_train, features + ['target'])
        df_train_scaled, metadata = du.apply_long_term_mean_scaling(
            df_train, ltm, features + ['target'],
            relative_scaling_vars=['T', 'SWE'],
            use_relative_target=True
        )
        
        # Testing: Apply same scaling
        df_test_scaled, _ = du.apply_long_term_mean_scaling(
            df_test, ltm, features,
            relative_scaling_vars=['T', 'SWE'],
            use_relative_target=False  # Don't scale target in test
        )
        
        # Simulate predictions (just copy scaled training target mean)
        df_test_scaled['prediction'] = df_train_scaled['target'].mean()
        
        # Apply inverse scaling
        df_test_restored = du.apply_inverse_long_term_mean_scaling(
            df_test_scaled,
            ltm,
            var_to_scale='prediction',
            var_used_for_scaling='target',
            scaling_metadata=metadata
        )
        
        # Check that predictions are in reasonable range
        assert df_test_restored['prediction'].min() > 0
        assert df_test_restored['prediction'].max() < 200
        
        # Check that features remain scaled
        assert df_test_restored['T_mean'].mean() < 10  # Should still be scaled
        assert df_test_restored['SWE_mean'].mean() < 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])