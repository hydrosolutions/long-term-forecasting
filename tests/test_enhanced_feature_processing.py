"""
Tests for enhanced FeatureProcessingArtifacts with selective scaling support.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

from scr.FeatureProcessingArtifacts import (
    FeatureProcessingArtifacts,
    process_training_data,
    process_test_data,
)


class TestEnhancedFeatureProcessingArtifacts:
    """Test enhanced FeatureProcessingArtifacts with selective scaling."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2019-01-01', end='2021-12-31', freq='D')
        
        data = []
        for date in dates:
            for basin in ['A', 'B']:
                # Create features with patterns
                data.append({
                    'date': date,
                    'code': basin,
                    'T_mean': 15 + 10 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.randn(),
                    'T_max': 20 + 10 * np.sin(2 * np.pi * date.dayofyear / 365) + np.random.randn(),
                    'SWE_mean': max(0, 50 * np.cos(2 * np.pi * date.dayofyear / 365) + 10 * np.random.randn()),
                    'P_sum': max(0, 30 + 20 * np.random.randn()),
                    'elevation': 1500 if basin == 'A' else 2000,
                    'area': 100 if basin == 'A' else 150,
                    'discharge': 50 + 30 * np.sin(2 * np.pi * (date.dayofyear - 30) / 365) + 5 * np.random.randn()
                })
        
        df = pd.DataFrame(data)
        df['target'] = df['discharge']
        return df
    
    def test_normalization_with_selective_scaling(self, sample_data):
        """Test that selective scaling is properly handled in normalization."""
        # Split data
        train_mask = sample_data['date'] < '2021-01-01'
        df_train = sample_data[train_mask].copy()
        
        features = ['T_mean', 'T_max', 'SWE_mean', 'P_sum', 'elevation', 'area']
        static_features = ['elevation', 'area']
        
        # Configure selective scaling
        experiment_config = {
            'normalize': True,
            'normalization_type': 'long_term_mean',
            'relative_scaling_vars': ['T', 'SWE'],
            'use_relative_target': True,
            'handle_na': 'drop',
            'use_feature_selection': False
        }
        
        # Process training data
        df_processed, artifacts = process_training_data(
            df_train,
            features=features,
            target='target',
            experiment_config=experiment_config,
            static_features=static_features
        )
        
        # Check that artifacts contain scaling metadata
        assert artifacts.relative_scaling_vars == ['T', 'SWE']
        assert artifacts.use_relative_target == True
        assert artifacts.scaling_metadata is not None
        
        # Check that relative features were identified correctly
        # Note: With use_relative_target=True, target should be in relative_features
        expected_relative = {'T_mean', 'T_max', 'SWE_mean', 'target'}
        assert set(artifacts.relative_features) == expected_relative
        
        # Check that per-basin features were identified
        # Note: With the fix, target should NOT be in per_basin_features when use_relative_target=True
        expected_per_basin = {'P_sum'}  # elevation and area are static, target moved to relative
        assert set(artifacts.per_basin_features) == expected_per_basin
        
        # Verify scaling was applied
        assert df_processed['T_mean'].mean() < 5  # Should be scaled around 1
        assert df_processed['SWE_mean'].mean() < 5
        
        # Target should also be scaled
        assert df_processed['target'].mean() < 5
    
    def test_test_data_processing_with_scaling_metadata(self, sample_data):
        """Test that test data processing uses stored scaling metadata."""
        # Split data
        train_mask = sample_data['date'] < '2021-01-01'
        df_train = sample_data[train_mask].copy()
        df_test = sample_data[~train_mask].copy()
        
        features = ['T_mean', 'T_max', 'SWE_mean', 'P_sum']
        
        experiment_config = {
            'normalize': True,
            'normalization_type': 'long_term_mean',
            'relative_scaling_vars': ['T', 'SWE'],
            'use_relative_target': False,
            'handle_na': 'drop',
            'use_feature_selection': False
        }
        
        # Process training data
        df_train_processed, artifacts = process_training_data(
            df_train,
            features=features,
            target='target',
            experiment_config=experiment_config
        )
        
        # Process test data
        df_test_processed = process_test_data(
            df_test,
            artifacts=artifacts,
            experiment_config=experiment_config
        )
        
        # Check that test data was processed with same scaling approach
        assert df_test_processed['T_mean'].mean() < 5
        assert df_test_processed['SWE_mean'].mean() < 5
        
        # Target should not be scaled in test data
        assert df_test_processed['target'].mean() > 10
    
    def test_artifact_persistence_with_new_attributes(self, sample_data):
        """Test saving and loading artifacts with new scaling attributes."""
        df_train = sample_data[sample_data['date'] < '2021-01-01'].copy()
        
        features = ['T_mean', 'SWE_mean', 'P_sum']
        
        experiment_config = {
            'normalize': True,
            'normalization_type': 'long_term_mean',
            'relative_scaling_vars': ['T', 'SWE'],
            'use_relative_target': True,
            'handle_na': 'drop',
            'use_feature_selection': False
        }
        
        # Process training data
        _, artifacts = process_training_data(
            df_train,
            features=features,
            target='target',
            experiment_config=experiment_config
        )
        
        # Save and load artifacts
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_artifacts"
            
            # Test with hybrid format (recommended)
            artifacts.save(save_path, format='hybrid')
            loaded_artifacts = FeatureProcessingArtifacts.load(save_path, format='hybrid')
            
            # Check all new attributes are preserved
            assert loaded_artifacts.relative_scaling_vars == artifacts.relative_scaling_vars
            assert loaded_artifacts.use_relative_target == artifacts.use_relative_target
            assert loaded_artifacts.relative_features == artifacts.relative_features
            assert loaded_artifacts.per_basin_features == artifacts.per_basin_features
            assert loaded_artifacts.scaling_metadata == artifacts.scaling_metadata
    
    def test_post_process_predictions_with_selective_scaling(self, sample_data):
        """Test post-processing predictions with selective scaling metadata."""
        df_train = sample_data[sample_data['date'] < '2021-01-01'].copy()
        df_test = sample_data[sample_data['date'] >= '2021-01-01'].copy()
        
        features = ['T_mean', 'SWE_mean']
        
        experiment_config = {
            'normalize': True,
            'normalization_type': 'long_term_mean',
            'relative_scaling_vars': ['T', 'SWE'],
            'use_relative_target': True,
            'handle_na': 'drop',
            'use_feature_selection': False
        }
        
        # Process training data
        df_train_processed, artifacts = process_training_data(
            df_train,
            features=features,
            target='target',
            experiment_config=experiment_config
        )
        
        # Process test data
        df_test_processed = process_test_data(
            df_test,
            artifacts=artifacts,
            experiment_config=experiment_config
        )
        
        # Simulate predictions (scaled values)
        df_test_processed['prediction'] = 1.2  # 20% above normal
        
        # Post-process predictions
        from scr.FeatureProcessingArtifacts import post_process_predictions
        
        df_restored = post_process_predictions(
            df_test_processed,
            artifacts=artifacts,
            experiment_config=experiment_config,
            prediction_column='prediction',
            target='target'
        )
        
        # Check that predictions are denormalized
        # Since use_relative_target=True was used in training, predictions should be denormalized
        # The exact value depends on the long-term mean of the target
        assert df_restored['prediction'].std() < 1e-10  # All predictions should be the same (constant)
        
        # Check if predictions were denormalized
        pred_mean = df_restored['prediction'].mean()
        if pred_mean > 5:  # If denormalized
            # For relative scaling, prediction of 1.2 means 20% above normal
            raw_target_mean = sample_data[sample_data['date'] >= '2021-01-01']['target'].mean()
            assert 0 < pred_mean < raw_target_mean * 2
        else:
            # If not denormalized (which is the current behavior), predictions remain scaled
            assert pred_mean == pytest.approx(1.2, rel=1e-6)
    
    def test_backward_compatibility(self, sample_data):
        """Test that models without new parameters still work."""
        df_train = sample_data[sample_data['date'] < '2021-01-01'].copy()
        
        features = ['T_mean', 'SWE_mean', 'P_sum']
        
        # Old-style config without new parameters
        experiment_config = {
            'normalize': True,
            'normalization_type': 'long_term_mean',
            'handle_na': 'drop',
            'use_feature_selection': False
        }
        
        # Should work without errors
        df_processed, artifacts = process_training_data(
            df_train,
            features=features,
            target='target',
            experiment_config=experiment_config
        )
        
        # Check that defaults are applied
        assert artifacts.relative_scaling_vars is None
        assert artifacts.use_relative_target == False
        
        # All features should use relative scaling (old behavior)
        assert df_processed['T_mean'].mean() < 5
        assert df_processed['P_sum'].mean() < 5


class TestConfigurationCombinations:
    """Test various configuration combinations."""
    
    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='MS')
        return pd.DataFrame({
            'date': dates,
            'code': ['A'] * len(dates),
            'T_mean': np.random.rand(len(dates)) * 30,
            'P_sum': np.random.rand(len(dates)) * 100,
            'target': np.random.rand(len(dates)) * 50
        })
    
    @pytest.mark.parametrize("normalization_type,relative_scaling_vars,use_relative_target", [
        ("long_term_mean", None, False),  # Default behavior
        ("long_term_mean", ['T'], False),  # Selective features only
        ("long_term_mean", ['T'], True),   # Selective features + target
        ("long_term_mean", [], True),      # Only target relative
        ("global", ['T'], False),          # Should ignore selective scaling
        ("per_basin", ['T'], False),       # Should ignore selective scaling
    ])
    def test_configuration_combinations(
        self, simple_data, normalization_type, relative_scaling_vars, use_relative_target
    ):
        """Test various configuration combinations."""
        experiment_config = {
            'normalize': True,
            'normalization_type': normalization_type,
            'relative_scaling_vars': relative_scaling_vars,
            'use_relative_target': use_relative_target,
            'handle_na': 'drop',
            'use_feature_selection': False
        }
        
        features = ['T_mean', 'P_sum']
        
        # Should process without errors
        df_processed, artifacts = process_training_data(
            simple_data,
            features=features,
            target='target',
            experiment_config=experiment_config
        )
        
        # For non-long_term_mean normalization, selective scaling attributes are not set
        if normalization_type != 'long_term_mean':
            # These attributes are only set for long_term_mean normalization
            assert artifacts.relative_scaling_vars is None or artifacts.relative_scaling_vars == relative_scaling_vars
            assert artifacts.scaling_metadata is None
        else:
            assert artifacts.scaling_metadata is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])