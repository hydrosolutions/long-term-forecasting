"""
Integration test for long-term mean scaling with real workflow.
"""

import pytest
import pandas as pd
import numpy as np
from scr.FeatureProcessingArtifacts import (
    FeatureProcessingArtifacts,
    process_training_data,
    process_test_data,
    post_process_predictions,
)
from scr import data_utils as du


def test_complete_long_term_mean_workflow():
    """Test complete workflow with long-term mean scaling."""
    # Create simple test data
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2020-12-31', freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'code': 'A',
        'T_mean': 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.randn(len(dates)),
        'discharge': 100 + 50 * np.sin(2 * np.pi * (dates.dayofyear - 30) / 365) + 10 * np.random.randn(len(dates))
    })
    df['target'] = df['discharge']
    
    # Split train/test
    train_mask = df['date'] < '2020-10-01'
    df_train = df[train_mask].copy()
    df_test = df[~train_mask].copy()
    
    # Configuration
    features = ['T_mean']
    experiment_config = {
        'normalize': True,
        'normalization_type': 'long_term_mean',
        'relative_scaling_vars': ['T'],
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
    
    # Verify training data is scaled
    assert df_train_processed['target'].mean() < 5  # Should be around 1
    assert df_train_processed['T_mean'].mean() < 5
    
    # Process test data
    df_test_processed = process_test_data(
        df_test,
        artifacts=artifacts,
        experiment_config=experiment_config
    )
    
    # Create mock predictions (copy scaled target as prediction)
    df_test_processed['prediction'] = df_test_processed['target'].mean()
    
    # Apply post-processing
    df_restored = post_process_predictions(
        df_test_processed.copy(),
        artifacts=artifacts,
        experiment_config=experiment_config,
        prediction_column='prediction',
        target='target'
    )
    
    # Check that predictions are denormalized
    # The prediction should be close to the mean discharge value
    assert 50 < df_restored['prediction'].mean() < 150  # Should be in discharge range
    assert df_restored['prediction'].iloc[0] == df_restored['prediction'].iloc[1]  # Constant prediction
    
    # Features should still be scaled
    assert df_restored['T_mean'].mean() < 5


def test_inverse_scaling_directly():
    """Test inverse scaling function directly."""
    # Create simple data
    dates = pd.date_range(start='2020-01-01', end='2020-03-31', freq='MS')
    df = pd.DataFrame({
        'date': dates,
        'code': 'A',
        'discharge': [100, 120, 110]
    })
    
    # Calculate long-term means
    ltm = du.get_long_term_mean_per_basin(df, ['discharge'])
    
    # Apply scaling
    df_scaled, metadata = du.apply_long_term_mean_scaling(
        df.copy(), ltm, ['discharge'],
        use_relative_target=True
    )
    
    # Create prediction data
    df_pred = df_scaled.copy()
    df_pred['prediction'] = 1.2  # 20% above normal
    
    # Apply inverse scaling
    df_restored = du.apply_inverse_long_term_mean_scaling(
        df_pred,
        ltm,
        var_to_scale='prediction',
        var_used_for_scaling='discharge',
        scaling_metadata=metadata
    )
    
    # Check results
    # Each prediction should be 1.2 times the corresponding month's mean
    for i, row in df_restored.iterrows():
        period = row['period']
        ltm_value = ltm[ltm['period'] == period]['discharge'].iloc[0, 0]
        expected = 1.2 * ltm_value
        assert abs(row['prediction'] - expected) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])