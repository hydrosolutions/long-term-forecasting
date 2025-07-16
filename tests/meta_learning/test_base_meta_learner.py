"""
Tests for base meta-learner module.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
from datetime import datetime

from monthly_forecasting.forecast_models.meta_learners.base_meta_learner import BaseMetaLearner


class TestBaseMetaLearner:
    """Test base meta-learner functionality."""
    
    def setup_method(self):
        """Set up test data and configurations."""
        # Create synthetic data
        np.random.seed(42)
        self.n_samples = 100
        
        # Create test data
        dates = pd.date_range('2020-01-01', periods=self.n_samples, freq='D')
        codes = ['BASIN01', 'BASIN02'] * (self.n_samples // 2)
        
        self.data = pd.DataFrame({
            'date': dates,
            'code': codes,
            'Q': np.random.randn(self.n_samples) * 10 + 50,
            'T': np.random.randn(self.n_samples) * 5 + 15,
            'P': np.random.randn(self.n_samples) * 20 + 100
        })
        
        self.static_data = pd.DataFrame({
            'code': ['BASIN01', 'BASIN02'],
            'area': [1000, 1500],
            'elevation': [500, 800]
        })
        
        # Create configurations
        self.general_config = {
            'model_name': 'test_meta_learner',
            'target_column': 'Q',
            'date_column': 'date',
            'code_column': 'code'
        }
        
        self.model_config = {
            'meta_learning': {
                'ensemble_method': 'weighted_mean',
                'weighting_strategy': 'performance_based',
                'performance_metric': 'rmse'
            }
        }
        
        self.feature_config = {
            'feature_columns': ['T', 'P'],
            'lag_features': [1, 2, 3]
        }
        
        self.path_config = {
            'model_dir': '/tmp/test_models',
            'output_dir': '/tmp/test_output'
        }
        
        # Create base model predictions
        self.base_model_predictions = self._create_test_predictions()
    
    def _create_test_predictions(self):
        """Create synthetic base model predictions."""
        predictions = {}
        
        # Create predictions for 3 base models
        for i, model_name in enumerate(['XGB', 'LGBM', 'CatBoost']):
            dates = pd.date_range('2020-01-01', periods=50, freq='D')
            codes = ['BASIN01', 'BASIN02'] * 25
            
            # Add some model-specific bias/noise
            obs = np.random.randn(50) * 10 + 50
            pred = obs + np.random.randn(50) * (1 + i * 0.5)  # Different noise levels
            
            predictions[model_name] = pd.DataFrame({
                'date': dates,
                'code': codes,
                'Q_obs': obs,
                'Q_pred': pred,
                'model': model_name
            })
        
        return predictions
    
    def test_initialization(self):
        """Test meta-learner initialization."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        assert meta_learner.name == 'test_meta_learner'
        assert meta_learner.ensemble_method == 'weighted_mean'
        assert meta_learner.weighting_strategy == 'performance_based'
        assert meta_learner.performance_metric == 'rmse'
        assert len(meta_learner.base_model_predictions) == 3
    
    def test_invalid_configuration(self):
        """Test initialization with invalid configuration."""
        invalid_config = self.model_config.copy()
        invalid_config['meta_learning']['ensemble_method'] = 'invalid_method'
        
        with pytest.raises(ValueError, match="Invalid ensemble_method"):
            ConcreteMetaLearner(
                data=self.data,
                static_data=self.static_data,
                general_config=self.general_config,
                model_config=invalid_config,
                feature_config=self.feature_config,
                path_config=self.path_config
            )
    
    def test_add_base_model_predictions(self):
        """Test adding base model predictions."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config
        )
        
        # Add predictions
        test_predictions = self.base_model_predictions['XGB']
        meta_learner.add_base_model_predictions('XGB', test_predictions)
        
        assert 'XGB' in meta_learner.base_model_predictions
        assert len(meta_learner.base_model_predictions['XGB']) == len(test_predictions)
    
    def test_invalid_prediction_format(self):
        """Test validation of prediction format."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config
        )
        
        # Missing required columns
        invalid_predictions = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=10),
            'Q_pred': np.random.randn(10)
            # Missing 'code' and 'Q_obs'
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            meta_learner.add_base_model_predictions('Invalid', invalid_predictions)
    
    def test_get_base_model_ids(self):
        """Test getting base model IDs."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        model_ids = meta_learner.get_base_model_ids()
        assert set(model_ids) == {'XGB', 'LGBM', 'CatBoost'}
    
    def test_get_common_prediction_index(self):
        """Test getting common prediction index."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        common_index = meta_learner.get_common_prediction_index()
        assert len(common_index) > 0
        assert common_index.names == ['date', 'code']
    
    def test_calculate_base_model_performance(self):
        """Test calculation of base model performance."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        # Calculate performance for one model
        performance = meta_learner.calculate_base_model_performance('XGB')
        
        expected_metrics = ['r2', 'rmse', 'nse', 'mae', 'kge', 'bias']
        for metric in expected_metrics:
            assert metric in performance
            assert not np.isnan(performance[metric])
    
    def test_calculate_base_model_performance_with_grouping(self):
        """Test calculation of base model performance with grouping."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        # Calculate performance grouped by code
        performance = meta_learner.calculate_base_model_performance('XGB', group_by=['code'])
        
        # Should have performance for each basin
        assert isinstance(performance, dict)
        # Check that we have basin-specific results
        assert any('BASIN' in str(key) for key in performance.keys())
    
    def test_calculate_all_base_model_performance(self):
        """Test calculation of performance for all base models."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        all_performance = meta_learner.calculate_all_base_model_performance()
        
        assert len(all_performance) == 3  # 3 base models
        assert 'XGB' in all_performance
        assert 'LGBM' in all_performance
        assert 'CatBoost' in all_performance
        
        # Check that each model has performance metrics
        for model_id, performance in all_performance.items():
            assert 'r2' in performance
            assert 'rmse' in performance
    
    def test_create_ensemble_predictions(self):
        """Test creation of ensemble predictions."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        # Test with custom weights
        weights = {'XGB': 0.5, 'LGBM': 0.3, 'CatBoost': 0.2}
        
        ensemble_predictions = meta_learner.create_ensemble_predictions(weights)
        
        assert len(ensemble_predictions) > 0
        assert 'date' in ensemble_predictions.columns
        assert 'code' in ensemble_predictions.columns
        assert 'Q_obs' in ensemble_predictions.columns
        assert 'Q_pred' in ensemble_predictions.columns
        assert 'model' in ensemble_predictions.columns
        
        # Check that ensemble predictions are reasonable
        assert not ensemble_predictions['Q_pred'].isna().all()
    
    def test_create_ensemble_predictions_no_weights(self):
        """Test creation of ensemble predictions without weights."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        # Mock compute_weights to return uniform weights
        meta_learner.compute_weights = Mock(return_value={'XGB': 1.0, 'LGBM': 1.0, 'CatBoost': 1.0})
        
        ensemble_predictions = meta_learner.create_ensemble_predictions()
        
        assert len(ensemble_predictions) > 0
        assert not ensemble_predictions['Q_pred'].isna().all()
    
    def test_evaluate_ensemble_performance(self):
        """Test evaluation of ensemble performance."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        # Mock compute_weights
        meta_learner.compute_weights = Mock(return_value={'XGB': 1.0, 'LGBM': 1.0, 'CatBoost': 1.0})
        
        # Create ensemble predictions
        ensemble_predictions = meta_learner.create_ensemble_predictions()
        
        # Evaluate performance
        performance = meta_learner.evaluate_ensemble_performance(ensemble_predictions)
        
        expected_metrics = ['r2', 'rmse', 'nse', 'mae', 'kge', 'bias']
        for metric in expected_metrics:
            assert metric in performance
            assert not np.isnan(performance[metric])
    
    def test_empty_base_model_predictions(self):
        """Test handling of empty base model predictions."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions={}
        )
        
        # Should raise error when trying to create ensemble without base models
        with pytest.raises(ValueError, match="No base model predictions available"):
            meta_learner.create_ensemble_predictions()
    
    def test_zero_weights(self):
        """Test handling of zero weights."""
        meta_learner = ConcreteMetaLearner(
            data=self.data,
            static_data=self.static_data,
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            path_config=self.path_config,
            base_model_predictions=self.base_model_predictions
        )
        
        # Test with zero weights
        zero_weights = {'XGB': 0.0, 'LGBM': 0.0, 'CatBoost': 0.0}
        
        with pytest.raises(ValueError, match="Total weight is zero"):
            meta_learner.create_ensemble_predictions(zero_weights)
    
    def test_abstract_methods_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        # Cannot instantiate BaseMetaLearner directly
        with pytest.raises(TypeError):
            BaseMetaLearner(
                data=self.data,
                static_data=self.static_data,
                general_config=self.general_config,
                model_config=self.model_config,
                feature_config=self.feature_config,
                path_config=self.path_config
            )


class ConcreteMetaLearner(BaseMetaLearner):
    """Concrete implementation of BaseMetaLearner for testing."""
    
    def compute_weights(self, **kwargs):
        """Mock implementation of compute_weights."""
        return {'XGB': 1.0, 'LGBM': 1.0, 'CatBoost': 1.0}
    
    def train_meta_model(self, **kwargs):
        """Mock implementation of train_meta_model."""
        pass
    
    def calibrate_model_and_hindcast(self):
        """Mock implementation of calibrate_model_and_hindcast."""
        return self.create_ensemble_predictions()
    
    def predict_operational(self, today=None):
        """Mock implementation of predict_operational."""
        return self.create_ensemble_predictions()
    
    def tune_hyperparameters(self):
        """Mock implementation of tune_hyperparameters."""
        return True, "Mock tuning successful"


if __name__ == '__main__':
    pytest.main([__file__])