"""
Tests for base_class module, focusing on BaseForecastModel interface and abstract methods.
"""

import pytest
import pandas as pd
import numpy as np
from abc import ABC
from forecast_models.base_class import BaseForecastModel


class TestBaseForecastModel:
    """Test the BaseForecastModel abstract base class."""
    
    def test_base_class_is_abstract(self):
        """Test that BaseForecastModel cannot be instantiated directly."""
        # Create mock data and configs
        data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10), 'value': range(10)})
        static_data = pd.DataFrame({'code': [1, 2], 'area': [100, 200]})
        general_config = {'model_name': 'test_model'}
        model_config = {'param1': 'value1'}
        feature_config = {'features': ['feature1']}
        path_config = {'model_path': '/tmp/test'}
        
        # Should raise TypeError when trying to instantiate abstract class
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseForecastModel(data, static_data, general_config, model_config, feature_config, path_config)
    
    def test_concrete_implementation_methods(self):
        """Test that concrete implementation must override abstract methods."""
        
        class IncompleteForecastModel(BaseForecastModel):
            """Incomplete implementation missing some abstract methods."""
            
            def predict_operational(self, data):
                return pd.DataFrame()
            
            def calibrate_model_and_hindcast(self, data):
                return pd.DataFrame()
            
            # Missing tune_hyperparameters, save_model, load_model
        
        # Create mock data and configs
        data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10), 'value': range(10)})
        static_data = pd.DataFrame({'code': [1, 2], 'area': [100, 200]})
        general_config = {'model_name': 'test_model'}
        model_config = {'param1': 'value1'}
        feature_config = {'features': ['feature1']}
        path_config = {'model_path': '/tmp/test'}
        
        # Should raise TypeError for incomplete implementation
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteForecastModel(data, static_data, general_config, model_config, feature_config, path_config)
    
    def test_complete_implementation_initialization(self):
        """Test that complete implementation initializes correctly."""
        
        class CompleteForecastModel(BaseForecastModel):
            """Complete implementation with all abstract methods."""
            
            def predict_operational(self, data):
                return pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'Q_pred': [100.0]})
            
            def calibrate_model_and_hindcast(self, data):
                return pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'Q_pred': [100.0]})
            
            def tune_hyperparameters(self):
                return True, "Hyperparameters tuned successfully"
            
            def save_model(self):
                pass
            
            def load_model(self):
                pass
        
        # Create mock data and configs
        data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10), 'value': range(10)})
        static_data = pd.DataFrame({'code': [1, 2], 'area': [100, 200]})
        general_config = {'model_name': 'test_model'}
        model_config = {'param1': 'value1'}
        feature_config = {'features': ['feature1']}
        path_config = {'model_path': '/tmp/test'}
        
        # Should create instance successfully
        model = CompleteForecastModel(data, static_data, general_config, model_config, feature_config, path_config)
        
        # Check initialization
        assert model.name == 'test_model'
        assert model.data is data
        assert model.static_data is static_data
        assert model.general_config == general_config
        assert model.model_config == model_config
        assert model.feature_config == feature_config
        assert model.path_config == path_config
    
    def test_abstract_method_signatures(self):
        """Test that abstract methods have correct signatures."""
        
        class CompleteForecastModel(BaseForecastModel):
            """Complete implementation for testing method signatures."""
            
            def predict_operational(self, data):
                # Test that we can call with DataFrame
                assert isinstance(data, pd.DataFrame)
                return pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'Q_pred': [100.0]})
            
            def calibrate_model_and_hindcast(self, data):
                # Test that we can call with DataFrame
                assert isinstance(data, pd.DataFrame)
                return pd.DataFrame({'date': [pd.Timestamp('2020-01-01')], 'Q_pred': [100.0]})
            
            def tune_hyperparameters(self):
                # Test that no parameters are required
                return True, "Success"
            
            def save_model(self):
                # Test that no parameters are required
                pass
            
            def load_model(self):
                # Test that no parameters are required
                pass
        
        # Create mock data and configs
        data = pd.DataFrame({'date': pd.date_range('2020-01-01', periods=10), 'value': range(10)})
        static_data = pd.DataFrame({'code': [1, 2], 'area': [100, 200]})
        general_config = {'model_name': 'test_model'}
        model_config = {'param1': 'value1'}
        feature_config = {'features': ['feature1']}
        path_config = {'model_path': '/tmp/test'}
        
        model = CompleteForecastModel(data, static_data, general_config, model_config, feature_config, path_config)
        
        # Test method calls
        test_data = pd.DataFrame({'test': [1, 2, 3]})
        
        operational_result = model.predict_operational(test_data)
        assert isinstance(operational_result, pd.DataFrame)
        assert 'Q_pred' in operational_result.columns
        
        hindcast_result = model.calibrate_model_and_hindcast(test_data)
        assert isinstance(hindcast_result, pd.DataFrame)
        assert 'Q_pred' in hindcast_result.columns
        
        tune_result = model.tune_hyperparameters()
        assert isinstance(tune_result, tuple)
        assert len(tune_result) == 2
        assert isinstance(tune_result[0], bool)
        assert isinstance(tune_result[1], str)
        
        # These should not raise exceptions
        model.save_model()
        model.load_model()
    
    def test_inheritance_structure(self):
        """Test that BaseForecastModel properly inherits from ABC."""
        # Test that BaseForecastModel is a subclass of ABC
        assert issubclass(BaseForecastModel, ABC)
        
        # Test that it has the expected abstract methods
        abstract_methods = BaseForecastModel.__abstractmethods__
        expected_methods = {
            'predict_operational',
            'calibrate_model_and_hindcast', 
            'tune_hyperparameters',
            'save_model',
            'load_model'
        }
        
        assert abstract_methods == expected_methods


if __name__ == "__main__":
    pytest.main([__file__, "-v"])