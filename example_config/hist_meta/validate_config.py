#!/usr/bin/env python3
"""
Configuration validation script for Historical Meta-Learning.

This script validates that all configuration files are properly formatted
and contain the required fields for the Historical Meta-Learner.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """Validates Historical Meta-Learning configuration files."""
    
    def __init__(self, config_dir: str = "example_config/hist_meta"):
        self.config_dir = Path(config_dir)
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> bool:
        """Validate all configuration files."""
        logger.info("Starting configuration validation...")
        
        # Validate each configuration file
        validators = [
            self._validate_general_config,
            self._validate_meta_learning_config,
            self._validate_base_model_config,
            self._validate_feature_config,
            self._validate_data_paths,
            self._validate_path_config,
            self._validate_experiment_config
        ]
        
        for validator in validators:
            try:
                validator()
            except Exception as e:
                self.errors.append(f"Error in {validator.__name__}: {str(e)}")
        
        # Report results
        self._report_results()
        
        return len(self.errors) == 0
    
    def _load_config(self, filename: str) -> Optional[Dict[str, Any]]:
        """Load a configuration file."""
        config_path = self.config_dir / filename
        if not config_path.exists():
            self.errors.append(f"Configuration file {filename} not found")
            return None
        
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON in {filename}: {str(e)}")
            return None
    
    def _validate_general_config(self):
        """Validate general_config.json."""
        config = self._load_config("general_config.json")
        if config is None:
            return
        
        required_fields = [
            "model_type", "base_models", "prediction_horizon", "feature_cols",
            "static_features", "target_column"
        ]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in general_config.json")
        
        # Validate model_type
        if config.get("model_type") != "historical_meta_learner":
            self.errors.append("model_type should be 'historical_meta_learner'")
        
        # Validate base_models
        base_models = config.get("base_models", [])
        valid_models = ["xgb", "lgbm", "catboost"]
        for model in base_models:
            if model not in valid_models:
                self.warnings.append(f"Unknown base model '{model}' in general_config.json")
        
        # Validate prediction_horizon
        if not isinstance(config.get("prediction_horizon"), int):
            self.errors.append("prediction_horizon must be an integer")
        
        logger.info("✓ General configuration validated")
    
    def _validate_meta_learning_config(self):
        """Validate meta_learning_config.json."""
        config = self._load_config("meta_learning_config.json")
        if config is None:
            return
        
        required_fields = [
            "ensemble_method", "weighting_strategy", "performance_metric",
            "basin_specific", "temporal_weighting", "weight_smoothing"
        ]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in meta_learning_config.json")
        
        # Validate ensemble_method
        valid_methods = ["weighted_mean", "mean", "median"]
        if config.get("ensemble_method") not in valid_methods:
            self.errors.append(f"ensemble_method must be one of {valid_methods}")
        
        # Validate weighting_strategy
        valid_strategies = ["performance_based", "uniform"]
        if config.get("weighting_strategy") not in valid_strategies:
            self.errors.append(f"weighting_strategy must be one of {valid_strategies}")
        
        # Validate performance_metric
        valid_metrics = ["rmse", "nrmse", "mae", "mape", "r2", "nse", "kge", "bias", "pbias"]
        if config.get("performance_metric") not in valid_metrics:
            self.errors.append(f"performance_metric must be one of {valid_metrics}")
        
        # Validate weight_smoothing
        smoothing = config.get("weight_smoothing", 0.1)
        if not isinstance(smoothing, (int, float)) or smoothing < 0 or smoothing > 1:
            self.errors.append("weight_smoothing must be a number between 0 and 1")
        
        # Validate max_weight_ratio
        max_ratio = config.get("max_weight_ratio", 100.0)
        if not isinstance(max_ratio, (int, float)) or max_ratio < 1:
            self.errors.append("max_weight_ratio must be a number >= 1")
        
        logger.info("✓ Meta-learning configuration validated")
    
    def _validate_base_model_config(self):
        """Validate base_model_config.json."""
        config = self._load_config("base_model_config.json")
        if config is None:
            return
        
        expected_models = ["xgb", "lgbm", "catboost"]
        
        for model in expected_models:
            if model not in config:
                self.warnings.append(f"Base model '{model}' not found in base_model_config.json")
                continue
            
            model_config = config[model]
            
            # Validate common parameters
            if model == "xgb":
                required_params = ["n_estimators", "learning_rate", "max_depth"]
            elif model == "lgbm":
                required_params = ["n_estimators", "learning_rate", "num_leaves"]
            elif model == "catboost":
                required_params = ["iterations", "depth", "learning_rate"]
            
            for param in required_params:
                if param not in model_config:
                    self.warnings.append(f"Missing parameter '{param}' for {model}")
        
        logger.info("✓ Base model configuration validated")
    
    def _validate_feature_config(self):
        """Validate feature_config.json."""
        config = self._load_config("feature_config.json")
        if config is None:
            return
        
        expected_features = ["discharge", "P", "T", "SWE", "ROF"]
        
        for feature in expected_features:
            if feature not in config:
                self.warnings.append(f"Feature '{feature}' not found in feature_config.json")
                continue
            
            feature_specs = config[feature]
            if not isinstance(feature_specs, list):
                self.errors.append(f"Feature '{feature}' must be a list of specifications")
                continue
            
            for spec in feature_specs:
                if "operation" not in spec:
                    self.errors.append(f"Missing 'operation' in feature spec for {feature}")
                if "windows" not in spec:
                    self.errors.append(f"Missing 'windows' in feature spec for {feature}")
                if "lags" not in spec:
                    self.errors.append(f"Missing 'lags' in feature spec for {feature}")
        
        logger.info("✓ Feature configuration validated")
    
    def _validate_data_paths(self):
        """Validate data_paths.json."""
        config = self._load_config("data_paths.json")
        if config is None:
            return
        
        required_paths = [
            "path_discharge", "path_forcing", "path_static_data",
            "model_home_path", "base_model_predictions_paths"
        ]
        
        for path_key in required_paths:
            if path_key not in config:
                self.errors.append(f"Missing required path '{path_key}' in data_paths.json")
        
        # Validate base_model_predictions_paths
        base_paths = config.get("base_model_predictions_paths", {})
        expected_models = ["xgb", "lgbm", "catboost"]
        
        for model in expected_models:
            if model not in base_paths:
                self.warnings.append(f"Missing prediction path for {model}")
        
        logger.info("✓ Data paths configuration validated")
    
    def _validate_path_config(self):
        """Validate path_config.json."""
        config = self._load_config("path_config.json")
        if config is None:
            return
        
        required_sections = ["data_paths", "model_paths", "output_paths"]
        
        for section in required_sections:
            if section not in config:
                self.errors.append(f"Missing required section '{section}' in path_config.json")
        
        # Validate data_paths section
        data_paths = config.get("data_paths", {})
        required_data_paths = ["discharge", "forcing", "static_data"]
        
        for path_key in required_data_paths:
            if path_key not in data_paths:
                self.errors.append(f"Missing data path '{path_key}' in path_config.json")
        
        logger.info("✓ Path configuration validated")
    
    def _validate_experiment_config(self):
        """Validate experiment_config.json."""
        config = self._load_config("experiment_config.json")
        if config is None:
            return
        
        required_fields = [
            "experiment_name", "model_type", "base_models",
            "meta_learning", "evaluation"
        ]
        
        for field in required_fields:
            if field not in config:
                self.errors.append(f"Missing required field '{field}' in experiment_config.json")
        
        # Validate meta_learning section
        meta_learning = config.get("meta_learning", {})
        required_meta_fields = ["ensemble_method", "weighting_strategy", "performance_metric"]
        
        for field in required_meta_fields:
            if field not in meta_learning:
                self.errors.append(f"Missing meta-learning field '{field}' in experiment_config.json")
        
        # Validate evaluation section
        evaluation = config.get("evaluation", {})
        required_eval_fields = ["metrics", "primary_metric"]
        
        for field in required_eval_fields:
            if field not in evaluation:
                self.errors.append(f"Missing evaluation field '{field}' in experiment_config.json")
        
        logger.info("✓ Experiment configuration validated")
    
    def _report_results(self):
        """Report validation results."""
        logger.info("\n" + "="*60)
        logger.info("CONFIGURATION VALIDATION RESULTS")
        logger.info("="*60)
        
        if self.errors:
            logger.error("ERRORS FOUND:")
            for error in self.errors:
                logger.error(f"  ❌ {error}")
        
        if self.warnings:
            logger.warning("WARNINGS:")
            for warning in self.warnings:
                logger.warning(f"  ⚠️  {warning}")
        
        if not self.errors and not self.warnings:
            logger.info("✅ All configurations are valid!")
        elif not self.errors:
            logger.info("✅ All configurations are valid (with warnings)")
        else:
            logger.error("❌ Configuration validation failed!")
        
        logger.info("="*60)


def main():
    """Main function to run configuration validation."""
    validator = ConfigValidator()
    success = validator.validate_all()
    
    if success:
        logger.info("Configuration validation completed successfully!")
        return 0
    else:
        logger.error("Configuration validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())