#!/usr/bin/env python3
"""
Example script demonstrating how to use the Historical Meta-Learning configuration files.

This script shows how to:
1. Load configuration files
2. Initialize the Historical Meta-Learner
3. Train the model
4. Generate ensemble predictions
5. Evaluate performance
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import the meta-learning framework
from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import (
    HistoricalMetaLearner,
)


def load_configurations(config_dir: str = "example_config/hist_meta"):
    """Load all configuration files from the specified directory."""
    config_path = Path(config_dir)
    configs = {}
    
    # Load each configuration file
    config_files = [
        "general_config.json",
        "meta_learning_config.json", 
        "base_model_config.json",
        "feature_config.json",
        "data_paths.json",
        "path_config.json",
        "experiment_config.json"
    ]
    
    for config_file in config_files:
        file_path = config_path / config_file
        if file_path.exists():
            with open(file_path, 'r') as f:
                configs[config_file.replace('.json', '')] = json.load(f)
            logger.info(f"Loaded {config_file}")
        else:
            logger.warning(f"Configuration file {config_file} not found")
    
    return configs


def create_sample_data():
    """Create sample data for demonstration purposes."""
    # Create sample discharge and forcing data
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range("2010-01-01", periods=n_samples, freq="D")
    codes = np.tile([1, 2, 3], (n_samples // 3 + 1))[:n_samples]
    
    data = pd.DataFrame({
        "date": dates,
        "code": codes,
        "Q": np.random.lognormal(2, 1, n_samples),  # Discharge
        "T": np.random.normal(10, 15, n_samples),   # Temperature
        "P": np.random.exponential(5, n_samples),   # Precipitation
        "SWE": np.random.gamma(2, 5, n_samples),    # Snow Water Equivalent
        "ROF": np.random.gamma(1, 3, n_samples),    # Runoff
    })
    
    # Create static data
    static_data = pd.DataFrame({
        "code": [1, 2, 3],
        "LAT": [42.0, 43.0, 44.0],
        "LON": [74.0, 75.0, 76.0],
        "gl_fr": [0.1, 0.2, 0.05],
        "h_mean": [2000, 2500, 1800],
        "h_min": [1500, 2000, 1200],
        "h_max": [3000, 3500, 2500],
        "slope": [0.3, 0.4, 0.2],
        "area_km2": [100, 150, 80],
        "code_str": ["basin_1", "basin_2", "basin_3"]
    })
    
    return data, static_data


def create_sample_base_model_predictions(data):
    """Create sample base model predictions."""
    np.random.seed(42)
    n_samples = len(data)
    
    base_predictions = {}
    
    # Create predictions for each base model
    for model_name in ["xgb", "lgbm", "catboost"]:
        # Add some noise and bias to make predictions realistic
        noise = np.random.normal(0, 0.2, n_samples)
        bias = np.random.normal(0, 0.1, 1)[0]
        
        predictions = pd.DataFrame({
            "date": data["date"],
            "code": data["code"],
            "Q_obs": data["Q"],
            "Q_pred": data["Q"] * (1 + bias) + noise,
        })
        
        base_predictions[model_name] = predictions
    
    return base_predictions


def main():
    """Main function demonstrating the Historical Meta-Learning workflow."""
    logger.info("Starting Historical Meta-Learning example")
    
    # 1. Load configuration files
    logger.info("Loading configuration files...")
    configs = load_configurations()
    
    # 2. Create sample data (in real usage, you would load your actual data)
    logger.info("Creating sample data...")
    data, static_data = create_sample_data()
    
    # 3. Create sample base model predictions
    logger.info("Creating sample base model predictions...")
    base_predictions = create_sample_base_model_predictions(data)
    
    # 4. Initialize the Historical Meta-Learner
    logger.info("Initializing Historical Meta-Learner...")
    meta_learner = HistoricalMetaLearner(
        data=data,
        static_data=static_data,
        general_config=configs["general_config"],
        model_config=configs["meta_learning_config"],
        feature_config=configs["feature_config"],
        path_config=configs["path_config"]
    )
    
    # 5. Add base model predictions
    logger.info("Adding base model predictions...")
    for model_name, predictions in base_predictions.items():
        meta_learner.add_base_model_predictions(model_name, predictions)
    
    # 6. Train the meta-learner
    logger.info("Training meta-learner...")
    meta_learner.train_meta_model()
    
    # 7. Calculate historical performance
    logger.info("Calculating historical performance...")
    historical_performance = meta_learner.calculate_historical_performance()
    
    # Display some results
    logger.info("Historical Performance Summary:")
    for model_id, perf_data in historical_performance.items():
        overall_rmse = perf_data["overall"].get("rmse", "N/A")
        logger.info(f"  {model_id}: RMSE = {overall_rmse:.3f}")
    
    # 8. Compute ensemble weights
    logger.info("Computing ensemble weights...")
    weights = meta_learner.compute_weights()
    
    logger.info("Ensemble Weights:")
    for model_id, weight in weights.items():
        logger.info(f"  {model_id}: {weight:.3f}")
    
    # 9. Create ensemble predictions
    logger.info("Creating ensemble predictions...")
    ensemble_predictions = meta_learner.create_ensemble_predictions()
    
    # 10. Evaluate ensemble performance
    logger.info("Evaluating ensemble performance...")
    ensemble_performance = meta_learner.evaluate_ensemble_performance()
    
    logger.info("Ensemble Performance:")
    for metric, value in ensemble_performance.items():
        logger.info(f"  {metric}: {value:.3f}")
    
    # 11. Demonstrate basin-specific and temporal weighting
    logger.info("Demonstrating basin-specific and temporal weighting...")
    
    # Basin-specific weights
    basin_weights = meta_learner.compute_basin_specific_weights(basin_code=1)
    logger.info("Basin-specific weights (Basin 1):")
    for model_id, weight in basin_weights.items():
        logger.info(f"  {model_id}: {weight:.3f}")
    
    # Temporal weights (for January first 10 days)
    temporal_weights = meta_learner.compute_temporal_weights(period="1-10")
    logger.info("Temporal weights (January 1-10):")
    for model_id, weight in temporal_weights.items():
        logger.info(f"  {model_id}: {weight:.3f}")
    
    # 12. Save the trained model
    logger.info("Saving trained model...")
    save_path = meta_learner.save_model()
    logger.info(f"Model saved to: {save_path}")
    
    # 13. Demonstrate operational prediction
    logger.info("Demonstrating operational prediction...")
    operational_predictions = meta_learner.predict_operational()
    
    logger.info(f"Generated {len(operational_predictions)} operational predictions")
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Base models: {list(base_predictions.keys())}")
    logger.info(f"Ensemble method: {configs['meta_learning_config']['ensemble_method']}")
    logger.info(f"Weighting strategy: {configs['meta_learning_config']['weighting_strategy']}")
    logger.info(f"Performance metric: {configs['meta_learning_config']['performance_metric']}")
    logger.info(f"Basin-specific weighting: {configs['meta_learning_config']['basin_specific']}")
    logger.info(f"Temporal weighting: {configs['meta_learning_config']['temporal_weighting']}")
    logger.info(f"Weight smoothing: {configs['meta_learning_config']['weight_smoothing']}")
    logger.info("="*60)
    
    logger.info("Historical Meta-Learning example completed successfully!")


if __name__ == "__main__":
    main()