#!/usr/bin/env python3
"""
Example usage of the meta-learning framework for monthly discharge forecasting.

This script demonstrates how to use the HistoricalMetaLearner for intelligent
ensemble weighting based on historical performance.
"""

import numpy as np
import pandas as pd
from datetime import datetime

# Import meta-learning components
from monthly_forecasting.forecast_models.meta_learners.historical_meta_learner import HistoricalMetaLearner
from monthly_forecasting.scr.evaluation_utils import calculate_all_metrics
from monthly_forecasting.scr.ensemble_utils import EnsembleBuilder


def create_synthetic_data():
    """Create synthetic data for demonstration."""
    np.random.seed(42)
    
    # Create multi-year data for LOOCV
    dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
    n_samples = len(dates)
    
    # Create seasonal pattern
    day_of_year = dates.dayofyear
    seasonal_pattern = 20 * np.sin(2 * np.pi * day_of_year / 365)
    
    # Base discharge with seasonal pattern
    base_discharge = 50 + seasonal_pattern + np.random.randn(n_samples) * 5
    
    # Create data DataFrame
    data = pd.DataFrame({
        'date': dates,
        'code': np.random.choice(['BASIN01', 'BASIN02'], n_samples),
        'Q': base_discharge,
        'T': np.random.randn(n_samples) * 5 + 15,  # Temperature
        'P': np.random.randn(n_samples) * 20 + 100  # Precipitation
    })
    
    # Static data
    static_data = pd.DataFrame({
        'code': ['BASIN01', 'BASIN02'],
        'area': [1000, 1500],
        'elevation': [500, 800]
    })
    
    return data, static_data


def create_base_model_predictions(data):
    """Create synthetic base model predictions for demonstration."""
    predictions = {}
    
    # Create predictions for 3 different models with different performance characteristics
    for i, model_name in enumerate(['XGBoost', 'LightGBM', 'CatBoost']):
        # Each model has different noise levels and biases
        noise_level = 3 + i * 1.5
        bias = i * 2
        
        # Create predictions based on observations with model-specific characteristics
        obs_values = data['Q'].values
        pred_values = obs_values + np.random.randn(len(obs_values)) * noise_level + bias
        
        predictions[model_name] = pd.DataFrame({
            'date': data['date'],
            'code': data['code'],
            'Q_obs': obs_values,
            'Q_pred': pred_values,
            'model': model_name
        })
    
    return predictions


def main():
    """Main demonstration function."""
    print("=== Meta-Learning Framework Demonstration ===")
    print()
    
    # Create synthetic data
    print("1. Creating synthetic data...")
    data, static_data = create_synthetic_data()
    print(f"   Created {len(data)} samples across {data['code'].nunique()} basins")
    
    # Create base model predictions
    print("\n2. Creating base model predictions...")
    base_predictions = create_base_model_predictions(data)
    print(f"   Created predictions for {len(base_predictions)} base models")
    
    # Show base model performance
    print("\n3. Base model performance:")
    for model_name, predictions in base_predictions.items():
        metrics = calculate_all_metrics(predictions['Q_obs'], predictions['Q_pred'])
        print(f"   {model_name}: R² = {metrics['r2']:.3f}, RMSE = {metrics['rmse']:.3f}, NSE = {metrics['nse']:.3f}")
    
    # Create meta-learning configurations
    general_config = {
        'model_name': 'HistoricalMetaLearner_Example',
        'target_column': 'Q',
        'date_column': 'date',
        'code_column': 'code'
    }
    
    model_config = {
        'meta_learning': {
            'ensemble_method': 'weighted_mean',
            'weighting_strategy': 'performance_based',
            'performance_metric': 'rmse',
            'basin_specific': True,
            'temporal_weighting': True,
            'min_samples_per_basin': 10,
            'weight_smoothing': 0.1
        }
    }
    
    feature_config = {
        'feature_columns': ['T', 'P'],
        'lag_features': [1, 2, 3]
    }
    
    path_config = {
        'model_dir': '/tmp/meta_learning_example',
        'output_dir': '/tmp/meta_learning_output'
    }
    
    # Initialize meta-learner
    print("\n4. Initializing Historical Meta-Learner...")
    meta_learner = HistoricalMetaLearner(
        data=data,
        static_data=static_data,
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        path_config=path_config,
        base_model_predictions=base_predictions
    )
    
    # Train meta-learner
    print("\n5. Training meta-learner...")
    meta_learner.train_meta_model()
    print("   Meta-learner trained successfully!")
    
    # Show learned weights
    print("\n6. Learned ensemble weights:")
    weights = meta_learner.compute_weights()
    for model_name, weight in weights.items():
        print(f"   {model_name}: {weight:.3f}")
    
    # Create ensemble predictions
    print("\n7. Creating ensemble predictions...")
    ensemble_predictions = meta_learner.create_ensemble_predictions()
    print(f"   Created {len(ensemble_predictions)} ensemble predictions")
    
    # Evaluate ensemble performance
    print("\n8. Ensemble performance:")
    ensemble_metrics = calculate_all_metrics(
        ensemble_predictions['Q_obs'], 
        ensemble_predictions['Q_pred']
    )
    print(f"   Ensemble: R² = {ensemble_metrics['r2']:.3f}, RMSE = {ensemble_metrics['rmse']:.3f}, NSE = {ensemble_metrics['nse']:.3f}")
    
    # Compare with simple average ensemble
    print("\n9. Comparison with simple average ensemble:")
    ensemble_builder = EnsembleBuilder(ensemble_method='mean')
    simple_ensemble = ensemble_builder.create_simple_ensemble(base_predictions)
    
    simple_metrics = calculate_all_metrics(
        simple_ensemble['Q_obs'], 
        simple_ensemble['Q_pred']
    )
    print(f"   Simple Average: R² = {simple_metrics['r2']:.3f}, RMSE = {simple_metrics['rmse']:.3f}, NSE = {simple_metrics['nse']:.3f}")
    
    # Show improvements
    print("\n10. Meta-learning improvements:")
    r2_improvement = (ensemble_metrics['r2'] - simple_metrics['r2']) / simple_metrics['r2'] * 100
    rmse_improvement = (simple_metrics['rmse'] - ensemble_metrics['rmse']) / simple_metrics['rmse'] * 100
    nse_improvement = (ensemble_metrics['nse'] - simple_metrics['nse']) / simple_metrics['nse'] * 100
    
    print(f"    R² improvement: {r2_improvement:+.1f}%")
    print(f"    RMSE improvement: {rmse_improvement:+.1f}%")
    print(f"    NSE improvement: {nse_improvement:+.1f}%")
    
    # Show basin-specific weights
    print("\n11. Basin-specific weights:")
    for basin_code in data['code'].unique():
        basin_weights = meta_learner.compute_basin_specific_weights(basin_code)
        print(f"   {basin_code}:")
        for model_name, weight in basin_weights.items():
            print(f"     {model_name}: {weight:.3f}")
    
    # Show temporal weights (example for January)
    print("\n12. Temporal weights (January):")
    temporal_weights = meta_learner.compute_temporal_weights(1)
    for model_name, weight in temporal_weights.items():
        print(f"   {model_name}: {weight:.3f}")
    
    # LOOCV Demonstration
    print("\n13. Performing LOOCV validation...")
    try:
        hindcast_df = meta_learner.calibrate_model_and_hindcast()
        print(f"   LOOCV completed with {len(hindcast_df)} hindcast predictions")
        
        # Show LOOCV performance
        hindcast_metrics = calculate_all_metrics(
            hindcast_df['Q_obs'], 
            hindcast_df['Q_pred']
        )
        print(f"   LOOCV Performance: R² = {hindcast_metrics['r2']:.3f}, RMSE = {hindcast_metrics['rmse']:.3f}, NSE = {hindcast_metrics['nse']:.3f}")
        
        # Show performance by test year
        print("   Performance by test year:")
        for year in sorted(hindcast_df['test_year'].unique()):
            year_data = hindcast_df[hindcast_df['test_year'] == year]
            year_metrics = calculate_all_metrics(year_data['Q_obs'], year_data['Q_pred'])
            print(f"     {year}: R² = {year_metrics['r2']:.3f}, RMSE = {year_metrics['rmse']:.3f}")
    
    except Exception as e:
        print(f"   LOOCV validation failed: {e}")
    
    # Operational prediction example
    print("\n14. Operational prediction example...")
    try:
        operational_df = meta_learner.predict_operational(today=datetime(2023, 6, 15))
        print(f"   Generated {len(operational_df)} operational predictions")
        
        if len(operational_df) > 0:
            print("   Sample operational predictions:")
            for i, row in operational_df.head(3).iterrows():
                print(f"     {row['date']}, {row['code']}: Q_pred = {row['Q_pred']:.2f}")
    
    except Exception as e:
        print(f"   Operational prediction failed: {e}")
    
    print("\n=== Meta-Learning Framework Demonstration Complete ===")


if __name__ == '__main__':
    main()