"""
Historical Meta-Learner for monthly discharge forecasting.

This module implements performance-based weighting using historical metrics,
supporting basin-specific and temporal weighting strategies.
"""

import pandas as pd
import numpy as np
import datetime
from typing import Dict, Any, List, Optional, Tuple
import logging

from .base_meta_learner import BaseMetaLearner
from monthly_forecasting.scr.evaluation_utils import calculate_all_metrics

logger = logging.getLogger(__name__)


class HistoricalMetaLearner(BaseMetaLearner):
    """
    Historical performance-based meta-learning model.
    
    This class implements meta-learning through historical performance weighting,
    supporting basin-specific and temporal weighting strategies using LOOCV.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        static_data: pd.DataFrame,
        general_config: Dict[str, Any],
        model_config: Dict[str, Any],
        feature_config: Dict[str, Any],
        path_config: Dict[str, Any],
        base_model_predictions: Dict[str, pd.DataFrame] = None,
    ) -> None:
        """
        Initialize the Historical Meta-Learner.
        
        Args:
            data: Historical data for training
            static_data: Static basin characteristics
            general_config: General configuration
            model_config: Meta-learning specific configuration
            feature_config: Feature engineering configuration
            path_config: Path configuration
            base_model_predictions: Dictionary of base model predictions
        """
        super().__init__(
            data, static_data, general_config, model_config, 
            feature_config, path_config, base_model_predictions
        )
        
        # Historical meta-learning specific configuration
        self.basin_specific = self.meta_learning_config.get('basin_specific', True)
        self.temporal_weighting = self.meta_learning_config.get('temporal_weighting', True)
        self.min_samples_per_basin = self.meta_learning_config.get('min_samples_per_basin', 12)
        self.weight_smoothing = self.meta_learning_config.get('weight_smoothing', 0.1)
        self.fallback_uniform = self.meta_learning_config.get('fallback_uniform', True)
        
        # Performance tracking
        self.historical_performance = {}
        self.basin_weights = {}
        self.temporal_weights = {}
        
        logger.info(f"Initialized HistoricalMetaLearner with basin_specific={self.basin_specific}, temporal_weighting={self.temporal_weighting}")
    
    def calculate_historical_performance(
        self,
        cv_predictions: Dict[str, pd.DataFrame] = None,
        metric: str = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate historical performance metrics using cross-validation predictions.
        
        Args:
            cv_predictions: Cross-validation predictions for each model
            metric: Performance metric to use (if None, use configured metric)
            
        Returns:
            Dictionary {model_id: {metric: performance}}
        """
        if cv_predictions is None:
            cv_predictions = self.base_model_predictions
        
        if metric is None:
            metric = self.performance_metric
        
        historical_perf = {}
        
        for model_id, predictions in cv_predictions.items():
            # Calculate overall performance
            overall_metrics = calculate_all_metrics(predictions['Q_obs'], predictions['Q_pred'])
            historical_perf[model_id] = {'overall': overall_metrics}
            
            # Basin-specific performance
            if self.basin_specific:
                basin_perf = {}
                for code in predictions['code'].unique():
                    basin_data = predictions[predictions['code'] == code]
                    if len(basin_data) >= self.min_samples_per_basin:
                        basin_metrics = calculate_all_metrics(basin_data['Q_obs'], basin_data['Q_pred'])
                        basin_perf[code] = basin_metrics
                    else:
                        # Fallback to overall performance
                        basin_perf[code] = overall_metrics
                
                historical_perf[model_id]['basin'] = basin_perf
            
            # Temporal performance (monthly)
            if self.temporal_weighting:
                temporal_perf = {}
                predictions_copy = predictions.copy()
                predictions_copy['month'] = pd.to_datetime(predictions_copy['date']).dt.month
                
                for month in range(1, 13):
                    month_data = predictions_copy[predictions_copy['month'] == month]
                    if len(month_data) >= 5:  # Minimum samples for monthly statistics
                        month_metrics = calculate_all_metrics(month_data['Q_obs'], month_data['Q_pred'])
                        temporal_perf[month] = month_metrics
                    else:
                        # Fallback to overall performance
                        temporal_perf[month] = overall_metrics
                
                historical_perf[model_id]['temporal'] = temporal_perf
        
        self.historical_performance = historical_perf
        logger.info(f"Calculated historical performance for {len(historical_perf)} models")
        
        return historical_perf
    
    def compute_performance_weights(
        self,
        performance_data: Dict[str, Dict[str, float]] = None,
        metric: str = None,
        invert_metric: bool = None
    ) -> Dict[str, float]:
        """
        Compute weights based on historical performance.
        
        Args:
            performance_data: Performance data for each model
            metric: Performance metric to use for weighting
            invert_metric: Whether to invert metric (True for error metrics like RMSE)
            
        Returns:
            Dictionary mapping model IDs to weights
        """
        if performance_data is None:
            if not self.historical_performance:
                raise ValueError("No historical performance data available")
            performance_data = {
                model_id: perf_data['overall'] 
                for model_id, perf_data in self.historical_performance.items()
            }
        
        if metric is None:
            metric = self.performance_metric
        
        if invert_metric is None:
            # Error metrics should be inverted (lower is better)
            invert_metric = metric in ['rmse', 'nrmse', 'mae', 'mape', 'bias', 'pbias']
        
        # Extract performance values
        performance_values = {}
        for model_id, metrics in performance_data.items():
            if metric in metrics and not np.isnan(metrics[metric]):
                performance_values[model_id] = metrics[metric]
        
        if not performance_values:
            logger.warning(f"No valid performance values for metric {metric}, using uniform weights")
            return {model_id: 1.0 for model_id in performance_data.keys()}
        
        # Calculate weights
        weights = {}
        
        if invert_metric:
            # For error metrics, use inverse weighting
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            inv_performance = {k: 1.0 / (v + epsilon) for k, v in performance_values.items()}
            total_inv_perf = sum(inv_performance.values())
            
            for model_id in performance_data.keys():
                if model_id in inv_performance:
                    weights[model_id] = inv_performance[model_id] / total_inv_perf
                else:
                    weights[model_id] = 0.0
        else:
            # For skill metrics, use direct weighting
            total_perf = sum(performance_values.values())
            
            if total_perf > 0:
                for model_id in performance_data.keys():
                    if model_id in performance_values:
                        weights[model_id] = performance_values[model_id] / total_perf
                    else:
                        weights[model_id] = 0.0
            else:
                # All performances are zero or negative, use uniform weights
                weights = {model_id: 1.0 for model_id in performance_data.keys()}
        
        # Apply smoothing to avoid extreme weights
        if self.weight_smoothing > 0:
            uniform_weight = 1.0 / len(weights)
            for model_id in weights:
                weights[model_id] = (1 - self.weight_smoothing) * weights[model_id] + self.weight_smoothing * uniform_weight
        
        return weights
    
    def compute_basin_specific_weights(
        self,
        basin_code: str,
        metric: str = None
    ) -> Dict[str, float]:
        """
        Compute weights for a specific basin.
        
        Args:
            basin_code: Basin code to compute weights for
            metric: Performance metric to use
            
        Returns:
            Dictionary mapping model IDs to basin-specific weights
        """
        if not self.basin_specific or not self.historical_performance:
            return self.compute_performance_weights(metric=metric)
        
        basin_performance = {}
        for model_id, perf_data in self.historical_performance.items():
            if 'basin' in perf_data and basin_code in perf_data['basin']:
                basin_performance[model_id] = perf_data['basin'][basin_code]
            else:
                # Fallback to overall performance
                basin_performance[model_id] = perf_data['overall']
        
        return self.compute_performance_weights(basin_performance, metric)
    
    def compute_temporal_weights(
        self,
        month: int,
        metric: str = None
    ) -> Dict[str, float]:
        """
        Compute weights for a specific month.
        
        Args:
            month: Month (1-12) to compute weights for
            metric: Performance metric to use
            
        Returns:
            Dictionary mapping model IDs to temporal weights
        """
        if not self.temporal_weighting or not self.historical_performance:
            return self.compute_performance_weights(metric=metric)
        
        temporal_performance = {}
        for model_id, perf_data in self.historical_performance.items():
            if 'temporal' in perf_data and month in perf_data['temporal']:
                temporal_performance[model_id] = perf_data['temporal'][month]
            else:
                # Fallback to overall performance
                temporal_performance[model_id] = perf_data['overall']
        
        return self.compute_performance_weights(temporal_performance, metric)
    
    def compute_weights(
        self,
        basin_code: str = None,
        month: int = None,
        metric: str = None
    ) -> Dict[str, float]:
        """
        Compute weights for ensemble combination.
        
        Args:
            basin_code: Basin code for basin-specific weights
            month: Month for temporal weights
            metric: Performance metric to use
            
        Returns:
            Dictionary mapping model IDs to weights
        """
        if not self.base_model_predictions:
            raise ValueError("No base model predictions available")
        
        # Ensure historical performance is calculated
        if not self.historical_performance:
            self.calculate_historical_performance()
        
        # Compute weights based on strategy
        if self.basin_specific and basin_code is not None:
            if self.temporal_weighting and month is not None:
                # Combined basin-specific and temporal weighting
                basin_weights = self.compute_basin_specific_weights(basin_code, metric)
                temporal_weights = self.compute_temporal_weights(month, metric)
                
                # Combine weights (geometric mean)
                combined_weights = {}
                for model_id in basin_weights:
                    if model_id in temporal_weights:
                        combined_weights[model_id] = np.sqrt(basin_weights[model_id] * temporal_weights[model_id])
                    else:
                        combined_weights[model_id] = basin_weights[model_id]
                
                # Normalize combined weights
                total_weight = sum(combined_weights.values())
                if total_weight > 0:
                    combined_weights = {k: v / total_weight for k, v in combined_weights.items()}
                
                return combined_weights
                
            else:
                # Basin-specific weighting only
                return self.compute_basin_specific_weights(basin_code, metric)
        
        elif self.temporal_weighting and month is not None:
            # Temporal weighting only
            return self.compute_temporal_weights(month, metric)
        
        else:
            # Overall performance weighting
            return self.compute_performance_weights(metric=metric)
    
    def train_meta_model(self, **kwargs) -> None:
        """
        Train the meta-model (not applicable for historical meta-learning).
        """
        logger.info("Historical meta-learning doesn't require explicit training")
        
        # Calculate historical performance if not already done
        if not self.historical_performance:
            self.calculate_historical_performance()
        
        # Cache overall weights
        self.performance_weights = self.compute_weights()
        
        logger.info("Historical meta-learning setup complete")
    
    def calibrate_model_and_hindcast(self) -> pd.DataFrame:
        """
        Calibrate the meta-model and perform hindcast using LOOCV.
        
        Returns:
            DataFrame with hindcast predictions
        """
        if not self.base_model_predictions:
            raise ValueError("No base model predictions available for calibration")
        
        logger.info("Starting historical meta-learning calibration and hindcast")
        
        # Get available years for LOOCV
        all_dates = pd.concat([
            pd.to_datetime(predictions['date']) 
            for predictions in self.base_model_predictions.values()
        ])
        available_years = sorted(all_dates.dt.year.unique())
        
        if len(available_years) < 2:
            raise ValueError("At least 2 years of data required for LOOCV")
        
        hindcast_predictions = []
        
        # Perform Leave-One-Year-Out cross-validation
        for test_year in available_years:
            logger.info(f"Processing test year: {test_year}")
            
            # Split data into train and test
            train_predictions = {}
            test_predictions = {}
            
            for model_id, predictions in self.base_model_predictions.items():
                pred_dates = pd.to_datetime(predictions['date'])
                
                train_mask = pred_dates.dt.year != test_year
                test_mask = pred_dates.dt.year == test_year
                
                train_predictions[model_id] = predictions[train_mask].copy()
                test_predictions[model_id] = predictions[test_mask].copy()
            
            # Calculate performance on training data
            train_performance = self.calculate_historical_performance(train_predictions)
            
            # Create ensemble predictions for test year
            if any(len(pred) > 0 for pred in test_predictions.values()):
                # Temporarily store original performance data
                original_performance = self.historical_performance
                self.historical_performance = train_performance
                
                # Get common test index
                test_indices = []
                for predictions in test_predictions.values():
                    if len(predictions) > 0:
                        test_indices.append(pd.MultiIndex.from_frame(predictions[['date', 'code']]))
                
                if test_indices:
                    common_test_index = test_indices[0]
                    for idx in test_indices[1:]:
                        common_test_index = common_test_index.intersection(idx)
                    
                    # Create ensemble predictions for each test sample
                    for date, code in common_test_index:
                        # Get month for temporal weighting
                        month = pd.to_datetime(date).month
                        
                        # Compute weights for this specific context
                        weights = self.compute_weights(
                            basin_code=code,
                            month=month
                        )
                        
                        # Collect predictions from all models
                        model_predictions = {}
                        obs_value = None
                        
                        for model_id, predictions in test_predictions.items():
                            mask = (predictions['date'] == date) & (predictions['code'] == code)
                            matching_preds = predictions[mask]
                            
                            if not matching_preds.empty:
                                model_predictions[model_id] = matching_preds['Q_pred'].iloc[0]
                                if obs_value is None:
                                    obs_value = matching_preds['Q_obs'].iloc[0]
                        
                        # Create weighted ensemble prediction
                        if model_predictions:
                            weighted_pred = 0.0
                            total_weight = 0.0
                            
                            for model_id, pred_value in model_predictions.items():
                                if model_id in weights and not np.isnan(pred_value):
                                    weighted_pred += weights[model_id] * pred_value
                                    total_weight += weights[model_id]
                            
                            if total_weight > 0:
                                ensemble_pred = weighted_pred / total_weight
                                
                                hindcast_predictions.append({
                                    'date': date,
                                    'code': code,
                                    'Q_obs': obs_value,
                                    'Q_pred': ensemble_pred,
                                    'model': self.name,
                                    'test_year': test_year
                                })
                
                # Restore original performance data
                self.historical_performance = original_performance
        
        if not hindcast_predictions:
            raise ValueError("No hindcast predictions generated")
        
        hindcast_df = pd.DataFrame(hindcast_predictions)
        
        # Final training on all data
        self.train_meta_model()
        
        logger.info(f"Completed hindcast with {len(hindcast_df)} predictions")
        
        return hindcast_df
    
    def predict_operational(self, today: datetime.datetime = None) -> pd.DataFrame:
        """
        Predict in operational mode.
        
        Args:
            today: Date to use as "today" for prediction
            
        Returns:
            DataFrame with operational predictions
        """
        if not self.base_model_predictions:
            raise ValueError("No base model predictions available")
        
        if today is None:
            today = datetime.datetime.now()
        
        logger.info(f"Generating operational predictions for {today}")
        
        # Ensure meta-model is trained
        if not self.performance_weights:
            self.train_meta_model()
        
        # For operational mode, we typically predict the next month
        # This requires base model predictions to be available
        
        # Get the most recent predictions from base models
        operational_predictions = []
        
        # Get common index for operational predictions
        common_index = self.get_common_prediction_index()
        
        for date, code in common_index:
            pred_date = pd.to_datetime(date)
            
            # Only use recent predictions for operational mode
            if pred_date >= today - pd.Timedelta(days=30):
                month = pred_date.month
                
                # Compute context-specific weights
                weights = self.compute_weights(
                    basin_code=code,
                    month=month
                )
                
                # Create weighted ensemble prediction
                weighted_pred = 0.0
                total_weight = 0.0
                obs_value = None
                
                for model_id, predictions in self.base_model_predictions.items():
                    mask = (predictions['date'] == date) & (predictions['code'] == code)
                    matching_preds = predictions[mask]
                    
                    if not matching_preds.empty:
                        pred_value = matching_preds['Q_pred'].iloc[0]
                        if not np.isnan(pred_value) and model_id in weights:
                            weighted_pred += weights[model_id] * pred_value
                            total_weight += weights[model_id]
                        
                        if obs_value is None:
                            obs_value = matching_preds['Q_obs'].iloc[0]
                
                if total_weight > 0:
                    ensemble_pred = weighted_pred / total_weight
                    
                    operational_predictions.append({
                        'date': date,
                        'code': code,
                        'Q_obs': obs_value,
                        'Q_pred': ensemble_pred,
                        'model': self.name
                    })
        
        if not operational_predictions:
            raise ValueError("No operational predictions generated")
        
        operational_df = pd.DataFrame(operational_predictions)
        
        logger.info(f"Generated {len(operational_df)} operational predictions")
        
        return operational_df
    
    def tune_hyperparameters(self) -> Tuple[bool, str]:
        """
        Tune hyperparameters for the meta-learning model.
        
        Returns:
            Tuple of (success, message)
        """
        logger.info("Tuning hyperparameters for HistoricalMetaLearner")
        
        # Historical meta-learning has limited hyperparameters to tune
        # Main parameters are weight_smoothing and minimum samples
        
        best_performance = -np.inf
        best_params = {}
        
        # Hyperparameter search space
        smoothing_values = [0.0, 0.05, 0.1, 0.2, 0.3]
        min_samples_values = [5, 10, 15, 20]
        
        for smoothing in smoothing_values:
            for min_samples in min_samples_values:
                # Temporarily set parameters
                original_smoothing = self.weight_smoothing
                original_min_samples = self.min_samples_per_basin
                
                self.weight_smoothing = smoothing
                self.min_samples_per_basin = min_samples
                
                try:
                    # Evaluate performance with these parameters
                    hindcast_df = self.calibrate_model_and_hindcast()
                    
                    # Calculate performance metric
                    metrics = calculate_all_metrics(hindcast_df['Q_obs'], hindcast_df['Q_pred'])
                    
                    # Use NSE as primary metric for hyperparameter tuning
                    performance = metrics['nse']
                    
                    if performance > best_performance:
                        best_performance = performance
                        best_params = {
                            'weight_smoothing': smoothing,
                            'min_samples_per_basin': min_samples
                        }
                
                except Exception as e:
                    logger.warning(f"Error with smoothing={smoothing}, min_samples={min_samples}: {e}")
                
                finally:
                    # Restore original parameters
                    self.weight_smoothing = original_smoothing
                    self.min_samples_per_basin = original_min_samples
        
        if best_params:
            # Set best parameters
            self.weight_smoothing = best_params['weight_smoothing']
            self.min_samples_per_basin = best_params['min_samples_per_basin']
            
            logger.info(f"Best hyperparameters: {best_params}, NSE: {best_performance:.4f}")
            
            return True, f"Hyperparameter tuning completed. Best NSE: {best_performance:.4f}"
        else:
            return False, "Hyperparameter tuning failed"