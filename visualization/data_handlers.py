"""
Data handler classes for the model evaluation dashboard.

This module provides classes for loading and processing metrics data and predictions
with caching support for improved performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import logging
from functools import lru_cache
import sys

# Add parent directory to path to import evaluation modules
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.prediction_loader import (
    load_all_predictions,
    MODEL_FAMILIES,
    scan_prediction_files,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsDataHandler:
    """Handle loading and processing of metrics data."""
    
    def __init__(self, metrics_path: str = "../monthly_forecasting_results/evaluation/metrics.csv"):
        """
        Initialize the metrics data handler.
        
        Args:
            metrics_path: Path to the metrics CSV file
        """
        self.metrics_path = Path(metrics_path)
        self._df = None
        self._available_models = None
        self._available_metrics = None
        self._available_codes = None
        self._available_months = None
        self._model_families = MODEL_FAMILIES
        
    @property
    def df(self) -> pd.DataFrame:
        """Lazy load the metrics dataframe."""
        if self._df is None:
            self._load_metrics()
        return self._df
    
    def _load_metrics(self) -> None:
        """Load the metrics CSV file."""
        logger.info(f"Loading metrics from {self.metrics_path}")
        self._df = pd.read_csv(self.metrics_path)
        
        # Convert month to appropriate type
        self._df['month'] = self._df['month'].replace('all', -1)
        self._df['month'] = pd.to_numeric(self._df['month'], errors='coerce')
        # Convert NaN months to -1 for overall evaluations
        self._df['month'] = self._df['month'].fillna(-1).astype(int)
        
        # Rename columns for consistency
        if 'model_name' in self._df.columns:
            self._df['model'] = self._df['model_name']
        
        # Sort by model and month if columns exist
        sort_cols = []
        if 'model' in self._df.columns:
            sort_cols.append('model')
        if 'month' in self._df.columns:
            sort_cols.append('month')
        
        if sort_cols:
            self._df = self._df.sort_values(by=sort_cols)
        
        # Add model family column if not already present
        if 'family' in self._df.columns:
            self._df['model_family'] = self._df['family']
        elif 'model' in self._df.columns:
            self._df['model_family'] = self._df['model'].apply(self._get_model_family)
        
        logger.info(f"Loaded {len(self._df)} records")
        
    def _get_model_family(self, model_name: str) -> str:
        """Get the family for a given model name."""
        for family, models in self._model_families.items():
            if model_name in models:
                return family
        return "Other"
    
    @property
    def available_models(self) -> List[str]:
        """Get list of available models."""
        if self._available_models is None:
            self._available_models = sorted(self.df['model'].unique())
        return self._available_models
    
    @property
    def available_metrics(self) -> List[str]:
        """Get list of available metrics."""
        if self._available_metrics is None:
            metric_cols = ['r2', 'rmse', 'nrmse', 'mae', 'mape', 'nse', 'kge', 'bias', 'pbias']
            self._available_metrics = [col for col in metric_cols if col in self.df.columns]
        return self._available_metrics
    
    @property
    def available_codes(self) -> List[str]:
        """Get list of available basin codes."""
        if self._available_codes is None:
            codes = self.df[self.df['code'].notna()]['code'].unique()
            self._available_codes = sorted(codes)
        return self._available_codes
    
    @property
    def available_months(self) -> List[int]:
        """Get list of available months."""
        if self._available_months is None:
            months = self.df[self.df['month'] > 0]['month'].unique()
            self._available_months = sorted(months)
        return self._available_months
    
    def get_filtered_data(
        self, 
        models: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        codes: Optional[List[str]] = None,
        months: Optional[List[int]] = None,
        evaluation_level: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get filtered metrics data.
        
        Args:
            models: List of model names to include
            metrics: List of metrics to include
            codes: List of basin codes to include
            months: List of months to include
            evaluation_level: Filter by evaluation level ('overall', 'per_code', 'per_month', 'per_code_month')
            
        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()
        
        if models is not None:
            df = df[df['model'].isin(models)]
            
        if codes is not None:
            df = df[df['code'].isin(codes)]
            
        if months is not None:
            df = df[df['month'].isin(months)]
            
        if evaluation_level is not None:
            # Check if we have a 'level' column
            if 'level' in df.columns:
                if evaluation_level == 'overall':
                    df = df[df['level'] == 'overall']
                elif evaluation_level == 'per_code':
                    df = df[df['level'] == 'per_code']
                elif evaluation_level == 'per_month':
                    df = df[df['level'] == 'per_month']
                elif evaluation_level == 'per_code_month':
                    df = df[df['level'] == 'per_code_month']
            else:
                # Fallback to old logic if 'level' column doesn't exist
                if evaluation_level == 'overall':
                    df = df[(df['code'].isna()) & (df['month'] == -1)]
                elif evaluation_level == 'per_code':
                    df = df[(df['code'].notna()) & (df['month'] == -1)]
                elif evaluation_level == 'per_month':
                    df = df[(df['code'].isna()) & (df['month'] > 0)]
                elif evaluation_level == 'per_code_month':
                    df = df[(df['code'].notna()) & (df['month'] > 0)]
                
        if metrics is not None:
            cols_to_keep = ['model', 'model_family', 'code', 'month'] + metrics
            cols_to_keep = [col for col in cols_to_keep if col in df.columns]
            df = df[cols_to_keep]
            
        return df
    
    def get_month_name(self, month: int) -> str:
        """Convert month number to name."""
        if month == -1:
            return "All Months"
        month_names = {
            1: "January", 2: "February", 3: "March", 4: "April",
            5: "May", 6: "June", 7: "July", 8: "August",
            9: "September", 10: "October", 11: "November", 12: "December"
        }
        return month_names.get(month, str(month))


class PredictionDataHandler:
    """Handle loading and caching of prediction data."""
    
    def __init__(self, results_dir: str = "../monthly_forecasting_results"):
        """
        Initialize the prediction data handler.
        
        Args:
            results_dir: Base directory containing prediction files
        """
        self.results_dir = Path(results_dir)
        self._all_predictions = None
        self._model_info = None
        self._available_predictions = None
        
    @property
    def available_predictions(self) -> Dict:
        """Get dict of available prediction files."""
        if self._available_predictions is None:
            self._available_predictions = scan_prediction_files(str(self.results_dir))
        return self._available_predictions
    
    def _load_all_predictions(self):
        """Load all predictions once and cache them."""
        if self._all_predictions is None:
            logger.info("Loading all predictions...")
            self._all_predictions, self._model_info = load_all_predictions(
                str(self.results_dir)
            )
            logger.info(f"Loaded predictions for {len(self._all_predictions)} models")
    
    def load_predictions(
        self, 
        model: str, 
        code: Optional[str] = None,
        include_ensemble_members: bool = False
    ) -> pd.DataFrame:
        """
        Load predictions for a specific model and optionally a specific basin.
        
        Args:
            model: Model name
            code: Basin code (optional)
            include_ensemble_members: Whether to include ensemble member predictions
            
        Returns:
            DataFrame with predictions
        """
        # Load all predictions if not already loaded
        self._load_all_predictions()
        
        # Get predictions for the specific model
        if model not in self._all_predictions:
            logger.warning(f"No prediction data found for model {model}")
            return pd.DataFrame()
        
        df = self._all_predictions[model].copy()
        
        # Filter by code if specified
        if code is not None:
            df = df[df['code'] == code]
            
        # Remove ensemble members if not requested
        if not include_ensemble_members and 'member' in df.columns:
            df = df[df['member'].isna()]
            
        return df
    
    def get_observed_vs_predicted(
        self,
        model: str,
        code: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get observed vs predicted data for plotting.
        
        Args:
            model: Model name
            code: Basin code
            start_date: Start date (optional)
            end_date: End date (optional)
            
        Returns:
            DataFrame with date, Q_obs, Q_pred columns
        """
        df = self.load_predictions(model, code)
        
        if df.empty:
            return df
            
        # Filter by date range if specified
        if start_date is not None:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df['date'] <= pd.to_datetime(end_date)]
            
        # Select relevant columns
        result = df[['date', 'Q_obs', 'Q_pred']].copy()
        result['model'] = model
        
        return result