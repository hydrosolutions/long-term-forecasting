import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any, Dict, Optional, Union
import logging
import pytorch_lightning as pl

logger = logging.getLogger(__name__)


class DeepLearningDataset(Dataset):
    """
    Enhanced dataset for deep learning forecasting models.
    
    Supports the required data structure:
    - x_past: (batch, past_time_steps, past_features) - past discharge, P, T, past predictions  
    - x_nan_mask: (batch, past_time_steps, past_features) - binary mask for missing features
    - x_future: (batch, future_time_steps, future_vars) - weather forecast, temporal features
    - x_now: (batch, 1, now_vars) - current predictions/errors from other models
    - x_static: static basin features
    
    Provides configurable NaN handling (mask or drop) and integration with existing
    FeatureExtractor workflow.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
        past_features: List[str],
        future_features: List[str],
        static_features: List[str],
        now_features: Optional[List[str]] = None,
        lookback: int = 365,
        future_known_steps: int = 30,
        unique_id_col: str = 'code',
        date_col: str = 'date',
        target_col: str = 'target',
        use_nan_mask: bool = True,
        drop_nan_samples: bool = False,
    ):
        """
        Initialize the deep learning dataset.
        
        Args:
            df: Long dataframe with time series data
            static_df: Wide dataframe with static basin characteristics
            past_features: List of feature names for past time steps
            future_features: List of feature names for future time steps  
            static_features: List of static feature names
            now_features: List of feature names for current time step (t)
            lookback: Number of past time steps to include
            future_known_steps: Number of future time steps to include
            unique_id_col: Column name for basin/code identifier
            date_col: Column name for date
            target_col: Column name for target variable
            use_nan_mask: If True, create NaN mask and fill NaN with 0; if False, drop NaN samples
            drop_nan_samples: If True, drop samples with any NaN values
        """
        self.df = df.copy()
        self.static_df = static_df.copy()
        self.past_features = past_features
        self.future_features = future_features
        self.static_features = static_features
        self.now_features = now_features or []
        self.lookback = lookback
        self.future_known_steps = future_known_steps
        self.unique_id_col = unique_id_col
        self.date_col = date_col
        self.target_col = target_col
        self.use_nan_mask = use_nan_mask
        self.drop_nan_samples = drop_nan_samples

        # Pre-sort the data for efficiency
        idx_cols = [self.unique_id_col, self.date_col]
        self.df = self.df.sort_values(idx_cols).reset_index(drop=True)
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[self.date_col]):
            self.df[self.date_col] = pd.to_datetime(self.df[self.date_col])
        
        # Precompute static features as dict for faster access
        self._build_static_features_dict()
        
        # Convert feature columns to numpy arrays for faster slicing
        self._build_feature_arrays()
        
        # Build valid windows based on NaN handling strategy
        self.valid_windows = self._get_valid_windows()
        
        logger.info(f"Dataset initialized with {len(self.valid_windows)} valid samples")
        logger.info(f"Past features: {len(self.past_features)}, Future features: {len(self.future_features)}")
        logger.info(f"Static features: {len(self.static_features)}, Now features: {len(self.now_features)}")

    def _build_static_features_dict(self):
        """Build dictionary for fast static feature lookup."""
        self.static_features_dict = {}
        
        if not self.static_features:
            # If no static features specified, create empty arrays
            for code in self.df[self.unique_id_col].unique():
                self.static_features_dict[code] = np.array([], dtype=np.float32)
            return
            
        for _, row in self.static_df.iterrows():
            code = row[self.unique_id_col]
            static_vals = row[self.static_features].values.astype(np.float32)
            
            # Handle NaN in static features
            if np.isnan(static_vals).any():
                if self.use_nan_mask:
                    static_vals = np.nan_to_num(static_vals, nan=0.0)
                else:
                    logger.warning(f"NaN values in static features for code {code}")
                    
            self.static_features_dict[code] = static_vals

    def _build_feature_arrays(self):
        """Convert feature columns to numpy arrays for efficient slicing."""
        # Past features array
        if self.past_features:
            self.past_array = self.df[self.past_features].values.astype(np.float32)
        else:
            self.past_array = np.empty((len(self.df), 0), dtype=np.float32)
            
        # Future features array
        if self.future_features:
            self.future_array = self.df[self.future_features].values.astype(np.float32)
        else:
            self.future_array = np.empty((len(self.df), 0), dtype=np.float32)
            
        # Now features array
        if self.now_features:
            self.now_array = self.df[self.now_features].values.astype(np.float32)
        else:
            self.now_array = np.empty((len(self.df), 0), dtype=np.float32)
        
        # Target array
        self.target_array = self.df[self.target_col].values.astype(np.float32)
        
        # Store code and date arrays for lookup
        self.code_array = self.df[self.unique_id_col].values
        self.date_array = self.df[self.date_col].values

    def _get_valid_windows(self) -> List[Tuple[int, int, int, Any]]:
        """
        Build list of valid sample windows based on NaN handling strategy.
        
        Returns:
            List of tuples (start_idx, today_idx, end_idx, code) for valid samples
        """
        windows = []
        
        # Group data by code for efficient processing
        group_indices = {}
        for i, code in enumerate(self.code_array):
            if code not in group_indices:
                group_indices[code] = []
            group_indices[code].append(i)
        
        # Process each code group
        for code, indices in group_indices.items():
            indices = np.array(indices)
            n = len(indices)
            
            # Check each possible today index
            for local_i in range(self.lookback, n - self.future_known_steps):
                global_i = indices[local_i]
                start_i = indices[local_i - self.lookback]
                end_i = indices[local_i + self.future_known_steps]
                
                if self._is_valid_sample(start_i, global_i, end_i):
                    windows.append((start_i, global_i, end_i, code))
                    
        return windows

    def _is_valid_sample(self, start_i: int, today_i: int, end_i: int) -> bool:
        """Check if a sample is valid based on NaN handling strategy."""
        # Always check target
        if np.isnan(self.target_array[today_i]):
            return False
            
        if self.drop_nan_samples:
            # Drop samples with any NaN values
            if self.past_features and np.isnan(self.past_array[start_i:today_i]).any():
                return False
            if self.future_features and np.isnan(self.future_array[start_i:end_i]).any():
                return False
            if self.now_features and np.isnan(self.now_array[today_i]).any():
                return False
        else:
            # Allow NaN values if using mask (will be handled in __getitem__)
            pass
            
        return True

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary with tensors following the required structure:
            - x_past: (past_time_steps, past_features)
            - x_nan_mask: (past_time_steps, past_features) - binary mask for NaN values
            - x_future: (future_time_steps, future_features)  
            - x_now: (1, now_features)
            - x_static: (static_features,)
            - y: scalar target value
            - metadata: dict with date, code, etc.
        """
        start_i, today_i, end_i, code = self.valid_windows[idx]
        
        # Extract past features and create NaN mask
        if self.past_features:
            x_past = self.past_array[start_i:today_i].copy()  # (lookback, n_past_features)
            x_nan_mask = np.isnan(x_past).astype(np.float32)  # Binary mask
            
            if self.use_nan_mask:
                # Fill NaN with 0 when using mask
                x_past = np.nan_to_num(x_past, nan=0.0)
        else:
            x_past = np.empty((self.lookback, 0), dtype=np.float32)
            x_nan_mask = np.empty((self.lookback, 0), dtype=np.float32)
        
        # Extract future features
        if self.future_features:
            x_future = self.future_array[start_i:end_i].copy()  # (lookback + future_steps, n_future_features)
            
            if self.use_nan_mask:
                x_future = np.nan_to_num(x_future, nan=0.0)
        else:
            x_future = np.empty((self.lookback + self.future_known_steps, 0), dtype=np.float32)
        
        # Extract now features (current time step)
        if self.now_features:
            x_now = self.now_array[today_i:today_i+1].copy()  # (1, n_now_features)
            
            if self.use_nan_mask:
                x_now = np.nan_to_num(x_now, nan=0.0)
        else:
            x_now = np.empty((1, 0), dtype=np.float32)
        
        # Extract static features
        x_static = self.static_features_dict[code].copy()  # (n_static_features,)
        
        # Extract target
        y = self.target_array[today_i]
        
        # Get metadata
        date = self.date_array[today_i]
        if isinstance(date, np.datetime64):
            date = pd.to_datetime(date)
        elif not isinstance(date, pd.Timestamp):
            date = pd.to_datetime(date)
            
        metadata = {
            'date': date,
            'code': code,
            'day': date.day,
            'month': date.month,
            'year': date.year,
        }
        
        return {
            'x_past': torch.tensor(x_past, dtype=torch.float32),
            'x_nan_mask': torch.tensor(x_nan_mask, dtype=torch.float32),
            'x_future': torch.tensor(x_future, dtype=torch.float32),
            'x_now': torch.tensor(x_now, dtype=torch.float32),
            'x_static': torch.tensor(x_static, dtype=torch.float32),
            'y': torch.tensor(y, dtype=torch.float32),
            'metadata': metadata,
        }

    def get_sample_info(self, idx: int) -> Tuple[pd.Timestamp, Any]:
        """Get date and code for a sample."""
        _, today_i, _, code = self.valid_windows[idx]
        date = self.date_array[today_i]
        if isinstance(date, np.datetime64):
            date = pd.to_datetime(date)
        return date, code


class DeepLearningDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for deep learning forecasting models.
    
    Handles data splitting, preprocessing, and DataLoader creation for training,
    validation, and testing phases.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
        past_features: List[str],
        future_features: List[str],
        static_features: List[str],
        now_features: Optional[List[str]] = None,
        train_years: Optional[List[int]] = None,
        val_years: Optional[List[int]] = None,
        test_years: Optional[List[int]] = None,
        lookback: int = 365,
        future_known_steps: int = 30,
        batch_size: int = 32,
        num_workers: int = 4,
        use_nan_mask: bool = True,
        drop_nan_samples: bool = False,
        **dataset_kwargs
    ):
        """
        Initialize the DataModule.
        
        Args:
            df: Main time series dataframe
            static_df: Static basin characteristics
            past_features: Features for past time steps
            future_features: Features for future time steps
            static_features: Static basin features
            now_features: Features for current time step
            train_years: Years to use for training
            val_years: Years to use for validation  
            test_years: Years to use for testing
            lookback: Number of past time steps
            future_known_steps: Number of future time steps
            batch_size: Batch size for DataLoaders
            num_workers: Number of worker processes
            use_nan_mask: Whether to use NaN masking
            drop_nan_samples: Whether to drop samples with NaN
            **dataset_kwargs: Additional arguments for dataset
        """
        super().__init__()
        
        self.df = df
        self.static_df = static_df
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # Dataset configuration
        self.dataset_kwargs = {
            'past_features': past_features,
            'future_features': future_features,
            'static_features': static_features,
            'now_features': now_features,
            'lookback': lookback,
            'future_known_steps': future_known_steps,
            'use_nan_mask': use_nan_mask,
            'drop_nan_samples': drop_nan_samples,
            **dataset_kwargs
        }

    def setup(self, stage: Optional[str] = None):
        """Set up datasets for different stages."""
        df = self.df.copy()
        
        # Add year column for splitting
        if 'year' not in df.columns:
            df['year'] = pd.to_datetime(df['date']).dt.year
        
        # Split data by years
        if self.train_years:
            train_df = df[df['year'].isin(self.train_years)]
        else:
            # Default: use 80% of years for training
            years = sorted(df['year'].unique())
            n_train = int(0.8 * len(years))
            train_df = df[df['year'].isin(years[:n_train])]
            
        if self.val_years:
            val_df = df[df['year'].isin(self.val_years)]
        else:
            # Default: use 10% of years for validation
            years = sorted(df['year'].unique())
            n_train = int(0.8 * len(years))
            n_val = int(0.1 * len(years))
            val_df = df[df['year'].isin(years[n_train:n_train+n_val])]
            
        if self.test_years:
            test_df = df[df['year'].isin(self.test_years)]
        else:
            # Default: use remaining 10% of years for testing
            years = sorted(df['year'].unique())
            n_train = int(0.8 * len(years))
            n_val = int(0.1 * len(years))
            test_df = df[df['year'].isin(years[n_train+n_val:])]
        
        # Create datasets
        self.train_dataset = DeepLearningDataset(
            train_df, self.static_df, **self.dataset_kwargs
        )
        self.val_dataset = DeepLearningDataset(
            val_df, self.static_df, **self.dataset_kwargs
        )
        self.test_dataset = DeepLearningDataset(
            test_df, self.static_df, **self.dataset_kwargs
        )
        
        logger.info(f"Dataset setup complete:")
        logger.info(f"  Train: {len(self.train_dataset)} samples")
        logger.info(f"  Val: {len(self.val_dataset)} samples")
        logger.info(f"  Test: {len(self.test_dataset)} samples")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0
        )

    def predict_dataloader(self):
        return self.test_dataloader()


def create_deep_learning_dataloader(
    df: pd.DataFrame,
    static_df: pd.DataFrame,
    past_features: List[str],
    future_features: List[str],
    static_features: List[str],
    now_features: Optional[List[str]] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4,
    **dataset_kwargs
) -> DataLoader:
    """
    Convenience function to create a DataLoader for deep learning models.
    
    Args:
        df: Time series dataframe
        static_df: Static features dataframe
        past_features: Past time step features
        future_features: Future time step features  
        static_features: Static basin features
        now_features: Current time step features
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        **dataset_kwargs: Additional dataset arguments
        
    Returns:
        DataLoader for the dataset
    """
    dataset = DeepLearningDataset(
        df=df,
        static_df=static_df,
        past_features=past_features,
        future_features=future_features,
        static_features=static_features,
        now_features=now_features,
        **dataset_kwargs
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0
    )