import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Any, Dict, Optional


import pytorch_lightning as pl

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


class MetaMonthDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
        past_features: List[str],
        future_features: List[str],
        static_features: List[str],
        base_learner_cols: List[str],
        base_learner_add_cols: List[str],
        train_years: List[int],
        val_years: List[int],
        test_years: List[int],
        lookback: int,
        future_known_steps: int,
        batch_size: int = 32,
        num_workers: int = 4,
        train_df: Optional[pd.DataFrame] = None,
        val_df: Optional[pd.DataFrame] = None,
        test_df: Optional[pd.DataFrame] = None,
        **dataset_kwargs,
    ):
        super().__init__()
        self.df = df
        self.static_df = static_df
        self.dataset_kwargs = dict(
            past_features=past_features,
            future_features=future_features,
            static_features=static_features,
            base_learner_cols=base_learner_cols,
            base_learner_add_cols=base_learner_add_cols,
            lookback=lookback,
            future_known_steps=future_known_steps,
            **dataset_kwargs,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_years = train_years
        self.val_years = val_years
        self.test_years = test_years

        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # check if self.df and self.train_df are None
        if self.df is None and self.train_df is None:
            raise ValueError("Either df or train_df must be provided.")

    def setup(self, stage=None):
        # instantiate full dataset
        if self.df is not None:
            df = self.df.copy()
            df["year"] = pd.to_datetime(df["date"]).dt.year

        if self.train_df is None:
            self.train_df = df[df["year"].isin(self.train_years)]
        if self.val_df is None:
            self.val_df = df[df["year"].isin(self.val_years)]
        if self.test_df is None:
            self.test_df = df[df["year"].isin(self.test_years)]

        self.train_ds = META_MONTH_DATA(
            self.train_df, self.static_df, **self.dataset_kwargs
        )
        self.val_ds = META_MONTH_DATA(
            self.val_df, self.static_df, **self.dataset_kwargs
        )
        self.test_ds = META_MONTH_DATA(
            self.test_df, self.static_df, **self.dataset_kwargs
        )

        # for prediction you might want the *entire* set or only test
        self.predict_ds = self.test_ds

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            pin_memory=True,
        )


class META_MONTH_DATA(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        static_df: pd.DataFrame,
        past_features: List[str],
        future_features: List[str],
        static_features: List[str],
        base_learner_cols: List[str],
        base_learner_add_cols: List[str],
        lookback: int,
        future_known_steps: int,
        unique_id_col: str = "code",
        date_col: str = "date",
        target_col: str = "target",
    ):
        """
        df: long dataframe with columns [unique_id_col, date_col, past_features..., future_features..., base_learner_cols..., target_col]
        static_df: wide dataframe indexed by unique_id_col, containing static_features
        """
        self.df = df.copy()
        self.static_df = static_df.copy()
        self.past_features = past_features
        self.future_features = future_features
        self.static_features = static_features
        self.base_learner_cols = base_learner_cols
        self.base_learner_add_cols = base_learner_add_cols
        self.lookback = lookback
        self.future_known_steps = future_known_steps
        self.unique_id_col = unique_id_col
        self.date_col = date_col
        self.target_col = target_col

        # Pre-sort the data - more efficient than sort_values
        idx_cols = [self.unique_id_col, self.date_col]
        self.df = self.df.sort_values(idx_cols).reset_index(drop=True)

        # Precompute static features as a dict for faster access
        self.static_features_dict = {}
        for _, row in self.static_df.iterrows():
            code = row[self.unique_id_col]
            self.static_features_dict[code] = row[self.static_features].values.astype(
                np.float32
            )

        # Convert target and base_learner columns to numpy arrays for faster slicing
        self.target_array = self.df[self.target_col].values
        self.base_learner_array = self.df[self.base_learner_cols].values
        self.base_learner_error_array = self.df[base_learner_add_cols].values

        # Also precompute past and future feature arrays
        self.past_array = self.df[past_features].values
        self.future_array = self.df[future_features].values

        # Store code and date arrays for faster lookup
        self.code_array = self.df[self.unique_id_col].values
        self.date_array = self.df[self.date_col].values

        # Build the list of valid windows once
        self.valid_windows = self.get_valid_index()

    def get_valid_index(self) -> List[Tuple[int, int, int, Any]]:
        """
        Scan each code-specific block and for each possible 'today' index i
        keep (start_idx, today_idx, end_idx, code) whenever:
          1. past_features have NO NaNs on [i - lookback : i)
          2. future_features have NO NaNs on [i - lookback : i + future_known_steps)
          3. target_col at i is not NaN
          4. all base_learner_cols at i are not NaN
        """
        windows = []
        # Get unique codes and their indices
        codes = self.df[self.unique_id_col].values
        dates = self.df[self.date_col].values

        # Use numpy for faster processing
        past_na_mask = np.isnan(self.past_array).any(axis=1)
        future_na_mask = np.isnan(self.future_array).any(axis=1)
        target_na_mask = np.isnan(self.target_array)
        base_na_mask = np.isnan(self.base_learner_array).any(axis=1)
        base_error_na_mask = np.isnan(self.base_learner_error_array).any(axis=1)

        # Process each group more efficiently
        group_indices = {}
        for i, code in enumerate(codes):
            if code not in group_indices:
                group_indices[code] = []
            group_indices[code].append(i)

        # Process each code group
        for code, indices in group_indices.items():
            indices = np.array(indices)
            n = len(indices)

            # Only consider indices where both past & future windows fit
            for local_i in range(self.lookback, n - self.future_known_steps):
                global_i = indices[local_i]
                start_i = indices[local_i - self.lookback]
                end_i = indices[local_i + self.future_known_steps]

                # Check conditions using precomputed masks
                if past_na_mask[start_i:global_i].any():
                    continue
                if future_na_mask[start_i:end_i].any():
                    continue
                if target_na_mask[global_i]:
                    continue
                if base_na_mask[global_i]:
                    continue
                if base_error_na_mask[global_i]:
                    continue

                windows.append((start_i, global_i, end_i, code))

        return windows

    def get_time_code(self, idx: int) -> Tuple[pd.Timestamp, Any]:
        """
        Given an integer idx in [0 .. len(self)-1],
        returns (date, code) for today_idx of that sample.
        """
        _, today_i, _, code = self.valid_windows[idx]
        date = self.date_array[today_i]
        return date, code

    def __len__(self):
        return len(self.valid_windows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        start_i, today_i, end_i, code = self.valid_windows[idx]

        # Use precomputed arrays for faster slicing
        past_np = self.past_array[start_i:today_i]
        future_np = self.future_array[start_i:end_i]

        y_val = self.target_array[today_i]
        base_np = self.base_learner_array[today_i]

        # Get static features from precomputed dict
        static_np = self.static_features_dict[code]

        raw_date = self.date_array[today_i]

        error_np = self.base_learner_error_array[today_i]

        # 1) If it's numpy.datetime64, convert to pandas.Timestamp first
        if isinstance(raw_date, np.datetime64):
            ts = pd.to_datetime(raw_date)
        elif isinstance(raw_date, pd.Timestamp):
            ts = raw_date
        else:
            # assume it's already a Python datetime.datetime
            ts = raw_date

        # 2) Now you can safely do:
        day = ts.day
        month = ts.month
        year = ts.year

        return {
            "past_input": torch.tensor(past_np, dtype=torch.float32),
            "future_input": torch.tensor(future_np, dtype=torch.float32),
            "y": torch.tensor(y_val, dtype=torch.float32),
            "base_learners": torch.tensor(base_np, dtype=torch.float32),
            "base_learner_errors": torch.tensor(error_np, dtype=torch.float32),
            "static_input": torch.tensor(static_np, dtype=torch.float32),
            "day": day,
            "month": month,
            "year": year,
            "code": code,
        }


class TabularDataset(Dataset):
    """
    Tabular Dataset.
    Returns a tensor in shape (batch_size, num_features)
    """

    def __init__(self, df: pd.DataFrame, features: List[str], target: str):
        self.df = df.copy()
        self.features = features
        self.y = df[target].values

        self.codes = self.df["code"].astype(int).values
        self.dates = pd.to_datetime(self.df["date"])
        self.days = self.dates.dt.day.values
        self.months = self.dates.dt.month.values
        self.years = self.dates.dt.year.values
        self.X = self.df[self.features].values

        logger.info(f"TabularDataset initialized with {len(self.X)} samples.")
        logger.info(f"Features: {self.features}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "X": torch.tensor(self.X[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "code": torch.tensor(self.codes[idx], dtype=torch.float32),
            "day": torch.tensor(self.days[idx], dtype=torch.float32),
            "month": torch.tensor(self.months[idx], dtype=torch.float32),
            "year": torch.tensor(self.years[idx], dtype=torch.float32),
        }


if __name__ == "__main__":
    """
    Test function for TabularDataset.
    Creates a synthetic dataset and demonstrates iteration and DataLoader usage.
    """
    print("Testing TabularDataset...")

    # Create synthetic test data
    np.random.seed(42)
    n_samples = 10**6
    n_features = 70

    # Generate synthetic data
    dates = pd.date_range(start="2020-01-01", periods=n_samples, freq="s")
    codes = np.random.choice([1, 2, 3, 4, 5], size=n_samples)

    # Create feature columns
    feature_names = [f"feature_{i}" for i in range(n_features)]
    features_data = np.random.randn(n_samples, n_features)

    # Create target (simple linear combination of features + noise)
    target = np.random.randn(n_samples)

    # Create DataFrame
    data = {
        "date": dates,
        "code": codes,
        "target": target,
    }
    for i, name in enumerate(feature_names):
        data[name] = features_data[:, i]

    df = pd.DataFrame(data)

    print(f"Created synthetic dataset with {len(df)} samples")
    print(f"Features: {feature_names}")
    print(f"DataFrame shape: {df.shape}")
    print(f"First few rows:")
    print(df.head())

    # Create TabularDataset
    dataset = TabularDataset(df=df, features=feature_names, target="target")

    print(f"\nDataset length: {len(dataset)}")

    # Test direct iteration through dataset
    print("\n=== Testing direct iteration ===")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        print(f"Sample {i}:")
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    print(f"  {key}: {value.item():.4f}")
                else:
                    print(f"  {key}: shape={value.shape}, mean={value.mean():.4f}")
            else:
                print(f"  {key}: {value}")
        print()

    # Test DataLoader
    print("=== Testing DataLoader ===")
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
    )

    print(f"DataLoader created with batch_size=8")
    print(f"Number of batches: {len(dataloader)}")

    print("About to create iterator...")
    train_iter = iter(dataloader)
    print("Iterator created, getting first batch...")
    batch = next(train_iter)
    print("First batch retrieved!")

    # Iterate through first few batches
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 2:  # Only show first 2 batches
            break

        print(f"\nBatch {batch_idx + 1}:")
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                if key == "X":
                    print(f"    X min/max: {value.min():.4f}/{value.max():.4f}")
                elif key == "y":
                    print(f"    y min/max: {value.min():.4f}/{value.max():.4f}")
                elif value.numel() <= 16:  # Show values for small tensors
                    print(
                        f"    values: {value.flatten()[:8]}..."
                    )  # Show first 8 values
            else:
                print(f"  {key}: {value}")

    # Test with missing values
    print("\n=== Testing with missing values ===")
    df_with_nan = df.copy()
    # Introduce some NaN values
    df_with_nan.loc[5:10, "feature_1"] = np.nan
    df_with_nan.loc[15:20, "target"] = np.nan

    dataset_with_nan = TabularDataset(
        df=df_with_nan, features=feature_names, target="target"
    )
    print(f"Dataset with NaN values created, length: {len(dataset_with_nan)}")

    # Test with a smaller batch
    print("\n=== Testing different batch sizes ===")
    small_dataloader = DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)
    batch = next(iter(small_dataloader))
    print(f"Small batch (size=3):")
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape={value.shape}")

    print("\n=== Testing completed successfully! ===")
    print(f"Total samples processed: {len(dataset)}")
    print(f"Features shape: {dataset.X.shape}")
    print(f"Target shape: {dataset.y.shape}")
    print(f"Unique codes: {sorted(np.unique(dataset.codes))}")
    print(f"Date range: {dataset.dates.min()} to {dataset.dates.max()}")
