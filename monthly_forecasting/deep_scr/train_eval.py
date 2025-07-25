import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import torch

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.metrics import r2_score, root_mean_squared_error
from tqdm import tqdm
import datetime
from joblib import dump, load

# Custom modules
from scr import data_loading as dl
from scr import data_utils as du
from scr import FeatureExtractor as FE
from scr import tree_utils
from scr import eval_helper

from config import PATH_CONFIG, EXPERIMENT_CONFIG, MODEL_CONFIG, FEATURE_CONFIG, PATH_BASE_LEARNER
from meta_config import META_CONFIG
from deep_scr import data_class, MLP, LSTM, NaiveEnsemble, UncertaintyNet, AL_UncertaintyNet
from deep_scr import callbacks_helper

# Shared logging
import logging
from scr.log_config import setup_logging
setup_logging()  

logger = logging.getLogger(__name__) 


def calculate_NSE(observed: np.ndarray, 
                 simulated: np.ndarray) -> float:
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) with robust error handling.
    
    NSE = 1 - Σ(O_i - S_i)² / Σ(O_i - O_mean)²
    where O_i are observed values and S_i are simulated values
    
    Parameters:
    -----------
    observed : np.ndarray
        Array of observed values
    simulated : np.ndarray
        Array of simulated values
    return_components : bool, optional
        If True, returns additional information about the calculation
        
    Returns:
    --------
    Union[float, Tuple[float, int, dict]]
        If return_components=False: returns NSE value
        If return_components=True: returns (NSE value, number of valid samples, components dict)
    """
    # Convert inputs to numpy arrays if they aren't already
    observed = np.asarray(observed)
    simulated = np.asarray(simulated)
    
    # Ensure same length
    if observed.shape != simulated.shape:
        raise ValueError(f"Observed and simulated arrays must have same shape. "
                        f"Got shapes {observed.shape} and {simulated.shape}")
    
    # Find valid indices (non-NaN in both arrays)
    valid_mask = ~np.isnan(observed) & ~np.isnan(simulated)
    valid_observed = observed[valid_mask]
    valid_simulated = simulated[valid_mask]
    
    # Check if we have enough valid data points
    n_valid = len(valid_observed)
    if n_valid < 2:  # Need at least 2 points to calculate variance
        return np.nan
    
    # Calculate observed mean
    obs_mean = np.mean(valid_observed)
    
    # Calculate denominator (variance of observations)
    denominator = np.sum((valid_observed - obs_mean) ** 2)
    
    # Check if denominator is too close to zero
    if denominator < 1e-10:
        return np.nan
    
    # Calculate numerator (sum of squared residuals)
    numerator = np.sum((valid_observed - valid_simulated) ** 2)
    
    # Calculate NSE
    nse = 1 - (numerator / denominator)
    
    # Handle edge cases
    if np.isinf(nse) or np.isnan(nse):
        return np.nan
    
    return nse


def normalize_train_data(
    df: pd.DataFrame,
    id_col: str = 'code',
    discharge_col: str = 'discharge',
    target_col: str = 'target'
):
    """
    Normalizes the training data with StandardScaler.
    - Scales `discharge_col` and `target_col` per-unique-ID (id_col) separately.
    - Scales all other numeric columns globally.
    
    Returns:
        df_train: DataFrame with normalized data
        scaler_discharge: Dict[id_value] = (mean, std) for discharge
        scaler_target: Dict[id_value] = (mean, std) for target
        scaler_features: Dict[column] = (mean, std) for other features
    """
    df_train = df.copy()
    
    # Per-ID scaling for discharge and target
    scaler_discharge = {}
    scaler_target = {}
    
    # Grouped means and stds (ddof=0 for population std)
    grp_disc = df_train.groupby(id_col)[discharge_col]
    grp_targ = df_train.groupby(id_col)[target_col]
    
    for gid, vals in grp_disc:
        mean = vals.mean()
        std = vals.std(ddof=0)
        scaler_discharge[gid] = (mean, std)
    for gid, vals in grp_targ:
        mean = vals.mean()
        std = vals.std(ddof=0)
        scaler_target[gid] = (mean, std)
    
    # Apply per-group transform
    df_train[discharge_col] = grp_disc.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    df_train[target_col]    = grp_targ.transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    
    base_learner_cols = META_CONFIG['base_learner_cols']
    #scale the base learner with the mean and std of the target
    for id in df_train[id_col].unique():
        mean, std = scaler_target[id]
        for col in base_learner_cols:
            if col in df_train.columns:
                df_train.loc[df_train[id_col] == id, col] = (df_train.loc[df_train[id_col] == id, col] - mean) / std
            else:
                logger.warning(f"Column {col} not found in DataFrame. Skipping scaling for this column.")
    # Global scaling for other features
    other_cols = [
        col for col in df_train.columns
        if col not in (id_col, discharge_col, target_col) 
           and pd.api.types.is_numeric_dtype(df_train[col])
           and col not in base_learner_cols
    ]
    scaler_features = {}
    for col in other_cols:
        mean = df_train[col].mean()
        std = df_train[col].std(ddof=0)
        scaler_features[col] = (mean, std)
        df_train[col] = (df_train[col] - mean) / std
    
    return df_train, scaler_discharge, scaler_target, scaler_features


def scale_data(
    df: pd.DataFrame,
    scaler_discharge: dict,
    scaler_target: dict,
    scaler_features: dict,
    id_col: str = 'code',
    discharge_col: str = 'discharge',
    target_col: str = 'target'
):
    """
    Scales new data using provided scalers.
    Args:
        df: DataFrame with data to be scaled
        scaler_discharge: Dict[id] = (mean, std) for discharge
        scaler_target:    Dict[id] = (mean, std) for target
        scaler_features:  Dict[feature] = (mean, std) for other features
    Returns:
        df_scaled: DataFrame with scaled data
    """
    df_scaled = df.copy()
    
    # Per-ID scaling
    def _scale_val(val, gid, scaler_dict):
        mean, std = scaler_dict.get(gid, (np.nan, np.nan))
        return (val - mean) / std
    
    df_scaled[discharge_col] = df_scaled.apply(
        lambda row: _scale_val(row[discharge_col], row[id_col], scaler_discharge),
        axis=1
    )
    df_scaled[target_col] = df_scaled.apply(
        lambda row: _scale_val(row[target_col], row[id_col], scaler_target),
        axis=1
    )

    # Scale base learner columns
    base_learner_cols = META_CONFIG['base_learner_cols']
    for id in df_scaled[id_col].unique():
        mean, std = scaler_target[id]
        for col in base_learner_cols:
            if col in df_scaled.columns:
                df_scaled.loc[df_scaled[id_col] == id, col] = (df_scaled.loc[df_scaled[id_col] == id, col] - mean) / std
            else:
                logger.warning(f"Column {col} not found in DataFrame. Skipping scaling for this column.")
    
    # Global features
    for col, (mean, std) in scaler_features.items():
        df_scaled[col] = (df_scaled[col] - mean) / std
    
    return df_scaled

def scale_static_features(
    static_df: pd.DataFrame,
    id_col: str = None
):
    """
    Scales static features with StandardScaler.
    If `id_col` is provided and present in static_df, it will be preserved.
    
    Returns:
        static_df_scaled: DataFrame with scaled static features
        scaler_static: Dict[feature] = (mean, std)
    """
    df = static_df.copy()
    scaler_static = {}
    
    # Identify columns to scale
    cols_to_scale = [
        col for col in df.columns
        if col != id_col and pd.api.types.is_numeric_dtype(df[col])
    ]
    
    for col in cols_to_scale:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        scaler_static[col] = (mean, std)
        df[col] = (df[col] - mean) / std
    
    return df, scaler_static


def get_error_distribution_of_base_learner(
    df: pd.DataFrame,
    id_col: str = 'code',
    target_col: str = 'target',
    date_col: str = 'date',
    pred_col: str = 'LR BASELINE'
    ) -> pd.DataFrame:
    """
    Computes the monthly error distribution for a base learner.
    Returns a DataFrame with columns:
      [id_col, month, base_learner_mean_error, base_learner_std_error]
    """
    dfc = df.copy()
    dfc[date_col] = pd.to_datetime(dfc[date_col])
    dfc['month'] = dfc[date_col].dt.month

    # error = actual - predicted
    dfc[f"error"] = dfc[target_col] - dfc[pred_col]

    # group by id and month
    dist = (
        dfc
        .groupby([id_col, 'month'])[f"error"]
        .agg(
            mean_error='mean',
            abs_mean_error=lambda x: x.abs().mean(),
            std_error=lambda x: x.std(ddof=0)
        )
        .reset_index()
    )

    # rename columns
    dist.rename(
        columns={
            'mean_error': f'{pred_col}_mean_error',
            'abs_mean_error': f'{pred_col}_abs_mean_error',
            'std_error': f'{pred_col}_std_error'
        },
        inplace=True
    )

    return dist



def apply_error_distribution(
    df: pd.DataFrame,
    error_distribution: pd.DataFrame,
    id_col: str = 'code',
    date_col: str = 'date',
    ) -> pd.DataFrame:
    """
    Merges in the monthly error stats and produces a scaled prediction column:
      prediction_scaled = (prediction - mean_error) / std_error
    Returns the enriched DataFrame.
    """
    dfc = df.copy()
    dfc[date_col] = pd.to_datetime(dfc[date_col])
    dfc['month'] = dfc[date_col].dt.month

    # merge stats
    merged = pd.merge(
        dfc,
        error_distribution,
        on=[id_col, 'month'],
        how='left'
    )
    
    return merged


 
def get_nse_distribution_of_base_learner(
    df: pd.DataFrame,
    id_col: str = 'code',
    target_col: str = 'target',
    date_col: str = 'date',
    pred_col: str = 'LR BASELINE'
) -> pd.DataFrame:
    """
    Computes the monthly NSE distribution for a base learner.
    Returns a DataFrame with columns:
      [id_col, month, base_learner_nse]
    """
    dfc = df.copy()
    dfc[date_col] = pd.to_datetime(dfc[date_col])
    dfc['month'] = dfc[date_col].dt.month
    
    #generate a unique time_id month-day
    dfc['is_end_of_month'] = dfc[date_col].dt.day >= 20
    dfc['suffix'] = dfc['is_end_of_month'].astype(str)
    dfc['unique_time'] = dfc['month'].astype(str) + '-' + dfc['suffix']

    logger.info(f"Unique time: {dfc['unique_time'].unique()}")
    # Group by id and month, then calculate NSE
    def safe_nse(group):
        return calculate_NSE(group[target_col].values, group[pred_col].values)

    nse_dist = (
        dfc
        .groupby([id_col, 'unique_time'])
        .apply(safe_nse)
        .reset_index(name=f'{pred_col}_nse')
    )

    return nse_dist


def apply_nse_distribution(
    df: pd.DataFrame,
    nse_distribution: pd.DataFrame,
    id_col: str = 'code',
    date_col: str = 'date',
) -> pd.DataFrame:
    """
    Merges the monthly NSE stats into the DataFrame and enriches it with NSE values.

    Args:
        df: DataFrame containing the data to enrich.
        nse_distribution: DataFrame containing the NSE distribution with columns [id_col, month, base_learner_nse].
        id_col: Column name for unique identifiers (e.g., station codes).
        date_col: Column name for the date.

    Returns:
        DataFrame enriched with NSE values.
    """
    dfc = df.copy()
    dfc[date_col] = pd.to_datetime(dfc[date_col])
    dfc['month'] = dfc[date_col].dt.month
    # Generate a unique time_id month-day
    dfc['is_end_of_month'] = dfc[date_col].dt.day >= 20
    dfc['suffix'] = dfc['is_end_of_month'].astype(str)
    dfc['unique_time'] = dfc['month'].astype(str) + '-' + dfc['suffix']

    # Merge the NSE distribution into the DataFrame
    merged = pd.merge(
        dfc,
        nse_distribution,
        on=[id_col, 'unique_time'],
        how='left'
    )

    # Drop the temporary columns
    merged.drop(columns=[ 'is_end_of_month', 'suffix', 'unique_time'], inplace=True)


    return merged



def prepare_data(df_train,df_val, df_test, static_df, save=False):
    df_train = df_train.copy()
    df_train['year'] = pd.to_datetime(df_train['date']).dt.year

    df_val = df_val.copy()
    df_val['year'] = pd.to_datetime(df_val['date']).dt.year

    df_test = df_test.copy()
    df_test['year'] = pd.to_datetime(df_test['date']).dt.year

    logger.info('Scaling data...')
    # Scale the data
    df_train, scaler_discharge, scaler_target, scaler_features = normalize_train_data(df_train)
    df_val = scale_data(df_val, scaler_discharge, scaler_target, scaler_features, id_col='code')
    df_test = scale_data(df_test, scaler_discharge, scaler_target, scaler_features, id_col='code')
    static_df, scaler_static = scale_static_features(static_df, id_col='code')

    # Apply error distribution
    if META_CONFIG['apply_error_distribution']:
        logger.info('Applying error distribution...')
        base_learner_error_cols = []
        for base_model in META_CONFIG['base_learner_cols']:
            if base_model in df_train.columns:
                error_distribution = get_error_distribution_of_base_learner(
                    df_train,
                    id_col='code',
                    target_col='target',
                    date_col='date',
                    pred_col=base_model
                )
                df_train = apply_error_distribution(df_train, error_distribution)
                df_val = apply_error_distribution(df_val, error_distribution)
                df_test = apply_error_distribution(df_test, error_distribution)

                error_cols_model = error_distribution.columns.tolist()
                error_cols_model.remove('code')
                error_cols_model.remove('month')

                base_learner_error_cols += error_cols_model

        META_CONFIG['base_learner_add_cols'] = base_learner_error_cols

    # Apply NSE distribution
    elif META_CONFIG['apply_nse_distribution']:
        logger.info('Applying NSE distribution...')
        nse_columns = []
        for base_model in META_CONFIG['base_learner_cols']:
            if base_model in df_train.columns:
                nse_distribution = get_nse_distribution_of_base_learner(
                    df_train,
                    id_col='code',
                    target_col='target',
                    date_col='date',
                    pred_col=base_model
                )
                df_train = apply_nse_distribution(df_train, nse_distribution)
                df_val = apply_nse_distribution(df_val, nse_distribution)
                df_test = apply_nse_distribution(df_test, nse_distribution)

                nse_columns += [f"{base_model}_nse"]
            
        META_CONFIG['base_learner_add_cols'] = nse_columns


    logger.info(f"Head of df_train: {df_train.head()}")
    logger.info(f"Head of df_val: {df_val.head()}")
    logger.info(f"Head of df_test: {df_test.head()}")


    if save:
        outdir = PATH_CONFIG['output_dir']

        scaler_dir = os.path.join(outdir, 'scalers')
        os.makedirs(scaler_dir, exist_ok=True)

        #save scalers a json
        with open(os.path.join(scaler_dir, 'scaler_discharge.json'), 'w') as f:
            json.dump(scaler_discharge, f)
        with open(os.path.join(scaler_dir, 'scaler_target.json'), 'w') as f:
            json.dump(scaler_target, f)
        with open(os.path.join(scaler_dir, 'scaler_features.json'), 'w') as f:
            json.dump(scaler_features, f)
        with open(os.path.join(scaler_dir, 'scaler_static.json'), 'w') as f:
            json.dump(scaler_static, f)
            


    return {
        'train': df_train,
        'val': df_val,
        'test': df_test,
        'static': static_df,
        'scaler_discharge': scaler_discharge,
        'scaler_target': scaler_target,
        'scaler_features': scaler_features,
        'scaler_static': scaler_static
    }


def calibrate(model, dm, load_ckpt=True):
    
    accelerator = 'cpu'
    if torch.cuda.is_available():
        accelerator = 'gpu'
    elif torch.backends.mps.is_available():
        accelerator = 'mps'

    # Initialize the callback
    metrics_callback = callbacks_helper.MetricsCallback()
    early_stop_cb = EarlyStopping(monitor='val_loss',
                                   patience=META_CONFIG['early_stop_patience'], 
                                   mode='min')
    
    lr_monitor_cb = LearningRateMonitor(logging_interval='epoch')

    checkpoint_cb = ModelCheckpoint(
        dirpath=META_CONFIG['checkpoint_dir'],
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='forecast-{epoch:02d}-{val_loss:.4f}'
    )

    # 4) train!
    trainer = Trainer(
        max_epochs=META_CONFIG['max_epochs'],
        accelerator=accelerator,
        log_every_n_steps=50,
        enable_progress_bar=True,
        gradient_clip_val=META_CONFIG['gradient_clip'],
        callbacks=[metrics_callback, early_stop_cb, lr_monitor_cb, checkpoint_cb],
    )
    if META_CONFIG['model_type'] != 'NaiveEnsemble':
        trainer.fit(model, datamodule=dm)
    
    else:
        metrics_callback = None

    trainer.validate(model, datamodule=dm)

    # Load the best model from its checkpoint
    best_model_path = checkpoint_cb.best_model_path
    if load_ckpt and best_model_path:
        ModelClass = type(model)
        model = ModelClass.load_from_checkpoint(best_model_path)
        logger.info(f"Loaded best model from {best_model_path}")

    return model, metrics_callback, trainer


def get_predictions(model, dm, trainer,  scaler_target):
    
    predictions = trainer.predict(model, datamodule=dm)

    # Concatenate all predictions
    predictions = pd.concat(predictions, axis=0)

    pred_cols = [ col for col in predictions.columns if 'Q' in col]

    for code in predictions['code'].unique():
        mean_ , std_ = scaler_target[code]
        predictions.loc[predictions['code'] == code, pred_cols] = predictions.loc[predictions['code'] == code, pred_cols] * std_ + mean_

    predictions['date'] = pd.to_datetime(predictions['date'])

    return predictions




def get_model(static_df):
        
    static_context_features = META_CONFIG['static_context_features']
    past_temporal_context_features = META_CONFIG['past_temporal_context_features']
    future_temporal_context_features = META_CONFIG['future_temporal_context_features'] 

    if static_context_features == 'all':
        static_context_features = static_df.columns.tolist()
        static_context_features.remove('code')

    base_learner_cols = META_CONFIG['base_learner_cols']
    past_dim  = len(past_temporal_context_features)
    future_dim = len(future_temporal_context_features)
    static_dim = len(static_context_features)
    base_learner_dim = len(base_learner_cols)
    base_learner_error_dim = len(META_CONFIG['base_learner_add_cols']) 
    quantiles = META_CONFIG['quantiles']
        
    if META_CONFIG['model_type'] == 'MLP':
        model = MLP.MLPForecast(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            base_learner_dim=base_learner_dim,
            base_learner_error_dim=base_learner_error_dim,
            lookback=META_CONFIG['lookback'],
            future_known_steps=META_CONFIG['future_known_steps'],
            hidden_dim=META_CONFIG['hidden_size'],
            lr=META_CONFIG['lr'],
            weight_decay=META_CONFIG['weight_decay'],
            output_dim=len(quantiles),
            loss_fn=META_CONFIG['loss_fn'],
            quantiles=quantiles,
            dropout=META_CONFIG['dropout'],
        )
    
    elif META_CONFIG['model_type'] == 'LSTM':
        model = LSTM.LSTMForecast(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            base_learner_dim=base_learner_dim,
            base_learner_error_dim=base_learner_error_dim,
            lookback=META_CONFIG['lookback'],
            future_known_steps=META_CONFIG['future_known_steps'],
            hidden_dim=META_CONFIG['hidden_size'],
            lr=META_CONFIG['lr'],
            weight_decay=META_CONFIG['weight_decay'],
            output_dim=len(quantiles),
            loss_fn=META_CONFIG['loss_fn'],
            quantiles=quantiles,
            dropout=META_CONFIG['dropout'],
        )

    elif META_CONFIG['model_type'] == 'NaiveEnsemble':
        model = NaiveEnsemble.NaiveEnsembleForecast(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            base_learner_dim=base_learner_dim,
            base_learner_error_dim=base_learner_error_dim,
            lookback=META_CONFIG['lookback'],
            future_known_steps=META_CONFIG['future_known_steps'],
            hidden_dim=META_CONFIG['hidden_size'],
            lr=META_CONFIG['lr'],
            weight_decay=META_CONFIG['weight_decay'],
            output_dim=len(quantiles),
            loss_fn=META_CONFIG['loss_fn'],
            quantiles=quantiles,
            dropout=META_CONFIG['dropout'],
        )

    elif META_CONFIG['model_type'] == 'UncertaintyNet':
        model = UncertaintyNet.Uncertainty_Forecast(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            base_learner_dim=base_learner_dim,
            base_learner_error_dim=base_learner_error_dim,
            lookback=META_CONFIG['lookback'],
            future_known_steps=META_CONFIG['future_known_steps'],
            hidden_dim=META_CONFIG['hidden_size'],
            lr=META_CONFIG['lr'],
            weight_decay=META_CONFIG['weight_decay'],
            output_dim=len(quantiles),
            loss_fn=META_CONFIG['loss_fn'],
            center_weight = META_CONFIG['center_weight'],
            quantiles=quantiles,
            dropout=META_CONFIG['dropout'],
        )
    elif META_CONFIG['model_type'] == 'AL_UncertaintyNet':
        model = AL_UncertaintyNet.AL_Uncertainty_Forecast(
            past_dim=past_dim,
            future_dim=future_dim,
            static_dim=static_dim,
            base_learner_dim=base_learner_dim,
            base_learner_error_dim=base_learner_error_dim,
            lookback=META_CONFIG['lookback'],
            future_known_steps=META_CONFIG['future_known_steps'],
            hidden_dim=META_CONFIG['hidden_size'],
            lr=META_CONFIG['lr'],
            weight_decay=META_CONFIG['weight_decay'],
            output_dim=len(quantiles),
            loss_fn=META_CONFIG['loss_fn'],
            adaptive_weighting=META_CONFIG['adaptive_weighting'],
            quantiles=quantiles,
            dropout=META_CONFIG['dropout'],
            correction_term=META_CONFIG['correction_term'],
            weight_by_metrics=META_CONFIG['apply_nse_distribution'],
        )
    else:
        raise ValueError(f"Unknown model type: {META_CONFIG['model_type']}")

    return model


def loo_cv(df, static_df):
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year
    static_df = static_df.copy()

    static_context_features = META_CONFIG['static_context_features']
    past_temporal_context_features = META_CONFIG['past_temporal_context_features']
    future_temporal_context_features = META_CONFIG['future_temporal_context_features'] 

    if static_context_features == 'all':
        static_context_features = static_df.columns.tolist()
        static_context_features.remove('code')

    base_learner_cols = META_CONFIG['base_learner_cols']

    all_predictions = pd.DataFrame()
    years = df['year'].unique()
    for year in years:
        logger.info(f"Running LOO CV for year: {year}")
        df_train = df[df['year'] != year]
        df_val = df[df['year'] == year]
        df_test = df[df['year'] == year]

        # Prepare data
        data_dict = prepare_data(df_train, df_val, df_test, static_df, save=False)

        # Create the data module
        dm = data_class.MetaMonthDataModule(
            df=None,
            static_df=data_dict['static'],
            static_features=static_context_features,
            past_features=past_temporal_context_features,
            future_features=future_temporal_context_features,
            base_learner_cols=base_learner_cols,
            base_learner_add_cols=META_CONFIG['base_learner_add_cols'],
            train_years = [],
            val_years = [],
            test_years = [],
            lookback=META_CONFIG['lookback'],
            future_known_steps=META_CONFIG['future_known_steps'],
            batch_size=META_CONFIG['batch_size'],
            num_workers=META_CONFIG['num_workers'],
            train_df=data_dict['train'],
            val_df=data_dict['val'],
            test_df=data_dict['test'],

        )

        # 3) instantiate the LightningModule
        model = get_model(static_df= static_df)

        
        model, metrics_callback, trainer = calibrate(model, dm, load_ckpt=False)

        # Get predictions
        predictions = get_predictions(model, dm, trainer, data_dict['scaler_target'])

        # Append predictions to all_predictions
        all_predictions = pd.concat([all_predictions, predictions], axis=0)

    
    return all_predictions




def fit_on_all(df, static_df, train_years, val_years, test_years):
    df = df.copy()
    df['year'] = pd.to_datetime(df['date']).dt.year

    static_df = static_df.copy()

    static_context_features = META_CONFIG['static_context_features']
    past_temporal_context_features = META_CONFIG['past_temporal_context_features']
    future_temporal_context_features = META_CONFIG['future_temporal_context_features']

    if static_context_features == 'all':
        static_context_features = static_df.columns.tolist()
        static_context_features.remove('code')

    base_learner_cols = META_CONFIG['base_learner_cols']

    df_train = df[df['year'].isin(train_years)].copy()
    df_val = df[df['year'].isin(val_years)].copy()
    df_test = df[df['year'].isin(test_years)].copy()
    # Prepare data
    data_dict = prepare_data(df_train, df_val, df_test, static_df, save=True)
    # Create the data module
    dm = data_class.MetaMonthDataModule(
        df=None,
        static_df=data_dict['static'],
        static_features=static_context_features,
        past_features=past_temporal_context_features,
        future_features=future_temporal_context_features,
        base_learner_cols=base_learner_cols,
        base_learner_add_cols=META_CONFIG['base_learner_add_cols'],
        train_years=train_years,
        val_years=val_years,
        test_years=test_years,
        lookback=META_CONFIG['lookback'],
        future_known_steps=META_CONFIG['future_known_steps'],
        batch_size=META_CONFIG['batch_size'],
        num_workers=META_CONFIG['num_workers'],
        train_df=data_dict['train'],
        val_df=data_dict['val'],
        test_df=data_dict['test'],

    )


    model = get_model(static_df= static_df)

    model, metrics_callback, trainer = calibrate(model, dm, load_ckpt=True)

    # Get predictions
    predictions = get_predictions(model, dm, trainer, data_dict['scaler_target'])

    
    return predictions, model, metrics_callback



