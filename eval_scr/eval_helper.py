import os
import sys
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

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

from config import PATH_CONFIG, EXPERIMENT_CONFIG, MODEL_CONFIG, FEATURE_CONFIG


import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_r2_rmse(df_predictions):
    df_predictions = df_predictions.copy()

    overall_r2 = r2_score(df_predictions['Q_obs'], df_predictions['Q_pred'])
    overall_rmse = root_mean_squared_error(df_predictions['Q_obs'], df_predictions['Q_pred'])

    r2_array = []
    nrmse = []
    for code in df_predictions.code.unique():
        df_code = df_predictions[df_predictions['code'] == code]
        r2 = r2_score(df_code['Q_obs'], df_code['Q_pred'])
        rmse = root_mean_squared_error(df_code['Q_obs'], df_code['Q_pred'])
        nrmse.append(rmse / df_code['Q_obs'].mean())
        r2_array.append(r2)

    save_dir = PATH_CONFIG['output_dir'] 
    #box plot of r2 and nrmse
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].boxplot(r2_array)
    ax[0].set_title(f"R2 overall {overall_r2:.3f}")
    ax[1].boxplot(nrmse)
    ax[1].set_title(f"NRMSE overall {overall_rmse:.3f}")
    plt.savefig(os.path.join(save_dir, 'r2_nrmse.png'))
    plt.show()