import os
import sys
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Optional
import datetime

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import forecast models
from monthly_forecasting.forecast_models.LINEAR_REGRESSION import LinearRegressionModel
from monthly_forecasting.forecast_models.SciRegressor import SciRegressor
from monthly_forecasting.scr import data_loading as dl


# Setup logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


# import environment variables saved in .env file
from dotenv import load_dotenv

load_dotenv()

PATH_TO_DISCHARGE = os.getenv("path_discharge")
PATH_TO_FORCING_ERA5 = os.getenv("path_forcing_era5")
PATH_TO_FORCING_OPERATIONAL = os.getenv("path_forcing_operational")
PATH_SWE_00003 = os.getenv("path_SWE_00003")
PATH_SWE_500m = os.getenv("path_SWE_500m")
PATH_ROF_00003 = os.getenv("path_ROF_00003")
PATH_ROF_500m = os.getenv("path_ROF_500m")
PATH_TO_SHP = os.getenv("path_to_shp")
PATH_TO_STATIC = os.getenv("path_to_static")

MODELS_OPERATIONAL = {
    "BaseCase": [
        ("LR", "LR_Q_T_P"),
        ("SciRegressor", "ShortTerm_Features"),
        ("SciRegressor", "NormBased"),
    ],
    "SnowMapper_Based": [
        ("LR", "LR_Q_dSWEdt_T_P"),
        ("LR", "LR_Q_SWE_T"),
        ("LR", "LR_Q_T_P_SWE"),
        ("LR", "LR_Q_SWE"),
        ("SciRegressor", "NormBased"),
        ("SciRegressor", "ShortTermLR"),
    ],
}

MODELS_DIR = "../monthly_forecasting_models"


### Load the Data Function
def load_discharge():
    df_discharge = pd.read_csv(PATH_TO_DISCHARGE, parse_dates=["date"])
    df_discharge["date"] = pd.to_datetime(df_discharge["date"], format="%Y-%m-%d")
    df_discharge["code"] = df_discharge["code"].astype(int)
    return df_discharge


def load_forcing():
    operational_T = pd.read_csv(
        os.path.join(PATH_TO_FORCING_OPERATIONAL, "00003_T_control_member.csv")
    )
    operational_P = pd.read_csv(
        os.path.join(PATH_TO_FORCING_OPERATIONAL, "00003_P_control_member.csv")
    )

    operational_T["date"] = pd.to_datetime(operational_T["date"], format="%Y-%m-%d")
    operational_P["date"] = pd.to_datetime(operational_P["date"], format="%Y-%m-%d")
    operational_T["code"] = operational_T["code"].astype(int)
    operational_P["code"] = operational_P["code"].astype(int)

    operational_data = pd.merge(operational_T, operational_P, on=["date", "code"])
    logger.info("Operational forcing data loaded successfully.")

    hindcast_T = pd.read_csv(
        os.path.join(PATH_TO_FORCING_ERA5, "00003_T_reanalysis.csv")
    )
    hindcast_P = pd.read_csv(
        os.path.join(PATH_TO_FORCING_ERA5, "00003_P_reanalysis.csv")
    )
    hindcast_T["date"] = pd.to_datetime(hindcast_T["date"], format="%Y-%m-%d")
    hindcast_P["date"] = pd.to_datetime(hindcast_P["date"], format="%Y-%m-%d")
    hindcast_T["code"] = hindcast_T["code"].astype(int)
    hindcast_P["code"] = hindcast_P["code"].astype(int)
    hindcast_data = pd.merge(hindcast_T, hindcast_P, on=["date", "code"])

    logger.info("Hindcast forcing data loaded successfully.")

    operationl_data = pd.concat([hindcast_data, operational_data], ignore_index=True)
    operationl_data = operationl_data.sort_values(by=["date", "code"]).reset_index(
        drop=True
    )

    # drop duplicates based on date and code - keep last occurrence
    operational_data = operationl_data.drop_duplicates(
        subset=["date", "code"], keep="last"
    )

    return operational_data


def load_snowmapper():
    swe_00003 = pd.read_csv(PATH_SWE_00003, parse_dates=["date"])
    swe_00003["date"] = pd.to_datetime(swe_00003["date"], format="%Y-%m-%d")
    swe_00003["code"] = swe_00003["code"].astype(int)

    swe_500m = pd.read_csv(PATH_SWE_500m, parse_dates=["date"])
    swe_500m["date"] = pd.to_datetime(swe_500m["date"], format="%Y-%m-%d")
    swe_500m["code"] = swe_500m["code"].astype(int)

    rof_00003 = pd.read_csv(PATH_ROF_00003, parse_dates=["date"])
    rof_00003["date"] = pd.to_datetime(rof_00003["date"], format="%Y-%m-%d")
    rof_00003["code"] = rof_00003["code"].astype(int)

    rof_500m = pd.read_csv(PATH_ROF_500m, parse_dates=["date"])
    rof_500m["date"] = pd.to_datetime(rof_500m["date"], format="%Y-%m-%d")
    rof_500m["code"] = rof_500m["code"].astype(int)

    return swe_00003, swe_500m, rof_00003, rof_500m


def load_static_data():
    df_static = pd.read_csv(PATH_TO_STATIC, parse_dates=["date"])

    if "CODE" in df_static.columns:
        df_static.rename(columns={"CODE": "code"}, inplace=True)
    df_static["code"] = df_static["code"].astype(int)

    return df_static


def create_data_frame(config: Dict[str, Any]) -> pd.DataFrame:
    data = load_discharge()
    forcing = load_forcing()
    swe_00003, swe_500m, rof_00003, rof_500m = load_snowmapper()
    static_data = load_static_data()

    data = data.merge(forcing, on=["date", "code"], how="left")

    SWE_HRU = config.get("HRU_SWE", None)

    if SWE_HRU == "HRU_00003":
        data = data.merge(swe_00003, on=["date", "code"], how="left")

    elif SWE_HRU == "KGZ500m":
        data = data.merge(swe_500m, on=["date", "code"], how="left")

    else:
        logger.info("No SWE data merged.")

    ROF_HRU = config.get("HRU_ROF", None)
    if ROF_HRU == "HRU_00003":
        data = data.merge(rof_00003, on=["date", "code"], how="left")
    elif ROF_HRU == "KGZ500m":
        data = data.merge(rof_500m, on=["date", "code"], how="left")
    else:
        logger.info("No ROF data merged.")

    return data, static_data


#### Load the Model Function


### Pedict Operational


### Check status


### check quality of forecast


def run_operational():
    pass


if __name__ == "__main__":
    run_operational()
