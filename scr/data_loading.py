import sys
import glob
import os
import re

import numpy as np
import pandas as pd

import geopandas as gpd
from scipy import stats
from pathlib import Path

# Shared logging
import logging
from log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


def reindex_dataframe(df, min_date, max_date):
    df = df.copy()
    # Create complete date range
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")
    codes = df["code"].unique()
    # Create multi-index DataFrame with all combinations
    index = pd.MultiIndex.from_product([date_range, codes], names=["date", "code"])

    # Reindex the DataFrame
    df_complete = df.set_index(["date", "code"]).reindex(index)

    # Add month column back
    df_complete["month"] = df_complete.index.get_level_values("date").month

    # Reset index to get date and code back as columns
    df_complete = df_complete.reset_index()

    return df_complete


def transform_data_gateway_data(df, var_name):
    df = df.copy()
    # rename unamed to date
    df.rename(columns={"Unnamed: 0": "date"}, inplace=True)

    df = df.iloc[4:]

    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    code_dict = {}

    for col in df.columns:
        if col != "date":
            # split by "_"
            new_col = col.split("_")
            if len(new_col) > 1:
                code = int(new_col[0])
                elevation_band = int(new_col[1])
                new_var_name = f"{var_name}_{elevation_band}"
            else:
                code = int(new_col[0])
                elevation_band = None
                new_var_name = var_name

            dates = df["date"]
            values = df[col].astype(float)
            # interpolate the values if gap < 15 days
            values = values.interpolate(limit=15)
            if code not in code_dict:
                code_dict[code] = {"date": dates, new_var_name: values}
            else:
                code_dict[code][new_var_name] = values

    new_df = pd.DataFrame()
    for code, data in code_dict.items():
        code_df = pd.DataFrame(data)
        code_df["code"] = code
        new_df = pd.concat([new_df, code_df], ignore_index=True)

    return new_df


def load_snow_data(path_to_file, var_name):
    """
    Load the snow data from a csv file from the data-gateway
    """
    df = pd.read_csv(path_to_file)

    df = transform_data_gateway_data(df, var_name)

    return df


def time_shift_sla_data(sla_df: pd.DataFrame) -> pd.DataFrame:
    """
    The sla data operates on a decadal basis.
    which means that the data for the 1 of the month is valid until the 10th day,
    the data for the 11th of the month is valid until the 20th day,
    and the data for the 21st of the month is valid until the end of the month.
    we need to adjust the date:
    if the day is the 1st or 11th of the month, we need to shift the date by 9 days
    if the day is the 21st of the month, it should be the last day of the month

    Parameters:
    -----------
    sla_df : pandas DataFrame
        DataFrame containing SLA data with a 'date' column.

    Returns:
    --------
    pandas DataFrame
    """

    # Create a copy of the DataFrame to avoid modifying the original
    df = sla_df.copy()

    # Convert 'date' column to datetime if not already
    df["date"] = pd.to_datetime(df["date"])
    unique_days_of_month = df["date"].dt.day.unique()
    print(f"Unique days of month in SLA data: {unique_days_of_month}")
    # Shift the date based on the day of the month
    df.loc[df["date"].dt.day.isin([1, 11]), "date"] += pd.DateOffset(days=9)
    df.loc[df["date"].dt.day == 21, "date"] = df["date"] + pd.offsets.MonthEnd()

    unique_days_of_month = df["date"].dt.day.unique()
    print(f"Unique days of month after SLA data shift: {unique_days_of_month}")

    return df


def load_data(
    path_discharge,
    path_forcing,
    path_static_data,
    path_to_sca,
    path_to_swe,
    path_to_hs,
    path_to_rof,
    HRU_SWE,
    HRU_HS,
    HRU_ROF,
    path_to_sla=None,
):
    # Load the discharge data
    discharge = pd.read_csv(path_discharge, parse_dates=True)
    min_date = discharge["date"].min()
    max_date = discharge["date"].max()
    discharge["date"] = pd.to_datetime(discharge["date"], format="mixed")

    # Load the forcing data
    forcing = pd.read_csv(path_forcing, parse_dates=True)
    forcing = forcing[["date", "code", "T", "P"]]
    forcing["date"] = pd.to_datetime(forcing["date"])

    hydro_ca = pd.merge(discharge, forcing, on=["date", "code"], how="left")

    # Load the static data
    static = pd.read_csv(path_static_data)
    static.rename(columns={"CODE": "code"}, inplace=True)

    # SCA Data
    try:
        sca_df = pd.read_csv(path_to_sca)
        sca_df["sca"] = sca_df["sca"] * 100

        sca_df["date"] = pd.to_datetime(sca_df["date"])
        sca_df["month"] = sca_df["date"].dt.month
        sca_df["year"] = sca_df["date"].dt.year

        df_pivoted = sca_df.pivot_table(
            index=["date", "CODE", "month"], columns="elevation_band", values="sca"
        ).reset_index()

        # Rename the elevation band columns
        df_pivoted.columns = ["date", "code", "month"] + [
            f"SCA_{int(col)}" for col in df_pivoted.columns[3:]
        ]

        df_pivoted["code"] = df_pivoted["code"].astype(int)

        df_complete_sca = reindex_dataframe(df_pivoted, min_date, max_date)

        hydro_ca = pd.merge(hydro_ca, df_complete_sca, on=["date", "code"], how="left")

    except Exception as e:
        print(f"Error loading SCA data: {str(e)}")

    # SWE DATA
    try:
        swe_df = pd.DataFrame()
        # get all the files in the path_to_swe folder
        all_swe_files = os.listdir(path_to_swe)
        for file in all_swe_files:
            if HRU_SWE in file:
                file_path = os.path.join(path_to_swe, file)
                swe_df_inter = load_snow_data(file_path, "SWE")
                swe_df = pd.concat([swe_df, swe_df_inter], ignore_index=True)

        # remove dublicates based on date and code
        swe_df = swe_df.drop_duplicates(subset=["date", "code"])
        hydro_ca = pd.merge(hydro_ca, swe_df, on=["date", "code"], how="left")
    except Exception as e:
        print(f"Error loading SWE data: {str(e)}")

    # HS DATA
    try:
        hs_df = pd.DataFrame()
        # get all the files in the path_to_hs folder
        all_hs_files = os.listdir(path_to_hs)
        for file in all_hs_files:
            if HRU_HS in file:
                file_path = os.path.join(path_to_hs, file)
                hs_df_inter = load_snow_data(file_path, "HS")
                hs_df = pd.concat([hs_df, hs_df_inter], ignore_index=True)

        # remove dublicates based on date and code
        hs_df = hs_df.drop_duplicates(subset=["date", "code"])

        hydro_ca = pd.merge(hydro_ca, hs_df, on=["date", "code"], how="left")
    except Exception as e:
        print(f"Error loading HS data: {str(e)}")

    # ROF DATA
    try:
        rof_df = pd.DataFrame()
        # get all the files in the path_to_rof folder
        all_rof_files = os.listdir(path_to_rof)
        for file in all_rof_files:
            if HRU_ROF in file:
                file_path = os.path.join(path_to_rof, file)
                rof_df_inter = load_snow_data(file_path, "ROF")
                rof_df = pd.concat([rof_df, rof_df_inter], ignore_index=True)

        # remove dublicates based on date and code
        rof_df = rof_df.drop_duplicates(subset=["date", "code"])
        hydro_ca = pd.merge(hydro_ca, rof_df, on=["date", "code"], how="left")
    except Exception as e:
        print(f"Error loading ROF data: {str(e)}")

    if path_to_sla is not None:
        # Load the SLA data
        hydro_ca["code"] = hydro_ca["code"].astype(int)
        hydro_ca["date"] = pd.to_datetime(hydro_ca["date"], format="%Y-%m-%d")
        static["code"] = static["code"].astype(int)

        try:
            sla_df = pd.read_csv(path_to_sla)
            sla_df["date"] = pd.to_datetime(sla_df["date"], format="%d.%m.%Y")
            sla_df = time_shift_sla_data(sla_df)

            sla_df = sla_df.rename(
                columns={"Code": "code", "gla_fsc": "gla_fsc_total", "fsc": "fsc_basin"}
            )

            # With this code that will work in all pandas versions:
            def safe_convert_to_int(value):
                try:
                    return int(value)
                except (ValueError, TypeError):
                    return np.nan

            # Apply the function to convert codes, preserving NaN for errors
            sla_df["code"] = sla_df["code"].apply(safe_convert_to_int)

            # Then drop rows with NaN codes if needed
            sla_df = sla_df.dropna(subset=["code"])
            # Convert remaining codes to int
            sla_df["code"] = sla_df["code"].astype(int)

            columns_of_interest = [
                "SLA_East",
                "SLA_West",
                "SLA_North",
                "SLA_South",
                "gla_area_below_sl50",
                "gla_fsc_total",
                "gla_fsc_below_sl50",
                "fsc_basin",
            ]

            sla_df = sla_df[["date", "code"] + columns_of_interest]

            # Add after loading sla_df but before merging
            sla_dates = set(sla_df["date"].dt.strftime("%Y-%m-%d"))
            hydro_dates = set(hydro_ca["date"].dt.strftime("%Y-%m-%d"))
            common_dates = sla_dates.intersection(hydro_dates)

            # Add this check
            sla_codes = set(sla_df["code"])
            hydro_codes = set(hydro_ca["code"])
            missing_codes = hydro_codes - sla_codes

            sla_df["date"] = sla_df["date"].dt.strftime("%Y-%m-%d")
            hydro_ca["date"] = hydro_ca["date"].dt.strftime("%Y-%m-%d")
            # retransform the date to be the same format
            sla_df["date"] = pd.to_datetime(sla_df["date"], format="%Y-%m-%d")
            hydro_ca["date"] = pd.to_datetime(hydro_ca["date"], format="%Y-%m-%d")

            sla_df["code"] = sla_df["code"].astype(int)
            hydro_ca["code"] = hydro_ca["code"].astype(int)

            hydro_ca = pd.merge(hydro_ca, sla_df, on=["date", "code"], how="left")

        except Exception as e:
            print(f"Error loading SLA data: {str(e)}")

    return hydro_ca, static


def load_base_learners(path_config_base_learners, static_df):
    base_learner_df = None

    for model_name, path in path_config_base_learners.items():
        model_df = pd.read_csv(path)
        model_df["date"] = pd.to_datetime(model_df["date"])
        model_df["code"] = model_df["code"].astype(int)

        # rename Q_pred to Q_pred_model
        model_df.rename(columns={"Q_pred": f"{model_name}"}, inplace=True)

        if static_df is not None:
            for code in model_df.code.unique():
                area = static_df[static_df["code"] == code]["area_km2"].values[0]
                # transform from m3/s to mm/day
                model_df.loc[model_df["code"] == code, f"{model_name}"] = (
                    model_df.loc[model_df["code"] == code, f"{model_name}"]
                    * 86.4
                    / area
                )

        if base_learner_df is None:
            base_learner_df = model_df
        else:
            # drop Q_obs from predictions
            model_df = model_df[["date", "code", f"{model_name}"]]
            base_learner_df = pd.merge(
                base_learner_df, model_df, on=["date", "code"], how="inner"
            )

    return base_learner_df
