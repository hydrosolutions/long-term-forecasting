import pandas as pd
import numpy as np
import scipy.stats

# Shared logging
import logging
from ..log_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)  # Use __name__ to get module-specific logger


# Calculate slope using linear regression on each rolling window
def rolling_slope(x):
    x_non_nan = x[~np.isnan(x)]
    if len(x_non_nan) < 2:
        return np.nan
    x_values = np.arange(len(x_non_nan))
    return np.polyfit(x_values, x_non_nan, 1)[0]  # Return slope coefficient


def last_value(x):
    """Computes the last value of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    x : nd-array
        Input from which last value is computed

    Returns
    -------
    float
        Last value result
    """
    x_non_nan = x[~np.isnan(x)]
    if len(x_non_nan) == 0:
        return np.nan
    else:
        return x_non_nan[-1]


def mean_difference(signal):
    """Computes the mean difference of the signal.
    For all non NaN values, it computes the mean of the differences between consecutive values.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which mean difference is computed

    Returns
    -------
    float
        Mean difference result
    """
    non_nan_signal = signal[~np.isnan(signal)]
    if len(non_nan_signal) < 2:
        return np.nan
    differences = np.diff(non_nan_signal)
    if len(differences) == 0:
        return np.nan
    return np.mean(differences)


def median_abs_deviation(signal):
    """Computes median absolute deviation of the signal.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which median absolute deviation is computed

    Returns
    -------
    float
        Mean absolute deviation result
    """
    return scipy.stats.median_abs_deviation(signal, scale=1)


def pk_pk_distance(signal):
    """Computes the peak to peak distance.

    Feature computational cost: 1

    Parameters
    ----------
    signal : nd-array
        Input from which peak to peak is computed

    Returns
    -------
    float
        peak to peak distance
    """
    return np.abs(np.nanmax(signal) - np.nanmin(signal))


def time_distance_from_peak(signal):
    loc_peak = np.nanargmax(signal)
    return len(signal) - loc_peak


def time_of_occurrence_last_value(signal):
    """
    Computes how many time steps ago the last non-NaN value occurred,
    measured from the end of the signal.

    Parameters
    ----------
    signal : np.ndarray
        Time series array with possible NaNs.

    Returns
    -------
    int or float
        Number of steps since the last valid value.
        Returns NaN if no valid values are present.
    """
    x_non_nan = signal[~np.isnan(signal)]
    if len(x_non_nan) == 0:
        return np.nan

    last_non_nan_index = np.where(~np.isnan(signal))[0][-1]
    return len(signal) - last_non_nan_index


def increasing_in_projection(signal):
    """
    Computes if the  signal is expected to increase in the next days.
     t- window // 2 is used as the reference, and t + window // 2 is used as the projection point.
     The feature then has to be shifted by the window // 2 to align with the projection point.

     Formula: projection / observed = x
     if x > 1.05 -> 1
     if x < 0.95 -> -1
     else -> 0
    """
    if len(signal) < 2:
        return np.nan

    window = len(signal) // 2
    window = int(window)  # Ensure window is an integer

    if window == 0:
        return np.nan

    observed = signal[: window - 1]
    projection = signal[window:]

    if np.sum(observed) == 0:
        return np.nan

    ratio = np.mean(projection) / np.mean(observed)

    if ratio > 1.05:
        return 1
    elif ratio < 0.95:
        return -1
    else:
        return 0


class StreamflowFeatureExtractor:
    """
    Feature extraction pipeline for streamflow prediction.
    Creates rolling window features and target variable for prediction.
    """

    def __init__(self, feature_configs, prediction_horizon=30, offset=None):
        """
        Initialize feature extractor.

        Parameters:
        -----------
        prediction_horizon : int, default=30
            Number of days ahead to predict average streamflow
        feature_configs : dict
            Configuration for feature creation
        offset : int, default=None
            Number of days to offset the target variable (default: prediction_horizon)
        """
        self.prediction_horizon = prediction_horizon

        if offset is None:
            self.offset = self.prediction_horizon
        else:
            self.offset = offset

        assert self.offset >= self.prediction_horizon, (
            "Offset must be greater or equal than prediction horizon"
        )

        # Define feature configurations
        self.feature_configs = feature_configs

    def _create_rolling_features(self, df, column, config):
        """
        Create rolling window features for a single column, handling multiple basins.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with 'code' column for basin identification
        column : str
            Column name to create features from
        config : dict
            Configuration for feature creation

        Returns:
        --------
        pandas.DataFrame
            DataFrame with rolling window features
        """
        features = pd.DataFrame(index=df.index)

        # Group by basin code and create features
        for code in df["code"].unique():
            basin_data = df[df["code"] == code][column]

            for window in config["windows"]:
                feature_name = f"{column}_roll_{config['operation']}_{window}"

                min_periods = min(int(window * 0.75), 15)
                # Calculate rolling feature for this basin
                if config["operation"] == "mean":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).mean()
                elif config["operation"] == "sum":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).sum()
                elif config["operation"] == "min":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).min()
                elif config["operation"] == "max":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).max()
                elif config["operation"] == "std":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).std()
                elif config["operation"] == "slope":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(rolling_slope, raw=True)
                elif config["operation"] == "peak_to_peak":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(pk_pk_distance, raw=True)
                elif config["operation"] == "median_abs_deviation":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(median_abs_deviation, raw=True)
                elif config["operation"] == "time_distance_from_peak":
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(time_distance_from_peak, raw=True)
                elif config["operation"] == "last_value":
                    min_periods = 1
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(last_value, raw=True)
                elif config["operation"] == "time_of_occurrence_last_value":
                    min_periods = 1
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(time_of_occurrence_last_value, raw=True)
                elif config["operation"] == "mean_difference":
                    min_periods = 2
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(mean_difference, raw=True)
                elif config["operation"] == "increasing_in_projection":
                    min_periods = 2
                    basin_feature = basin_data.rolling(
                        window=window, min_periods=min_periods
                    ).apply(increasing_in_projection, raw=True)
                else:
                    raise ValueError(f"Unsupported operation: {config['operation']}")

                # First add the base feature (no lag)
                features.loc[basin_data.index, feature_name] = basin_feature

                # Handle lags, which could be a nested dictionary
                if config["lags"]:
                    for window_key, lag_list in config["lags"].items():
                        # Only apply lags for the matching window size
                        window_key = int(window_key)
                        window = int(window)
                        if window == window_key:
                            for lag in lag_list:
                                # Create both positive (past) and negative (future) lags
                                lagged_feature = basin_feature.shift(lag)
                                lag_direction = "lag" if lag > 0 else "lead"
                                lag_value = abs(lag)
                                features.loc[
                                    basin_data.index,
                                    f"{feature_name}_{lag_direction}_{lag_value}",
                                ] = lagged_feature

        return features

    def create_target(self, df, column="discharge"):
        """
        Create target variable: average discharge for next N days, by basin.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame with 'code' column
        column : str, default='discharge'
            Column to create target from

        Returns:
        --------
        pandas.Series
            Target variable
        """
        min_periods = min(int(self.prediction_horizon * 0.75), 15)
        target = pd.Series(index=df.index, dtype=float)

        # Calculate target for each basin separately
        for code in df["code"].unique():
            basin_data = df[df["code"] == code][column]

            # Calculate future average
            future_avg = (
                basin_data.rolling(
                    window=self.prediction_horizon, min_periods=min_periods
                )
                .mean()
                .shift(-self.offset)
            )

            # Assign values back to the correct rows
            target.loc[basin_data.index] = future_avg

        return target

    def _create_time_features(self, df):
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week

        df["day_of_year_sin"] = np.sin(2 * np.pi * df["date"].dt.dayofyear / 365)
        df["day_of_year_cos"] = np.cos(2 * np.pi * df["date"].dt.dayofyear / 365)

        df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
        df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df[
            [
                "week_sin",
                "week_cos",
                "month_sin",
                "month_cos",
                "day_of_year_sin",
                "day_of_year_cos",
            ]
        ]

    def create_all_features(self, df):
        """
        Create all features based on configuration, handling multiple basins.

        Parameters:
        -----------
        df : pandas.DataFrame
            Input DataFrame containing all required columns and 'code' column

        Returns:
        --------
        pandas.DataFrame
            DataFrame with date index, code, features, and target columns
        """
        # sort the df by date
        df = df.sort_values(by="date")

        # Create features for each variable
        feature_dfs = []
        for column, config in self.feature_configs.items():
            logger.info(f"Creating features for column: {column}")
            features_with_column = [col for col in df.columns if column in col]
            logger.info(f"Features with column '{column}': {features_with_column}")
            logger.info(f"Configuration for column '{column}': {config}")
            for c in config:
                for col in features_with_column:
                    features = self._create_rolling_features(df, col, c)
                    feature_dfs.append(features)

        # Combine all features
        X = pd.concat(feature_dfs, axis=1)

        # Create target
        y = self.create_target(df)

        # Create time features
        time_features = self._create_time_features(df)

        # Combine everything into a single DataFrame in one operation
        # Start with base information
        base_data = {"date": df["date"], "code": df["code"], "target": y}

        # Add all feature columns from X
        for col in X.columns:
            base_data[col] = X[col]

        # Add all time feature columns
        for col in time_features.columns:
            base_data[col] = time_features[col]

        # Create the final DataFrame in one go
        final_df = pd.DataFrame(base_data, index=df.index)

        return final_df

    def get_feature_names(self):
        """
        Get list of all feature names that will be created.

        Returns:
        --------
        list
            List of feature names
        """
        feature_names = []
        for column, config in self.feature_configs.items():
            for window in config["windows"]:
                feature_names.append(f"{column}_roll_{config['operation']}_{window}")
        return feature_names
