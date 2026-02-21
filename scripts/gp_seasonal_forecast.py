"""
Gaussian Process Seasonal Discharge Forecasting Script

Standalone script for GP-based seasonal (April-September) discharge forecasting
for Kyrgyzstan with four issue dates, implementing three GP variants plus
linear regression baseline.

Run: uv run python scripts/gp_seasonal_forecast.py
"""

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct, WhiteKernel
from sklearn.linear_model import BayesianRidge, Lasso, LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class IssueDate:
    """Configuration for a forecast issue date."""

    month: int
    day: int
    name: str

    def __str__(self) -> str:
        return self.name


ISSUE_DATES = [
    IssueDate(1, 25, "Jan25"),
    IssueDate(2, 25, "Feb25"),
    IssueDate(3, 25, "Mar25"),
    IssueDate(4, 25, "Apr25"),
]


@dataclass
class LOOCVResult:
    """Result from LOOCV for a single test year/basin."""

    code: int
    test_year: int
    y_true: float
    y_pred: float
    y_std: float | None = None

    def in_interval(self, coverage: float) -> bool | None:
        """Check if true value is within the specified coverage interval."""
        if self.y_std is None or self.y_std <= 0:
            return None
        # For Gaussian: 50% interval = ±0.674σ, 90% interval = ±1.645σ
        z = stats.norm.ppf((1 + coverage) / 2)
        lower = self.y_pred - z * self.y_std
        upper = self.y_pred + z * self.y_std
        return lower <= self.y_true <= upper


@dataclass
class BasinScaler:
    """Per-basin scaler for features and target."""

    code: int
    feature_scaler: StandardScaler
    target_mean: float
    target_std: float


@dataclass
class ModelResults:
    """Results for a model across all issue dates."""

    model_name: str
    results: dict[str, list[LOOCVResult]] = field(default_factory=dict)
    lengthscales: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)


# =============================================================================
# Model Protocol and Implementations
# =============================================================================


class ForecastModel(Protocol):
    """Protocol for forecast models."""

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None: ...

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]: ...


class LocalGP:
    """Independent GP per basin with linear kernel."""

    name = "LocalGP"

    def __init__(self, n_restarts: int = 2):
        self.n_restarts = n_restarts
        self.model: GaussianProcessRegressor | None = None

    def _create_kernel(self):
        # Linear kernel: k(x, y) = sigma_0^2 + x · y
        return ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(
            sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        kernel = self._create_kernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not fitted")
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std

    def get_lengthscales(self) -> np.ndarray | None:
        # Linear kernel has no lengthscales
        return None


class GlobalGP:
    """Global GP trained on all basins pooled together with linear kernel."""

    name = "GlobalGP"

    def __init__(self, n_restarts: int = 2):
        self.n_restarts = n_restarts
        self.model: GaussianProcessRegressor | None = None

    def _create_kernel(self):
        # Linear kernel: k(x, y) = sigma_0^2 + x · y
        return ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(
            sigma_0=1.0, sigma_0_bounds=(1e-3, 1e3)
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        kernel = self._create_kernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not fitted")
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std

    def get_lengthscales(self) -> np.ndarray | None:
        # Linear kernel has no lengthscales
        return None


class EmpiricalBayesGP:
    """Per-basin GP with global warm start from pooled data (linear kernel)."""

    name = "EmpiricalBayesGP"

    def __init__(self, global_sigma_0: float | None = None, n_restarts: int = 1):
        self.global_sigma_0 = global_sigma_0
        self.n_restarts = n_restarts
        self.model: GaussianProcessRegressor | None = None

    def _create_kernel(self):
        # Linear kernel with optional warm-start sigma_0
        sigma_0 = self.global_sigma_0 if self.global_sigma_0 is not None else 1.0
        return ConstantKernel(1.0, (1e-3, 1e3)) * DotProduct(
            sigma_0=sigma_0, sigma_0_bounds=(1e-3, 1e3)
        ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        kernel = self._create_kernel()
        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts,
            normalize_y=True,
            random_state=42,
        )
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not fitted")
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std

    def get_lengthscales(self) -> np.ndarray | None:
        # Linear kernel has no lengthscales
        return None


class RidgeBaseline:
    """Ridge regression baseline model."""

    name = "Ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Ridge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = Ridge(alpha=self.alpha)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X), None


class LinearRegressionBaseline:
    """Standard linear regression baseline model (OLS)."""

    name = "OLS"

    def __init__(self):
        self.model: LinearRegression | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = LinearRegression()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X), None


class LassoBaseline:
    """Lasso regression baseline model."""

    name = "Lasso"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model: Lasso | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = Lasso(alpha=self.alpha, max_iter=10000)
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, None]:
        if self.model is None:
            raise ValueError("Model not fitted")
        return self.model.predict(X), None


class BayesianRidgeModel:
    """Bayesian Ridge Regression - provides uncertainty estimates."""

    name = "BayesianRidge"

    def __init__(self):
        self.model: BayesianRidge | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model = BayesianRidge()
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.model is None:
            raise ValueError("Model not fitted")
        y_pred, y_std = self.model.predict(X, return_std=True)
        return y_pred, y_std


# =============================================================================
# Data Loading Functions
# =============================================================================


def load_discharge(path: str) -> pd.DataFrame:
    """Load discharge data."""
    logger.info(f"Loading discharge from {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["code"] = df["code"].astype(int)
    logger.info(
        f"  Shape: {df.shape}, date range: {df['date'].min()} to {df['date'].max()}"
    )
    logger.info(f"  Basins: {df['code'].nunique()}")
    return df


def load_forcing(path_p: str, path_t: str) -> pd.DataFrame:
    """Load and merge P and T forcing data."""
    logger.info(f"Loading forcing data from {path_p} and {path_t}")
    df_p = pd.read_csv(path_p)
    df_t = pd.read_csv(path_t)

    df_p["date"] = pd.to_datetime(df_p["date"], format="mixed")
    df_t["date"] = pd.to_datetime(df_t["date"], format="mixed")

    df = df_p.merge(df_t[["date", "code", "T"]], on=["date", "code"], how="outer")
    df["code"] = df["code"].astype(int)
    logger.info(
        f"  Shape: {df.shape}, date range: {df['date'].min()} to {df['date'].max()}"
    )
    return df


def load_swe(path: str) -> pd.DataFrame:
    """Load SWE data (already aggregated per basin)."""
    logger.info(f"Loading SWE from {path}")
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df["code"] = df["code"].astype(int)

    logger.info(f"  Shape: {df.shape}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Basins: {df['code'].nunique()}")
    return df[["date", "code", "SWE"]]


def load_static(path: str) -> pd.DataFrame:
    """Load static basin attributes."""
    logger.info(f"Loading static data from {path}")
    df = pd.read_csv(path)
    if "CODE" in df.columns:
        df = df.rename(columns={"CODE": "code"})
    df["code"] = df["code"].astype(int)
    logger.info(f"  Shape: {df.shape}, basins: {len(df)}")
    return df


def merge_data(
    discharge: pd.DataFrame,
    forcing: pd.DataFrame,
    swe: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources."""
    logger.info("Merging data sources...")

    # Merge discharge and forcing
    df = discharge.merge(
        forcing[["date", "code", "P", "T"]], on=["date", "code"], how="inner"
    )

    # Merge SWE
    df = df.merge(swe, on=["date", "code"], how="left")

    # Forward-fill SWE within each basin (SWE data may have gaps)
    df = df.sort_values(["code", "date"])
    df["SWE"] = df.groupby("code")["SWE"].ffill()

    logger.info(f"  Merged shape: {df.shape}")
    logger.info(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    logger.info(f"  Basins: {df['code'].nunique()}")
    return df


# =============================================================================
# Feature Extraction
# =============================================================================


def compute_seasonal_target(
    df: pd.DataFrame, year: int, target_start_month: int = 4, target_end_month: int = 9
) -> pd.DataFrame:
    """Compute average seasonal discharge (Apr-Sep) for each basin in a given year."""
    start_date = pd.Timestamp(year, target_start_month, 1)
    # End of September
    end_date = pd.Timestamp(year, target_end_month, 30)

    mask = (df["date"] >= start_date) & (df["date"] <= end_date)
    seasonal = df[mask].groupby("code")["discharge"].mean().reset_index()
    seasonal.columns = ["code", "Q_seasonal"]
    seasonal["year"] = year
    return seasonal


def extract_features_for_issue_date(
    df: pd.DataFrame, issue_date: IssueDate, year: int
) -> pd.DataFrame:
    """Extract features for a given issue date and year."""
    issue = pd.Timestamp(year, issue_date.month, issue_date.day)

    # SWE: past 10 days (up to and including issue date)
    swe_start = issue - pd.Timedelta(days=10)
    swe_mask = (df["date"] > swe_start) & (df["date"] <= issue)

    # P: past 90 days (longer lookback for cumulative precipitation)
    p_start = issue - pd.Timedelta(days=90)
    p_mask = (df["date"] > p_start) & (df["date"] <= issue)

    # T: past 30 days
    t_start = issue - pd.Timedelta(days=30)
    t_mask = (df["date"] > t_start) & (df["date"] <= issue)

    # Discharge: past 30 days
    q_start = issue - pd.Timedelta(days=30)
    q_mask = (df["date"] > q_start) & (df["date"] <= issue)

    # Aggregate features per basin
    swe_features = df[swe_mask].groupby("code")["SWE"].mean().reset_index()
    swe_features.columns = ["code", "SWE_10d_mean"]

    p_features = df[p_mask].groupby("code")["P"].sum().reset_index()
    p_features.columns = ["code", "P_90d_sum"]

    t_features = df[t_mask].groupby("code")["T"].mean().reset_index()
    t_features.columns = ["code", "T_30d_mean"]

    q_features = df[q_mask].groupby("code")["discharge"].mean().reset_index()
    q_features.columns = ["code", "Q_30d_mean"]

    # Merge features
    features = swe_features.merge(p_features, on="code", how="outer")
    features = features.merge(t_features, on="code", how="outer")
    features = features.merge(q_features, on="code", how="outer")
    features["year"] = year
    features["issue_date"] = str(issue_date)

    return features


def prepare_dataset(
    df: pd.DataFrame, issue_date: IssueDate, years: list[int]
) -> pd.DataFrame:
    """Prepare full dataset with features and targets for all years."""
    all_features = []
    all_targets = []

    for year in years:
        features = extract_features_for_issue_date(df, issue_date, year)
        targets = compute_seasonal_target(df, year)
        all_features.append(features)
        all_targets.append(targets)

    features_df = pd.concat(all_features, ignore_index=True)
    targets_df = pd.concat(all_targets, ignore_index=True)

    # Merge features and targets
    dataset = features_df.merge(targets_df, on=["code", "year"], how="inner")

    # Drop rows with missing values
    dataset = dataset.dropna()

    logger.info(
        f"  Dataset for {issue_date}: {len(dataset)} samples, {dataset['code'].nunique()} basins"
    )

    return dataset


# =============================================================================
# LOOCV Engine
# =============================================================================


def run_loocv_local_gp(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with LocalGP (independent GP per basin). No log transform for linear kernel."""
    results = []
    years = sorted(dataset["year"].unique())
    codes = sorted(dataset["code"].unique())

    for test_year in years:
        train_mask = dataset["year"] != test_year
        test_mask = dataset["year"] == test_year

        for code in codes:
            basin_train = dataset[train_mask & (dataset["code"] == code)]
            basin_test = dataset[test_mask & (dataset["code"] == code)]

            if len(basin_train) < 3 or len(basin_test) == 0:
                continue

            X_train = basin_train[feature_cols].values
            y_train = basin_train["Q_seasonal"].values  # No log transform
            X_test = basin_test[feature_cols].values
            y_test = basin_test["Q_seasonal"].values

            # Per-basin normalization (fit on train only)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit and predict
            model = LocalGP(n_restarts=2)
            try:
                model.fit(X_train_scaled, y_train)
                y_pred, y_std = model.predict(X_test_scaled)

                # Ensure non-negative predictions
                y_pred = np.maximum(0, y_pred)

                for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
                    results.append(
                        LOOCVResult(
                            code=code,
                            test_year=test_year,
                            y_true=float(yt),
                            y_pred=float(yp),
                            y_std=float(y_std[i]) if y_std is not None else None,
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"LocalGP failed for basin {code}, year {test_year}: {e}"
                )

    return results, {}


def run_loocv_global_gp(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with GlobalGP (all basins pooled). Note: O(n³) complexity - slow!"""
    results = []
    years = sorted(dataset["year"].unique())

    for test_year in years:
        train_mask = dataset["year"] != test_year
        test_mask = dataset["year"] == test_year

        train_data = dataset[train_mask]
        test_data = dataset[test_mask]

        if len(train_data) < 10:
            continue

        X_train = train_data[feature_cols].values
        y_train_raw = train_data["Q_seasonal"].values
        X_test = test_data[feature_cols].values
        y_test_raw = test_data["Q_seasonal"].values
        test_codes = test_data["code"].values

        # Log-transform target
        y_train = np.log(y_train_raw + 1e-6)

        # Global normalization (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit global model
        model = GlobalGP(n_restarts=2)
        try:
            model.fit(X_train_scaled, y_train)
            y_pred_log, y_std_log = model.predict(X_test_scaled)

            # Inverse transform
            y_pred = np.exp(y_pred_log) - 1e-6
            y_pred = np.maximum(0, y_pred)

            for i, (code, yt, yp) in enumerate(zip(test_codes, y_test_raw, y_pred)):
                results.append(
                    LOOCVResult(
                        code=int(code),
                        test_year=test_year,
                        y_true=float(yt),
                        y_pred=float(yp),
                        y_std=float(y_std_log[i]) if y_std_log is not None else None,
                    )
                )
        except Exception as e:
            logger.warning(f"GlobalGP failed for year {test_year}: {e}")

    return results, {}


def run_loocv_empirical_bayes_gp(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with EmpiricalBayesGP (per-basin with global warm start)."""
    results = []
    years = sorted(dataset["year"].unique())
    codes = sorted(dataset["code"].unique())

    for test_year in years:
        train_mask = dataset["year"] != test_year
        test_mask = dataset["year"] == test_year

        train_all = dataset[train_mask]

        if len(train_all) < 10:
            continue

        # Step 1: Fit global GP on all training data to get sigma_0
        X_train_all = train_all[feature_cols].values
        y_train_all = np.log(train_all["Q_seasonal"].values + 1e-6)

        global_scaler = StandardScaler()
        X_train_all_scaled = global_scaler.fit_transform(X_train_all)

        global_model = GlobalGP(n_restarts=1)
        global_sigma_0 = None
        try:
            global_model.fit(X_train_all_scaled, y_train_all)
            # Extract sigma_0 from fitted kernel
            if global_model.model is not None:
                kernel = global_model.model.kernel_
                for k in [kernel, getattr(kernel, "k1", None)]:
                    if (
                        k is not None
                        and hasattr(k, "k2")
                        and isinstance(k.k2, DotProduct)
                    ):
                        global_sigma_0 = k.k2.sigma_0
                        break
        except Exception as e:
            logger.warning(f"Global prior fitting failed for year {test_year}: {e}")

        # Step 2: Fit per-basin models with warm start
        for code in codes:
            basin_train = dataset[train_mask & (dataset["code"] == code)]
            basin_test = dataset[test_mask & (dataset["code"] == code)]

            if len(basin_train) < 3 or len(basin_test) == 0:
                continue

            X_train = basin_train[feature_cols].values
            y_train_raw = basin_train["Q_seasonal"].values
            X_test = basin_test[feature_cols].values
            y_test_raw = basin_test["Q_seasonal"].values

            # Log-transform target
            y_train = np.log(y_train_raw + 1e-6)

            # Per-basin normalization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Fit with warm start
            model = EmpiricalBayesGP(global_sigma_0=global_sigma_0, n_restarts=1)
            try:
                model.fit(X_train_scaled, y_train)
                y_pred_log, y_std_log = model.predict(X_test_scaled)

                # Inverse transform
                y_pred = np.exp(y_pred_log) - 1e-6
                y_pred = np.maximum(0, y_pred)

                for i, (yt, yp) in enumerate(zip(y_test_raw, y_pred)):
                    results.append(
                        LOOCVResult(
                            code=code,
                            test_year=test_year,
                            y_true=float(yt),
                            y_pred=float(yp),
                            y_std=float(y_std_log[i])
                            if y_std_log is not None
                            else None,
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"EmpiricalBayesGP failed for basin {code}, year {test_year}: {e}"
                )

    return results, {}


def run_loocv_ridge(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with Ridge regression baseline."""
    return _run_loocv_linear_model(
        dataset, feature_cols, RidgeBaseline(alpha=1.0), "Ridge"
    )


def run_loocv_bayesian_ridge(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with Bayesian Ridge (provides uncertainty)."""
    results = []
    years = sorted(dataset["year"].unique())
    codes = sorted(dataset["code"].unique())

    for test_year in years:
        train_mask = dataset["year"] != test_year
        test_mask = dataset["year"] == test_year

        for code in codes:
            basin_train = dataset[train_mask & (dataset["code"] == code)]
            basin_test = dataset[test_mask & (dataset["code"] == code)]

            if len(basin_train) < 3 or len(basin_test) == 0:
                continue

            X_train = basin_train[feature_cols].values
            y_train = basin_train["Q_seasonal"].values
            X_test = basin_test[feature_cols].values
            y_test = basin_test["Q_seasonal"].values

            # Per-basin normalization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            model = BayesianRidgeModel()
            try:
                model.fit(X_train_scaled, y_train)
                y_pred, y_std = model.predict(X_test_scaled)

                # Ensure non-negative
                y_pred = np.maximum(0, y_pred)

                for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
                    results.append(
                        LOOCVResult(
                            code=code,
                            test_year=test_year,
                            y_true=float(yt),
                            y_pred=float(yp),
                            y_std=float(y_std[i]) if y_std is not None else None,
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"BayesianRidge failed for basin {code}, year {test_year}: {e}"
                )

    return results, {}


def run_loocv_ols(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with OLS (standard linear regression)."""
    return _run_loocv_linear_model(
        dataset, feature_cols, LinearRegressionBaseline(), "OLS"
    )


def run_loocv_lasso(
    dataset: pd.DataFrame, feature_cols: list[str]
) -> tuple[list[LOOCVResult], dict]:
    """Run LOOCV with Lasso regression."""
    return _run_loocv_linear_model(
        dataset, feature_cols, LassoBaseline(alpha=0.1), "Lasso"
    )


def _run_loocv_linear_model(
    dataset: pd.DataFrame, feature_cols: list[str], model_template, model_name: str
) -> tuple[list[LOOCVResult], dict]:
    """Generic LOOCV for linear models."""
    results = []
    years = sorted(dataset["year"].unique())
    codes = sorted(dataset["code"].unique())

    for test_year in years:
        train_mask = dataset["year"] != test_year
        test_mask = dataset["year"] == test_year

        for code in codes:
            basin_train = dataset[train_mask & (dataset["code"] == code)]
            basin_test = dataset[test_mask & (dataset["code"] == code)]

            if len(basin_train) < 3 or len(basin_test) == 0:
                continue

            X_train = basin_train[feature_cols].values
            y_train = basin_train["Q_seasonal"].values
            X_test = basin_test[feature_cols].values
            y_test = basin_test["Q_seasonal"].values

            # Per-basin normalization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Create fresh model instance
            if isinstance(model_template, RidgeBaseline):
                model = RidgeBaseline(alpha=model_template.alpha)
            elif isinstance(model_template, LassoBaseline):
                model = LassoBaseline(alpha=model_template.alpha)
            else:
                model = LinearRegressionBaseline()

            try:
                model.fit(X_train_scaled, y_train)
                y_pred, _ = model.predict(X_test_scaled)

                # Ensure non-negative
                y_pred = np.maximum(0, y_pred)

                for i, (yt, yp) in enumerate(zip(y_test, y_pred)):
                    results.append(
                        LOOCVResult(
                            code=code,
                            test_year=test_year,
                            y_true=float(yt),
                            y_pred=float(yp),
                            y_std=None,
                        )
                    )
            except Exception as e:
                logger.warning(
                    f"{model_name} failed for basin {code}, year {test_year}: {e}"
                )

    return results, {}


# =============================================================================
# Evaluation Metrics
# =============================================================================


def compute_coverage(results: list[LOOCVResult], coverage: float) -> float | None:
    """Compute empirical coverage of prediction intervals.

    Returns the fraction of true values falling within the specified coverage interval.
    For a well-calibrated GP, this should match the nominal coverage.
    """
    in_interval = [r.in_interval(coverage) for r in results]
    valid = [x for x in in_interval if x is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)


def compute_r2_per_basin(results: list[LOOCVResult]) -> dict[int, float]:
    """Compute R² score per basin from LOOCV results."""
    # Group by basin
    basin_results: dict[int, tuple[list[float], list[float]]] = {}
    for r in results:
        if r.code not in basin_results:
            basin_results[r.code] = ([], [])
        basin_results[r.code][0].append(r.y_true)
        basin_results[r.code][1].append(r.y_pred)

    # Compute R² per basin
    r2_scores = {}
    for code, (y_true, y_pred) in basin_results.items():
        y_true_arr = np.array(y_true)
        y_pred_arr = np.array(y_pred)

        if len(y_true_arr) < 2:
            continue

        ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
        ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)

        if ss_tot == 0:
            r2_scores[code] = 0.0
        else:
            r2_scores[code] = 1 - ss_res / ss_tot

    return r2_scores


def results_to_dataframe(
    all_results: dict[str, dict[str, list[LOOCVResult]]],
) -> pd.DataFrame:
    """Convert all results to a DataFrame with per-basin R² scores."""
    rows = []
    for model_name, issue_results in all_results.items():
        for issue_name, results in issue_results.items():
            r2_scores = compute_r2_per_basin(results)
            for code, r2 in r2_scores.items():
                rows.append(
                    {
                        "model": model_name,
                        "issue_date": issue_name,
                        "code": code,
                        "r2": r2,
                    }
                )
    return pd.DataFrame(rows)


# =============================================================================
# Visualization
# =============================================================================


def plot_r2_comparison(
    metrics_df: pd.DataFrame, issue_date: str, output_dir: Path
) -> None:
    """Create boxplot/violin of R² distribution across models for an issue date."""
    fig, ax = plt.subplots(figsize=(10, 6))

    subset = metrics_df[metrics_df["issue_date"] == issue_date]

    sns.violinplot(data=subset, x="model", y="r2", ax=ax, inner="box")
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.5)
    ax.set_title(f"R² Distribution - Issue Date: {issue_date}")
    ax.set_xlabel("Model")
    ax.set_ylabel("R² Score")
    ax.set_ylim(-1, 1)

    plt.tight_layout()
    plt.savefig(output_dir / f"r2_comparison_{issue_date}.png", dpi=150)
    plt.close()


def plot_r2_scatter(
    metrics_df: pd.DataFrame,
    model1: str,
    model2: str,
    issue_date: str,
    output_dir: Path,
) -> None:
    """Create scatter plot comparing R² between two models."""
    subset = metrics_df[metrics_df["issue_date"] == issue_date]

    df1 = subset[subset["model"] == model1][["code", "r2"]].rename(
        columns={"r2": "r2_1"}
    )
    df2 = subset[subset["model"] == model2][["code", "r2"]].rename(
        columns={"r2": "r2_2"}
    )
    merged = df1.merge(df2, on="code")

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.scatter(merged["r2_1"], merged["r2_2"], alpha=0.6)
    ax.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel(f"{model1} R²")
    ax.set_ylabel(f"{model2} R²")
    ax.set_title(f"{model1} vs {model2} - {issue_date}")
    ax.set_aspect("equal")

    # Count improvements
    improved = (merged["r2_2"] > merged["r2_1"]).sum()
    total = len(merged)
    ax.text(
        0.05,
        0.95,
        f"{model2} better: {improved}/{total}",
        transform=ax.transAxes,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / f"r2_scatter_{model1}_vs_{model2}_{issue_date}.png", dpi=150
    )
    plt.close()


def plot_r2_per_basin(
    metrics_df: pd.DataFrame, model: str, issue_date: str, output_dir: Path
) -> None:
    """Create sorted bar chart of R² per basin for a model."""
    subset = metrics_df[
        (metrics_df["model"] == model) & (metrics_df["issue_date"] == issue_date)
    ].sort_values("r2", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = ["green" if r2 > 0 else "red" for r2 in subset["r2"]]
    ax.bar(range(len(subset)), subset["r2"], color=colors)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Basin (sorted by R²)")
    ax.set_ylabel("R² Score")
    ax.set_title(f"R² per Basin - {model} - {issue_date}")

    plt.tight_layout()
    plt.savefig(output_dir / f"r2_per_basin_{model}_{issue_date}.png", dpi=150)
    plt.close()


# =============================================================================
# Main Orchestration
# =============================================================================


def main() -> None:
    """Main execution function."""
    # Load environment variables
    load_dotenv()

    # Get paths from environment
    path_discharge = os.getenv("kgz_path_base_pred")
    path_forcing_p = os.getenv("PATH_TO_FORCING_ERA5")
    path_swe = os.getenv(
        "path_SWE_00003"
    )  # Using 00003 SWE (already aggregated per basin)
    path_static = os.getenv("PATH_TO_STATIC")

    if not all([path_discharge, path_forcing_p, path_swe, path_static]):
        logger.error("Missing required environment variables")
        logger.error(f"  kgz_path_base_pred: {path_discharge}")
        logger.error(f"  PATH_TO_FORCING_ERA5: {path_forcing_p}")
        logger.error(f"  path_SWE_00003: {path_swe}")
        logger.error(f"  PATH_TO_STATIC: {path_static}")
        sys.exit(1)

    # Construct forcing paths (P and T are in the same directory structure)
    # Based on the data structure, P and T files have format: 00003_P_reanalysis.csv
    # But actually they contain all basins, so we just use them directly
    path_forcing_t = path_forcing_p  # Same directory, different file pattern

    # Actually the forcing files contain all basins in one file
    # Let me construct paths correctly
    forcing_dir = Path(path_forcing_p)
    path_p = forcing_dir / "00003_P_reanalysis.csv"
    path_t = forcing_dir / "00003_T_reanalysis.csv"

    # Setup output directory
    output_dir = Path("gp_seasonal_results")
    output_dir.mkdir(exist_ok=True)

    # Load data
    logger.info("=" * 60)
    logger.info("Loading data...")
    logger.info("=" * 60)

    discharge = load_discharge(path_discharge)
    forcing = load_forcing(str(path_p), str(path_t))
    swe = load_swe(path_swe)

    # Merge data
    data = merge_data(discharge, forcing, swe)

    # Determine available years (need Apr-Sep data)
    data["year"] = data["date"].dt.year
    years_with_data = []
    for year in data["year"].unique():
        year_data = data[data["year"] == year]
        # Check if we have data through September
        max_month = year_data["date"].dt.month.max()
        if max_month >= 9:
            years_with_data.append(int(year))

    years = sorted(years_with_data)
    logger.info(f"Years with complete seasonal data: {years}")

    # Feature columns (SWE 10d, P 90d, T 30d, Q 30d)
    feature_cols = ["SWE_10d_mean", "P_90d_sum", "T_30d_mean", "Q_30d_mean"]

    # Results storage
    all_results: dict[str, dict[str, list[LOOCVResult]]] = {}

    # Run models for each issue date
    # LocalGP and BayesianRidge provide uncertainty estimates
    models_to_run = [
        ("LocalGP", run_loocv_local_gp),
        ("BayesianRidge", run_loocv_bayesian_ridge),
        ("OLS", run_loocv_ols),
        ("Ridge", run_loocv_ridge),
        ("Lasso", run_loocv_lasso),
    ]

    for issue_date in ISSUE_DATES:
        logger.info("=" * 60)
        logger.info(f"Processing issue date: {issue_date}")
        logger.info("=" * 60)

        # Prepare dataset for this issue date
        dataset = prepare_dataset(data, issue_date, years)

        if len(dataset) < 10:
            logger.warning(f"Insufficient data for {issue_date}, skipping")
            continue

        for model_name, run_func in models_to_run:
            logger.info(f"  Running {model_name}...")

            results, _ = run_func(dataset, feature_cols)

            if model_name not in all_results:
                all_results[model_name] = {}

            all_results[model_name][str(issue_date)] = results

            # Quick summary
            r2_scores = compute_r2_per_basin(results)
            if r2_scores:
                mean_r2 = np.mean(list(r2_scores.values()))
                median_r2 = np.median(list(r2_scores.values()))
                logger.info(f"    Mean R²: {mean_r2:.3f}, Median R²: {median_r2:.3f}")

            # Coverage analysis for models with uncertainty
            if model_name in ["LocalGP", "BayesianRidge"]:
                cov_50 = compute_coverage(results, 0.50)
                cov_90 = compute_coverage(results, 0.90)
                if cov_50 is not None and cov_90 is not None:
                    logger.info(
                        f"    Coverage: 50%={cov_50:.1%} (expect 50%), 90%={cov_90:.1%} (expect 90%)"
                    )

    # Convert results to DataFrame
    logger.info("=" * 60)
    logger.info("Generating outputs...")
    logger.info("=" * 60)

    metrics_df = results_to_dataframe(all_results)
    metrics_df.to_csv(output_dir / "all_metrics.csv", index=False)
    logger.info(f"Saved metrics to {output_dir / 'all_metrics.csv'}")

    # Generate plots
    for issue_date in ISSUE_DATES:
        issue_str = str(issue_date)
        if issue_str not in metrics_df["issue_date"].values:
            continue

        # R² comparison boxplot
        plot_r2_comparison(metrics_df, issue_str, output_dir)

        # R² scatter: LocalGP vs BayesianRidge (both have uncertainty)
        plot_r2_scatter(metrics_df, "LocalGP", "BayesianRidge", issue_str, output_dir)

        # R² per basin for each model
        for model_name in ["LocalGP", "BayesianRidge", "OLS", "Ridge", "Lasso"]:
            if model_name in metrics_df["model"].values:
                plot_r2_per_basin(metrics_df, model_name, issue_str, output_dir)

    # Summary statistics
    logger.info("=" * 60)
    logger.info("Summary Statistics")
    logger.info("=" * 60)

    summary = metrics_df.groupby(["model", "issue_date"])["r2"].agg(
        ["mean", "median", "std", "count"]
    )
    logger.info("\n" + summary.to_string())

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
