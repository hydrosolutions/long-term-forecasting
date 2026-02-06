"""
Bayesian Post-Processing Module for Monthly Discharge Forecasts

This module implements Bayesian ratio-based correction for probabilistic forecasts,
using historical discharge ratios (R = Q_curr / Q_prev) to constrain predictions.

Key features:
- Ratio-based Bayesian correction with configurable prior strength
- Support for quantile-based probabilistic forecasts
- Optional soft bounds based on historical constraints
- Multi-horizon correction support

The approach transforms forecasts into ratio space, performs Bayesian updating
with historical ratio statistics, and transforms back to discharge space.
"""

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# =============================================================================
# Constants and Helper Functions
# =============================================================================

LT_FORECAST_BASE_COLUMNS = ["date", "code", "valid_from", "valid_to"]


def infer_q_columns(df: pd.DataFrame) -> List[str]:
    """
    Infer quantile columns (Q5, Q10, Q25, Q50, Q75, Q90, Q95) from DataFrame.

    Args:
        df: DataFrame to inspect

    Returns:
        List of column names matching the pattern Q followed by digits
    """
    return [c for c in df.columns if re.fullmatch(r"Q\d+", c)]


# =============================================================================
# Configuration and Data Classes
# =============================================================================


@dataclass
class RatioPriorConfig:
    """
    Configuration for prior strength in ratio-based Bayesian correction.

    The prior strength determines how much the correction is influenced by
    historical discharge ratios vs the model forecast.

    Attributes:
        prior_precision_scale: Multiplier for climatology precision.
            - 1.0 = natural precision from historical data (default)
            - >1.0 = stronger prior (more weight to historical ratios)
            - <1.0 = weaker prior (more weight to forecast)
            - 0.0 = no prior influence (forecast unchanged)

        min_effective_n: Minimum effective sample size for prior calculations.
            Prevents very weak priors when historical data is sparse.
            Default: 3

        apply_soft_bounds: Whether to apply soft bounds based on historical
            constraints (e.g., if all historical ratios < 1, constrain forecast).
            Default: True

        constraint_confidence_scale: Scale for constraint confidence.
            Higher values = more confident in historical bounds.
            Default: 1.0

        apply_peak_capping: Whether to cap forecasts at the seasonal peak value.
            After the climatological peak month, forecasts are constrained to not
            exceed the peak value (either observed or forecasted).
            Default: True

        peak_cap_softness: How soft the peak capping is (0-1).
            - 0.0 = hard cap (strict limit at peak)
            - 1.0 = no cap (disabled)
            - 0.1-0.3 = soft cap (recommended, allows small exceedance)
            Default: 0.1

    Example:
        >>> # Strong prior - heavily weight historical ratios
        >>> config = RatioPriorConfig(prior_precision_scale=2.0)

        >>> # Weak prior - mostly trust the forecast
        >>> config = RatioPriorConfig(prior_precision_scale=0.5)

        >>> # With peak capping (default)
        >>> config = RatioPriorConfig(apply_peak_capping=True)

        >>> # Hard peak cap
        >>> config = RatioPriorConfig(apply_peak_capping=True, peak_cap_softness=0.0)
    """

    prior_precision_scale: float = 1.0
    min_effective_n: int = 3
    apply_soft_bounds: bool = True
    constraint_confidence_scale: float = 1.0
    apply_peak_capping: bool = True
    peak_cap_softness: float = 0.1

    def __post_init__(self):
        if self.prior_precision_scale < 0:
            raise ValueError("prior_precision_scale must be non-negative")
        if self.min_effective_n < 1:
            raise ValueError("min_effective_n must be at least 1")
        if self.constraint_confidence_scale < 0:
            raise ValueError("constraint_confidence_scale must be non-negative")
        if not 0.0 <= self.peak_cap_softness <= 1.0:
            raise ValueError("peak_cap_softness must be between 0.0 and 1.0")


@dataclass
class RatioStats:
    """
    Statistics for historical discharge ratios R = Q_curr / Q_prev for a (basin, month) pair.

    Attributes:
        mean_R: Mean of historical ratios
        std_R: Standard deviation of historical ratios
        n: Sample size (number of historical ratios)
        R_max: Maximum observed ratio (used for soft bounds)
        constraint_exists: Whether all historical ratios satisfy R < 1
            (indicates declining flow pattern for this month)
    """

    mean_R: float
    std_R: float
    n: int
    R_max: float
    constraint_exists: bool


@dataclass
class RatioCorrection:
    """
    Diagnostic information from a ratio-based Bayesian correction.

    Attributes:
        original_forecast: Original forecast value before correction
        corrected_forecast: Corrected forecast value
        Q_prev: Previous month's observed discharge used as reference
        R_forecast: Forecast ratio (Q_curr_fc / Q_prev)
        R_posterior: Posterior ratio after Bayesian update
        sigma_R_eff: Effective climatology uncertainty (inflated for small n)
        sigma_R_fc: Forecast uncertainty in ratio space
        prior_weight: Effective weight given to prior (0-1)
        constraint_applied: Whether soft bounds were applied
    """

    original_forecast: float
    corrected_forecast: float
    Q_prev: float
    R_forecast: float
    R_posterior: float
    sigma_R_eff: float
    sigma_R_fc: float
    prior_weight: float
    constraint_applied: bool


# =============================================================================
# Helper Functions
# =============================================================================


def compute_climatological_peak_months(
    observations_df: pd.DataFrame,
    basin_col: str = "code",
    month_col: str = "month",
    value_col: str = "Q_obs",
) -> Dict[int, int]:
    """
    Compute the climatological peak month for each basin.

    The peak month is the month with the highest mean discharge across all years.

    Args:
        observations_df: DataFrame with monthly observations
        basin_col: Column name for basin identifier
        month_col: Column name for month (1-12)
        value_col: Column name for discharge values

    Returns:
        Dictionary mapping basin code to peak month (1-12)
    """
    peak_months: Dict[int, int] = {}

    for basin, basin_df in observations_df.groupby(basin_col):
        monthly_means = basin_df.groupby(month_col)[value_col].mean()
        if len(monthly_means) > 0:
            peak_month = monthly_means.idxmax()
            peak_months[basin] = int(peak_month)

    logger.info(f"Computed peak months for {len(peak_months)} basins")
    return peak_months


def is_after_peak_month(target_month: int, peak_month: int) -> bool:
    """
    Check if target_month is in the recession period after the peak.

    Handles wrap-around for basins with winter peaks.

    Args:
        target_month: Month being forecasted (1-12)
        peak_month: Climatological peak month (1-12)

    Returns:
        True if target_month is after peak (in recession)
    """
    # Simple case: target is after peak in same year
    if target_month > peak_month:
        return True

    # Handle wrap-around: if peak is late in year (e.g., Nov/Dec)
    # and target is early (e.g., Jan/Feb), it's still recession
    # But only for a few months after peak
    if peak_month >= 10 and target_month <= 3:
        return True

    return False


def get_previous_month_value(
    observations_df: pd.DataFrame,
    basin: int,
    year: int,
    month: int,
    value_col: str = "Q_obs",
) -> Optional[float]:
    """
    Get the observed discharge value from the previous month.

    For January (month=1), retrieves December of the previous year.

    Args:
        observations_df: DataFrame with columns: code, year, month, and value_col
        basin: Basin code
        year: Year of the target month
        month: Target month (1-12)
        value_col: Column name for discharge values

    Returns:
        Previous month's discharge value, or None if not found
    """
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year

    mask = (
        (observations_df["code"] == basin)
        & (observations_df["year"] == prev_year)
        & (observations_df["month"] == prev_month)
    )

    matched = observations_df.loc[mask, value_col]

    if len(matched) == 0:
        return None

    return matched.iloc[0]


def compute_historical_ratios(
    observations_df: pd.DataFrame,
    basin_col: str = "code",
    month_col: str = "month",
    year_col: str = "year",
    value_col: str = "Q_obs",
) -> Dict[Tuple[int, int], RatioStats]:
    """
    Compute historical discharge ratios R = Q_curr / Q_prev for each (basin, month) pair.

    Args:
        observations_df: DataFrame with monthly observations containing:
            - basin_col: Basin identifier
            - year_col: Year
            - month_col: Month (1-12)
            - value_col: Discharge value
        basin_col: Column name for basin identifier
        month_col: Column name for month
        year_col: Column name for year
        value_col: Column name for discharge values

    Returns:
        Dictionary mapping (basin, month) tuples to RatioStats
    """
    ratios_by_key: Dict[Tuple[int, int], List[float]] = {}

    df_sorted = observations_df.sort_values([basin_col, year_col, month_col]).copy()

    for basin, basin_df in df_sorted.groupby(basin_col):
        basin_df = basin_df.reset_index(drop=True)

        for idx in range(len(basin_df)):
            row = basin_df.iloc[idx]
            year = row[year_col]
            month = row[month_col]
            Q_curr = row[value_col]

            Q_prev = get_previous_month_value(
                observations_df=observations_df,
                basin=basin,
                year=year,
                month=month,
                value_col=value_col,
            )

            if Q_prev is None or Q_prev <= 0 or pd.isna(Q_curr) or Q_curr < 0:
                continue

            R = Q_curr / Q_prev

            key = (basin, month)
            if key not in ratios_by_key:
                ratios_by_key[key] = []
            ratios_by_key[key].append(R)

    result: Dict[Tuple[int, int], RatioStats] = {}

    for key, ratios in ratios_by_key.items():
        if len(ratios) < 1:
            continue

        ratios_arr = np.array(ratios)
        mean_R = np.mean(ratios_arr)
        std_R = np.std(ratios_arr, ddof=1) if len(ratios_arr) > 1 else mean_R * 0.5
        n = len(ratios_arr)
        R_max = np.max(ratios_arr)
        constraint_exists = np.all(ratios_arr < 1.0)

        result[key] = RatioStats(
            mean_R=mean_R,
            std_R=std_R,
            n=n,
            R_max=R_max,
            constraint_exists=constraint_exists,
        )

    logger.info(f"Computed historical ratios for {len(result)} (basin, month) pairs")

    return result


# =============================================================================
# Main Corrector Class
# =============================================================================


class RatioBayesianCorrector:
    """
    Bayesian correction using discharge ratios with configurable prior strength.

    This corrector transforms forecasts and observations into ratio space
    (R = Q_curr / Q_prev), performs Bayesian updating, and transforms back.

    Mathematical formulation:
    1. Historical ratios provide prior: N(mu_R, sigma_R^2)
    2. Forecast provides likelihood: N(R_fc, sigma_R_fc^2)
    3. Posterior combines both with precision weighting
    4. Prior strength is controlled via prior_precision_scale

    The posterior mean is:
        R_post = (mu_R * tau_clim * scale + R_fc * tau_fc) / (tau_clim * scale + tau_fc)

    where:
        - tau_clim = 1 / sigma_R_eff^2 (climatology precision)
        - tau_fc = 1 / sigma_R_fc^2 (forecast precision)
        - scale = prior_precision_scale

    Example:
        >>> # Default configuration
        >>> ratios = compute_historical_ratios(observations_df)
        >>> corrector = RatioBayesianCorrector(ratios)
        >>> corrected_df = corrector.correct_forecasts_df(forecasts_df, observations_df)

        >>> # Strong prior configuration
        >>> config = RatioPriorConfig(prior_precision_scale=2.0)
        >>> corrector = RatioBayesianCorrector(ratios, config=config)

        >>> # Single forecast correction
        >>> Q_corrected, diagnostics = corrector.correct_forecast(
        ...     basin=12345,
        ...     target_month=8,
        ...     Q_curr_fc=50.0,
        ...     Q_prev=60.0,
        ...     sigma_fc=5.0
        ... )
    """

    def __init__(
        self,
        historical_ratios: Dict[Tuple[int, int], RatioStats],
        config: Optional[RatioPriorConfig] = None,
        default_cv: float = 0.3,
    ):
        """
        Initialize the ratio-based Bayesian corrector.

        Args:
            historical_ratios: Dictionary from compute_historical_ratios()
            config: Prior configuration. If None, uses default RatioPriorConfig.
            default_cv: Default coefficient of variation for forecast uncertainty
                       when sigma_fc is not provided (sigma_fc = default_cv * Q_curr_fc)
        """
        self.historical_ratios = historical_ratios
        self.config = config or RatioPriorConfig()
        self.default_cv = default_cv

    def correct_forecast(
        self,
        basin: int,
        target_month: int,
        Q_curr_fc: float,
        Q_prev: float,
        sigma_fc: Optional[float] = None,
    ) -> Tuple[float, RatioCorrection]:
        """
        Apply Bayesian ratio correction to a single forecast.

        Args:
            basin: Basin code
            target_month: Target forecast month (1-12)
            Q_curr_fc: Forecast value for current month
            Q_prev: Observed discharge from previous month
            sigma_fc: Forecast uncertainty (std dev). If None, estimated from default_cv.

        Returns:
            Tuple of (corrected_forecast, RatioCorrection diagnostics)
        """
        # Handle edge cases
        if Q_prev is None or Q_prev <= 0 or pd.isna(Q_prev):
            logger.debug(
                f"Basin {basin}, month {target_month}: Q_prev invalid ({Q_prev}), "
                "skipping correction"
            )
            return Q_curr_fc, self._empty_correction(Q_curr_fc, Q_prev)

        if pd.isna(Q_curr_fc) or Q_curr_fc < 0:
            logger.debug(
                f"Basin {basin}, month {target_month}: Q_curr_fc invalid ({Q_curr_fc}), "
                "skipping correction"
            )
            return Q_curr_fc, self._empty_correction(Q_curr_fc, Q_prev)

        # Check if prior precision is zero (no correction)
        if self.config.prior_precision_scale == 0:
            return Q_curr_fc, self._empty_correction(Q_curr_fc, Q_prev)

        # Check for historical ratios
        key = (basin, target_month)
        if key not in self.historical_ratios:
            logger.debug(
                f"Basin {basin}, month {target_month}: no historical ratios, "
                "skipping correction"
            )
            return Q_curr_fc, self._empty_correction(Q_curr_fc, Q_prev)

        stats = self.historical_ratios[key]

        # Get climatology statistics
        mu_R = stats.mean_R
        sigma_R = stats.std_R
        n = max(stats.n, self.config.min_effective_n)

        # Effective climatology uncertainty (inflate for small samples)
        sigma_R_eff = sigma_R * np.sqrt(1 + 1 / n)

        # Convert forecast to ratio space
        R_fc = Q_curr_fc / Q_prev

        # Estimate forecast uncertainty in ratio space
        if sigma_fc is None:
            sigma_fc = self.default_cv * Q_curr_fc
        sigma_R_fc = max(sigma_fc / Q_prev, 1e-6)

        # Bayesian update with scaled prior precision
        tau_clim = self.config.prior_precision_scale / (sigma_R_eff**2)
        tau_fc = 1 / (sigma_R_fc**2)
        tau_post = tau_clim + tau_fc

        # Posterior mean
        mu_R_post = (mu_R * tau_clim + R_fc * tau_fc) / tau_post

        # Compute effective prior weight for diagnostics
        prior_weight = tau_clim / tau_post

        # Apply soft bounds if configured and constraint exists
        constraint_applied = False
        if self.config.apply_soft_bounds and stats.constraint_exists:
            R_max = stats.R_max
            if mu_R_post > R_max:
                # Constraint confidence scaled by config and sample size
                p_constraint = min(
                    1.0, (n / (n + 2)) * self.config.constraint_confidence_scale
                )
                mu_R_bounded = min(mu_R_post, R_max)
                mu_R_post = p_constraint * mu_R_bounded + (1 - p_constraint) * mu_R_post
                constraint_applied = True

        # Recover corrected forecast
        Q_corrected = max(Q_prev * mu_R_post, 0.0)

        diagnostics = RatioCorrection(
            original_forecast=Q_curr_fc,
            corrected_forecast=Q_corrected,
            Q_prev=Q_prev,
            R_forecast=R_fc,
            R_posterior=mu_R_post,
            sigma_R_eff=sigma_R_eff,
            sigma_R_fc=sigma_R_fc,
            prior_weight=prior_weight,
            constraint_applied=constraint_applied,
        )

        return Q_corrected, diagnostics

    def _empty_correction(
        self, Q_curr_fc: float, Q_prev: Optional[float]
    ) -> RatioCorrection:
        """Create an empty correction diagnostic for skipped forecasts."""
        return RatioCorrection(
            original_forecast=Q_curr_fc,
            corrected_forecast=Q_curr_fc,
            Q_prev=Q_prev if Q_prev is not None else 0.0,
            R_forecast=np.nan,
            R_posterior=np.nan,
            sigma_R_eff=np.nan,
            sigma_R_fc=np.nan,
            prior_weight=0.0,
            constraint_applied=False,
        )

    def correct_forecasts_df(
        self,
        forecasts_df: pd.DataFrame,
        observations_df: pd.DataFrame,
        pred_col: str = "Q50",
        basin_col: str = "code",
        date_col: str = "valid_from",
        sigma_col: Optional[str] = None,
        value_col: str = "Q_obs",
        inplace: bool = False,
    ) -> pd.DataFrame:
        """
        Apply Bayesian ratio correction to a DataFrame of forecasts.

        Args:
            forecasts_df: DataFrame with forecast quantiles
            observations_df: DataFrame with monthly observations (code, year, month, Q_obs)
            pred_col: Column name for central prediction to correct
            basin_col: Column name for basin identifier
            date_col: Column name for forecast valid date (to extract year/month)
            sigma_col: Optional column with forecast uncertainty. If None, uses default_cv.
            value_col: Column name for observed values in observations_df
            inplace: If True, modify forecasts_df in place

        Returns:
            DataFrame with corrected forecasts. All quantile columns are adjusted
            by the same multiplicative factor as the central prediction.
        """
        if not inplace:
            forecasts_df = forecasts_df.copy()

        q_cols = infer_q_columns(forecasts_df)

        # Convert quantile columns to float
        for col in q_cols:
            if col in forecasts_df.columns:
                forecasts_df[col] = forecasts_df[col].astype(float)

        corrections_applied = 0
        corrections_skipped = 0

        for idx in forecasts_df.index:
            row = forecasts_df.loc[idx]
            basin = row[basin_col]

            date_val = pd.to_datetime(row[date_col])
            year = date_val.year
            month = date_val.month

            Q_curr_fc = row[pred_col]

            Q_prev = get_previous_month_value(
                observations_df=observations_df,
                basin=basin,
                year=year,
                month=month,
                value_col=value_col,
            )

            sigma_fc = row[sigma_col] if sigma_col and sigma_col in row else None

            Q_corrected, diagnostics = self.correct_forecast(
                basin=basin,
                target_month=month,
                Q_curr_fc=Q_curr_fc,
                Q_prev=Q_prev,
                sigma_fc=sigma_fc,
            )

            if (
                pd.isna(diagnostics.R_forecast)
                or diagnostics.R_forecast == diagnostics.R_posterior
            ):
                corrections_skipped += 1
                continue

            corrections_applied += 1

            if Q_curr_fc > 0:
                adjustment_factor = Q_corrected / Q_curr_fc
            else:
                adjustment_factor = 1.0

            for col in q_cols:
                if col in forecasts_df.columns:
                    original_val = forecasts_df.loc[idx, col]
                    forecasts_df.loc[idx, col] = max(
                        original_val * adjustment_factor, 0.0
                    )

        logger.info(
            f"Ratio Bayesian correction: {corrections_applied} forecasts corrected, "
            f"{corrections_skipped} skipped (prior_precision_scale="
            f"{self.config.prior_precision_scale})"
        )

        return forecasts_df


# =============================================================================
# Convenience Functions
# =============================================================================


def get_forecast_as_prev_month(
    forecast_df: pd.DataFrame,
    basin: int,
    year: int,
    month: int,
    pred_col: str = "Q50",
    basin_col: str = "code",
    date_col: str = "valid_from",
) -> Optional[float]:
    """
    Get the forecasted value for the previous month from a forecast DataFrame.

    Used for cascading correction where horizon h uses horizon h-1's forecast as Q_prev.

    Args:
        forecast_df: DataFrame with forecasts (already corrected for previous horizon)
        basin: Basin code
        year: Year of the target month
        month: Target month (1-12)
        pred_col: Column name for prediction values
        basin_col: Column name for basin identifier
        date_col: Column name for valid date

    Returns:
        Previous month's forecasted value, or None if not found
    """
    # Compute previous month and year
    if month == 1:
        prev_month = 12
        prev_year = year - 1
    else:
        prev_month = month - 1
        prev_year = year

    # Find the forecast for previous month
    mask = forecast_df[basin_col] == basin

    matched = forecast_df.loc[mask].copy()
    if len(matched) == 0:
        return None

    # Filter by valid_from date matching previous month
    matched[date_col] = pd.to_datetime(matched[date_col])
    matched = matched[
        (matched[date_col].dt.year == prev_year)
        & (matched[date_col].dt.month == prev_month)
    ]

    if len(matched) == 0:
        return None

    return matched[pred_col].iloc[0]


def apply_ratio_bayesian_correction(
    forecasts_by_horizon: Dict[int, pd.DataFrame],
    observations_df: pd.DataFrame,
    config: Optional[RatioPriorConfig] = None,
    pred_col: str = "Q50",
    basin_col: str = "code",
    date_col: str = "valid_from",
    value_col: str = "Q_obs",
    default_cv: float = 0.3,
) -> Dict[int, pd.DataFrame]:
    """
    Apply Bayesian ratio correction to forecasts across multiple horizons.

    This is the main entry point for ratio-based Bayesian correction.
    It computes historical ratios once and applies corrections with proper
    cascading of Q_prev values:

    - **Horizon 0**: No correction (current month forecast, no prior available)
    - **Horizon 1**: Uses observations as Q_prev (previous month is already observed)
    - **Horizon 2+**: Uses previous horizon's corrected forecast as Q_prev (cascading)

    Args:
        forecasts_by_horizon: Dictionary mapping horizon (0, 1, 2, ...) to forecast DataFrame
        observations_df: DataFrame with monthly observations containing:
            - basin_col: Basin identifier
            - year: Year
            - month: Month (1-12)
            - value_col: Discharge value
        config: Prior configuration. If None, uses default RatioPriorConfig.
        pred_col: Column name for central prediction
        basin_col: Column name for basin identifier
        date_col: Column name for forecast valid date
        value_col: Column name for observed values
        default_cv: Default coefficient of variation for forecast uncertainty

    Returns:
        Dictionary mapping horizon to corrected forecast DataFrame

    Example:
        >>> # Default correction with cascading
        >>> corrected = apply_ratio_bayesian_correction(
        ...     forecasts_by_horizon=forecasts,
        ...     observations_df=observations
        ... )

        >>> # Strong prior
        >>> config = RatioPriorConfig(prior_precision_scale=2.0)
        >>> corrected = apply_ratio_bayesian_correction(
        ...     forecasts_by_horizon=forecasts,
        ...     observations_df=observations,
        ...     config=config
        ... )
    """
    config = config or RatioPriorConfig()

    logger.info("Computing historical discharge ratios...")
    historical_ratios = compute_historical_ratios(
        observations_df=observations_df,
        basin_col=basin_col,
        value_col=value_col,
    )

    if len(historical_ratios) == 0:
        logger.warning(
            "No historical ratios computed. Returning uncorrected forecasts."
        )
        return {h: df.copy() for h, df in forecasts_by_horizon.items()}

    # Compute peak months for peak capping
    peak_months: Dict[int, int] = {}
    if config.apply_peak_capping:
        logger.info("Computing climatological peak months...")
        peak_months = compute_climatological_peak_months(
            observations_df=observations_df,
            basin_col=basin_col,
            value_col=value_col,
        )

    corrector = RatioBayesianCorrector(
        historical_ratios=historical_ratios,
        config=config,
        default_cv=default_cv,
    )

    corrected_by_horizon: Dict[int, pd.DataFrame] = {}

    # Track peak values for each (basin, year) for peak capping
    # Key: (basin, year), Value: peak Q value seen so far
    peak_values: Dict[Tuple[int, int], float] = {}

    # Process horizons in order for proper cascading
    for horizon in sorted(forecasts_by_horizon.keys()):
        forecasts = forecasts_by_horizon[horizon]

        # Horizon 0: No correction (current month, no reliable Q_prev)
        if horizon == 0:
            logger.info(f"Horizon {horizon}: Skipping correction (current month)")
            corrected_by_horizon[horizon] = forecasts.copy()
            continue

        # Horizon 1: Use observations as Q_prev
        if horizon == 1:
            logger.info(
                f"Horizon {horizon}: Applying correction with observations as Q_prev"
            )
            corrected = corrector.correct_forecasts_df(
                forecasts_df=forecasts,
                observations_df=observations_df,
                pred_col=pred_col,
                basin_col=basin_col,
                date_col=date_col,
                value_col=value_col,
                inplace=False,
            )
            # Update peak values from this horizon
            if config.apply_peak_capping:
                _update_peak_values(
                    corrected, peak_values, peak_months, pred_col, basin_col, date_col
                )
            corrected_by_horizon[horizon] = corrected
            continue

        # Horizon 2+: Use previous horizon's corrected forecast as Q_prev
        logger.info(
            f"Horizon {horizon}: Applying correction with horizon {horizon - 1} "
            "forecast as Q_prev (cascading)"
        )

        prev_horizon_forecast = corrected_by_horizon.get(horizon - 1)
        if prev_horizon_forecast is None:
            logger.warning(
                f"Horizon {horizon}: Previous horizon forecast not available, "
                "skipping correction"
            )
            corrected_by_horizon[horizon] = forecasts.copy()
            continue

        # Apply correction with cascading Q_prev from previous horizon
        corrected = _correct_with_cascading_prev(
            corrector=corrector,
            forecasts_df=forecasts,
            prev_horizon_df=prev_horizon_forecast,
            observations_df=observations_df,
            pred_col=pred_col,
            basin_col=basin_col,
            date_col=date_col,
            value_col=value_col,
        )

        # Apply peak capping if enabled
        if config.apply_peak_capping:
            corrected = _apply_peak_capping(
                forecasts_df=corrected,
                peak_values=peak_values,
                peak_months=peak_months,
                softness=config.peak_cap_softness,
                pred_col=pred_col,
                basin_col=basin_col,
                date_col=date_col,
            )
            # Update peak values from this horizon
            _update_peak_values(
                corrected, peak_values, peak_months, pred_col, basin_col, date_col
            )

        corrected_by_horizon[horizon] = corrected

    return corrected_by_horizon


def _update_peak_values(
    forecasts_df: pd.DataFrame,
    peak_values: Dict[Tuple[int, int], float],
    peak_months: Dict[int, int],
    pred_col: str,
    basin_col: str,
    date_col: str,
) -> None:
    """
    Update the peak values dictionary with forecasts at or before peak month.

    Only updates if the forecast is for the peak month or earlier.
    """
    for idx in forecasts_df.index:
        row = forecasts_df.loc[idx]
        basin = row[basin_col]
        date_val = pd.to_datetime(row[date_col])
        year = date_val.year
        month = date_val.month
        q_val = row[pred_col]

        if pd.isna(q_val):
            continue

        peak_month = peak_months.get(basin)
        if peak_month is None:
            continue

        key = (basin, year)

        # Update peak if this is at or before peak month
        if not is_after_peak_month(month, peak_month):
            if key not in peak_values or q_val > peak_values[key]:
                peak_values[key] = q_val


def _apply_peak_capping(
    forecasts_df: pd.DataFrame,
    peak_values: Dict[Tuple[int, int], float],
    peak_months: Dict[int, int],
    softness: float,
    pred_col: str,
    basin_col: str,
    date_col: str,
) -> pd.DataFrame:
    """
    Apply peak capping to forecasts in recession period.

    For months after the climatological peak, cap the forecast at the
    peak value (observed or forecasted).

    Args:
        forecasts_df: Forecasts to cap
        peak_values: Dictionary of (basin, year) -> peak Q value
        peak_months: Dictionary of basin -> peak month
        softness: How soft the cap is (0=hard, 1=no cap)
        pred_col: Prediction column
        basin_col: Basin column
        date_col: Date column

    Returns:
        Capped forecast DataFrame
    """
    forecasts_df = forecasts_df.copy()
    q_cols = infer_q_columns(forecasts_df)

    capped_count = 0

    for idx in forecasts_df.index:
        row = forecasts_df.loc[idx]
        basin = row[basin_col]
        date_val = pd.to_datetime(row[date_col])
        year = date_val.year
        month = date_val.month

        peak_month = peak_months.get(basin)
        if peak_month is None:
            continue

        # Only apply capping after peak month
        if not is_after_peak_month(month, peak_month):
            continue

        key = (basin, year)
        peak_val = peak_values.get(key)
        if peak_val is None:
            continue

        q_curr = row[pred_col]
        if pd.isna(q_curr):
            continue

        # Check if capping needed
        if q_curr > peak_val:
            capped_count += 1

            # Soft capping: blend between capped and uncapped
            # softness=0 -> full cap, softness=1 -> no cap
            q_capped = peak_val
            q_new = softness * q_curr + (1 - softness) * q_capped

            # Apply same ratio adjustment to all quantiles
            if q_curr > 0:
                adjustment_factor = q_new / q_curr
                for col in q_cols:
                    if col in forecasts_df.columns:
                        forecasts_df.loc[idx, col] = (
                            forecasts_df.loc[idx, col] * adjustment_factor
                        )

    if capped_count > 0:
        logger.info(f"Peak capping applied to {capped_count} forecasts")

    return forecasts_df


def _correct_with_cascading_prev(
    corrector: RatioBayesianCorrector,
    forecasts_df: pd.DataFrame,
    prev_horizon_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    pred_col: str = "Q50",
    basin_col: str = "code",
    date_col: str = "valid_from",
    value_col: str = "Q_obs",
) -> pd.DataFrame:
    """
    Apply correction using previous horizon's forecast as Q_prev.

    For horizon h >= 2, we don't have observations for the previous month
    at forecast time. Instead, we use the corrected forecast from horizon h-1.

    Args:
        corrector: Configured RatioBayesianCorrector instance
        forecasts_df: Forecasts to correct
        prev_horizon_df: Corrected forecasts from previous horizon (h-1)
        observations_df: Observations (for historical ratio lookup, not Q_prev)
        pred_col: Prediction column name
        basin_col: Basin column name
        date_col: Valid date column name
        value_col: Observation value column name

    Returns:
        Corrected forecast DataFrame
    """
    forecasts_df = forecasts_df.copy()

    q_cols = infer_q_columns(forecasts_df)
    for col in q_cols:
        if col in forecasts_df.columns:
            forecasts_df[col] = forecasts_df[col].astype(float)

    corrections_applied = 0
    corrections_skipped = 0

    for idx in forecasts_df.index:
        row = forecasts_df.loc[idx]
        basin = row[basin_col]

        date_val = pd.to_datetime(row[date_col])
        year = date_val.year
        month = date_val.month

        Q_curr_fc = row[pred_col]

        # Get Q_prev from previous horizon's forecast (not observations)
        Q_prev = get_forecast_as_prev_month(
            forecast_df=prev_horizon_df,
            basin=basin,
            year=year,
            month=month,
            pred_col=pred_col,
            basin_col=basin_col,
            date_col=date_col,
        )

        if Q_prev is None:
            # Fall back to observations if cascade not available
            Q_prev = get_previous_month_value(
                observations_df=observations_df,
                basin=basin,
                year=year,
                month=month,
                value_col=value_col,
            )

        Q_corrected, diagnostics = corrector.correct_forecast(
            basin=basin,
            target_month=month,
            Q_curr_fc=Q_curr_fc,
            Q_prev=Q_prev,
        )

        if (
            pd.isna(diagnostics.R_forecast)
            or diagnostics.R_forecast == diagnostics.R_posterior
        ):
            corrections_skipped += 1
            continue

        corrections_applied += 1

        if Q_curr_fc > 0:
            adjustment_factor = Q_corrected / Q_curr_fc
        else:
            adjustment_factor = 1.0

        for col in q_cols:
            if col in forecasts_df.columns:
                original_val = forecasts_df.loc[idx, col]
                forecasts_df.loc[idx, col] = max(original_val * adjustment_factor, 0.0)

    logger.info(
        f"Cascading correction: {corrections_applied} forecasts corrected, "
        f"{corrections_skipped} skipped"
    )

    return forecasts_df
