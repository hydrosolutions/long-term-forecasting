"""
Performance metrics tracking for meta-learning in monthly discharge forecasting.

This module provides functionality for tracking and analyzing model performance
over time, supporting basin-specific and temporal performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import warnings

from .evaluation_utils import calculate_all_metrics

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Track and analyze model performance over time.

    This class provides functionality for tracking model performance
    across different basins, time periods, and conditions.
    """

    def __init__(self, track_basin_specific: bool = True, track_temporal: bool = True):
        """
        Initialize performance tracker.

        Args:
            track_basin_specific: Whether to track basin-specific performance
            track_temporal: Whether to track temporal performance patterns
        """
        self.track_basin_specific = track_basin_specific
        self.track_temporal = track_temporal

        # Performance storage
        self.performance_history = {}
        self.basin_performance = {}
        self.temporal_performance = {}

        logger.info(
            f"Initialized PerformanceTracker with basin_specific={track_basin_specific}, temporal={track_temporal}"
        )

    def add_performance_record(
        self,
        model_id: str,
        predictions: pd.DataFrame,
        timestamp: datetime = None,
        metadata: Dict[str, Any] = None,
    ) -> None:
        """
        Add a performance record for a model.

        Args:
            model_id: Unique identifier for the model
            predictions: DataFrame with predictions (columns: date, code, Q_obs, Q_pred)
            timestamp: Timestamp of the performance record
            metadata: Additional metadata about the performance record
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate overall performance
        overall_metrics = calculate_all_metrics(
            predictions["Q_obs"], predictions["Q_pred"]
        )

        # Store performance record
        if model_id not in self.performance_history:
            self.performance_history[model_id] = []

        record = {
            "timestamp": timestamp,
            "overall_metrics": overall_metrics,
            "n_samples": len(predictions),
            "n_basins": predictions["code"].nunique(),
            "metadata": metadata or {},
        }

        self.performance_history[model_id].append(record)

        # Calculate basin-specific performance
        if self.track_basin_specific:
            self._update_basin_performance(model_id, predictions, timestamp)

        # Calculate temporal performance
        if self.track_temporal:
            self._update_temporal_performance(model_id, predictions, timestamp)

        logger.info(f"Added performance record for {model_id} at {timestamp}")

    def _update_basin_performance(
        self, model_id: str, predictions: pd.DataFrame, timestamp: datetime
    ) -> None:
        """Update basin-specific performance tracking."""
        if model_id not in self.basin_performance:
            self.basin_performance[model_id] = {}

        for code in predictions["code"].unique():
            basin_data = predictions[predictions["code"] == code]

            if len(basin_data) >= 3:  # Minimum samples for reliable statistics
                basin_metrics = calculate_all_metrics(
                    basin_data["Q_obs"], basin_data["Q_pred"]
                )

                if code not in self.basin_performance[model_id]:
                    self.basin_performance[model_id][code] = []

                basin_record = {
                    "timestamp": timestamp,
                    "metrics": basin_metrics,
                    "n_samples": len(basin_data),
                }

                self.basin_performance[model_id][code].append(basin_record)

    def _update_temporal_performance(
        self, model_id: str, predictions: pd.DataFrame, timestamp: datetime
    ) -> None:
        """Update temporal performance tracking."""
        if model_id not in self.temporal_performance:
            self.temporal_performance[model_id] = {}

        # Add month column
        predictions_copy = predictions.copy()
        predictions_copy["month"] = pd.to_datetime(predictions_copy["date"]).dt.month

        for month in range(1, 13):
            month_data = predictions_copy[predictions_copy["month"] == month]

            if len(month_data) >= 3:  # Minimum samples for reliable statistics
                month_metrics = calculate_all_metrics(
                    month_data["Q_obs"], month_data["Q_pred"]
                )

                if month not in self.temporal_performance[model_id]:
                    self.temporal_performance[model_id][month] = []

                temporal_record = {
                    "timestamp": timestamp,
                    "metrics": month_metrics,
                    "n_samples": len(month_data),
                }

                self.temporal_performance[model_id][month].append(temporal_record)

    def get_latest_performance(
        self, model_id: str, metric: str = "nse"
    ) -> Optional[float]:
        """
        Get latest performance metric for a model.

        Args:
            model_id: Model identifier
            metric: Metric name to retrieve

        Returns:
            Latest performance metric value or None if not available
        """
        if (
            model_id not in self.performance_history
            or not self.performance_history[model_id]
        ):
            return None

        latest_record = self.performance_history[model_id][-1]
        return latest_record["overall_metrics"].get(metric, None)

    def get_performance_trend(
        self, model_id: str, metric: str = "nse", window: int = 5
    ) -> Optional[float]:
        """
        Get performance trend for a model.

        Args:
            model_id: Model identifier
            metric: Metric name to analyze
            window: Number of recent records to consider

        Returns:
            Trend value (positive = improving, negative = deteriorating)
        """
        if (
            model_id not in self.performance_history
            or len(self.performance_history[model_id]) < 2
        ):
            return None

        records = self.performance_history[model_id][-window:]

        if len(records) < 2:
            return None

        # Extract metric values
        values = [record["overall_metrics"].get(metric, np.nan) for record in records]
        values = [v for v in values if not np.isnan(v)]

        if len(values) < 2:
            return None

        # Calculate trend (simple linear regression slope)
        x = np.arange(len(values))
        y = np.array(values)

        # Simple slope calculation
        slope = np.polyfit(x, y, 1)[0]

        return slope

    def get_basin_performance(
        self, model_id: str, basin_code: str, metric: str = "nse"
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific basin.

        Args:
            model_id: Model identifier
            basin_code: Basin code
            metric: Metric name to retrieve

        Returns:
            List of performance records for the basin
        """
        if (
            model_id not in self.basin_performance
            or basin_code not in self.basin_performance[model_id]
        ):
            return []

        basin_records = self.basin_performance[model_id][basin_code]

        return [
            {
                "timestamp": record["timestamp"],
                "value": record["metrics"].get(metric, np.nan),
                "n_samples": record["n_samples"],
            }
            for record in basin_records
        ]

    def get_temporal_performance(
        self, model_id: str, month: int, metric: str = "nse"
    ) -> List[Dict[str, Any]]:
        """
        Get performance history for a specific month.

        Args:
            model_id: Model identifier
            month: Month (1-12)
            metric: Metric name to retrieve

        Returns:
            List of performance records for the month
        """
        if (
            model_id not in self.temporal_performance
            or month not in self.temporal_performance[model_id]
        ):
            return []

        month_records = self.temporal_performance[model_id][month]

        return [
            {
                "timestamp": record["timestamp"],
                "value": record["metrics"].get(metric, np.nan),
                "n_samples": record["n_samples"],
            }
            for record in month_records
        ]

    def compare_models(
        self, model_ids: List[str], metric: str = "nse", comparison_type: str = "latest"
    ) -> Dict[str, float]:
        """
        Compare performance across multiple models.

        Args:
            model_ids: List of model identifiers to compare
            metric: Metric name to compare
            comparison_type: 'latest', 'average', 'best'

        Returns:
            Dictionary mapping model IDs to performance values
        """
        comparison_results = {}

        for model_id in model_ids:
            if model_id not in self.performance_history:
                comparison_results[model_id] = np.nan
                continue

            records = self.performance_history[model_id]

            if not records:
                comparison_results[model_id] = np.nan
                continue

            if comparison_type == "latest":
                comparison_results[model_id] = records[-1]["overall_metrics"].get(
                    metric, np.nan
                )
            elif comparison_type == "average":
                values = [
                    record["overall_metrics"].get(metric, np.nan) for record in records
                ]
                values = [v for v in values if not np.isnan(v)]
                comparison_results[model_id] = np.mean(values) if values else np.nan
            elif comparison_type == "best":
                values = [
                    record["overall_metrics"].get(metric, np.nan) for record in records
                ]
                values = [v for v in values if not np.isnan(v)]
                comparison_results[model_id] = np.max(values) if values else np.nan

        return comparison_results

    def get_performance_summary(
        self, model_id: str, include_basin: bool = True, include_temporal: bool = True
    ) -> Dict[str, Any]:
        """
        Get comprehensive performance summary for a model.

        Args:
            model_id: Model identifier
            include_basin: Whether to include basin-specific summary
            include_temporal: Whether to include temporal summary

        Returns:
            Dictionary with performance summary
        """
        if model_id not in self.performance_history:
            return {}

        records = self.performance_history[model_id]

        if not records:
            return {}

        # Overall summary
        summary = {
            "model_id": model_id,
            "n_records": len(records),
            "latest_performance": records[-1]["overall_metrics"],
            "first_record": records[0]["timestamp"],
            "latest_record": records[-1]["timestamp"],
        }

        # Calculate average performance
        all_metrics = {}
        for record in records:
            for metric_name, metric_value in record["overall_metrics"].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                if not np.isnan(metric_value):
                    all_metrics[metric_name].append(metric_value)

        summary["average_performance"] = {
            metric_name: np.mean(values) if values else np.nan
            for metric_name, values in all_metrics.items()
        }

        # Basin-specific summary
        if (
            include_basin
            and self.track_basin_specific
            and model_id in self.basin_performance
        ):
            basin_summary = {}
            for basin_code, basin_records in self.basin_performance[model_id].items():
                if basin_records:
                    latest_basin_metrics = basin_records[-1]["metrics"]
                    basin_summary[basin_code] = {
                        "n_records": len(basin_records),
                        "latest_nse": latest_basin_metrics.get("nse", np.nan),
                        "latest_rmse": latest_basin_metrics.get("rmse", np.nan),
                    }

            summary["basin_performance"] = basin_summary

        # Temporal summary
        if (
            include_temporal
            and self.track_temporal
            and model_id in self.temporal_performance
        ):
            temporal_summary = {}
            for month, month_records in self.temporal_performance[model_id].items():
                if month_records:
                    latest_month_metrics = month_records[-1]["metrics"]
                    temporal_summary[month] = {
                        "n_records": len(month_records),
                        "latest_nse": latest_month_metrics.get("nse", np.nan),
                        "latest_rmse": latest_month_metrics.get("rmse", np.nan),
                    }

            summary["temporal_performance"] = temporal_summary

        return summary

    def export_performance_data(self, output_path: str, format: str = "csv") -> None:
        """
        Export performance data to file.

        Args:
            output_path: Path to save the performance data
            format: Export format ('csv', 'json')
        """
        if format == "csv":
            # Export overall performance
            overall_data = []
            for model_id, records in self.performance_history.items():
                for record in records:
                    row = {
                        "model_id": model_id,
                        "timestamp": record["timestamp"],
                        "n_samples": record["n_samples"],
                        "n_basins": record["n_basins"],
                    }
                    row.update(record["overall_metrics"])
                    overall_data.append(row)

            overall_df = pd.DataFrame(overall_data)
            overall_df.to_csv(output_path, index=False)

        elif format == "json":
            import json

            export_data = {
                "performance_history": self.performance_history,
                "basin_performance": self.basin_performance,
                "temporal_performance": self.temporal_performance,
            }

            # Convert datetime objects to strings for JSON serialization
            def convert_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: convert_datetime(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_datetime(item) for item in obj]
                else:
                    return obj

            export_data = convert_datetime(export_data)

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        logger.info(f"Exported performance data to {output_path}")

    def clear_performance_history(self, model_id: str = None) -> None:
        """
        Clear performance history for a model or all models.

        Args:
            model_id: Model identifier to clear (if None, clear all)
        """
        if model_id is None:
            self.performance_history.clear()
            self.basin_performance.clear()
            self.temporal_performance.clear()
            logger.info("Cleared all performance history")
        else:
            if model_id in self.performance_history:
                del self.performance_history[model_id]
            if model_id in self.basin_performance:
                del self.basin_performance[model_id]
            if model_id in self.temporal_performance:
                del self.temporal_performance[model_id]
            logger.info(f"Cleared performance history for {model_id}")


def calculate_performance_weights(
    performance_data: Dict[str, float],
    weighting_method: str = "inverse_error",
    epsilon: float = 1e-10,
) -> Dict[str, float]:
    """
    Calculate weights based on performance metrics.

    Args:
        performance_data: Dictionary mapping model IDs to performance values
        weighting_method: Method for calculating weights
        epsilon: Small value to avoid division by zero

    Returns:
        Dictionary mapping model IDs to weights
    """
    if not performance_data:
        return {}

    # Filter out invalid values
    valid_performance = {
        model_id: perf
        for model_id, perf in performance_data.items()
        if not np.isnan(perf) and np.isfinite(perf)
    }

    if not valid_performance:
        # Return uniform weights if no valid performance data
        return {model_id: 1.0 for model_id in performance_data.keys()}

    weights = {}

    if weighting_method == "inverse_error":
        # Assume performance metric is an error metric (lower is better)
        inv_performance = {k: 1.0 / (v + epsilon) for k, v in valid_performance.items()}
        total_inv_perf = sum(inv_performance.values())

        for model_id in performance_data.keys():
            if model_id in inv_performance:
                weights[model_id] = inv_performance[model_id] / total_inv_perf
            else:
                weights[model_id] = 0.0

    elif weighting_method == "direct_skill":
        # Assume performance metric is a skill metric (higher is better)
        total_perf = sum(valid_performance.values())

        if total_perf > 0:
            for model_id in performance_data.keys():
                if model_id in valid_performance:
                    weights[model_id] = valid_performance[model_id] / total_perf
                else:
                    weights[model_id] = 0.0
        else:
            # All performances are zero or negative, use uniform weights
            weights = {model_id: 1.0 for model_id in performance_data.keys()}

    elif weighting_method == "softmax":
        # Softmax weighting
        performance_values = list(valid_performance.values())
        exp_values = np.exp(performance_values - np.max(performance_values))
        softmax_weights = exp_values / np.sum(exp_values)

        weight_mapping = dict(zip(valid_performance.keys(), softmax_weights))

        for model_id in performance_data.keys():
            weights[model_id] = weight_mapping.get(model_id, 0.0)

    else:
        # Uniform weights as fallback
        weights = {model_id: 1.0 for model_id in performance_data.keys()}

    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}

    return weights


def analyze_performance_stability(
    performance_tracker: PerformanceTracker,
    model_id: str,
    metric: str = "nse",
    window: int = 10,
) -> Dict[str, float]:
    """
    Analyze performance stability for a model.

    Args:
        performance_tracker: PerformanceTracker instance
        model_id: Model identifier
        metric: Metric to analyze
        window: Window size for stability analysis

    Returns:
        Dictionary with stability metrics
    """
    if model_id not in performance_tracker.performance_history:
        return {}

    records = performance_tracker.performance_history[model_id]

    if len(records) < window:
        return {}

    # Extract recent performance values
    recent_records = records[-window:]
    values = [
        record["overall_metrics"].get(metric, np.nan) for record in recent_records
    ]
    values = [v for v in values if not np.isnan(v)]

    if len(values) < 2:
        return {}

    # Calculate stability metrics
    stability_metrics = {
        "mean": np.mean(values),
        "std": np.std(values),
        "cv": np.std(values) / np.mean(values) if np.mean(values) != 0 else np.nan,
        "min": np.min(values),
        "max": np.max(values),
        "range": np.max(values) - np.min(values),
    }

    return stability_metrics
