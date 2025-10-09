import os
import sys
from typing import Dict, Any, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb


# Add the parent directory to sys.path to allow imports from monthly_forecasting
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from monthly_forecasting.scr import FeatureExtractor as FE
from monthly_forecasting.scr import data_loading as dl

# supress logging from matplotlib
import logging

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


data_config = {
    "path_discharge": "../../../data/discharge/digitized_kyrgyz_hydromet/kyrgyz_hydromet_discharge_daily_2000_2023_kgz_filtered_v2.csv",
    "path_forcing": "../../../data/forcing/ERA5_krg/HRU00003_forcing_2000-2023.csv",
    "path_static_data": "../../../GIS/ML_Sandro/ML_basin_attributes_v2.csv",
    "path_to_sla": "../../../data/sla_silvan/fsc_sla_timeseries_gapfilled.csv",
    "path_to_nir": "../../../data/sla_silvan/meanNIR_TS_allBasins.csv",
    "path_to_sca": None,
    "path_to_hru_shp": None,
    "path_to_swe": "../../../data/snow/kyrgyzstan_ts/SWE",
    "path_to_hs": "../../../data/snow/kyrgyzstan_ts/HS",
    "path_to_rof": "../../../data/snow/kyrgyzstan_ts/RoF",
    "HRU_SWE": "HRU_00003",
    "HRU_HS": "HRU_00003",
    "HRU_ROF": "HRU_00003",
    "model_home_path": "../../monthly_forecasting_models/GlacierMapper_Based",
}

feature_config = {
    "discharge": [
        {
            "operation": "mean",
            "windows": [
                15,
                30,
            ],
            "lags": {},
        },
    ],
    "P": [{"operation": "sum", "windows": [15, 30], "lags": {}}],
    "T": [{"operation": "mean", "windows": [3, 15, 30], "lags": {}}],
    "fsc_basin": [{"operation": "last_value", "windows": [30], "lags": {}}],
    "NIR": [
        {"operation": "last_value", "windows": [30], "lags": {}},
        {"operation": "mean", "windows": [30], "lags": {}},
    ],
    "SWE": [{"operation": "mean", "windows": [15, 30], "lags": {}}],
}


def load_data(data_config: Dict[str, Any], path_config: Dict[str, Any]) -> tuple:
    """
    Load data using the data loading utilities.

    Args:
        data_config: Data configuration
        path_config: Path configuration

    Returns:
        Tuple of (data, static_data)
    """
    # -------------- 1. Load Data ------------------------------
    hydro_ca, static_df = dl.load_data(
        path_discharge=path_config["path_discharge"],
        path_forcing=path_config["path_forcing"],
        path_static_data=path_config["path_static_data"],
        path_to_sca=path_config["path_to_sca"],
        path_to_swe=path_config["path_to_swe"],
        path_to_hs=path_config["path_to_hs"],
        path_to_rof=path_config["path_to_rof"],
        HRU_SWE=path_config["HRU_SWE"],
        HRU_HS=path_config["HRU_HS"],
        HRU_ROF=path_config["HRU_ROF"],
        path_to_sla=path_config.get("path_to_sla", None),
        path_to_nir=path_config.get("path_to_nir", None),
    )

    # if log_discharge in columns - drop
    if "log_discharge" in hydro_ca.columns:
        hydro_ca.drop(columns=["log_discharge"], inplace=True)

    hydro_ca = hydro_ca.sort_values("date")

    hydro_ca["code"] = hydro_ca["code"].astype(int)

    if "CODE" in static_df.columns:
        static_df.rename(columns={"CODE": "code"}, inplace=True)
    static_df["code"] = static_df["code"].astype(int)

    return hydro_ca, static_df


def main():
    save_dir = "../monthly_forecasting_results/figures/methods"
    os.makedirs(save_dir, exist_ok=True)
    selected_basin = 16936

    # Load the data
    hydro_ca, static_df = load_data(data_config, data_config)

    print("Data loaded successfully.")

    # get the dimensions of the data
    print(f"hydro_ca shape: {hydro_ca.shape}")

    # get the dtype of each column
    print(hydro_ca.dtypes)

    # print head
    print(hydro_ca.head())

    # Use FeatureExtractor for time series features
    extractor = FE.StreamflowFeatureExtractor(
        feature_configs=feature_config,
        prediction_horizon=30,
        offset=30,
    )

    data = extractor.create_all_features(hydro_ca)

    # only keep the last day of the month
    data = data[data["date"].dt.is_month_end]

    print("Features extracted successfully.")
    print(f"Feature data shape: {data.shape}")
    # columns
    print(f"Feature data columns: {data.columns.tolist()}")

    # shift the date + 30 days to represent the prediction date
    data["date"] = data["date"] + pd.DateOffset(days=30)

    data["month"] = data["date"].dt.month

    feature_1 = "month"
    feature_2 = "T_roll_mean_15"

    # rename the features
    data.rename(columns={feature_1: "Month", feature_2: "T"}, inplace=True)

    feature_1 = "Month"
    feature_2 = "T"

    data = data[data["code"] == selected_basin].copy()

    # Drop rows with missing values in target or features
    data_clean = data[["date", feature_1, feature_2, "target"]].dropna()

    print(f"\nClean data shape: {data_clean.shape}")
    print(
        f"Feature 1 ({feature_1}) range: {data_clean[feature_1].min():.2f} - {data_clean[feature_1].max():.2f}"
    )
    print(
        f"Feature 2 ({feature_2}) range: {data_clean[feature_2].min():.2f} - {data_clean[feature_2].max():.2f}"
    )
    print(
        f"Target range: {data_clean['target'].min():.2f} - {data_clean['target'].max():.2f}"
    )

    # ============= PART 1: Regression Tree with Feature 1 Only =============
    print("\n" + "=" * 60)
    print("PART 1: Regression Tree with Feature 1 (month) only")
    print("=" * 60)

    X_single = data_clean[[feature_1]].values
    y = data_clean["target"].values
    dates = data_clean["date"].values

    # Split data into train and validation sets (80/20 split)
    X_train_single, X_val_single, y_train, y_val, dates_train, dates_val = (
        train_test_split(
            X_single, y, dates, test_size=0.2, random_state=42, shuffle=False
        )
    )

    print(f"Training set size: {len(X_train_single)}")
    print(f"Validation set size: {len(X_val_single)}")

    # Create a lean regression tree for visualization
    tree_single = DecisionTreeRegressor(
        max_depth=4, random_state=42, min_samples_split=2, min_samples_leaf=2
    )
    tree_single.fit(X_train_single, y_train)

    # Predict on validation set
    y_pred_single = tree_single.predict(X_val_single)

    # Calculate metrics on validation set
    r2_single = r2_score(y_val, y_pred_single)
    mae_single = mean_absolute_error(y_val, y_pred_single)
    rmse_single = np.sqrt(mean_squared_error(y_val, y_pred_single))

    print(f"Single feature model (validation) R²: {r2_single:.3f}")
    print(f"Single feature model (validation) MAE: {mae_single:.3f}")
    print(f"Single feature model (validation) RMSE: {rmse_single:.3f}")

    # Visualize the tree and predictions for single feature
    visualize_tree_and_predictions(
        tree_single,
        X_val_single,
        y_val,
        y_pred_single,
        feature_names=[feature_1],
        save_path=os.path.join(save_dir, "tree_single_feature.png"),
        title_prefix="Single Feature (Month) - Validation Set",
        metrics={"R²": r2_single, "MAE": mae_single, "RMSE": rmse_single},
        dates=dates_val,
    )

    # ============= PART 2: Regression Tree with Both Features =============
    print("\n" + "=" * 60)
    print("PART 2: Regression Tree with Both Features")
    print("=" * 60)

    X_dual = data_clean[[feature_1, feature_2]].values

    # Split data into train and validation sets
    X_train_dual, X_val_dual, _, _ = train_test_split(
        X_dual, y, test_size=0.2, random_state=42, shuffle=False
    )

    # Create regression tree with both features
    tree_dual = DecisionTreeRegressor(
        max_depth=4, random_state=42, min_samples_split=2, min_samples_leaf=2
    )
    tree_dual.fit(X_train_dual, y_train)

    # Predict on validation set
    y_pred_dual = tree_dual.predict(X_val_dual)

    # Calculate metrics on validation set
    r2_dual = r2_score(y_val, y_pred_dual)
    mae_dual = mean_absolute_error(y_val, y_pred_dual)
    rmse_dual = np.sqrt(mean_squared_error(y_val, y_pred_dual))

    print(f"Dual feature model (validation) R²: {r2_dual:.3f}")
    print(f"Dual feature model (validation) MAE: {mae_dual:.3f}")
    print(f"Dual feature model (validation) RMSE: {rmse_dual:.3f}")

    # Visualize the tree and predictions for dual features
    visualize_tree_and_predictions(
        tree_dual,
        X_val_dual,
        y_val,
        y_pred_dual,
        feature_names=[feature_1, feature_2],
        save_path=os.path.join(save_dir, "tree_dual_features.png"),
        title_prefix="Dual Features (Month + SWE) - Validation Set",
        metrics={"R²": r2_dual, "MAE": mae_dual, "RMSE": rmse_dual},
        dates=dates_val,
    )

    # ============= PART 3: Comparison Plots =============
    print("\n" + "=" * 60)
    print("PART 3: Creating comparison plots")
    print("=" * 60)

    create_comparison_plots(
        X_val_single,
        X_val_dual,
        y_val,
        y_pred_single,
        y_pred_dual,
        feature_1,
        feature_2,
        save_path=os.path.join(save_dir, "tree_comparison.png"),
        metrics_single={"R²": r2_single, "MAE": mae_single, "RMSE": rmse_single},
        metrics_dual={"R²": r2_dual, "MAE": mae_dual, "RMSE": rmse_dual},
        dates=dates_val,
    )

    # ============= PART 4: XGBoost Gradient Boosting =============
    print("\n" + "=" * 60)
    print("PART 4: XGBoost Gradient Boosting Visualization")
    print("=" * 60)

    # Train XGBoost model with few boosting rounds for visualization
    xgb_model = xgb.XGBRegressor(
        n_estimators=5,  # Few boosting rounds to visualize the process
        max_depth=3,
        learning_rate=0.3,
        random_state=42,
        min_child_weight=5,
    )
    xgb_model.fit(X_train_dual, y_train)
    y_pred_xgb = xgb_model.predict(X_val_dual)

    # Calculate metrics
    r2_xgb = r2_score(y_val, y_pred_xgb)
    mae_xgb = mean_absolute_error(y_val, y_pred_xgb)
    rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))

    print(f"XGBoost model (validation) R²: {r2_xgb:.3f}")
    print(f"XGBoost model (validation) MAE: {mae_xgb:.3f}")
    print(f"XGBoost model (validation) RMSE: {rmse_xgb:.3f}")

    # Visualize how XGBoost works iteratively
    visualize_xgboost_boosting_process(
        xgb_model,
        X_train_dual,
        y_train,
        X_val_dual,
        y_val,
        feature_names=[feature_1, feature_2],
        save_path=os.path.join(save_dir, "xgboost_boosting_process.png"),
        dates=dates_val,
    )

    print(f"\nAll visualizations saved to: {save_dir}")


def visualize_tree_and_predictions(
    tree_model: DecisionTreeRegressor,
    X: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_names: list[str],
    save_path: str,
    title_prefix: str = "",
    metrics: dict[str, float] | None = None,
    dates: np.ndarray | None = None,
) -> None:
    """
    Create a visualization showing a single regression tree and its predictions.

    Args:
        tree_model: Trained DecisionTreeRegressor
        X: Feature matrix
        y_true: True target values
        y_pred: Predicted target values
        feature_names: List of feature names
        save_path: Path to save the figure
        title_prefix: Prefix for plot titles
        metrics: Dictionary of metrics to display
        dates: Optional array of dates for x-axis
    """
    # Create figure with subplots - tree on top (2/3), time series below (1/3)
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1], hspace=0.35)

    # Plot the regression tree (simplified: only show decision variables)
    ax_tree = fig.add_subplot(gs[0])
    plot_tree(
        tree_model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        fontsize=11,
        ax=ax_tree,
        impurity=False,  # Don't show error/impurity
        proportion=False,  # Don't show percentage of data
    )
    ax_tree.set_title(
        f"Regression Tree Structure", fontsize=14, fontweight="bold", pad=10
    )

    # Plot time series: predictions vs observations
    ax_pred = fig.add_subplot(gs[1])

    # Use dates for x-axis if provided, otherwise use index
    if dates is not None:
        x_vals = pd.to_datetime(dates)
        ax_pred.plot(x_vals, y_true, "k-", linewidth=2, label="Observed", alpha=0.8)
        ax_pred.plot(x_vals, y_pred, "r-", linewidth=2.5, label="Predicted", alpha=0.9)
        ax_pred.set_xlabel("Date", fontsize=13, fontweight="bold")
        # Rotate x-axis labels for better readability
        plt.setp(ax_pred.xaxis.get_majorticklabels(), rotation=45, ha="right")
    else:
        # Fallback to index-based plotting if no dates provided
        ax_pred.plot(
            range(len(y_true)), y_true, "k-", linewidth=2, label="Observed", alpha=0.8
        )
        ax_pred.plot(
            range(len(y_pred)),
            y_pred,
            "r-",
            linewidth=2.5,
            label="Predicted",
            alpha=0.9,
        )
        ax_pred.set_xlabel("Sample Index", fontsize=13, fontweight="bold")

    ax_pred.set_ylabel("Discharge", fontsize=13, fontweight="bold")
    ax_pred.set_title(f"Predictions vs Observations", fontsize=14, fontweight="bold")
    ax_pred.legend(fontsize=12, loc="best", framealpha=0.9)
    ax_pred.grid(True, alpha=0.3, linestyle="--")

    # Add metrics text if provided
    if metrics:
        metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        ax_pred.text(
            0.45,
            0.98,
            metrics_text,
            transform=ax_pred.transAxes,
            fontsize=11,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
        )

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved tree visualization to: {save_path}")


def create_comparison_plots(
    X_single: np.ndarray,
    X_dual: np.ndarray,
    y_true: np.ndarray,
    y_pred_single: np.ndarray,
    y_pred_dual: np.ndarray,
    feature_1: str,
    feature_2: str,
    save_path: str,
    metrics_single: dict[str, float],
    metrics_dual: dict[str, float],
    dates: np.ndarray | None = None,
) -> None:
    """
    Create side-by-side comparison of single and dual feature models (time series only).

    Args:
        X_single: Single feature matrix
        X_dual: Dual feature matrix
        y_true: True target values
        y_pred_single: Predictions from single feature model
        y_pred_dual: Predictions from dual feature model
        feature_1: Name of first feature
        feature_2: Name of second feature
        save_path: Path to save the figure
        metrics_single: Metrics for single feature model
        metrics_dual: Metrics for dual feature model
        dates: Optional array of dates for x-axis
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    plt.style.use("seaborn-v0_8-darkgrid")

    # Prepare x-axis values
    if dates is not None:
        x_vals = pd.to_datetime(dates)
        xlabel = "Date"
    else:
        x_vals = range(len(y_true))
        xlabel = "Sample Index"

    # Single feature - time series
    axes[0].plot(x_vals, y_true, "k-", linewidth=2, label="Observed", alpha=0.8)
    axes[0].plot(
        x_vals, y_pred_single, "r-", linewidth=2.5, label="Predicted", alpha=0.9
    )
    axes[0].set_xlabel(xlabel, fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Discharge", fontsize=12, fontweight="bold")
    axes[0].set_title(
        "Single Feature Model - Predictions", fontsize=13, fontweight="bold"
    )
    axes[0].legend(fontsize=11, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle="--")
    if dates is not None:
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add metrics
    metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_single.items()])
    axes[0].text(
        0.02,
        0.98,
        metrics_text,
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.7),
    )

    # Dual features - time series
    axes[1].plot(x_vals, y_true, "k-", linewidth=2, label="Observed", alpha=0.8)
    axes[1].plot(x_vals, y_pred_dual, "r-", linewidth=2.5, label="Predicted", alpha=0.9)
    axes[1].set_xlabel(xlabel, fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Discharge", fontsize=12, fontweight="bold")
    axes[1].set_title(
        "Dual Features Model - Predictions", fontsize=13, fontweight="bold"
    )
    axes[1].legend(fontsize=11, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle="--")
    if dates is not None:
        plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add metrics
    metrics_text = "\n".join([f"{k}: {v:.3f}" for k, v in metrics_dual.items()])
    axes[1].text(
        0.02,
        0.98,
        metrics_text,
        transform=axes[1].transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    plt.suptitle(
        f"Model Comparison: Single ({feature_1}) vs Dual ({feature_1} + {feature_2})",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison visualization to: {save_path}")


def visualize_xgboost_boosting_process(
    xgb_model: xgb.XGBRegressor,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
    save_path: str,
    dates: np.ndarray | None = None,
) -> None:
    """
    Visualize how XGBoost gradient boosting works iteratively.

    Shows the progressive improvement as boosting rounds are added,
    demonstrating the ensemble learning concept at a high level.

    Args:
        xgb_model: Trained XGBoost model
        X_train: Training features
        y_train: Training targets
        X_val: Validation features
        y_val: Validation targets
        feature_names: List of feature names
        save_path: Path to save the figure
        dates: Optional array of dates for x-axis
    """
    n_estimators = xgb_model.n_estimators

    # Create figure with subplots showing progressive predictions
    fig = plt.figure(figsize=(20, 12))

    # Calculate number of rows needed (show up to 6 stages)
    n_stages = min(n_estimators + 1, 6)  # +1 for initial prediction
    n_cols = 2
    n_rows = (n_stages + 1) // 2

    plt.style.use("seaborn-v0_8-darkgrid")

    # Prepare x-axis values
    if dates is not None:
        x_vals = pd.to_datetime(dates)
        xlabel = "Date"
    else:
        x_vals = range(len(y_val))
        xlabel = "Sample Index"

    # Track predictions at each stage
    cumulative_predictions = []
    stage_indices = []

    # Get predictions at each boosting stage
    if n_estimators > 0:
        # XGBoost provides iteration_range parameter for staged predictions
        stage_step = max(1, n_estimators // (n_stages - 1))

        for i in range(0, n_estimators + 1, stage_step):
            if i == 0:
                # Initial prediction (mean)
                pred = np.full(len(X_val), y_train.mean())
            else:
                # Prediction up to iteration i
                xgb_model_staged = xgb.XGBRegressor(
                    n_estimators=i,
                    max_depth=xgb_model.max_depth,
                    learning_rate=xgb_model.learning_rate,
                    random_state=xgb_model.random_state,
                    min_child_weight=xgb_model.min_child_weight,
                )
                xgb_model_staged.fit(X_train, y_train, verbose=False)
                pred = xgb_model_staged.predict(X_val)

            cumulative_predictions.append(pred)
            stage_indices.append(i)

            if len(cumulative_predictions) >= n_stages:
                break

    # Plot each stage
    for idx, (stage_num, pred) in enumerate(zip(stage_indices, cumulative_predictions)):
        row = idx // n_cols
        col = idx % n_cols
        ax = plt.subplot(n_rows, n_cols, idx + 1)

        # Plot observed and predicted
        ax.plot(x_vals, y_val, "k-", linewidth=2, label="Observed", alpha=0.7)
        ax.plot(
            x_vals,
            pred,
            "r-",
            linewidth=2.5,
            label=f"Predicted (n_trees={stage_num})",
            alpha=0.9,
        )

        # Calculate metrics for this stage
        r2 = r2_score(y_val, pred)
        mae = mean_absolute_error(y_val, pred)

        # Set labels and title
        ax.set_xlabel(xlabel, fontsize=10, fontweight="bold")
        ax.set_ylabel("Discharge", fontsize=10, fontweight="bold")

        if stage_num == 0:
            ax.set_title(
                f"Stage 0: Initial Prediction (Mean)\nR²={r2:.3f}, MAE={mae:.1f}",
                fontsize=11,
                fontweight="bold",
            )
        else:
            ax.set_title(
                f"Stage: {stage_num} Trees\nR²={r2:.3f}, MAE={mae:.1f}",
                fontsize=11,
                fontweight="bold",
            )

        ax.legend(fontsize=9, loc="best", framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        if dates is not None:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add overall title and explanation
    plt.suptitle(
        "XGBoost Gradient Boosting Process: Progressive Improvement\n"
        + "Each stage shows how adding more trees iteratively improves predictions",
        fontsize=14,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved XGBoost boosting process visualization to: {save_path}")


if __name__ == "__main__":
    main()
