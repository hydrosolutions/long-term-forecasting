"""Tests for static plotting utilities in create_plots.py.

These tests only verify that plot functions execute without raising errors on a
small synthetic dataset. They do not validate visual correctness.
"""

from pathlib import Path
import pandas as pd
import numpy as np

from dev_tools.visualization.create_plots import (
    plot_bar_mean_std,
    plot_box_distribution,
)
import matplotlib.pyplot as plt


def _synthetic_df() -> pd.DataFrame:
    models = ["A", "B", "C"]
    codes = [f"{100+i}" for i in range(8)]
    rows = []
    rng = np.random.default_rng(42)
    for m in models:
        for c in codes:
            rows.append(
                {
                    "model": m,
                    "model_family": "Test",
                    "code": c,
                    "month": -1,  # per-code overall
                    "nse": rng.normal(loc=0.7, scale=0.05),
                }
            )
    return pd.DataFrame(rows)


def test_plot_bar_mean_std(tmp_path: Path):
    df = _synthetic_df()
    fig, ax = plt.subplots()
    ax = plot_bar_mean_std(df, metric="nse", models=["A", "B", "C"], ax=ax)
    out = tmp_path / "bar.png"
    fig.savefig(out)
    assert out.exists()
    plt.close(fig)


def test_plot_box_distribution(tmp_path: Path):
    df = _synthetic_df()
    fig, ax = plt.subplots()
    ax = plot_box_distribution(df, metric="nse", models=["A", "B", "C"], ax=ax)
    out = tmp_path / "box.png"
    fig.savefig(out)
    assert out.exists()
    plt.close(fig)
