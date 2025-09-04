import os
import pandas as pd
import numpy as np
import datetime
import json
import warnings
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm as progress_bar
from scipy import stats
import matplotlib.pyplot as plt

# Shared logging
import logging
from monthly_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

warnings.filterwarnings("ignore")


class MixtureModel:
    def __init__(self, distribution_type: str = "ALD"):
        self.distribution_type = distribution_type

    def __get_distribution_fn__(self, params):
        if self.distribution_type == "ALD":
            # 1) New-style class from the generator
            ALD = stats.make_distribution(stats.laplace_asymmetric)
            # 2) Instantiate with shape-only (kappa)
            base = ALD(kappa=params["asymmetry"])
            # 3) Apply scale/loc as affine transforms (this "freezes" it)
            dist = params["scale"] * base + params["loc"]

        elif self.distribution_type == "Gaussian":
            # Uniform approach across SciPy versions:
            Normal = stats.make_distribution(stats.norm)
            base = Normal()  # no shape params for Normal
            dist = params["sigma"] * base + params["mu"]

            # (If you're on a recent SciPy, this also works:
            # from scipy.stats import Normal
            # dist = Normal(mu=params["mu"], sigma=params["sigma"])
            # but keeping the affine style makes ALD & Gaussian consistent.)

        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        return dist  # <- new-style ContinuousDistribution, ready for Mixture

    def __create_mixture__(self, parameter_dict: Dict[str, Dict[str, np.float_]]):
        components = []
        weights = []
        names = []

        for key, params in parameter_dict.items():
            # if any item in params is nan continue
            if any(np.isnan(v) for v in params.values()):
                continue
            dist = self.__get_distribution_fn__(params)
            components.append(dist)
            weights.append(params.get("weight", 1.0))
            names.append(key)

        weights = np.array(weights) / np.sum(
            weights
        )  # Normalize weights to sum up to 1
        weights = weights.tolist()

        self.mixture = stats.Mixture(components, weights=weights)
        self.names = names
        self.components = components
        self.weights = weights

    def get_statistic(
        self, parameter_dict: Dict[str, Dict[str, float]], quantiles: List[float]
    ) -> Dict[str, float]:
        self.__create_mixture__(parameter_dict)

        mean = self.mixture.mean()

        # Compute quantiles using inverse CDF (icdf)
        quantile_values = {}
        for q in quantiles:
            quantile_values[f"Q{int(q * 100)}"] = self.mixture.icdf(q)

        if "Q50" in quantile_values:
            median = self.mixture.median()
            ratio = quantile_values["Q50"] / median
            epsilon = 1e-4
            if abs(ratio - 1) > epsilon:
                logger.warning(
                    f"Warning: Q50/Median ratio is {ratio:.4f}, which is outside the expected range. This indicates a wrong quantile calculation."
                )

        # Combine mean and quantiles
        result = {"mean": mean}
        result.update(quantile_values)

        return result

    def plot_distributions(
        self,
        x_range: Tuple[float, float] = None,
        n_points: int = 1000,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """
        Simple plot showing individual components and mixture distribution.

        Args:
            x_range: Tuple of (min, max) values for x-axis. If None, uses data range.
            n_points: Number of points to evaluate distributions at.
            figsize: Figure size as (width, height).

        Returns:
            matplotlib Figure object
        """
        if not hasattr(self, "mixture"):
            raise ValueError("Mixture not created yet. Call get_statistic() first.")

        # Set up x values
        if x_range is None:
            means = [comp.mean() for comp in self.components]
            stds = [comp.standard_deviation() for comp in self.components]
            x_min = min(means) - 3 * max(stds)
            x_max = max(means) + 3 * max(stds)
        else:
            x_min, x_max = x_range

        x = np.linspace(x_min, x_max, n_points)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot individual components (weighted)
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.components)))

        for i, (comp, weight, name, color) in enumerate(
            zip(self.components, self.weights, self.names, colors)
        ):
            comp_pdf = comp.pdf(x)
            weighted_pdf = comp_pdf * weight

            ax.plot(
                x,
                weighted_pdf,
                "--",
                color=color,
                alpha=0.7,
                label=f"{name} (w={weight:.3f})",
                linewidth=2,
            )

        # Plot mixture distribution
        mixture_pdf = self.mixture.pdf(x)
        ax.plot(
            x, mixture_pdf, "k-", linewidth=3, label="Mixture Distribution", alpha=0.9
        )

        ax.set_title(
            "Component Distributions vs Mixture Distribution",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_xlabel("Value", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig


if __name__ == "__main__":
    # Create dummy asymmetric Laplace distributions
    parameter_dict = {
        "Component_1": {
            "asymmetry": 0.5,  # Left-skewed
            "loc": -2.0,
            "scale": 1.0,
            "weight": 0.3,
        },
        "Component_2": {
            "asymmetry": 2.0,  # Right-skewed
            "loc": 0.0,
            "scale": 0.8,
            "weight": 0.25,
        },
        "Component_3": {
            "asymmetry": 1.0,  # Symmetric
            "loc": 1.5,
            "scale": 1.2,
            "weight": 0.2,
        },
        "Component_4": {
            "asymmetry": 0.3,  # Strongly left-skewed
            "loc": -1.0,
            "scale": 0.6,
            "weight": 0.15,
        },
        "Component_5": {
            "asymmetry": 3.0,  # Strongly right-skewed
            "loc": 3.0,
            "scale": 1.5,
            "weight": 0.1,
        },
    }

    import time

    start_time = time.time()
    # Create mixture model
    model = MixtureModel(distribution_type="ALD")

    # Define quantiles to compute
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95]

    # Calculate statistics
    print("Calculating mixture statistics...")
    stats_result = model.get_statistic(parameter_dict, quantiles)
    end_time = time.time()

    print(
        f"time it took for statistics calculation: {end_time - start_time:.4f} seconds"
    )

    print("\nMixture Statistics:")
    print("=" * 50)
    for key, value in stats_result.items():
        print(f"{key:>8}: {value:.4f}")

    # Create and display plots
    print("\nCreating plots...")

    # Detailed plot
    fig_detailed = model.plot_distributions()
    plt.figure(fig_detailed.number)
    plt.suptitle(
        "Mixture Model Analysis: Components vs Mixture", fontsize=16, fontweight="bold"
    )

    # Show plots
    plt.show()

    print("\nDemo completed successfully!")
    print("Check the generated PNG files for visualizations.")
