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
from lt_forecasting.log_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Suppress matplotlib debug logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)


class MixtureModel:
    # Class-level distribution generators (created once)
    _ALD_DIST = None
    _NORM_DIST = None

    def __init__(self, distribution_type: str = "ALD"):
        self.distribution_type = distribution_type
        self._dist_cache = {}  # Cache for distribution objects

        # Initialize distribution generators once at class level
        if MixtureModel._ALD_DIST is None:
            MixtureModel._ALD_DIST = stats.make_distribution(stats.laplace_asymmetric)
        if MixtureModel._NORM_DIST is None:
            MixtureModel._NORM_DIST = stats.make_distribution(stats.norm)

    def __get_distribution_fn__(self, params):
        # Create a hashable cache key from parameters
        if self.distribution_type == "ALD":
            cache_key = ("ALD", params["loc"], params["scale"], params["asymmetry"])
        elif self.distribution_type == "Gaussian":
            cache_key = ("Gaussian", params["mu"], params["sigma"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

        # Check cache first
        if cache_key in self._dist_cache:
            return self._dist_cache[cache_key]

        # Create distribution if not cached
        if self.distribution_type == "ALD":
            tau = params["asymmetry"]
            kappa = np.sqrt(tau / (1.0 - tau))
            scale = params["scale"] / np.sqrt(tau * (1.0 - tau))
            base = self._ALD_DIST(kappa=kappa)
            dist = scale * base + params["loc"]
        elif self.distribution_type == "Gaussian":
            base = self._NORM_DIST()
            dist = params["sigma"] * base + params["mu"]

        # Cache the distribution
        self._dist_cache[cache_key] = dist
        return dist

    def __create_mixture__(self, parameter_dict: Dict[str, Dict[str, np.float_]]):
        components = []
        weights = []
        names = []

        for key, params in parameter_dict.items():
            # Skip if any parameter is NaN
            if any(np.isnan(v) for v in params.values()):
                continue
            dist = self.__get_distribution_fn__(params)
            components.append(dist)
            weights.append(params.get("weight", 1.0))
            names.append(key)

        # Normalize weights to sum to 1
        weights = np.array(weights)
        weights = weights / weights.sum()

        self.mixture = stats.Mixture(components, weights=weights.tolist())
        self.names = names
        self.components = components
        self.weights = weights.tolist()

    def get_statistic(
        self, parameter_dict: Dict[str, Dict[str, float]], quantiles: List[float]
    ) -> Dict[str, float]:
        self.__create_mixture__(parameter_dict)

        mean = self.mixture.mean()

        # Vectorized quantile calculation - compute all at once
        quantiles_array = np.array(quantiles)
        quantile_values_array = self.mixture.icdf(quantiles_array)

        # Build result dictionary
        result = {"mean": mean}
        for q, val in zip(quantiles, quantile_values_array):
            result[f"Q{int(q * 100)}"] = val

        # Optional validation check for Q50
        if "Q50" in result:
            median = self.mixture.median()
            ratio = result["Q50"] / median
            epsilon = 1e-4
            if abs(ratio - 1) > epsilon:
                logger.warning(
                    f"Warning: Q50/Median ratio is {ratio:.4f}, which is outside the expected range. "
                    f"This indicates a wrong quantile calculation."
                )

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

    def clear_cache(self):
        """Clear the distribution cache to free memory if needed."""
        self._dist_cache.clear()


class FastMixtureModel:
    def __init__(self, distribution_type: str = "ALD"):
        self.distribution_type = distribution_type
        self._dist_cache = {}

        # Pre-compile the distribution classes once
        if distribution_type == "ALD":
            from scipy.stats import laplace_asymmetric

            self._base_dist = laplace_asymmetric
        else:
            from scipy.stats import norm

            self._base_dist = norm

    def get_statistic_fast(
        self, parameter_dict: Dict[str, Dict[str, float]], quantiles: List[float]
    ) -> Dict[str, float]:
        """
        Fast version that computes quantiles directly without creating mixture object
        """
        # Collect valid components
        valid_params = []
        weights = []

        for key, params in parameter_dict.items():
            if any(np.isnan(v) for v in params.values()):
                continue
            valid_params.append(params)
            weights.append(params.get("weight", 1.0))

        if not valid_params:
            return {f"Q{int(q * 100)}": np.nan for q in quantiles} | {"mean": np.nan}

        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()

        # Convert quantiles to array
        quantiles_array = np.array(quantiles)
        n_quantiles = len(quantiles)
        n_components = len(valid_params)

        # Pre-allocate result array
        component_quantiles = np.zeros((n_components, n_quantiles))
        component_means = np.zeros(n_components)

        # Compute quantiles for each component
        for i, params in enumerate(valid_params):
            if self.distribution_type == "ALD":
                tau = params["asymmetry"]
                kappa = np.sqrt(tau / (1.0 - tau))
                scale = params["scale"] / np.sqrt(tau * (1.0 - tau))

                # Direct computation without creating distribution object
                component_quantiles[i] = self._base_dist.ppf(
                    quantiles_array, kappa=kappa, loc=params["loc"], scale=scale
                )
                component_means[i] = params["loc"]  # For ALD, mean = loc

            else:  # Gaussian
                component_quantiles[i] = self._base_dist.ppf(
                    quantiles_array, loc=params["mu"], scale=params["sigma"]
                )
                component_means[i] = params["mu"]

        # Weighted average of quantiles (approximation for mixture)
        mixed_quantiles = np.average(component_quantiles, weights=weights, axis=0)
        mixed_mean = np.average(component_means, weights=weights)

        # Build result
        result = {"mean": mixed_mean}
        for q, val in zip(quantiles, mixed_quantiles):
            result[f"Q{int(q * 100)}"] = val

        return result


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
