#!/usr/bin/env python
"""Setup script for lt-forecasting package."""

from setuptools import setup, find_packages
import os
import re


# Read version from __init__.py
def get_version():
    """Extract version from __init__.py"""
    init_file = os.path.join("lt_forecasting", "__init__.py")
    with open(init_file, "r") as f:
        content = f.read()
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', content)
        if match:
            return match.group(1)
        else:
            raise ValueError("Version not found in __init__.py")


# Read requirements from pyproject.toml dependencies
def get_requirements():
    """Extract dependencies from pyproject.toml"""
    return [
        "catboost==1.2.3",
        "dash>=2.14.0",
        "dash-bootstrap-components>=1.7.1",
        "fiona==1.9.5",
        "geopandas==0.14.3",
        "lightgbm==4.5.0",
        "matplotlib==3.8.2",
        "numpy==1.26.4",
        "optuna==3.5.0",
        "pandas==2.2.0",
        "pe-oudin==0.3",
        "plotly>=5.18.0",
        "pyarrow==15.0.0",
        "scikit-learn==1.4.0",
        "scipy==1.15.2",
        "seaborn==0.13.2",
        "tqdm==4.66.2",
        "xgboost==2.0.3",
    ]


# Read long description from README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="lt-forecasting",
    version=get_version(),
    author="Sandro Hunziker",
    author_email="hunziker@hydrosolutions.ch",
    description="A machine learning package for long-term hydrological forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hydrosolutions/lt_forecasting",
    packages=find_packages(
        exclude=[
            "dev_tools",
            "dev_tools.*",
            "tests",
            "tests.*",
            "scripts",
            "scripts.*",
            "notebooks",
            "notebooks.*",
            "docs",
            "docs.*",
            "old_files",
            "old_files.*",
            "example_config",
            "example_config.*",
            "scratchpads",
            "scratchpads.*",
            "catboost_info",
            "catboost_info.*",
            "logs",
            "logs.*",
            "tests_output",
            "tests_output.*",
        ]
    ),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: OSI Approved :: MIT License",  # Update with actual license
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.11",
    install_requires=get_requirements(),
    extras_require={
        "dev": [
            "pytest==8.4.1",
            "ruff>=0.12.2",
        ],
    },
    entry_points={
        # Add any command-line scripts here if needed
    },
    include_package_data=True,
    zip_safe=False,
)
