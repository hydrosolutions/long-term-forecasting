# Development and Production Workflow

## Overview

This document outlines the development workflow for the monthly discharge forecasting package, covering both development iterations and production deployment strategies.

## Project Philosophy

The forecasting package is designed with clear separation between:

1. **Production Code** (`lt_forecasting/`): Core functionality for operational forecasting
2. **Development Tools** (`dev_tools/`): Evaluation, visualization, and analysis tools
3. **Tests** (`tests/`): Comprehensive test suite ensuring code quality

This separation enables:
- Lightweight production deployments (only core package)
- Rich development environment with analysis tools
- Clear dependency management
- Stable version control and rollback capabilities

## Project Structure

### Production Package (`lt_forecasting/`)

```
lt_forecasting/
├── __init__.py                 # Package initialization with version info
├── forecast_models/            # Model implementations
│   ├── base_class.py           # Abstract base class
│   ├── LINEAR_REGRESSION.py    # Linear regression baseline
│   ├── SciRegressor.py         # Tree-based models (XGB, LGBM, CatBoost)
│   └── deep_models/            # Deep learning models
│       └── uncertainty_mixture.py  # MDN meta-learner
├── scr/                        # Data processing and feature engineering
│   ├── FeatureExtractor.py     # Time-series feature extraction
│   ├── FeatureProcessingArtifacts.py  # Preprocessing pipeline
│   ├── data_loading.py         # Data ingestion
│   └── data_utils.py           # Utility functions
└── log_config.py               # Logging configuration
```

**Purpose**: Minimal, stable codebase for operational forecasting.

### Development Tools (`dev_tools/`)

```
dev_tools/
├── evaluation/                 # Evaluation pipeline (excluded from production)
│   ├── evaluate_pipeline.py    # Main evaluation orchestrator
│   ├── ensemble_builder.py     # Ensemble creation
│   └── prediction_loader.py    # Load and process predictions
├── visualization/              # Interactive dashboard (excluded from production)
│   ├── dashboard.py            # Streamlit dashboard
│   └── plotting_utils.py       # Visualization functions
└── eval_scr/                   # Evaluation utilities (excluded from production)
    ├── metric_functions.py     # NSE, KGE, R², etc.
    └── eval_helper.py          # Helper functions
```

**Purpose**: Rich tooling for model development, evaluation, and analysis.

### Test Suite (`tests/`)

```
tests/
├── unit/                       # Unit tests (test individual functions)
├── functionality/              # Functionality tests (test workflows)
└── integration/                # Integration tests (test with mock data)
```

**Purpose**: Ensure code quality and catch regressions.

## Package Configuration

### Version Management

**Location**: `lt_forecasting/__init__.py`

```python
__version__ = "0.1.0"
```

See [VERSION_MANAGEMENT.md](VERSION_MANAGEMENT.md) for detailed version control strategy.

### Dependencies (`pyproject.toml`)

The package uses `uv` for dependency management:

**Production Dependencies**:
```toml
[project]
name = "lt_forecasting"
version = "0.1.0"
requires-python = ">=3.11"

dependencies = [
    "numpy>=1.24.0",
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "xgboost>=2.0.0",
    "lightgbm>=4.0.0",
    "catboost>=1.2.0",
    "optuna>=3.0.0",
    "joblib>=1.3.0",
]
```

**Development Dependencies**:
```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "ruff>=0.1.0",
    "streamlit>=1.28.0",
    "plotly>=5.17.0",
    "matplotlib>=3.7.0",
]
```

**Installation**:
```bash
# Production environment
uv sync

# Development environment
uv sync --extra dev
```

## Development Workflow

### Phase 1: Feature Development

**Typical development cycle on main branch**:

```bash
# 1. Ensure environment is up to date
uv sync --extra dev

# 2. Create feature branch (optional for complex features)
git checkout -b feature/improve-feature-engineering

# 3. Make changes to code
# Edit files in lt_forecasting/, dev_tools/, or tests/

# 4. Format code
uv run ruff format

# 5. Run tests
uv run pytest tests/ -v

# 6. Test specific functionality
uv run python scripts/calibrate_hindcast.py --config_path example_config/DUMMY_MODEL

# 7. Commit changes
git add .
git commit -m "feat: improve feature engineering for snow variables"

# 8. Push to remote
git push origin feature/improve-feature-engineering
```

### Phase 2: Testing and Validation

**Run comprehensive tests**:

```bash
# Unit tests
uv run pytest tests/unit/ -v

# Functionality tests
uv run pytest tests/functionality/ -v

# Integration tests (if applicable)
uv run pytest tests/integration/ -v

# Code quality checks
uv run ruff check
```

**Manual validation**:

```bash
# Test calibration
uv run python scripts/calibrate_hindcast.py --config_path example_config/DUMMY_MODEL

# Test hyperparameter tuning
uv run python scripts/tune_hyperparams.py --config_path example_config/DUMMY_MODEL

# Launch dashboard for visual inspection
uv run python -m dev_tools.visualization.dashboard
```

### Phase 3: Release Process

**Version release workflow**:

```bash
# 1. Update version number
# Edit lt_forecasting/__init__.py
__version__ = "0.2.0"

# 2. Commit version bump
git add lt_forecasting/__init__.py
git commit -m "chore: bump version to 0.2.0"
git push origin main

# 3. Create git tag
git tag -a v0.2.0 -m "Release version 0.2.0"
git push origin v0.2.0

# 4. Create release notes (GitHub/GitLab)
# Document changes, new features, bug fixes
```

**Semantic Versioning**:
- **Major (X.0.0)**: Breaking API changes
- **Minor (0.X.0)**: New features, backward compatible
- **Patch (0.0.X)**: Bug fixes, no breaking changes

Example: `v0.1.0` → `v0.2.0` (new feature) → `v0.2.1` (bug fix)

## Production Deployment

### Deployment Options

#### Option 1: Direct Installation

For production environments, install only the core package:

```bash
# Clone repository
git clone https://github.com/your-org/monthly_forecasting.git
cd monthly_forecasting

# Checkout specific version
git checkout v0.2.0

# Install production dependencies only
uv sync

# Verify installation
uv run python -c "from lt_forecasting import __version__; print(__version__)"
```

#### Option 2: Package Installation

Create distributable package:

```bash
# Build package
uv build

# Install from wheel
uv pip install dist/lt_forecasting-0.2.0-py3-none-any.whl
```

### Production Best Practices

**Environment Separation**:
```bash
# Production: minimal dependencies
uv sync

# Development: full tooling
uv sync --extra dev
```

**Configuration Management**:
- Store configs outside codebase (`/etc/forecasting/` or cloud storage)
- Use environment variables for sensitive data
- Version control config templates only

**Model Artifacts**:
- Store trained models separately from code
- Use cloud storage (S3, Azure Blob) for large models
- Maintain model version alongside code version

**Logging and Monitoring**:
```python
# Production logging configuration
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/forecasting/app.log'),
        logging.StreamHandler()
    ]
)
```

## Common Development Tasks

### Adding a New Model

See [how_to_add_model.md](how_to_add_model.md) for detailed guide.

**Quick steps**:
1. Create `lt_forecasting/forecast_models/MY_MODEL.py`
2. Inherit from `BaseForecastModel`
3. Implement required abstract methods
4. Create configuration in `example_config/MY_MODEL/`
5. Write tests in `tests/unit/test_my_model.py`
6. Update documentation

### Updating Feature Engineering

**Steps**:
1. Modify `lt_forecasting/scr/FeatureExtractor.py`
2. Update `feature_config.json` examples
3. Test with existing models
4. Run full evaluation pipeline
5. Update `docs/feature_engineering.md`

### Tuning Hyperparameters

```bash
# Tune single model
uv run python scripts/tune_hyperparams.py --config_path example_config/XGB_MODEL

# Tune multiple models (parallel)
./tune_and_calibrate_script.sh
```

## Troubleshooting

### Common Issues

**Issue**: `ImportError: cannot import name 'BaseForecastModel'`

**Solution**:
```bash
# Ensure package is installed
uv sync

# Or run from project root with PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv run python scripts/calibrate_hindcast.py
```

---

**Issue**: Model artifacts fail to load

**Solution**:
```bash
# Check artifact format compatibility
# Use hybrid format for better compatibility
save_artifacts_for_production(artifacts, path, format='hybrid')
```

---

**Issue**: Tests fail after update

**Solution**:
```bash
# Clear cache and reinstall
rm -rf .pytest_cache __pycache__
uv sync --reinstall

# Run tests with verbose output
uv run pytest tests/ -vv
```

## Quick Reference

### Essential Commands

```bash
# Setup
uv sync --extra dev

# Development
uv run ruff format                    # Format code
uv run pytest tests/ -v               # Run tests
uv run python scripts/calibrate_hindcast.py  # Train model

# Quality Checks
uv run ruff check                     # Lint code
uv run pytest --cov=lt_forecasting    # Test coverage

# Release
git tag -a v0.X.0 -m "Release X"      # Tag version
git push origin v0.X.0                # Push tag
```

### Documentation Links

- [Feature Engineering](feature_engineering.md) - Feature extraction and preprocessing
- [Model Descriptions](model_description.md) - Detailed model specifications
- [How to Add Models](how_to_add_model.md) - Guide for new model implementations
- [Version Management](VERSION_MANAGEMENT.md) - Detailed version control strategy

---

**Last Updated**: 2025-01-25