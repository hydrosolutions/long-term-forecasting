# Migration Guide for Restructured Monthly Forecasting Project

This guide helps you migrate your code to work with the new restructured project layout.

## Overview of Changes

The project has been restructured to separate production code from development tools:

- **Production code** is now in the `monthly_forecasting` package
- **Development tools** are in the `dev_tools` directory
- **Scripts** have moved to the `scripts` directory
- **Tests** are organized into `unit`, `functionality`, and `integration` subdirectories

## Import Changes

### Production Code Imports

**Old:**
```python
from forecast_models.base_class import BaseForecastModel
from scr import data_utils as du
from log_config import setup_logging
```

**New:**
```python
from monthly_forecasting.forecast_models.base_class import BaseForecastModel
from monthly_forecasting.scr import data_utils as du
from monthly_forecasting.log_config import setup_logging
```

### Development Tools Imports

**Old:**
```python
from evaluation.evaluate_pipeline import run_evaluation_pipeline
from visualization.dashboard import app
from eval_scr import metric_functions
```

**New:**
```python
from dev_tools.evaluation.evaluate_pipeline import run_evaluation_pipeline
from dev_tools.visualization.dashboard import app
from dev_tools.eval_scr import metric_functions
```

## Running Scripts

### Command Line Usage

**Old:**
```bash
uv run calibrate_hindcast.py --config_dir path/to/config
uv run tune_hyperparams.py --config_dir path/to/config
```

**New:**
```bash
uv run python scripts/calibrate_hindcast.py --config_dir path/to/config
uv run python scripts/tune_hyperparams.py --config_dir path/to/config
```

### Shell Scripts

The shell scripts have been updated with the new paths. No changes needed for users.

## Running Tests

**Old:**
```bash
uv run pytest tests/
```

**New (more specific):**
```bash
uv run pytest -v                    # Run all tests
uv run pytest tests/unit/ -v        # Run unit tests only
uv run pytest tests/functionality/ -v  # Run functionality tests only
```

## Dashboard Usage

**Old:**
```bash
uv run python visualization/dashboard.py
```

**New:**
```bash
uv run python -m dev_tools.visualization.dashboard
```

## Installing the Package

The package can now be installed for production use:

```bash
pip install -e .  # Development installation
# or
pip install .     # Production installation
```

This will install only the production code (`monthly_forecasting` package), excluding development tools.

## Configuration Files

No changes needed - configuration files remain in the same format and location.

## For ForecastTools Integration

When integrating as a git submodule:

```python
# In ForecastTools
from libs.monthly_forecasting.monthly_forecasting.forecast_models import SciRegressor
from libs.monthly_forecasting.monthly_forecasting.scr import data_loading
```

## Common Issues and Solutions

### ModuleNotFoundError

If you get `ModuleNotFoundError: No module named 'forecast_models'`:
- Update your imports to use `monthly_forecasting.forecast_models`

### Path Issues in Scripts

If scripts can't find modules:
- Make sure you're running from the project root directory
- Use `uv run python scripts/script_name.py` instead of direct execution

### Test Discovery Issues

If pytest can't find tests:
- Run pytest from the project root directory
- The `pyproject.toml` is configured to find tests automatically

## Gradual Migration

If you need to migrate gradually:

1. Update imports in your scripts first
2. Test each script individually
3. Update any custom scripts or notebooks you have
4. Verify shell scripts still work correctly

## Getting Help

If you encounter issues not covered here:

1. Check the updated documentation in each module
2. Review the test files for usage examples
3. Submit an issue on GitHub with details about the problem