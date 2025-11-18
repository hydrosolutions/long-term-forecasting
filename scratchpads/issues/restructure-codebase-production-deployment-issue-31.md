# Restructure Codebase for Production Deployment - Issue 31

## Objective
Restructure the lt_forecasting codebase to align with the development production plan outlined in `docs/development_production_plan.md`. Separate development tools from production code and create a proper package structure for integration into ForecastTools.

## Context
- [GitHub Issue #31](https://github.com/sandrohuni/lt_forecasting/issues/31)
- Development plan: `docs/development_production_plan.md`
- Current structure is flat with mixed production and development code
- Need clean separation for git submodule integration

## Plan

### Phase 1: Create New Package Structure
- [ ] Create `lt_forecasting/lt_forecasting/` directory
- [ ] Add `lt_forecasting/__init__.py` with version info (`__version__ = "0.1.0"`)
- [ ] Create `scripts/` directory for development scripts

### Phase 2: Move Production Code
- [ ] Move `forecast_models/` to `lt_forecasting/lt_forecasting/forecast_models/`
- [ ] Move `scr/` to `lt_forecasting/lt_forecasting/scr/`
- [ ] Move `log_config.py` to `lt_forecasting/lt_forecasting/log_config.py`
- [ ] Update imports in all moved modules

### Phase 3: Separate Development Tools
- [ ] Create `dev_tools/` directory
- [ ] Move `visualization/` to `dev_tools/visualization/`
- [ ] Move `evaluation/` to `dev_tools/evaluation/`
- [ ] Move `eval_scr/` to `dev_tools/eval_scr/`
- [ ] Update imports in development tools

### Phase 4: Reorganize Tests
- [ ] Create test subdirectories: `tests/unit/`, `tests/functionality/`, `tests/integration/`
- [ ] Move and categorize tests:
  - Unit tests: `test_base_class.py`, `test_data_utils.py`, `test_sci_utils.py`
  - Functionality tests: `test_feature_processing.py`, `test_linear_regression.py`, `test_sciregressor.py`, `test_long_term_mean_enhancements.py`
- [ ] Update test imports

### Phase 5: Move Scripts
- [ ] Move `tune_hyperparams.py` to `scripts/`
- [ ] Move `calibrate_hindcast.py` to `scripts/`
- [ ] Move evaluation scripts to `scripts/`
- [ ] Update script imports

### Phase 6: Create Setup Configuration
- [ ] Create `setup.py` with proper package configuration
- [ ] Update `pyproject.toml` to exclude dev_tools

### Phase 7: Update Shell Scripts
- [ ] Update `run_model_workflow.sh` paths
- [ ] Update `tune_and_calibrate_script.sh` paths
- [ ] Update `run_evaluation_pipeline.sh` paths

### Phase 8: Documentation Updates
- [ ] Update README.md with new structure
- [ ] Create migration guide for existing users

## Implementation Notes

### Import Changes Pattern
Before:
```python
from forecast_models.base_class import BaseModel
from scr.data_utils import load_data
from evaluation.evaluate_models import evaluate
```

After:
```python
from lt_forecasting.forecast_models.base_class import BaseModel
from lt_forecasting.scr.data_utils import load_data
from dev_tools.evaluation.evaluate_models import evaluate  # for dev scripts
```

### Setup.py Template
```python
from setuptools import setup, find_packages

setup(
    name="lt-forecasting",
    version="0.1.0",
    packages=find_packages(exclude=[
        "dev_tools", 
        "dev_tools.*", 
        "tests", 
        "scripts",
        "notebooks"
    ]),
    install_requires=[
        # Extract from requirements.txt
    ],
    python_requires=">=3.8",
)
```

## Testing Strategy
1. Run existing tests after each major phase
2. Ensure all imports are updated correctly
3. Test package installation with `pip install -e .`
4. Verify shell scripts still work
5. Check that dev tools are excluded from package

## Review Points
- All imports updated correctly
- No production code depends on dev_tools
- Tests passing with new structure
- Shell scripts functioning
- Package installable
- Development workflow preserved