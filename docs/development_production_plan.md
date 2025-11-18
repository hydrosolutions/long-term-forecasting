# ML Package Development and Production Plan

## Overview

This document outlines the development workflow for integrating the ML package into ForecastTools with direct folder access, minimal code footprint, and stable version control.

## Project Structure

### ML Package Repository (Development)
```
lt_forecasting/
├── lt_forecasting/                 # Core package (production code)
│   ├── __init__.py            # Version info
│   ├── forecast_models/                # Model classes
│   ├── scr/                    # Utilities
│   ├── log_config.py           # Logger setup
├── dev_tools/                 # Development only (excluded)
│   ├── visualization/         # Plotting, charts
│   ├── evaluation/            # Metrics, analysis
│   └── eval_scr/             # Utilities for evaluation
├── tests/                     # tests
│   └── unit/                   # Unit tests
│   └── functionality/          # funtionality tests
│   └── integration/          # integration tests  with mock data          
├── setup.py                   # Package configuration
├── requirements.txt           # Dependencies
└── README.md
```

### ForecastTools Structure (Operational Use)
```
machine_learning_monthly/
├── forecasting/               # Main forecasting system
├── libs/
│   └── lt_forecasting/            # ML package as submodule
│       ├── lt_forecasting/        # Only core code (no dev_tools)
│       └── setup.py
├── tests/
│   └── test_ml_integration.py # Integration tests
├── requirements.txt
└── README.md
```

## Setup Configuration

### ML Package setup.py
Look at pyproject.toml
```python
from setuptools import setup, find_packages

setup(
    name="ml-package",
    version="0.1.0",
    packages=find_packages(exclude=[
        "dev_tools", 
        "dev_tools.*", 
        "tests", 
        "notebooks"
    ]),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "optuna>=3.0.0",
    ],
    extras_require={
        "dev": ["pytest", "black", "flake8", "matplotlib", "seaborn"],
    },
    python_requires=">=3.8",
)
```

### Version Management
- Version defined in `ml_package/__init__.py`:
```python
__version__ = "0.1.0"
```

## Daily Development Workflow

### 1. ML Package Development

**Regular development on main branch:**
```bash
cd ml-package/
git checkout main
git pull origin main  # Get latest changes

# Make your changes to models, tuning, prediction code
# ... edit files ...

# Test locally
pytest tests/
python -c "from ml_package.models import YourModel; print('Basic test passed')"

# Commit changes
git add .
git commit -m "Improve hyperparameter tuning algorithm"
git push origin main
```

### 2. Integration Testing in ForecastTools

**Update ML package to latest development version:**
```bash
cd ForecastTools/libs/ml-package
git pull origin main  # Get latest ML package changes
cd ../..
```

**Run integration tests:**
```bash
python -m pytest tests/test_ml_integration.py -v
```

**If tests pass** ✅ - Continue development  
**If tests fail** ❌ - Fix in ML package and repeat

### 3. Version Release and Testing Process

#### Step 1: Create New Version

**When ready to release a stable version:**
```bash
cd ml-package/

# Update version number
# Edit ml_package/__init__.py: __version__ = "0.2.0"
git add ml_package/__init__.py
git commit -m "Bump version to 0.2.0"
git push origin main

# Create tag for stable version
git tag v0.2.0
git push origin v0.2.0
```

#### Step 2: Test New Version in ForecastTools

**Update to tagged version:**
```bash
cd ForecastTools/libs/ml-package

# Check what version you're currently on
git describe --tags  # Shows current version (e.g., v0.1.0)

# Switch to new version
git fetch origin  # Get latest tags
git checkout v0.2.0
cd ../..
```

**Run comprehensive tests:**
```bash
# Run all integration tests
python -m pytest tests/test_ml_integration.py -v

# Optional: Run additional operational tests
python -m pytest tests/ -k "operational"

# Manual testing if needed
python scripts/test_full_workflow.py
```

#### Step 3A: If Tests Pass ✅

**Commit the version update:**
```bash
cd ForecastTools/
git add libs/ml-package
git commit -m "Update ml-package to v0.2.0 - all tests passing"
git push origin main
```

**You're now running the new stable version!**

#### Step 3B: If Tests Fail ❌

**Rollback to previous working version:**
```bash
cd ForecastTools/libs/ml-package

# Check available versions
git tag --list  # Shows: v0.1.0, v0.1.1, v0.2.0

# Rollback to previous working version
git checkout v0.1.1  # or whatever was the last working version
cd ../..

# Verify rollback works
python -m pytest tests/test_ml_integration.py -v
```

**Fix issues in ML package:**
```bash
cd ml-package/

# Fix the compatibility issues
# ... make necessary changes ...

git add .
git commit -m "Fix compatibility issues with ForecastTools"
git push origin main

# Create patch version
# Edit __init__.py: __version__ = "0.2.1"
git add ml_package/__init__.py
git commit -m "Bump version to 0.2.1"
git tag v0.2.1
git push origin v0.2.1
```

**Test fixed version:**
```bash
cd ForecastTools/libs/ml-package
git fetch origin
git checkout v0.2.1
cd ../..
python -m pytest tests/test_ml_integration.py -v
```

### 4. Version History Management

**Check version history:**
```bash
cd ForecastTools/libs/ml-package

# See all available versions
git tag --list --sort=-version:refname
# Output: v0.2.1, v0.2.0, v0.1.1, v0.1.0

# See what version you're currently using
git describe --tags
# Output: v0.2.1

# See commit history for current version
git log --oneline -5
```

**Emergency rollback to any previous version:**
```bash
cd ForecastTools/libs/ml-package

# Rollback to any specific version
git checkout v0.1.0  # Go back to known stable version
cd ../..

# Always test after rollback
python -m pytest tests/test_ml_integration.py
```

## Release Workflow

### 1. Prepare Release on Main Branch

**Update version directly on main:**
```python
# ml_package/__init__.py
__version__ = "0.2.0"
```

```bash
cd ml-package/
git add ml_package/__init__.py
git commit -m "Bump version to 0.2.0"
git push origin main
```

### 2. Create Stable Release

**Tag stable version:**
```bash
cd ml-package/
git tag v0.2.0
git push origin v0.2.0
```

### 3. Update ForecastTools

**Pull latest stable version:**
```bash
cd ForecastTools/libs/ml-package
git fetch origin
git checkout v0.2.0  # Use tagged version
cd ../..
python -m pytest tests/test_ml_integration.py  # Quick verification
git add libs/ml-package
git commit -m "Update ml-package to v0.2.0"
```

## Environment Management

### Development Environment
- **ML Package**: main branch + dev tools
- **ForecastTools**: main branch for testing

### Production Environment
- **ML Package**: tagged version (vX.X.X)
- **ForecastTools**: stable, tested version

## Version Control Strategy

### Branch Strategy
- **main**: Active development and releases
- **tags**: Final stable versions (v0.1.0, v0.2.0, etc.)

### Semantic Versioning
- **X.Y.Z** format
- **Patch (Z)**: Bug fixes, no breaking changes
- **Minor (Y)**: New features, backward compatible
- **Major (X)**: Breaking changes

### Rollback Strategy
```bash
# If issues found in production
cd ForecastTools/libs/ml-package
git checkout v0.1.0  # Previous stable version
cd ../..
python -m pytest tests/test_ml_integration.py  # Verify works
```

## Key Benefits

✅ **Direct folder access**: ML package code visible in ForecastTools structure  
✅ **Minimal footprint**: Only core functionality, no dev tools  
✅ **Stable releases**: Clear versioning with rollback capability  
✅ **Integration testing**: Catch operational issues before production  
✅ **Development flexibility**: Continue evolving ML package independently  

## Commands Cheat Sheet

### Daily Development
```bash
# Update ML package in ForecastTools to latest development
cd ForecastTools/libs/ml-package && git pull origin main

# Run integration tests
cd ForecastTools && python -m pytest tests/test_ml_integration.py
```

### Version Management
```bash
# Check current version
cd ForecastTools/libs/ml-package && git describe --tags

# See all available versions
git tag --list --sort=-version:refname

# Update to specific version
git checkout v0.X.0

# Rollback to previous version
git checkout v0.1.0  # Replace with known working version
```

### Release Process
```bash
# Create new version in ML package
cd ml-package && git tag v0.X.0 && git push origin v0.X.0

# Test new version in ForecastTools
cd ForecastTools/libs/ml-package && git checkout v0.X.0
cd ../.. && python -m pytest tests/test_ml_integration.py

# If tests pass, commit the update
git add libs/ml-package && git commit -m "Update to ml-package v0.X.0"

# If tests fail, rollback immediately
cd libs/ml-package && git checkout v0.1.0  # Previous working version
```

### Troubleshooting
```bash
# Always test after any version change
cd ForecastTools && python -m pytest tests/test_ml_integration.py

# Check what changed between versions
cd ForecastTools/libs/ml-package
git log --oneline v0.1.0..v0.2.0  # See changes between versions

# Emergency rollback to last known working state
git checkout v0.1.0 && cd ../.. && pytest tests/test_ml_integration.py
```

This workflow ensures stable, tested releases while maintaining direct access to ML package code within the ForecastTools project structure.