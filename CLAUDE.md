# Claude AI Assistant Configuration

This project is configured to work with Claude AI assistants for automated issue resolution and code maintenance. This guide provides comprehensive instructions for environment setup, command usage, issue resolution workflows, and GitHub issue creation.

## Environment Setup & Activation

**Operating System:** macOS
**Shell:** zsh
**Python Version:** 3.10+
**Project Type:** Machine Learning Hydrology - Monthly Forecasting

### Environment Activation

Before running any Claude commands or scripts, ensure your Python environment is properly activated:

```bash
# Navigate to project directory
cd "/path/to/monthly_forecasting"

# Activate Python environment (choose one method):

# Method 1: If using conda
conda activate your_ml_env

# Method 2: If using venv
source venv/bin/activate

# Method 3: If using pyenv
pyenv shell 3.10.x

# Verify Python version
python3 --version  # Should show Python 3.10+

# Install/update dependencies
pip install -r requirements.txt
```

### Required Dependencies

Ensure these core packages are installed:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost optuna pytest black flake8 mypy
```

## Project Structure

This project follows a clean, modular architecture with the following key components:

```
monthly_forecasting/
├── scr/                    # Core source code modules
│   ├── data_loading.py     # Data loading utilities
│   ├── data_utils.py       # Data manipulation utilities
│   ├── FeatureExtractor.py # Feature engineering
│   ├── sci_utils.py        # ML model utilities
│   └── tree_utils.py       # Tree-based model utilities
├── forecast_models/        # Model implementations
│   ├── base_class.py       # Base model interface
│   ├── LINEAR_REGRESSION.py
│   └── SciRegressor.py     # Main regressor implementation
├── eval_scr/              # Evaluation utilities
│   ├── eval_helper.py     # Evaluation helpers
│   └── metric_functions.py # Performance metrics
├── tests/                 # Test suite
│   ├── test_*.py          # Unit and integration tests
│   └── README_*.md        # Test documentation
└── logs/                  # Application logs
```

## Code Standards

This project follows strict Python coding standards as defined in `.github/copilot-instructions.md`. All Claude assistants must adhere to these standards:

### Core Principles (from .github/copilot-instructions.md)
- **Python 3.10+ compatibility** with modern syntax and features (use `str | None` instead of `Optional[str]`)
- **PEP 8 compliance** with 88-character line limits
- **Comprehensive type annotations** using modern Python typing
- **Google-style docstrings** for all public functions and classes
- **Robust error handling** with specific exceptions and logging
- **Clean architecture** following SOLID principles
- **Comprehensive testing** using pytest framework

### Import Organization
```python
# Standard library imports
import os
import logging
from typing import Dict, Any, List

# Third-party imports
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

# Local application imports
from scr import data_utils as du
from forecast_models.base_class import BaseModel
```

### Type Annotations (Python 3.10+ Style)
```python
def process_data(
    items: list[dict[str, int | float]], 
    threshold: float = 0.5
) -> tuple[list[dict[str, int | float]], int]:
    """Process data items above the specified threshold."""
    filtered_items = [item for item in items if item.get('value', 0) > threshold]
    return filtered_items, len(filtered_items)
```

## Command Reference

The `.claude/commands/` directory contains automated scripts for common tasks. All commands should be run from the project root directory.

### 🔧 fix-issue - Automated Issue Resolution

The primary workflow command for resolving GitHub issues:

```bash
# Basic usage
./.claude/commands/fix-issue <issue_number> [commit_message]

# Examples
./.claude/commands/fix-issue 42
./.claude/commands/fix-issue 42 "Add comprehensive type annotations to optimize_hyperparams"
./.claude/commands/fix-issue 15 "Fix categorical feature handling in XGBoost objective"
```

**What it does:**
1. Creates new branch `fix/issue-{number}`
2. Validates environment (Python 3.10+, git, dependencies)
3. Waits for you to implement the fix
4. Runs code quality checks (black, flake8)
5. Executes comprehensive tests (pytest)
6. Creates conventional commit with proper messaging
7. Opens pull request (if GitHub CLI available)

### 🧪 run-tests - Execute Test Suite

```bash
# Run all tests
./.claude/commands/run-tests

# The script automatically:
# - Sets PYTHONPATH to include project root
# - Runs pytest with verbose output
# - Shows test durations and coverage
```

### 🎨 lint-code - Code Quality Checks

```bash
# Format and lint code
./.claude/commands/lint-code

# What it does:
# - Installs black and flake8 if needed
# - Formats code with black (88-char line limit)
# - Runs flake8 style checks
# - Reports any quality issues
```

### 📚 update-docs - Documentation Maintenance

```bash
# Check and update documentation
./.claude/commands/update-docs

# What it checks:
# - Docstring compliance (Google style)
# - README file completeness
# - API documentation consistency
```

### 🐛 GitHub Issue Creation Scripts

```bash
# Create bug report
./.claude/commands/create-bug-issue.sh "Description" "file.py" "function_name"

# Create feature request  
./.claude/commands/create-feature-issue.sh "Feature description" "target_module"

# Create refactoring issue (analyzes actual file)
./.claude/commands/create-refactor-issue.sh "Improvement description" "path/to/file.py"
```

## Issue Resolution Workflow

### Step-by-Step Process

1. **Identify or Create Issue**
   ```bash
   # Use Claude CLI to create detailed issues
   claude chat --file scr/sci_utils.py "Create a GitHub issue for improving type annotations"
   ```

2. **Start Automated Resolution**
   ```bash
   # Note the GitHub issue number (e.g., #42)
   ./.claude/commands/fix-issue 42 "Brief description of fix"
   ```

3. **Implement the Fix**
   - The script creates a new branch and waits
   - Make your code changes following the coding standards
   - Ensure all changes follow Python 3.10+ patterns

4. **Automated Quality Checks**
   - Script automatically runs linting and formatting
   - Executes full test suite
   - Creates commit and pull request

### When to Use Each Command

| Scenario | Command | Purpose |
|----------|---------|---------|
| Bug fix needed | `fix-issue` | Complete automated workflow |
| Code quality check | `lint-code` | Format and style validation |
| Test verification | `run-tests` | Ensure all tests pass |
| Documentation update | `update-docs` | Check doc completeness |
| Create new issue | `create-*-issue.sh` | Generate GitHub issues |

## GitHub Issue Writing with Claude CLI

### Quick Setup

```bash
# Install Claude CLI
pip install claude-cli

# Configure with your API key
claude configure
```

### Writing Issues with Context

```bash
# Analyze code and create issue
claude chat --file scr/sci_utils.py "
Create a GitHub issue for improving the optimize_hyperparams function.
Focus on:
- Adding comprehensive type hints (Python 3.10+ style)
- Improving error handling
- Adding input validation
- Following our clean code standards
"

# Create issue for multiple files
claude chat --file scr/sci_utils.py --file tests/test_sciregressor.py "
Create an issue for improving test coverage based on these files
"

# Bug report with specific context
claude chat "
Create a bug report for XGBoost model failing with categorical features:
- File: scr/sci_utils.py
- Function: _objective_xgb  
- Expected: Handle categorical data gracefully
- Environment: macOS, Python 3.10+
"
```

### Issue Templates

**Bug Report:**
```bash
./.claude/commands/create-bug-issue.sh "Model crashes with NaN values" "scr/sci_utils.py" "fit_model"
```

**Feature Request:**
```bash
./.claude/commands/create-feature-issue.sh "Add cross-validation support" "sci_utils"
```

**Code Quality:**
```bash
./.claude/commands/create-refactor-issue.sh "Modernize type annotations" "scr/sci_utils.py"
```

### Issue Quality Standards

All GitHub issues should include:
- **Clear problem description** with specific examples
- **File and function references** for code-related issues
- **Acceptance criteria** with testable requirements
- **Technical requirements** following our coding standards
- **Testing strategy** for validation

## Scratchpad Usage for Claude Assistants

Claude assistants should use scratchpads for complex problem analysis and planning:

### When to Use Scratchpads

1. **Complex Issue Analysis**
   ```bash
   # Create analysis file
   touch scratchpad_issue_42_analysis.md
   # Document findings, approach, and implementation plan
   ```

2. **Multi-File Refactoring**
   ```bash
   # Plan refactoring across multiple files
   echo "# Refactoring Plan for Issue #42" > scratchpad_refactoring_plan.md
   ```

3. **Debugging Complex Problems**
   ```bash
   # Document debugging process
   echo "# Debug Analysis for Performance Issue" > scratchpad_debug_issue_15.md
   ```

### Scratchpad Best Practices

- **Use descriptive filenames** with issue numbers or task descriptions
- **Document thought process** step by step  
- **Include code snippets** and analysis
- **Plan implementation** before making changes
- **Clean up scratchpads** after task completion

Example scratchpad structure:
```markdown
# Scratchpad: Issue #42 - Type Annotation Improvements

## Problem Analysis
- Current state: Missing type hints in optimize_hyperparams
- Required changes: Add Python 3.10+ style annotations

## Implementation Plan
1. Add parameter type hints
2. Add return type annotation  
3. Update internal variable annotations
4. Test type checking with mypy

## Files to Modify
- scr/sci_utils.py (lines 160-220)
- tests/test_sciregressor.py (add type checking tests)
```

## Dependencies

Core dependencies include:
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost, lightgbm, catboost
- **Hyperparameter Optimization:** optuna
- **Testing:** pytest, pytest-cov
- **Code Quality:** black, flake8, mypy
- **Documentation:** pydocstyle

Install all dependencies:
```bash
pip install -r requirements.txt
```

## Claude Assistant Workflow

Claude assistants working on this project should:

1. **Use scratchpads** for complex analysis and planning
2. **Follow the automated issue resolution workflow** defined in `.claude/commands/`
3. **Maintain code quality** by adhering to standards in `.github/copilot-instructions.md`
4. **Run comprehensive tests** before proposing changes
5. **Create descriptive commit messages** following conventional commit format
6. **Ensure proper documentation** for all code changes

## Issue Resolution Process

When addressing GitHub issues, Claude assistants will:

1. **Create scratchpad** for issue analysis and planning
2. **Use automated workflow:** `./.claude/commands/fix-issue <issue_number>`
3. **Implement fix** following clean code standards from `.github/copilot-instructions.md`
4. **Validate changes** with automated testing and linting
5. **Create conventional commit** with descriptive messaging
6. **Open pull request** with detailed description

## Testing Requirements

All code changes must:
- Pass existing unit tests (`pytest tests/`)
- Include new tests for added functionality
- Maintain or improve code coverage
- Follow testing patterns in the `tests/` directory
- Use proper mocking for external dependencies
- Include both positive and negative test cases

## Logging and Debugging

The project uses structured logging:
- Logs are stored in the `logs/` directory
- Log configuration is centralized in `log_config.py`
- Use appropriate log levels (DEBUG, INFO, WARNING, ERROR)
- Include context in error messages
- Log important state changes and decisions

## Getting Started

To work with this project:

1. **Environment Setup**
   ```bash
   # Ensure Python 3.10+ is installed
   python3 --version
   
   # Activate your environment
   conda activate your_ml_env  # or source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Verify Installation**
   ```bash
   # Run tests to ensure everything works
   ./.claude/commands/run-tests
   
   # Check code quality
   ./.claude/commands/lint-code
   ```

3. **Create Your First Issue**
   ```bash
   # Install Claude CLI
   pip install claude-cli
   claude configure
   
   # Create an issue
   claude chat "Create a GitHub issue for adding documentation to a specific function"
   ```

4. **Fix an Issue**
   ```bash
   # Use automated workflow
   ./.claude/commands/fix-issue <issue_number> "Brief description"
   ```

## Reference Documentation

For detailed information, see:
- `.github/copilot-instructions.md` - Complete coding standards
- `.claude/CLAUDE_CLI_ISSUE_GUIDE.md` - Comprehensive GitHub issue writing guide
- `README.md` - Project overview and usage
- `WORKFLOW_DOCUMENTATION.md` - Detailed workflow documentation  
- `tests/README_*.md` - Testing guidelines and examples

## Troubleshooting

### Common Issues

**Environment Problems:**
```bash
# Check Python version
python3 --version

# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**Git Issues:**
```bash
# Ensure you're in a git repository
git status

# Check remote configuration
git remote -v
```

**Permission Issues:**
```bash
# Make scripts executable
chmod +x .claude/commands/*
```

**Test Failures:**
```bash
# Run specific test file
pytest tests/test_specific_module.py -v

# Run with debugging
pytest tests/ -v --tb=long
```
