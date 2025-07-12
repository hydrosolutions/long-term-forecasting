# GitHub Configuration

This directory contains GitHub-specific configuration files for the monthly forecasting project.

## Structure

### workflows/
Contains GitHub Actions workflow definitions:

- **ci.yml**: Continuous Integration workflow
  - Runs on push and pull requests
  - Executes test suite
  - Performs code quality checks (ruff)
  - Validates model configurations

## Workflow Details

### Continuous Integration (CI)

The CI workflow ensures code quality and functionality by:
1. Setting up Python environment with uv
2. Installing project dependencies
3. Running the full test suite
4. Checking code formatting with ruff
5. Validating configuration files

Triggers:
- Push to main/develop branches
- Pull requests
- Manual workflow dispatch

## Adding New Workflows

To add a new workflow:
1. Create a new `.yml` file in `workflows/`
2. Define triggers, jobs, and steps
3. Test locally using `act` if available
4. Push and verify in GitHub Actions tab

## Best Practices

- Keep workflows focused and single-purpose
- Use environment variables for sensitive data
- Cache dependencies for faster runs
- Add status badges to main README
- Monitor workflow run times and optimize as needed