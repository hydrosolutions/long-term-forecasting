# Version Management and Release Strategy

## Overview

This project uses **git tags** for version management to maintain stable releases while continuing active development. This approach is optimized for internal use as a dependency in operational applications.

## Versioning Scheme

We follow [Semantic Versioning (SemVer)](https://semver.org/):

```
v{MAJOR}.{MINOR}.{PATCH}

Examples:
- v1.0.0 - Initial stable release
- v1.1.0 - New features, backward compatible
- v1.0.1 - Bug fixes only
- v2.0.0 - Breaking changes
```

### When to increment:

- **MAJOR**: Breaking changes (API changes, removed features)
- **MINOR**: New features, backward compatible enhancements
- **PATCH**: Bug fixes, minor improvements

## Creating a New Release

### 1. Pre-release checklist

```bash
# Ensure all tests pass
uv run pytest -v

# Format code
uv run ruff format

# Verify clean git status
git status
```

### 2. Update version (optional)

You can optionally update the version in `pyproject.toml`:

```toml
[project]
version = "1.1.0"  # Update as needed
```

### 3. Create and push the tag

```bash
# Create annotated tag with descriptive message
git tag -a v1.1.0 -m "Release v1.1.0: Add glacier mapper features support"

# Push tag to remote
git push origin v1.1.0
```

### 4. Verify the tag

```bash
# List all tags
git tag -l

# View tag details
git show v1.1.0
```

## Using Versions in Operational Applications

### Install specific version

In your operational application's `pyproject.toml`:

```toml
dependencies = [
    "lt-forecasting @ git+https://github.com/hydrosolutions/long-term-forecasting.git@v1.0.0"
]
```

Then install:

```bash
uv sync
```

### Upgrade to newer version

1. Change the tag in `pyproject.toml`:
   ```toml
   "lt-forecasting @ git+https://github.com/hydrosolutions/long-term-forecasting.git@v1.1.0"
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

### Use latest development version (not recommended for production)

```toml
dependencies = [
    "lt-forecasting @ git+https://github.com/hydrosolutions/long-term-forecasting.git@main"
]
```

## Development Workflow

### Active development

Continue working on `main` branch normally:

```bash
# Make changes
git add .
git commit -m "Add new feature"
git push origin main
```

**Important**: Changes to `main` do **not** affect existing tags. Applications using `v1.0.0` will continue getting the exact same code.

### Bug fixes for released versions

#### Option 1: Fix on main and tag new patch release

```bash
# Fix bug on main
git commit -m "Fix: resolve data validation bug"
git push origin main

# Create patch release
git tag -a v1.0.1 -m "Release v1.0.1: Fix data validation bug"
git push origin v1.0.1
```

#### Option 2: Maintain release branch (for major versions only)

If you need to maintain older major versions while developing v2.x:

```bash
# Create release branch from v1.0.0 tag
git checkout -b release/v1.x v1.0.0
git push origin release/v1.x

# Apply fixes
git commit -m "Fix: critical bug in v1.x"
git tag -a v1.0.1 -m "Release v1.0.1: Critical bug fix"
git push origin release/v1.x
git push origin v1.0.1

# Return to main for v2.x development
git checkout main
```

## Version History

### v1.0.0 (2025-01-18)
- Initial stable release
- Multi-basin monthly discharge forecasting
- Support for dynamic features (discharge, weather, snow, glacier mapper)
- Static basin features
- Comprehensive test suite
- Production-ready package structure

## Best Practices

1. **Always test before tagging**: Run full test suite before creating a release tag
2. **Use descriptive tag messages**: Explain what's new or fixed in the tag message
3. **Tag stable commits only**: Don't tag broken or incomplete code
4. **Document breaking changes**: Clearly communicate API changes when incrementing major version
5. **Keep main stable**: Merge only tested, working code to main
6. **Pin versions in production**: Always use specific tags in operational applications, never `@main`

## Troubleshooting

### Tag already exists

```bash
# Delete local tag
git tag -d v1.0.0

# Delete remote tag
git push origin :refs/tags/v1.0.0

# Recreate tag
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

### Application not picking up new version

```bash
# Clear uv cache and reinstall
rm -rf .venv
uv sync
```

### View all available versions

```bash
git tag -l
```

## Additional Resources

- [Semantic Versioning](https://semver.org/)
- [Git Tagging Documentation](https://git-scm.com/book/en/v2/Git-Basics-Tagging)
- [UV Documentation](https://github.com/astral-sh/uv)