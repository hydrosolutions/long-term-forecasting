# Rename Package: lt_forecasting → lt_forecasting (Issue #52)

**Issue**: https://github.com/hydrosolutions/lt_forecasting/issues/52
**Status**: Complete - PR #53 Created
**Branch**: rename-package-to-lt-forecasting

## Objective

Rename the package from `lt_forecasting` to `lt_forecasting` (Long-Term Forecasting) before v0.1.0 release to better reflect its general-purpose nature for long-term hydrological forecasting.

## Context

- **Current**: ~126 Python file occurrences + ~4 markdown occurrences
- **Safe to rename**: No production users, pre-release v0.1.0
- **Comprehensive plan**: Issue #52 provides detailed 12-phase implementation plan

## Implementation Strategy

I'll follow the plan from issue #52 with slight modifications for efficiency:

### Phase 1: Preparation
1. Create new branch
2. Use `git mv` to rename directory (preserves history)

### Phase 2-8: Automated Bulk Changes
Execute automated sed commands for:
- Python imports
- Documentation files
- Configuration files
- Shell scripts

### Phase 9: Manual Updates
Review and manually update:
- `pyproject.toml`
- `setup.py`
- Any contextual references needing careful handling

### Phase 10: Verification
- Run tests: `uv run pytest -v`
- Run formatter: `uv run ruff format`
- Check for remaining references
- Test local installation

### Phase 11: Cleanup & Commit
- Remove caches
- Stage all changes
- Create comprehensive commit message
- Push for review

## Testing Strategy

### Automated
```bash
uv run pytest -v
```

### Manual
```bash
# Test imports
python -c "from lt_forecasting.forecast_models import SciRegressor"

# Test scripts
uv run python scripts/calibrate_hindcast.py --config_path example_config/DUMMY_MODEL

# Search for lingering references
grep -r "lt_forecasting" --include="*.py" --include="*.md" . | grep -v ".git" | grep -v "__pycache__"
```

## Risk Mitigation

- ✅ All changes are reversible via git
- ✅ Comprehensive test suite exists
- ✅ No production users
- ✅ Pre-release version

## Notes

- Deferring notebook kernel updates (83 notebooks) - can be done post-release
- Focus on core package, tests, scripts, and documentation
- Use git mv to preserve history

## Checklist

- [x] Create branch
- [x] Rename directory with git mv
- [x] Automated bulk updates (Python)
- [x] Automated bulk updates (docs/scripts)
- [x] Manual updates (pyproject.toml, setup.py)
- [x] Run tests
- [x] Run formatter
- [x] Verify no remaining references
- [x] Test installation
- [x] Commit and push
- [x] Create PR

## Results

- **PR Created**: https://github.com/hydrosolutions/monthly_forecasting/pull/53
- **Tests**: 167 passed, 1 skipped
- **Package**: Successfully renamed and importable
- **Version**: 0.1.0 maintained
