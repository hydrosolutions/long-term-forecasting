# MONTHLY DISCHARGE FORECASTING

## Information about the forecasting setup (condensed):

The aim is to predict monthly discharge. We have a multi-basin setup and the following input variables.
**Dynamic Features**
Observed Discharge (only available until forecast date t) (daily resolution)
Weather Data (P, T) (available until t+ 15d) (daily resolution)
Snow Information from Physical Model (available t + 10 days) (daily resolution)
Glacier Mapper features (Earth Observation) (available every 10th day until t) (10d resolution)

**Static Data**
Static Basin features (area, lat., lon, aridity, glacier fraction...)


## PROJECT STRUCTURE

The project has been restructured for production deployment:

```
lt_forecasting/
├── lt_forecasting/     # Core production package
│   ├── forecast_models/     # Model implementations
│   ├── scr/                 # Data processing utilities
│   └── log_config.py        # Logging configuration
├── dev_tools/               # Development-only tools
│   ├── evaluation/          # Evaluation pipeline
│   ├── visualization/       # Dashboard and plotting
│   └── eval_scr/            # Evaluation metrics
├── scripts/                 # Development scripts
│   ├── calibrate_hindcast.py
│   └── tune_hyperparams.py
└── tests/                   # Test suite
    ├── unit/                # Unit tests
    ├── functionality/       # Functionality tests
    └── integration/         # Integration tests
```

## COMMON BASH COMMANDS:

1. activate environment
As we are working we do not need to activate the venv specifically but can just use
bash 
'
uv run python scripts/some_script.py
'

2. Run ruff
bash
'uv run ruff format'

3. run the tests:
bash 
'
uv run pytest -v
'

4. Run specific test categories:
bash
'
uv run pytest tests/unit/ -v       # Run unit tests
uv run pytest tests/functionality/ -v  # Run functionality tests
'

5. Run the shell scripts:
bash
'
./tune_and_calibrate_script.sh
'


## Scratchpad System

Use structured markdown files for complex work planning and documentation.

### Directory Structure

```bash
scratchpads/
├── issues/          # Issue-specific work (/project:fix-issue command)
├── planning/        # Feature planning and design
└── research/        # Technical exploration and spikes
```

### Naming Convention

`scratchpads/{type}/{brief-description}.md`

Examples:

- `scratchpads/issues/fix-data-validation-bug-123.md`
- `scratchpads/planning/user-authentication-system.md`  
- `scratchpads/research/performance-optimization-analysis.md`

### Standard Template

```markdown
# [Task Name]

## Objective
[Clear description of what needs to be accomplished]

## Context
[Background information, links to issues, previous work]

## Plan
- [ ] Step 1: [Specific actionable item]
- [ ] Step 2: [Specific actionable item]
- [ ] Step 3: [Specific actionable item]

## Implementation Notes
[Code snippets, architectural decisions, API changes]

## Testing Strategy
[How to verify the solution works]

## Review Points
[Areas requiring special attention during code review]
```


## Custom Slash Commands

Available project-specific commands:

### `/project:issue <issue-number>`

Complete issue-driven development workflow following TDD principles.

**Usage**: `/project:issue 123`

**What it does**:

1. Analyzes GitHub issue via `gh issue view`
2. Searches codebase for relevant context
3. Creates implementation plan in scratchpad
4. Implements solution using TDD approach
5. Creates pull request with proper documentation


### Error Handling Philosophy

The `scripts/dev.py` includes robust error handling:

- **Graceful degradation**: Warns about missing tools instead of failing
- **Detailed diagnostics**: Shows exact commands and exit codes
- **Clear reporting**: ✅/❌ status summary for all operations
- **CI compatibility**: Identical behavior locally and in GitHub Actions

## Best Practices Summary

### Daily Development

1. **Use scratchpads**: Document complex work for context preservation
2.  **Write maintainable code** write functions which can be re-used: if a lot of code is repeated write a function for it.

### Code Quality

1. **Test incrementally**: Run specific tests during development
2. **Think deeply**: Use extended thinking for complex architectural decisions

### Collaboration  

1. **Document decisions**: Use scratchpads for complex reasoning
2. **Create clear PRs**: Use `/project:create-pr` for consistent formatting
3. **Link issues**: Always reference relevant GitHub issues
4. **Review thoroughly**: Focus on areas highlighted in PR descriptions