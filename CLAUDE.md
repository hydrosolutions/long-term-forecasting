# MONTHLY DISCHARGE FORECASTING


## COMMON BASH COMMANDS:

1. activate environment
bash 
'
source "/Users/sandrohunziker/Documents/sapphire_venv/monthly_forecast/bin/activate"
'

2. run the tests:
bash 
'
python -m pytest -ra 
'

3. Run the  shell scripts:
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