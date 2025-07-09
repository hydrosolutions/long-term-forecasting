# Scratchpads Directory

## Purpose
Work planning and documentation area for complex development tasks, issue tracking, and technical exploration. Follows a structured approach to maintain context and enable effective collaboration.

## Directory Structure
- `issues/`: Issue-specific implementation planning and tracking
- `planning/`: Feature design and architecture planning documents

## Contents

### issues/
- `comprehensive-sciregressor-testing-issue-10.md`: Testing strategy for SciRegressor
- `comprehensive-test-coverage-issue-6.md`: Overall test coverage improvement plan
- `fix-r2-scaling-issue-4.md`: R-squared metric scaling bug fix documentation
- `improve-test-suite-efficiency-issue-12.md`: Test performance optimization
- `length-mismatch-long-term-mean-scaling-issue-8.md`: Data alignment issue resolution

### planning/
- `github-issue-test-suite.md`: Test suite design for GitHub integration
- `test-suite-improvement.md`: Overall testing strategy improvements

## Document Structure

Each scratchpad follows a standard template:
```markdown
# [Task Name]

## Objective
Clear description of what needs to be accomplished

## Context
Background information, links to issues, previous work

## Plan
- [ ] Specific actionable steps
- [ ] Implementation checkpoints
- [ ] Testing requirements

## Implementation Notes
Code snippets, architectural decisions, API changes

## Testing Strategy
How to verify the solution works

## Review Points
Areas requiring special attention during code review
```

## Usage Guidelines

### When to Create a Scratchpad
- Complex multi-step tasks (3+ distinct steps)
- Non-trivial implementation requiring planning
- Bug fixes requiring root cause analysis
- Feature implementations with architectural impact
- Performance optimization investigations

### Naming Convention
- Issues: `{description}-issue-{number}.md`
- Planning: `{feature-description}.md`
- Research: `{topic}-analysis.md`

## Benefits
- Preserves implementation context
- Enables knowledge transfer
- Supports asynchronous collaboration
- Documents decision rationale
- Facilitates code review

## Integration Points
- Referenced in pull requests
- Linked from GitHub issues
- Used during code reviews
- Guides test development
- Informs documentation updates