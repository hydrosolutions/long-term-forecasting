# Claude Configuration

This directory contains Claude-specific configuration and custom commands for the monthly forecasting project.

## Structure

### commands/
Contains custom slash commands for Claude:

- **update_docs.md**: Command for updating project documentation
  - Analyzes codebase structure
  - Updates README files
  - Creates overview documentation
  - Generates workflow diagrams

## Custom Commands

### /update_docs
Updates all project documentation by:
1. Scanning project structure
2. Reading and analyzing code
3. Checking recent changes
4. Updating README files for each component
5. Creating high-level documentation (Overview.md, model_description.md)

Usage:
```
/update_docs
```

## Adding New Commands

To create a new custom command:
1. Create a new `.md` file in `commands/`
2. Define the command behavior and steps
3. Test the command functionality
4. Document usage in this README

## Command Best Practices

- Make commands specific and focused
- Include clear step-by-step instructions
- Add validation steps where appropriate
- Document expected outputs
- Consider edge cases and error handling