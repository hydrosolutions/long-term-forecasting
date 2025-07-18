Please analyze and fix the GitHub review of the Pull Request: $ARGUMENTS.

Follow these steps:


#  PLAN
1. Use 'gh' to get the pull request and the comment details
2. Understand the problem and comments described in the PR
3. Ask clarifiying questions if necessary
4. Understand the prior art for this issue
   -    Search the scratchpads for previous thoughts related to the issue / PR
   -    Search the codebase for relevant files
5. Think harder about how to break the issue down into a series of small manageable tasks.
6. Document your plan in a new scratchpad
   - include the issue name in the filename
   - include a link to the issue in the scratchpad
  
# CREATE
1. Ensure that you work on the correct branch.
2. solve the issue in small, manageble steps, according to your plan
3. Commit the changes after each step.

# TEST
1. Write test functions to describe the expected behaviour of your code
2. Run the full test suite to ensure you haven't broken anything
3. If the tests are failing, fix them
4. Ensure that all tests are passing before moving to the next step.

# DEPLOY
- Commit to the PR and write what you have changed.

Remember to use the Github CLI ('gh') for all Github-related tasks
