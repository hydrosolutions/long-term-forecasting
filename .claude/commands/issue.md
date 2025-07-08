Please analyze and fix the GitHub issue: $ARGUMENTS.

Follow these steps:


#  PLAN
1. Use 'gh issue view' to get the issue details
2. Understand the problem described in the issue
3. Ask clarifiying questions if necessary
4. Understand the prior art for this issue
   -    Search the scratchpads for previous thoughts related to the issue
   -    Search PRs to see if you can find history of this issue
   -    Search the codebase for relevant files
5. Think harder about how to break the issue donw into a series of small manageable tasks.
6. Document your plan in a new scratchpad
   - include the issue name in the filename
   - include a link to the issue in the scratchpad
  
# CREATE
1. Create a new branch for the issue
2. solve the issue in small, manageble steps, according to your plan
3. Commit the changes after each step.

# TEST
1. Write test functions to describe the expected behaviour of your code
2. Run the full test suite to ensure you haven't broken anything
3. If the tests are failing, fix them
4. Ensure that all tests are passing before moving to the next step.

# DEPLOY
- Open a PR and request a review

Remember to use the Github CLI ('gh') for all Github-related tasks
