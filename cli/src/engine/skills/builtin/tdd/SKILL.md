---
description: Change code test-first: write a failing test, see it fail, make it pass with the smallest change, then rerun the tests
---
Change behaviour test-first.

1. Find the test command (AGENTS.md, makefile, pyproject or package.json).
2. Write or change one test for the behaviour. Run it and see it fail for the intended reason.
3. Make the smallest change that makes it pass.
4. Rerun that test, then the related tests.
5. Report the commands you ran and their results.

Rules:
- Never weaken, skip or delete a test to make it pass.
- Use `todo` to track multi-step work.
