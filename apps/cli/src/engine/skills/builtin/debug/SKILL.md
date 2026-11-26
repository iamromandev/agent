---
description: Find the root cause of a bug, failing test or crash before changing code, then fix it test-first and verify
---
Find the cause before you change anything.

1. Reproduce the failure. Record the exact command and its output. If you cannot reproduce it, stop and say so.
2. Read the code path with `read` and `grep`.
3. State the root cause with evidence (file:line, output) before any `edit`.
4. Where the project has tests, add a failing test that shows the bug.
5. Fix the cause, not the symptom. Rerun that test and the related tests.

If you cannot find the cause, stop and report what you ruled out instead of guessing a fix.
