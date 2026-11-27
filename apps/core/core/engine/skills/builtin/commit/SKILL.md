---
description: Write a conventional commit message from the staged changes and commit them; only when the user asks to commit
---
Commit only when the user asked you to.

1. Run `git status` and `git diff --staged` with `bash`.
2. Stage only what the user named. Ask before any `git add`; never run `git add -A` unasked.
3. Stop and ask if the staged diff has unrelated changes or secret-looking files (`.env`, keys, tokens).
4. Write the message as `type(scope): summary`. Types: `feat`, `fix`, `docs`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`.
   - Imperative mood, at most 72 characters in the subject.
   - Add a body that explains why when that is not obvious.
5. Pass the message safely through `bash`:
   - Use `git commit -m '<subject>' -m '<body>'` with single quotes; write each single quote inside as `'\''`.
   - Or write the message to a temp file with `write` and run `git commit -F <file>`.
   - Never put `$` or backticks inside double quotes.
6. Show `git log -1 --stat`.

Rules:
- Never push, amend, use `--force` or `--no-verify`, or change the git config unless the user asks.
- If a hook fails, fix the cause and make a new commit. Never amend. Do not bypass the hook.
