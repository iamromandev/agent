"""The built-in slash commands of the app."""

from __future__ import annotations

# Built-in commands and what `/` suggests for them. Must equal the cases of `AgentApp._command` (a test checks).
BUILTINS: dict[str, str] = {
    "help": "this help",
    "new": "new session",
    "clear": "new session",
    "sessions": "switch session",
    "mcp": "MCP servers and their tools",
    "permissions": "saved permission rules",
    "guardrails": "sandbox, protected paths and redaction",
    "hooks": "configured hooks",
    "skills": "loaded skills",
    "agents": "loaded subagents",
    "compact": "shrink the context",
    "model": "switch model (/model <name>)",
    "init": "write AGENTS.md",
    "undo": "undo the last turn",
    "diff": "show what the last turn changed",
    "usage": "show the session's token totals",
    "quit": "exit",
}
