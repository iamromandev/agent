"""Ollama implementation of the provider contract."""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncGenerator, Sequence
from typing import Any
from uuid import uuid4

from src.client import OllamaClient, OllamaError, OllamaUnreachable

from .base import Done, Message, ProviderError, ProviderEvent, TextDelta, ToolCall, ToolCallEvent, ToolSpec


def _message_payload(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        payload: dict[str, Any] = {"role": "tool", "content": message.content}
        if message.name is not None:
            payload["tool_name"] = message.name
        return payload
    payload = {"role": message.role, "content": message.content}
    if message.role == "assistant" and message.tool_calls:
        payload["tool_calls"] = [
            {"function": {"name": call.name, "arguments": call.arguments}} for call in message.tool_calls
        ]
    return payload


def _tool_payload(tool: ToolSpec) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {"name": tool.name, "description": tool.description, "parameters": tool.parameters},
    }


def _arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw) if raw.strip() else {}
        except ValueError:
            return {"__invalid__": raw}
        return parsed if isinstance(parsed, dict) else {"__invalid__": raw}
    return {} if raw is None else {"__invalid__": str(raw)}


def _tool_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function")
    if not isinstance(function, dict):
        function = {}
    call_id = raw.get("id")
    name = function.get("name")
    return ToolCall(
        id=call_id if isinstance(call_id, str) and call_id else f"call_{uuid4().hex[:8]}",
        name=name if isinstance(name, str) else "",
        arguments=_arguments(function.get("arguments")),
    )


def _count(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class OllamaProvider:
    def __init__(self, client: OllamaClient, model: str, num_ctx: int) -> None:
        self._client = client
        self.model = model
        self.context_window = num_ctx

    async def stream(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> AsyncGenerator[ProviderEvent]:
        payload = {
            "model": self.model,
            "messages": [_message_payload(m) for m in messages],
            "tools": [_tool_payload(t) for t in tools],
            "stream": True,
            "options": {"num_ctx": self.context_window},
        }
        try:
            async with contextlib.aclosing(self._client.chat(payload)) as chunks:
                async for chunk in chunks:
                    message = chunk.get("message")
                    if not isinstance(message, dict):
                        message = {}
                    content = message.get("content")
                    if content and isinstance(content, str):
                        yield TextDelta(content)
                    for raw in message.get("tool_calls") or []:
                        if isinstance(raw, dict):
                            yield ToolCallEvent(_tool_call(raw))
                    if chunk.get("done"):
                        yield Done(_count(chunk.get("prompt_eval_count")), _count(chunk.get("eval_count")))
                        return
        except OllamaError as error:
            yield ProviderError(str(error))
        except (KeyError, TypeError, ValueError, AttributeError):
            yield ProviderError("Ollama sent an unexpected response")

    async def check(self) -> str | None:
        try:
            names = await self._client.models()
        except OllamaUnreachable:
            return f"Cannot reach Ollama at {self._client.base_url} — start it with `ollama serve`"
        except OllamaError as error:
            return str(error)
        tagged = ":" in self.model.rsplit("/", 1)[-1]
        wanted = self.model if tagged else f"{self.model}:latest"
        if self.model in names or wanted in names:
            return None
        return f"Model {self.model} is not pulled — run `ollama pull {self.model}`"
