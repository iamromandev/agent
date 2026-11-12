from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from src.agent.provider.base import (
    Done,
    Message,
    Provider,
    ProviderError,
    TextDelta,
    ToolCall,
    ToolCallEvent,
    ToolSpec,
)
from src.agent.provider.ollama import OllamaProvider
from src.client import OllamaClient

URL = "http://ollama.test"


def _provider(handler, model: str = "foo", num_ctx: int = 4096) -> OllamaProvider:
    client = OllamaClient(URL, transport=httpx.MockTransport(handler))
    return OllamaProvider(client, model, num_ctx)


def _ndjson(*lines: dict[str, Any]) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


def _serving(body: bytes) -> Callable[[httpx.Request], Any]:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body)

    return handler


async def _events(provider: OllamaProvider, messages: list[Message] | None = None, tools=()) -> list:
    return [e async for e in provider.stream(messages or [Message("user", "hi")], list(tools))]


def _call_line(call: Any) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": "", "tool_calls": [call]}, "done": False}


DONE_LINE = {"message": {"role": "assistant", "content": ""}, "done": True, "prompt_eval_count": 7, "eval_count": 3}


async def test_recorded_tool_call_stream_maps_to_events(fixture_path: Callable[[str], Path]) -> None:
    body = fixture_path("ollama/tool_call.ndjson").read_bytes()
    events = await _events(_provider(_serving(body)))

    assert events == [
        ToolCallEvent(ToolCall("call_zf6mn04t", "grep", {"path": "cli/src/main.py", "pattern": "^def.*:.*$"})),
        Done(225, 31),
    ]


async def test_recorded_text_stream_yields_deltas_in_order(fixture_path: Callable[[str], Path]) -> None:
    body = fixture_path("ollama/text.ndjson").read_bytes()
    events = await _events(_provider(_serving(body)))

    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    expected = [
        json.loads(line)["message"]["content"] for line in body.splitlines() if json.loads(line)["message"]["content"]
    ]
    assert deltas == expected
    assert isinstance(events[-1], Done)
    assert events[-1].prompt_tokens > 0 and events[-1].completion_tokens > 0
    assert sum(isinstance(e, Done) for e in events) == 1


async def test_missing_id_gets_a_generated_one() -> None:
    body = _ndjson(
        {"message": {"content": "ok"}, "done": False},
        _call_line({"function": {"name": "read", "arguments": {"path": "a.py"}}}),
        DONE_LINE,
    )
    events = await _events(_provider(_serving(body)))

    assert events[0] == TextDelta("ok")
    call = events[1].call
    assert (call.name, call.arguments) == ("read", {"path": "a.py"})
    assert call.id.startswith("call_") and len(call.id) == len("call_") + 8
    assert events[2] == Done(7, 3)


async def test_string_arguments_are_parsed() -> None:
    body = _ndjson(_call_line({"id": "c1", "function": {"name": "read", "arguments": '{"path": "a.py"}'}}), DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events[0] == ToolCallEvent(ToolCall("c1", "read", {"path": "a.py"}))


async def test_invalid_string_arguments_are_flagged() -> None:
    body = _ndjson(_call_line({"id": "c1", "function": {"name": "read", "arguments": "{oops"}}), DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events[0] == ToolCallEvent(ToolCall("c1", "read", {"__invalid__": "{oops"}))


async def test_request_payload_has_model_options_tools_and_messages() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_ndjson(DONE_LINE))

    messages = [
        Message("system", "be brief"),
        Message("user", "read a.py"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "read", {"path": "a.py"})]),
        Message("tool", "contents", tool_call_id="c1", name="read"),
    ]
    tools = [ToolSpec("read", "Read a file", {"type": "object", "properties": {}, "required": []})]
    await _events(_provider(handler, model="foo", num_ctx=4096), messages, tools)

    payload = captured[0]
    assert payload["model"] == "foo"
    assert payload["stream"] is True
    assert payload["options"] == {"num_ctx": 4096}
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }
    ]
    assert payload["messages"] == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "read a.py"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "read", "arguments": {"path": "a.py"}}}],
        },
        {"role": "tool", "content": "contents", "tool_name": "read"},
    ]


async def test_stream_dropped_after_two_chunks_yields_a_provider_error() -> None:
    async def body() -> AsyncIterator[bytes]:
        yield _ndjson({"message": {"content": "a"}, "done": False})
        yield _ndjson({"message": {"content": "b"}, "done": False})
        raise httpx.ReadError("dropped")

    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=body())

    events = await _events(_provider(handler))
    assert events[:2] == [TextDelta("a"), TextDelta("b")]
    assert len(events) == 3 and isinstance(events[2], ProviderError)


async def test_http_500_yields_a_provider_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "model crashed"})

    assert await _events(_provider(handler)) == [ProviderError("model crashed")]


async def test_provider_exposes_model_and_context_window() -> None:
    provider: Provider = _provider(_serving(b""), model="foo", num_ctx=4096)
    assert (provider.model, provider.context_window) == ("foo", 4096)


def _tags(*names: str) -> Callable[[httpx.Request], Any]:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": n} for n in names]})

    return handler


async def test_check_reports_an_unreachable_server() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    assert await _provider(handler).check() == f"Cannot reach Ollama at {URL} — start it with `ollama serve`"


async def test_check_reports_a_missing_model() -> None:
    result = await _provider(_tags("other:7b"), model="foo").check()
    assert result == "Model foo is not pulled — run `ollama pull foo`"


async def test_check_accepts_a_tagless_name_matching_latest() -> None:
    assert await _provider(_tags("foo:latest"), model="foo").check() is None


async def test_check_accepts_an_exact_match() -> None:
    assert await _provider(_tags("qwen2.5:7b-instruct"), model="qwen2.5:7b-instruct").check() is None


async def test_check_does_not_match_a_different_tag() -> None:
    assert await _provider(_tags("foo:7b"), model="foo").check() is not None


async def test_check_returns_the_server_error_for_http_500() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "disk full"})

    assert await _provider(handler).check() == "disk full"


async def test_check_returns_the_error_for_an_invalid_model_list() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"nom": "x"}]})

    assert await _provider(handler).check() == "Ollama sent an invalid model list"


async def test_check_reports_unreachable_on_connect_timeout() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("slow", request=request)

    assert await _provider(handler).check() == f"Cannot reach Ollama at {URL} — start it with `ollama serve`"


async def test_check_treats_a_registry_port_name_as_untagged() -> None:
    result = await _provider(_tags("localhost:5000/foo:latest"), model="localhost:5000/foo").check()
    assert result is None


async def test_check_does_not_treat_a_registry_port_as_a_tag() -> None:
    assert await _provider(_tags("localhost:5000/foo:7b"), model="localhost:5000/foo").check() is not None


async def test_tool_message_without_a_name_omits_tool_name() -> None:
    captured: list[dict[str, Any]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        return httpx.Response(200, content=_ndjson(DONE_LINE))

    await _events(_provider(handler), [Message("tool", "contents", tool_call_id="c1")])
    assert captured[0]["messages"] == [{"role": "tool", "content": "contents"}]


async def test_second_done_line_is_not_emitted() -> None:
    body = _ndjson(DONE_LINE, DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events == [Done(7, 3)]


async def test_non_dict_message_is_ignored() -> None:
    body = _ndjson({"message": "str", "done": True, "prompt_eval_count": 7, "eval_count": 3})
    assert await _events(_provider(_serving(body))) == [Done(7, 3)]


async def test_non_dict_tool_call_is_skipped() -> None:
    body = _ndjson(_call_line("x"), DONE_LINE)
    assert await _events(_provider(_serving(body))) == [Done(7, 3)]


async def test_non_dict_function_yields_a_call_with_an_empty_name() -> None:
    body = _ndjson(_call_line({"id": "c1", "function": "x"}), DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events == [ToolCallEvent(ToolCall("c1", "", {})), Done(7, 3)]


@pytest.mark.parametrize("name", [None, 5, ["a"], ""])
async def test_non_string_or_empty_tool_name_becomes_an_empty_name(name: Any) -> None:
    body = _ndjson(_call_line({"id": "c1", "function": {"name": name, "arguments": {}}}), DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events[0] == ToolCallEvent(ToolCall("c1", "", {}))


@pytest.mark.parametrize("call_id", [None, 5, ["a"], ""])
async def test_non_string_tool_call_id_is_replaced_with_a_generated_one(call_id: Any) -> None:
    body = _ndjson(_call_line({"id": call_id, "function": {"name": "read", "arguments": {}}}), DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events[0].call.id.startswith("call_") and events[0].call.name == "read"


@pytest.mark.parametrize(
    ("prompt", "completion", "expected"),
    [("x", "y", (0, 0)), ([1], {"a": 1}, (0, 0)), (None, None, (0, 0)), (4.0, "5", (4, 5))],
)
async def test_token_counts_are_coerced_to_int_or_zero(prompt: Any, completion: Any, expected: tuple[int, int]) -> None:
    body = _ndjson(
        {"message": {"content": ""}, "done": True, "prompt_eval_count": prompt, "eval_count": completion}
    )
    assert await _events(_provider(_serving(body))) == [Done(*expected)]


async def test_unmappable_chunk_yields_an_unexpected_response_error() -> None:
    body = _ndjson({"message": {"tool_calls": 5}, "done": False}, DONE_LINE)
    events = await _events(_provider(_serving(body)))
    assert events == [ProviderError("Ollama sent an unexpected response")]


class _TrackedStream(httpx.AsyncByteStream):
    def __init__(self, first: bytes, hang: bool = True) -> None:
        self.first = first
        self.hang = hang
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.first
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self) -> None:
        self.closed = True


def _tracked_provider(stream: _TrackedStream) -> OllamaProvider:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, stream=stream)

    return _provider(handler)


async def test_consumer_stopping_early_closes_the_response() -> None:
    stream = _TrackedStream(_ndjson({"message": {"content": "a"}, "done": False}))
    events = cast(AsyncGenerator, _tracked_provider(stream).stream([Message("user", "hi")], []))
    assert await anext(events) == TextDelta("a")
    await events.aclose()
    assert stream.closed


async def test_cancelling_mid_stream_closes_the_response() -> None:
    stream = _TrackedStream(_ndjson({"message": {"content": "a"}, "done": False}))
    provider = _tracked_provider(stream)
    got_first = asyncio.Event()

    async def consume() -> None:
        async for _ in provider.stream([Message("user", "hi")], []):
            got_first.set()

    task = asyncio.create_task(consume())
    await got_first.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stream.closed
