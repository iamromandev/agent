from __future__ import annotations

import asyncio

from src.agent.provider.base import Done, Message, Provider, ProviderError, TextDelta, ToolSpec

from tests.fakes import ScriptedProvider


def test_scripted_provider_satisfies_the_provider_protocol() -> None:
    provider: Provider = ScriptedProvider([])
    assert provider.model == "fake"
    assert provider.context_window == 1000


async def test_it_records_requests_and_replays_the_script() -> None:
    provider = ScriptedProvider([[TextDelta("hi"), Done(1, 2)]])
    messages = [Message("user", "q")]
    tools = [ToolSpec("read", "d", {"type": "object"})]

    events = [event async for event in provider.stream(messages, tools)]
    messages.append(Message("user", "later"))

    assert events == [TextDelta("hi"), Done(1, 2)]
    assert provider.requests == [([Message("user", "q")], ["read"])]


async def test_an_exhausted_script_yields_an_error() -> None:
    provider = ScriptedProvider([])
    events = [event async for event in provider.stream([], [])]
    assert events == [ProviderError("script exhausted")]


async def test_check_returns_the_configured_result() -> None:
    assert await ScriptedProvider([], check_result="down").check() == "down"
    assert await ScriptedProvider([]).check() is None


async def test_delay_sleeps_between_events() -> None:
    provider = ScriptedProvider([[TextDelta("a"), Done(0, 0)]], delay=0.05)
    loop = asyncio.get_running_loop()
    start = loop.time()
    _ = [event async for event in provider.stream([], [])]
    assert loop.time() - start >= 0.09
