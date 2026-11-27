from __future__ import annotations

from core.engine.approval import Grants
from core.engine.mode import Mode


def test_a_keyed_grant_is_not_covered_in_plan_mode() -> None:
    grants = Grants()
    grants.add("web_fetch", {"url": "u"}, key="url")
    assert not grants.covers("web_fetch", {"url": "u"}, Mode.PLAN, key="url")


def test_revoking_a_tool_removes_its_tool_grant_and_its_keyed_grants() -> None:
    grants = Grants()
    grants.add("mcp__a__t", {})
    grants.add("web_fetch", {"url": "https://x"}, "url")
    grants.add("other", {})

    grants.revoke("mcp__a__t")
    grants.revoke("web_fetch")

    assert grants.covers("mcp__a__t", {}, Mode.BUILD) is False
    assert grants.covers("web_fetch", {"url": "https://x"}, Mode.BUILD, "url") is False
    assert grants.covers("other", {}, Mode.BUILD) is True


def test_an_approval_request_names_no_subagent_by_default() -> None:
    from core.engine.approval import ApprovalRequest

    assert ApprovalRequest("bash", "bash ls", None, True).subagent == ""
