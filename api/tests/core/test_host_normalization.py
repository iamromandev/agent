"""One spelling of a host for every comparison the service makes."""

from __future__ import annotations

from src.core.common import host_of, normalize_host


def test_a_url_reduces_to_its_comparable_host() -> None:
    assert normalize_host(host_of("http://Login-AB.Test:3000/")) == "login-ab.test:3000"
    assert normalize_host(host_of("https://wise-crane-8343.login.exateks.com")) == (
        "wise-crane-8343.login.exateks.com"
    )
    assert normalize_host(host_of("http://localhost.:3000")) == "localhost:3000"

