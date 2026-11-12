"""Headers set on an HTTPException must survive the error envelope.

This service answers errors with a rendered envelope rather than FastAPI's
own body, so the handler rebuilds the response — and a rebuilt response only
carries the headers it is told to carry. It used to be told about exactly one
name, ``WWW-Authenticate``, which was indistinguishable from correct for as
long as that was the only header anything set.

The login lockout added ``Retry-After`` and it vanished on the way out. The
body said "try again later" and nothing said when; the tests were green and
the types checked, because neither can see a header that was never copied.

So this pins the rule rather than the names: whatever the raiser set arrives.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from src.core.error import init_global_errors


def _client() -> TestClient:
    app = FastAPI()
    init_global_errors(app)

    @app.get("/locked")
    async def locked() -> None:
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Try again later.",
            headers={"Retry-After": "845"},
        )

    @app.get("/unauthorized")
    async def unauthorized() -> None:
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Basic"},
        )

    @app.get("/plain")
    async def plain() -> None:
        raise HTTPException(status_code=404, detail="Nothing here")

    return TestClient(app, raise_server_exceptions=False)


def test_retry_after_reaches_the_caller() -> None:
    resp = _client().get("/locked")
    assert resp.status_code == 429
    assert resp.headers["Retry-After"] == "845"
