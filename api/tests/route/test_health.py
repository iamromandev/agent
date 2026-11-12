"""``GET /health/check`` answers the success envelope around the service's report."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient
from src.core.error import init_global_errors
from src.core.type import Status
from src.data.schema.health import DatabaseSchema, HealthSchema
from src.route import router
from src.service import get_health_service


class _StubHealthService:
    async def check_health(self, host: str | None = None, port: int | None = None) -> HealthSchema:
        return HealthSchema(
            version="9.9.9",
            host=host,
            port=port,
            environment="local",
            uptime="1s",
            db=DatabaseSchema(status=Status.SUCCESS, version="PostgreSQL 18"),
        )


def _client() -> TestClient:
    app = FastAPI()
    init_global_errors(app)
    app.include_router(router)
    app.dependency_overrides[get_health_service] = lambda: _StubHealthService()
    return TestClient(app)


def test_health_check_wraps_the_report_in_a_success_envelope() -> None:
    response = _client().get("/health/check")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["code"] == 200
    assert body["data"]["version"] == "9.9.9"
    assert body["data"]["db"] == {"status": "success", "version": "PostgreSQL 18"}

