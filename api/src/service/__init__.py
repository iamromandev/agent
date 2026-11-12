from collections.abc import AsyncGenerator

from src.service.health.health_service import HealthService


async def get_health_service() -> AsyncGenerator[HealthService]:
    yield HealthService()
