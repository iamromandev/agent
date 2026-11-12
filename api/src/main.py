from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_settings
from src.config.logging import configure_logging
from src.core.common import get_app_version
from src.core.error import init_global_errors
from src.data.db import init_db
from src.middleware import trust_proxies
from src.route import router as _router


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """App-specific startup/shutdown; Tortoise is registered via ``init_db`` (merged lifespan).

    Tortoise's own startup runs first — ``register_tortoise`` merges it ahead of
    this one — so the database is reachable here.
    """
    yield


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    app = FastAPI(
        title="Agent service",
        description="Agent API.",
        version=get_app_version(),
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Outermost, so every layer and route below sees the caller rather than a load balancer.
    trust_proxies(app, settings.trusted_proxy_list)

    init_global_errors(app)

    _routers = [
        _router
    ]
    for router in _routers:
        app.include_router(router)

    # Must run while building the app (not inside lifespan): ``register_tortoise`` merges
    # its init/shutdown into ``lifespan``; calling it from inside lifespan is too late.
    init_db(app)

    return app


app = create_app()


def run() -> None:
    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        loop="uvloop",
    )
