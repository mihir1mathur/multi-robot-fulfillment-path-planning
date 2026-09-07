"""FastAPI application factory.

    create_app()  ->  a configured FastAPI instance

WHY A FACTORY (not a module-level ``app = FastAPI()``)
-----------------------------------------------------
A factory takes parameters (a test `Settings`, a test `Database`), so the same
wiring builds the production app and an isolated test app. Nothing global is
created at import time, so importing this module has no side effects.

WHAT create_app WIRES UP
------------------------
    1. structured logging
    2. Settings + Database on ``app.state`` (one engine per process)
    3. the schema:
         - dev / tests: ``Base.metadata.create_all`` (``auto_create_tables``)
         - production / Docker: NOTHING - Alembic owns the schema
           (``alembic upgrade head`` runs before the app starts)
    4. an optional bootstrap admin (so a fresh database is never locked out)
    5. a correlation-id + timing middleware
    6. the exception handlers (one error body shape)
    7. every router, including ``/auth``
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from robotics.api.errors import install_exception_handlers
from robotics.api.logging_config import configure_logging, correlation_id_var
from robotics.api.routers import (
    allocation,
    auth,
    coordination,
    health,
    planning,
    recovery,
    robots,
    tasks,
)
from robotics.persistence.config import Settings, get_settings
from robotics.persistence.database import Database

# Importing the models registers every table on ``Base.metadata``.
from robotics.persistence import models  # noqa: F401
from robotics.services.auth_service import AuthService

logger = logging.getLogger("robotics.api")

API_TITLE = "Autonomous Fulfillment & Path-Planning API"
API_VERSION = "1.1.0"
API_DESCRIPTION = (
    "REST access to the deterministic warehouse simulation: robot and task "
    "records, single-robot path planning (A* / Dijkstra), fleet task "
    "allocation (greedy / CP-SAT), prioritized multi-robot coordination, and "
    "dynamic replanning / fault recovery. Reads and writes fleet state and run "
    "history to PostgreSQL. The API layer orchestrates the existing algorithms "
    "and enforces JWT authentication + role-based authorization; it contains no "
    "planning or optimisation logic of its own."
)


def create_app(
    settings: Settings | None = None,
    database: Database | None = None,
    create_tables: bool | None = None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    database = database or Database(settings)

    # `create_tables` argument wins if given; otherwise follow the setting.
    should_create = (
        settings.auto_create_tables if create_tables is None else create_tables
    )
    if should_create:
        database.create_all()

    if settings.is_postgres and settings.jwt_secret_is_default:
        logger.warning(
            "app.insecure_jwt_secret",
            extra={"hint": "set JWT_SECRET_KEY for any non-local environment"},
        )

    # Bootstrap admin (only creates one if configured AND the users table is empty).
    try:
        with database.session_scope() as session:
            created = AuthService(session, settings).ensure_bootstrap_admin()
        if created:
            logger.info("app.bootstrap_admin", extra={"username": created})
    except Exception:  # noqa: BLE001 - a DB that is not ready must not stop import-time wiring
        logger.warning("app.bootstrap_admin_skipped", extra={"reason": "database not ready"})

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # --- startup ---------------------------------------------------
        # Wait (briefly, with backoff) for the database. The app starts even if
        # it never answers; /ready then reports 503 until it does.
        db_ready = database.wait_until_ready()
        logger.info(
            "app.ready",
            extra={"database_ready": db_ready, "version": API_VERSION},
        )
        try:
            yield
        finally:
            # --- shutdown ---------------------------------------------
            # Return every pooled connection so the process exits cleanly.
            database.dispose()
            logger.info("app.stopped", extra={"app": settings.app_name})

    app = FastAPI(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        lifespan=_lifespan,
    )
    app.state.settings = settings
    app.state.database = database

    @app.middleware("http")
    async def _correlate_and_time(request: Request, call_next):
        correlation_id = request.headers.get("x-correlation-id") or uuid.uuid4().hex[:12]
        request.state.correlation_id = correlation_id
        token = correlation_id_var.set(correlation_id)
        started = time.perf_counter_ns()
        try:
            response = await call_next(request)
        finally:
            correlation_id_var.reset(token)
        duration_ms = (time.perf_counter_ns() - started) / 1_000_000
        response.headers["x-correlation-id"] = correlation_id
        logger.info(
            "http.request",
            extra={
                "correlation_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(duration_ms, 3),
            },
        )
        return response

    install_exception_handlers(app)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(robots.router)
    app.include_router(tasks.router)
    app.include_router(planning.router)
    app.include_router(allocation.router)
    app.include_router(coordination.router)
    app.include_router(recovery.router)

    logger.info(
        "app.started",
        extra={
            "app": settings.app_name,
            "database": settings.safe_database_url(),
            "version": API_VERSION,
        },
    )
    return app


# Run in production with:
#     alembic upgrade head
#     uvicorn robotics.api.app:create_app --factory
# The factory reads DATABASE_URL, JWT_SECRET_KEY (and the other settings) from
# the environment. Importing this module has NO side effects - no engine, no
# file, no schema - so tests build their own app with
# create_app(settings=..., database=...).
