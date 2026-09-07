"""Health and readiness endpoints.

    GET /health  - liveness. Does the process respond? No DB call. Always 200
                   while the app is up.
    GET /ready   - readiness. Can it serve DB-backed requests? Runs SELECT 1.
                   200 when the database answers, 503 when it does not.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response, status

from robotics.api.dependencies import get_database, get_settings
from robotics.api.schemas.health import HealthResponse, ReadinessResponse
from robotics.persistence.config import Settings
from robotics.persistence.database import Database

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness probe - the process is up and the event loop is turning."""
    return HealthResponse(status="ok", app=settings.app_name)


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={503: {"model": ReadinessResponse}},
)
def ready(
    response: Response,
    database: Database = Depends(get_database),
) -> ReadinessResponse:
    """Readiness probe - the database is reachable right now."""
    if database.ping():
        return ReadinessResponse(status="ready", checks={"database": "ok"})
    response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(
        status="not_ready", checks={"database": "unavailable"}
    )
