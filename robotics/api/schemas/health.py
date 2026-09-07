"""Health and readiness schemas.

HEALTH != READINESS
-------------------
    /health  - "is this process alive and serving?"  Never touches the
               database. Used by a load balancer to decide whether to keep the
               instance in rotation / restart it.
    /ready   - "can this process serve DB-backed requests RIGHT NOW?"  Runs a
               real ``SELECT 1``. Used to decide whether to send traffic yet
               (e.g. just after deploy, before the DB connection is up).

A process can be healthy but not ready (DB still starting). Returning ready
when the database is down would send traffic that only 500s.
"""

from __future__ import annotations

from typing import Dict

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str


class ReadinessResponse(BaseModel):
    status: str  # "ready" | "not_ready"
    checks: Dict[str, str]  # e.g. {"database": "ok"} / {"database": "unavailable"}
