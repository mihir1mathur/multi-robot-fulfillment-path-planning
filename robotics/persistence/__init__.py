"""Database persistence for fleet state and operation history.

WHAT THIS PACKAGE OWNS
----------------------
Turning long-lived facts - the robots in the fleet, the tasks in the queue,
and a history of every planning / allocation / coordination / recovery run -
into rows in a relational database, and back again.

    domain objects  (robotics.robots.Robot, robotics.tasks.Task, ...)
            |  mapped by
            v
    ORM records     (RobotRecord, TaskRecord, PlanningRun, ...)   models.py
            |  read / written through
            v
    repositories    (RobotRepository, TaskRepository, RunRepository)
            |  using a
            v
    SQLAlchemy Session   -> Engine -> connection pool -> PostgreSQL / SQLite

WHAT IT DOES NOT OWN
-------------------
No path planning, no optimisation, no coordination logic. This layer stores
and retrieves; the `robotics.services` layer decides what to compute and calls
the existing `robotics.planning` / `robotics.allocation` / `robotics.coordination`
/ `robotics.recovery` code.

PRODUCTION TARGET vs TESTS
-------------------------
The production target is PostgreSQL (``postgresql+psycopg://...`` in
``DATABASE_URL``). The schema is plain SQLAlchemy Core/ORM with no
PostgreSQL-only column types, so the same models run on SQLite, which is what
the deterministic unit / service / API tests use (an isolated in-memory
database per test). Tests that would need PostgreSQL-specific behaviour are
marked and skipped when no PostgreSQL URL is configured.
"""

from robotics.persistence.config import Settings, get_settings
from robotics.persistence.database import (
    Base,
    Database,
    build_engine,
    session_scope,
)
from robotics.persistence.models import (
    AllocationRun,
    CoordinationRun,
    PlanningRun,
    RecoveryEvent,
    RobotRecord,
    TaskRecord,
    UserRecord,
)
from robotics.persistence.repositories import (
    RobotRepository,
    RunRepository,
    TaskRepository,
)

__all__ = [
    "Settings",
    "get_settings",
    "Base",
    "Database",
    "build_engine",
    "session_scope",
    "RobotRecord",
    "UserRecord",
    "TaskRecord",
    "PlanningRun",
    "AllocationRun",
    "CoordinationRun",
    "RecoveryEvent",
    "RobotRepository",
    "TaskRepository",
    "RunRepository",
]
