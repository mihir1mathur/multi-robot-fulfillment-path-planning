"""HTTP API for the fulfillment / path-planning system.

    HTTP request
        |
        v
    FastAPI router        (robotics/api/routers/*.py)  - parse + shape only
        |
        v
    Pydantic validation   (robotics/api/schemas/*.py)  - reject bad input
        |
        v
    service layer         (robotics/services/*.py)     - orchestrate
        |
        +----------------------------+
        |                            |
        v                            v
    robotics.planning /         robotics.persistence
    allocation / coordination   (SQLAlchemy -> PostgreSQL / SQLite)
    / recovery
        |
        v
    Pydantic response schema -> HTTP response

The router layer contains NO robotics algorithms and NO SQL. Build the app
with `create_app()`.
"""

from robotics.api.app import create_app

__all__ = ["create_app"]
