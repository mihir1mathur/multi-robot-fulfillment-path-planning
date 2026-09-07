"""Application startup / shutdown (the FastAPI ``lifespan``).

    startup   - probe the database (bounded retry); the app starts either way
    shutdown  - dispose the connection pool so the process exits clean

and the database helpers the lifespan uses:

    Database.wait_until_ready()  - retry a SELECT 1, never raise, never hang
    Database.dispose()           - close every pooled connection
"""

from __future__ import annotations

from robotics.common.retry import RetryError
from robotics.persistence.config import Settings
from robotics.persistence.database import Database


# --- wait_until_ready ------------------------------------------
def test_wait_until_ready_is_true_for_a_live_database(api_database):
    assert api_database.wait_until_ready(sleep=lambda _s: None) is True


def test_wait_until_ready_is_false_and_does_not_raise_when_unreachable(monkeypatch):
    settings = Settings(
        database_url="sqlite+pysqlite://",
        db_connect_attempts=3,
        db_connect_base_delay_s=0.0,
        jwt_secret_key="x" * 32,
        auto_create_tables=False,
    )
    database = Database(settings=settings)

    def _always_fails():
        raise OSError("connection refused")

    monkeypatch.setattr(database.engine, "connect", _always_fails)
    slept: list[float] = []
    assert database.wait_until_ready(sleep=slept.append) is False
    assert len(slept) == 2  # 3 attempts, slept between them, not after the last


def test_wait_until_ready_recovers_if_the_database_appears_mid_retry(monkeypatch, api_database):
    calls = {"n": 0}
    real_connect = api_database.engine.connect

    def _flaky_connect(*a, **k):
        calls["n"] += 1
        if calls["n"] < 2:
            raise OSError("not up yet")
        return real_connect(*a, **k)

    monkeypatch.setattr(api_database.engine, "connect", _flaky_connect)
    assert api_database.wait_until_ready(sleep=lambda _s: None) is True
    assert calls["n"] == 2


# --- dispose --------------------------------------------------
def test_dispose_closes_the_pool_and_is_idempotent(api_database):
    api_database.dispose()
    api_database.dispose()  # second call must not raise
    # the engine still works afterwards (a fresh connection is made on demand)
    assert api_database.ping() is True


# --- the lifespan actually runs -------------------------------
def test_lifespan_startup_and_shutdown_fire(api_app, monkeypatch):
    from fastapi.testclient import TestClient

    events: list[str] = []
    real_wait = api_app.state.database.wait_until_ready
    real_dispose = api_app.state.database.dispose

    monkeypatch.setattr(
        api_app.state.database, "wait_until_ready",
        lambda **k: (events.append("startup"), real_wait(**k))[1],
    )
    monkeypatch.setattr(
        api_app.state.database, "dispose",
        lambda: (events.append("shutdown"), real_dispose())[1],
    )

    with TestClient(api_app) as client:
        assert client.get("/health").status_code == 200
        assert events == ["startup"]
    assert events == ["startup", "shutdown"]


def test_app_starts_and_serves_even_if_the_database_is_down_at_boot(api_database, monkeypatch):
    """Startup probe fails -> app still starts -> /health ok, /ready 503."""
    from fastapi.testclient import TestClient

    from robotics.api.app import create_app

    monkeypatch.setattr(api_database, "wait_until_ready", lambda **k: False)
    monkeypatch.setattr(api_database, "ping", lambda: False)
    app = create_app(settings=api_database.settings, database=api_database, create_tables=False)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 503
