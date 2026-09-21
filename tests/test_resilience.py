"""Behaviour when the database is unavailable, and secret redaction.

    - /ready must report 503 (never a false 'ready')
    - a DB error DURING a request must surface as a controlled 500 with no
      secret / connection string / stack trace in the body
    - the redacted URL helper never reveals the password
    - the structured logger never emits the password or a token
"""

from __future__ import annotations

import json
import logging

from robotics.persistence.config import Settings


# --- readiness when the database is down --------------------------
def test_readiness_is_503_when_the_database_ping_fails(api_app, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_app.state.database, "ping", lambda: False)
    with TestClient(api_app) as client:
        r = client.get("/ready")
    assert r.status_code == 503
    assert r.json() == {"status": "not_ready", "checks": {"database": "unavailable"}}


def test_health_still_ok_when_the_database_is_down(api_app, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setattr(api_app.state.database, "ping", lambda: False)
    with TestClient(api_app) as client:
        assert client.get("/health").status_code == 200


def test_a_database_error_mid_request_is_a_controlled_500_without_secrets(
    api_app, api_database, monkeypatch
):
    from fastapi.testclient import TestClient

    from tests.conftest import _token_for

    # Mint the token BEFORE the database is broken (needs a working session).
    token = _token_for(api_database, "operator")

    def _boom():
        raise RuntimeError("connection to server at localhost:5432 failed: password authentication failed for user 'fulfillment'")

    monkeypatch.setattr(api_database, "session", _boom)

    # raise_server_exceptions=False surfaces the app's real 500 response body
    # (the controlled error) instead of TestClient re-raising for the caller.
    with TestClient(api_app, raise_server_exceptions=False) as client:
        client.headers["Authorization"] = f"Bearer {token}"
        # session() is patched, so even resolving get_current_user fails
        r = client.get("/robots")
    assert r.status_code == 500
    body = r.text.lower()
    assert "password" not in body
    assert "5432" not in body
    assert "traceback" not in body
    assert "incident" in r.json()["message"]


# --- redaction ---------------------------------------------------
def test_safe_database_url_hides_the_password():
    s = Settings(database_url="postgresql+psycopg://fulfillment:s3cr3t-pw@db.local:5432/fulfillment")
    safe = s.safe_database_url()
    assert "s3cr3t-pw" not in safe
    assert safe == "postgresql+psycopg://fulfillment:***@db.local:5432/fulfillment"


def test_safe_database_url_handles_a_url_with_no_credentials():
    s = Settings(database_url="sqlite+pysqlite:///./fulfillment.db")
    assert s.safe_database_url() == "sqlite+pysqlite:///./fulfillment.db"


# --- logging never leaks secrets --------------------------------
def test_structured_logger_output_contains_no_password_or_token(api_client, caplog):
    """Drive an auth failure + a normal request; scan every captured log line."""
    api_client.get("/auth/me")  # authenticated request -> http.request log line
    with caplog.at_level(logging.INFO, logger="robotics"):
        api_client.post("/auth/login", json={"username": "operator_user", "password": "operator-pw-123"})
        api_client.get("/robots")

    joined = "\n".join(r.getMessage() + " " + json.dumps(getattr(r, "__dict__", {}), default=str)
                       for r in caplog.records)
    assert "operator-pw-123" not in joined       # no plaintext password
    assert "Bearer " not in joined                # no Authorization header value
    assert "password_hash" not in joined
