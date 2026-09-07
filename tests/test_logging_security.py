"""Structured logging: it carries the right context and never a secret.

Covered:
  * the per-request correlation id reaches EVERY log line emitted while that
    request is handled (via the ContextVar + logging filter), and is returned
    in the ``x-correlation-id`` response header
  * a caller-supplied ``x-correlation-id`` is honoured
  * passwords, tokens, password hashes and connection strings never appear in
    any log record

The ``robotics`` logger has ``propagate=False``, so pytest's ``caplog`` (which
captures at the root logger) does not see its records. These tests attach their
own collecting handler directly to the ``robotics`` logger instead.
"""

from __future__ import annotations

import json
import logging
import contextlib

from robotics.api.logging_config import CorrelationIdFilter, correlation_id_var


@contextlib.contextmanager
def _collect_robotics_logs(level=logging.INFO):
    """Capture every record the ``robotics`` logger tree emits, with the same
    CorrelationIdFilter the production handler uses.

    ``robotics`` has ``propagate=False`` and other tests / pytest's logging
    plugin move levels and the global ``logging.disable`` around, so this
    pins everything it needs and restores it afterwards.
    """
    records: list[logging.LogRecord] = []

    class _Collector(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Collector()
    handler.addFilter(CorrelationIdFilter())
    lg = logging.getLogger("robotics")
    previous_level = lg.level
    previous_disable = logging.root.manager.disable
    logging.disable(logging.NOTSET)   # undo any global suppression
    lg.setLevel(level)
    lg.addHandler(handler)
    # A prior in-process `alembic upgrade` calls logging.config.fileConfig, which
    # sets `.disabled = True` on every logger it does not itself name - including
    # this project's "robotics.*" tree. Re-enable it for the capture and put it
    # back exactly as it was afterwards.
    disabled_loggers = [
        obj
        for name, obj in logging.Logger.manager.loggerDict.items()
        if name == "robotics" or name.startswith("robotics.")
        if isinstance(obj, logging.Logger) and obj.disabled
    ]
    for obj in disabled_loggers:
        obj.disabled = False
    try:
        yield records
    finally:
        lg.removeHandler(handler)
        lg.setLevel(previous_level)
        logging.disable(previous_disable)
        for obj in disabled_loggers:
            obj.disabled = True


def _joined(records) -> str:
    return "\n".join(
        r.getMessage() + " " + json.dumps(getattr(r, "__dict__", {}), default=str)
        for r in records
    )


# --- correlation id -------------------------------------------
def test_response_carries_a_correlation_id_header(api_client):
    assert api_client.get("/health").headers.get("x-correlation-id")


def test_a_supplied_correlation_id_is_echoed_back(api_client):
    r = api_client.get("/health", headers={"x-correlation-id": "trace-abc-123"})
    assert r.headers["x-correlation-id"] == "trace-abc-123"


def test_correlation_id_is_attached_to_service_log_lines(api_client):
    """A log line emitted deep in the service layer (robot.created) carries the
    request's correlation id, without that call site passing it explicitly."""
    with _collect_robotics_logs() as records:
        api_client.post(
            "/robots",
            json={"robot_id": "R1", "position": {"row": 0, "col": 0}},
            headers={"x-correlation-id": "corr-xyz-999"},
        )

    created = [r for r in records if r.getMessage() == "robot.created"]
    assert created, "expected a robot.created log line"
    assert getattr(created[0], "correlation_id", None) == "corr-xyz-999"

    # and the middleware's own summary line has it too
    http_lines = [r for r in records if r.getMessage() == "http.request"]
    assert http_lines and http_lines[0].correlation_id == "corr-xyz-999"


def test_context_var_is_reset_after_the_request(api_client):
    api_client.get("/health")
    assert correlation_id_var.get() is None


# --- no secrets in logs --------------------------------------
def test_logs_never_contain_a_password_token_or_connection_string(api_client):
    with _collect_robotics_logs(level=logging.DEBUG) as records:
        api_client.post("/auth/login",
                        json={"username": "operator_user", "password": "operator-pw-123"})
        api_client.get("/robots")
        api_client.get("/auth/me")

    text = _joined(records)
    assert "operator-pw-123" not in text
    assert "password_hash" not in text
    assert "$2b$" not in text                # a bcrypt hash prefix
    assert "sqlite+pysqlite" not in text     # the connection string
    assert "jwt_secret" not in text.lower()
    # a raw JWT (three base64 segments starting 'eyJ') must never be logged
    assert "eyJ" not in text


def test_app_start_log_uses_the_redacted_url():
    from robotics.persistence.config import Settings
    from robotics.persistence.database import Database

    settings = Settings(
        database_url="postgresql+psycopg://fulfillment:sup3r-s3cret@db:5432/fulfillment",
        jwt_secret_key="x" * 40,
        auto_create_tables=False,
    )
    with _collect_robotics_logs() as records:
        from robotics.api.app import create_app

        with contextlib.suppress(Exception):
            create_app(settings=settings, database=Database(settings=settings),
                       create_tables=False)

    text = _joined(records)
    assert "sup3r-s3cret" not in text
    assert "fulfillment:***@db:5432" in text
