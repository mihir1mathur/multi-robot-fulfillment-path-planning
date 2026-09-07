"""Structured application logging.

WHY STRUCTURED
--------------
A line like ``robot.created robot_id=R1 position=[0, 1]`` can be grepped and
parsed; a hand-formatted English sentence cannot. Each log record is emitted as
one JSON object: timestamp, level, logger, message, plus whatever key/values
the call passed in ``extra=``.

WHAT IS NEVER LOGGED
-------------------
Passwords, database URLs with credentials, environment secrets. The config
object logs only ``settings.safe_database_url()`` (password redacted). There is
also no logging inside the path-planning inner loops - only at service / API
boundaries - so a busy planner does not drown the log.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from typing import Any

# LogRecord attributes that are always present; anything else on the record was
# passed by the caller via ``extra=`` and should be surfaced.
_STANDARD_ATTRS = set(
    logging.makeLogRecord({}).__dict__
) | {"message", "asctime", "taskName"}

# The correlation id of the request currently being handled. The API middleware
# sets this at the start of every request; every log line emitted while handling
# that request then carries it automatically, without each call site having to
# pass ``correlation_id=`` by hand. A ContextVar is task/thread-local, so
# concurrent requests never see each other's id.
correlation_id_var: ContextVar[str | None] = ContextVar("correlation_id", default=None)


class CorrelationIdFilter(logging.Filter):
    """Attach the current request's correlation id to every record that does
    not already carry one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "correlation_id"):
            current = correlation_id_var.get()
            if current is not None:
                record.correlation_id = current
        return True


class JsonLogFormatter(logging.Formatter):
    """Render each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_ATTRS or key.startswith("_"):
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except TypeError:
                payload[key] = repr(value)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON formatter on the ``robotics`` logger once.

    Idempotent: calling it again only adjusts the level, so repeated app
    creation in tests does not stack handlers.
    """
    global _CONFIGURED
    root = logging.getLogger("robotics")
    root.setLevel(level.upper())

    if not _CONFIGURED:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(JsonLogFormatter())
        handler.addFilter(CorrelationIdFilter())
        root.addHandler(handler)
        root.propagate = False
        _CONFIGURED = True
    else:
        for handler in root.handlers:
            handler.setLevel(level.upper())
