"""Engine, session factory and transaction helpers.

THE SQLALCHEMY CHAIN
--------------------
    Database URL
        |  build_engine()
        v
    Engine            one per process; owns the connection pool
        |
        v
    Connection pool   reuses TCP connections instead of reconnecting per query
        |  sessionmaker()
        v
    Session           a "unit of work": tracks objects, batches writes,
        |             owns one transaction at a time
        v
    Transaction       BEGIN ... COMMIT / ROLLBACK
        |
        v
    ORM objects / queries

SESSION LIFECYCLE (per API request)
-----------------------------------
    open session
      -> run the request's reads / writes
      -> no error  -> COMMIT
      -> exception -> ROLLBACK   (never leave a half-written transaction)
    always -> close (return the connection to the pool)

`session_scope()` implements exactly that for non-request callers (the demo,
the benchmark, ``init_db``). The FastAPI dependency in ``robotics.api`` does
the same thing per request.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from robotics.common.retry import RetryError, RetryPolicy, retry_call
from robotics.persistence.config import Settings, get_settings

logger = logging.getLogger("robotics.persistence.database")


class Base(DeclarativeBase):
    """Declarative base class every ORM model inherits from.

    Holds the shared ``MetaData`` - the in-memory description of every table -
    which ``Base.metadata.create_all(engine)`` uses to build the schema.
    """


def build_engine(settings: Settings | None = None) -> Engine:
    """Create a SQLAlchemy `Engine` for the configured database.

    SQLite needs two extra touches that a real server does not:
      * ``check_same_thread=False`` so the TestClient (which may touch the
        session from a worker thread) works;
      * a ``PRAGMA foreign_keys=ON`` on every connection, because SQLite
        ignores foreign keys unless asked - PostgreSQL always enforces them.
    """
    settings = settings or get_settings()

    connect_args: dict = {}
    engine_kwargs: dict = {
        "echo": settings.sql_echo,
        "future": True,
        # pre_ping issues a cheap check-out probe and silently replaces a
        # connection the server has dropped - the single most useful reliability
        # setting for a long-lived pool.
        "pool_pre_ping": True,
    }

    if settings.is_sqlite:
        connect_args["check_same_thread"] = False
    else:
        # A real server: size the pool explicitly (and modestly) instead of
        # relying on the driver defaults, and recycle old connections.
        engine_kwargs.update(
            pool_size=settings.db_pool_size,
            max_overflow=settings.db_max_overflow,
            pool_timeout=settings.db_pool_timeout_s,
            pool_recycle=settings.db_pool_recycle_s,
        )

    engine = create_engine(
        settings.database_url, connect_args=connect_args, **engine_kwargs
    )

    if settings.is_sqlite:
        _enforce_sqlite_foreign_keys(engine)

    return engine


def _enforce_sqlite_foreign_keys(engine: Engine) -> None:
    """Run ``PRAGMA foreign_keys=ON`` on every new SQLite connection.

    SQLite ignores foreign keys unless this is set per-connection; PostgreSQL
    always enforces them. Registering the listener more than once is harmless.
    """

    @event.listens_for(engine, "connect")
    def _enable(dbapi_connection, _record):  # pragma: no cover - trivial
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


class Database:
    """Bundles an engine and its session factory.

    One `Database` per process (or per test). ``create_all`` builds the schema;
    ``session()`` hands out sessions bound to this engine.
    """

    def __init__(self, settings: Settings | None = None, engine: Engine | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = engine or build_engine(self.settings)
        # An engine passed in by a test skips build_engine, so make sure SQLite
        # foreign keys are enforced here too (PostgreSQL always enforces them).
        if self.settings.is_sqlite:
            _enforce_sqlite_foreign_keys(self.engine)
        self.session_factory = sessionmaker(
            bind=self.engine, autoflush=False, autocommit=False, future=True
        )

    def create_all(self) -> None:
        """Create every table that does not yet exist."""
        Base.metadata.create_all(self.engine)

    def drop_all(self) -> None:
        """Drop every table. Used by tests for a clean slate."""
        Base.metadata.drop_all(self.engine)

    def session(self) -> Session:
        """A new session bound to this database's engine."""
        return self.session_factory()

    def ping(self) -> bool:
        """Lightweight connectivity check for the readiness endpoint.

        Runs ``SELECT 1``. Returns True on success, False on any failure -
        it never raises, because readiness must report "not ready", not crash.
        """
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:  # noqa: BLE001 - readiness must not propagate
            return False

    def wait_until_ready(self, *, sleep=None) -> bool:
        """Try to reach the database a few times with exponential backoff.

        Called once from the app lifespan on startup: a container's database
        often takes a second or two longer to accept connections than the API
        process. Returns True as soon as a connection succeeds, False if every
        attempt failed. NEVER raises and NEVER blocks forever - the app starts
        either way and ``/ready`` reports the truth from then on.
        """
        policy = RetryPolicy(
            attempts=self.settings.db_connect_attempts,
            base_delay=self.settings.db_connect_base_delay_s,
            max_delay=self.settings.db_connect_max_delay_s,
            jitter=0.0,
        )

        def _probe() -> None:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))

        kwargs = {"sleep": sleep} if sleep is not None else {}
        try:
            retry_call(
                _probe, policy=policy, operation="database.connect", **kwargs
            )
            return True
        except RetryError as exc:
            logger.warning(
                "database.startup_unreachable",
                extra={
                    "attempts": policy.attempts,
                    "error_type": type(exc.last_exc).__name__,
                },
            )
            return False

    def dispose(self) -> None:
        """Close every pooled connection. Called from the app lifespan on
        shutdown so the process exits without leaking sockets."""
        self.engine.dispose()
        logger.info("database.disposed")

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        """A transactional session: commit on success, roll back on error."""
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()


@contextmanager
def session_scope(database: Database | None = None) -> Iterator[Session]:
    """Module-level convenience wrapper around ``Database.session_scope``.

    Builds a default `Database` from the environment if one is not supplied.
    """
    database = database or Database()
    with database.session_scope() as session:
        yield session
