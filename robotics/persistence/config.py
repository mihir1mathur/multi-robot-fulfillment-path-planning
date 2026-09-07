"""Typed, environment-driven configuration.

WHY A TYPED SETTINGS OBJECT
--------------------------
Reading ``os.environ`` in a dozen places gives no defaults, no validation and
no single list of what the app needs. `Settings` is a pydantic-settings model:
every value has a type and a default, it is populated from the environment (or
a local ``.env`` file), and an invalid value fails loudly at startup instead of
deep inside a request.

SECRETS
-------
`database_url` and `jwt_secret_key` may contain sensitive material. They are
read from the environment only - never hard-coded, never logged. `.env` is
git-ignored; `.env.example` carries placeholders only.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, populated from the environment / ``.env``.

    Attributes:
        database_url: SQLAlchemy URL for the database. The production target is
            PostgreSQL, e.g.
            ``postgresql+psycopg://user:password@host:5432/fulfillment``.
            The default is a local SQLite file so the app and its tests run
            with zero external setup.
        test_database_url: optional PostgreSQL URL used only by the dedicated
            integration tests (``tests/test_postgres_integration.py``). When it
            is unset those tests skip.
        auto_create_tables: when True (dev / tests), the app builds the schema
            with ``Base.metadata.create_all`` on startup. Set False in
            production / Docker, where Alembic owns the schema.
        sql_echo: when True, SQLAlchemy logs every statement. Debugging only.
        log_level: root log level for the application logger.
        app_name: identifier used in logs and the health payload.
        jwt_secret_key: HMAC signing key for access tokens. MUST be overridden
            in any real deployment (a fixed dev default is provided only so the
            app starts with zero config).
        jwt_algorithm: JWS algorithm, HS256.
        access_token_expire_minutes: token lifetime.
        bootstrap_admin_username / bootstrap_admin_password: if both are set and
            the users table is empty, the app creates one admin at startup so
            there is a way in. Leave unset in environments that provision users
            another way.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = Field(default="sqlite+pysqlite:///./fulfillment.db")
    test_database_url: Optional[str] = Field(default=None)
    auto_create_tables: bool = Field(default=True)
    sql_echo: bool = Field(default=False)
    log_level: str = Field(default="INFO")
    app_name: str = Field(default="fulfillment-api")

    # --- database connection pool (server databases only; ignored for SQLite) --
    # Deliberately modest defaults. A pool reuses TCP connections instead of
    # reconnecting per request; it is NOT a claim about high-scale throughput.
    #   db_pool_size      - connections kept open and handed out in rotation
    #   db_max_overflow   - extra connections allowed briefly under a spike
    #   db_pool_timeout_s - how long a request waits for a free connection
    #                       before failing (rather than hanging forever)
    #   db_pool_recycle_s - drop a connection older than this (avoids a stale
    #                       socket a proxy/DB silently closed); -1 disables
    db_pool_size: int = Field(default=5, ge=1)
    db_max_overflow: int = Field(default=5, ge=0)
    db_pool_timeout_s: float = Field(default=30.0, gt=0)
    db_pool_recycle_s: int = Field(default=1800)

    # --- startup database readiness (used by the app lifespan) ----------------
    # On boot the app tries to reach the database a few times with exponential
    # backoff - a container often starts before its database finishes starting.
    # If it never answers, the app STILL starts; /ready then reports 503 until
    # the database appears. It never blocks forever.
    db_connect_attempts: int = Field(default=5, ge=1)
    db_connect_base_delay_s: float = Field(default=0.2, ge=0)
    db_connect_max_delay_s: float = Field(default=5.0, ge=0)

    # --- authentication ------------------------------------------------
    jwt_secret_key: str = Field(default="dev-insecure-jwt-key-change-me-in-any-real-environment")
    jwt_algorithm: str = Field(default="HS256")
    access_token_expire_minutes: int = Field(default=30, gt=0)
    bootstrap_admin_username: Optional[str] = Field(default=None)
    bootstrap_admin_password: Optional[str] = Field(default=None)

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    @property
    def jwt_secret_is_default(self) -> bool:
        return self.jwt_secret_key == "dev-insecure-jwt-key-change-me-in-any-real-environment"

    @staticmethod
    def _redact_url(url: str) -> str:
        if "@" not in url or "://" not in url:
            return url
        scheme, rest = url.split("://", 1)
        creds, host = rest.split("@", 1)
        if ":" in creds:
            creds = f"{creds.split(':', 1)[0]}:***"
        return f"{scheme}://{creds}@{host}"

    def safe_database_url(self) -> str:
        """The database URL with any password redacted, for logging."""
        return self._redact_url(self.database_url)


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide `Settings`, built once and cached.

    Cached so every caller sees the same object and the environment is read
    only once. Tests clear the cache (``get_settings.cache_clear()``) or
    override the FastAPI dependency to inject a test configuration.
    """
    return Settings()
