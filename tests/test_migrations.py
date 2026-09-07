"""Alembic migrations: upgrade, downgrade, re-upgrade, and 'matches the models'.

Runs against a throwaway SQLite file (fast, no server). CI additionally runs
`alembic upgrade head` + `alembic check` against real PostgreSQL - see
.github/workflows/ci.yml and tests/test_postgres_integration.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_APP_TABLES = {
    "users", "robots", "tasks",
    "planning_runs", "allocation_runs", "coordination_runs", "recovery_events",
}


@pytest.fixture
def sqlite_url(tmp_path) -> str:
    return f"sqlite+pysqlite:///{tmp_path / 'migrate.db'}"


def _cfg(url: str) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    cfg.set_main_option("sqlalchemy.url", url)
    return cfg


def _tables(url: str) -> set[str]:
    engine = create_engine(url)
    try:
        return set(inspect(engine).get_table_names())
    finally:
        engine.dispose()


def test_upgrade_creates_every_table(sqlite_url):
    command.upgrade(_cfg(sqlite_url), "head")
    tables = _tables(sqlite_url)
    assert _APP_TABLES <= tables
    assert "alembic_version" in tables


def test_downgrade_removes_every_app_table(sqlite_url):
    cfg = _cfg(sqlite_url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    tables = _tables(sqlite_url)
    assert _APP_TABLES.isdisjoint(tables)


def test_downgrade_then_re_upgrade_is_clean(sqlite_url):
    cfg = _cfg(sqlite_url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")  # must not raise
    assert _APP_TABLES <= _tables(sqlite_url)


def test_migration_matches_the_orm_models(sqlite_url):
    """`alembic check` finds no difference between the migration and the models."""
    from alembic.autogenerate import compare_metadata
    from alembic.runtime.migration import MigrationContext

    from robotics.persistence.database import Base
    from robotics.persistence import models  # noqa: F401  (registers tables)

    command.upgrade(_cfg(sqlite_url), "head")
    engine = create_engine(sqlite_url)
    try:
        with engine.connect() as conn:
            ctx = MigrationContext.configure(conn, opts={"compare_type": True})
            diff = compare_metadata(ctx, Base.metadata)
    finally:
        engine.dispose()
    assert diff == [], f"migration drifted from the models: {diff}"


def test_migrated_schema_equals_create_all_schema(sqlite_url, tmp_path):
    """The two ways of building the schema produce the same tables + indexes."""
    from robotics.persistence.database import Base
    from robotics.persistence import models  # noqa: F401

    command.upgrade(_cfg(sqlite_url), "head")
    migrated = create_engine(sqlite_url)

    ca_url = f"sqlite+pysqlite:///{tmp_path / 'createall.db'}"
    created = create_engine(ca_url)
    Base.metadata.create_all(created)

    try:
        mi, ci = inspect(migrated), inspect(created)
        assert set(mi.get_table_names()) - {"alembic_version"} == set(ci.get_table_names())
        for table in _APP_TABLES:
            m_idx = {i["name"] for i in mi.get_indexes(table)}
            c_idx = {i["name"] for i in ci.get_indexes(table)}
            assert m_idx == c_idx, f"{table}: {m_idx} != {c_idx}"
    finally:
        migrated.dispose()
        created.dispose()


def test_running_a_migration_does_not_disable_the_app_loggers(sqlite_url):
    """`alembic/env.py` calls logging.config.fileConfig, which by default sets
    `.disabled = True` on every logger it does not name. Running a migration
    in-process (tests, CI, a programmatic command.upgrade) must leave this
    project's own "robotics.*" loggers untouched - env.py passes
    disable_existing_loggers=False for exactly this reason.
    """
    import logging

    # Make sure the loggers exist and are enabled before the migration runs.
    for name in ("robotics", "robotics.api", "robotics.services", "robotics.persistence.database"):
        logging.getLogger(name).disabled = False

    command.upgrade(_cfg(sqlite_url), "head")

    for name in ("robotics", "robotics.api", "robotics.services", "robotics.persistence.database"):
        assert logging.getLogger(name).disabled is False, f"{name} was disabled by the migration"


def test_head_revision_is_single_and_known():
    from alembic.script import ScriptDirectory

    script = ScriptDirectory.from_config(_cfg("sqlite+pysqlite://"))
    heads = script.get_heads()
    assert tuple(heads) == ("0001",), heads
