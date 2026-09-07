"""Alembic migration environment.

    ORM models  (robotics/persistence/models.py)
        v  registered on
    Base.metadata   (robotics/persistence/database.py)
        v  compared against the live DB by
    Alembic autogenerate / this env
        v  emits
    a migration script  (alembic/versions/*.py)
        v  applied to
    the PostgreSQL / SQLite schema

The database URL is NOT read from alembic.ini - it comes from the same
`Settings` the application uses (``DATABASE_URL`` in the environment / ``.env``),
so migrations always target the same database as the running service and no
credentials live in a committed file.
"""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make ``robotics`` importable when alembic is run from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.persistence.config import get_settings  # noqa: E402
from robotics.persistence.database import Base  # noqa: E402

# Importing the models registers every table on Base.metadata.
from robotics.persistence import models  # noqa: E402,F401

config = context.config

if config.config_file_name is not None:
    # disable_existing_loggers=False: this configures alembic's own logging
    # (root / sqlalchemy / alembic) from alembic.ini, but running a migration
    # in-process (tests, CI, a programmatic command.upgrade) must NOT silently
    # switch off the application's own "robotics.*" loggers, which fileConfig
    # does by default for every logger it does not itself name.
    fileConfig(config.config_file_name, disable_existing_loggers=False)

target_metadata = Base.metadata


def _include_name(name, type_, parent_names):
    """Limit autogenerate / ``alembic check`` to tables this project owns.

    A shared database (a developer's local PostgreSQL, a managed instance) can
    hold tables that are outside this migration chain. Without this filter
    ``alembic check`` reports every such table as a spurious "remove_table"
    diff. Columns, indexes and constraints of the project's own tables are
    still compared normally.
    """
    if type_ == "table":
        return name == "alembic_version" or name in target_metadata.tables
    return True


def _database_url() -> str:
    """Resolve the target URL, in priority order:

    1. ``-x db_url=...`` on the alembic command line (one-off runs / CI)
    2. ``config.attributes["db_url"]`` or a non-empty ``sqlalchemy.url`` in the
       Config (set by programmatic callers such as the PostgreSQL
       integration-test fixture). ``attributes`` is preferred because it bypasses
       configparser, whose ``%`` interpolation mangles URL-encoded passwords.
    3. the application ``Settings`` (``DATABASE_URL`` in the environment / .env)
    """
    x_args = context.get_x_argument(as_dictionary=True)
    if "db_url" in x_args:
        return x_args["db_url"]
    attr_url = config.attributes.get("db_url")
    if attr_url:
        return attr_url
    ini_url = config.get_main_option("sqlalchemy.url")
    if ini_url:
        return ini_url
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Emit SQL to a script without a live connection."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        render_as_batch=True,
        include_name=_include_name,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()
    connectable = engine_from_config(
        section,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            include_name=_include_name,
            # batch mode lets ALTER work on SQLite; harmless on PostgreSQL.
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
