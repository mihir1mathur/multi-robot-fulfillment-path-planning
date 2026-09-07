"""Reproducible database initialisation (dev / SQLite convenience).

    python -m robotics.persistence.init_db            # create tables
    python -m robotics.persistence.init_db --drop     # drop then recreate

ALEMBIC OWNS THE SCHEMA
----------------------
For PostgreSQL / production / Docker the schema is created and evolved by
Alembic migrations:

    alembic upgrade head

``create_all`` here is a shortcut for local development and for the fast
SQLite-backed unit / API tests, where a full migration run per test would be
wasted work. It builds exactly the schema the migrations produce (an
``alembic check`` in CI enforces that they stay in sync).
"""

from __future__ import annotations

import argparse
import sys

from robotics.persistence.config import get_settings
from robotics.persistence.database import Database

# Importing the models registers them on ``Base.metadata`` so ``create_all``
# knows about every table.
from robotics.persistence import models  # noqa: F401


def initialise(drop: bool = False) -> None:
    settings = get_settings()
    database = Database(settings)
    if drop:
        database.drop_all()
        print("dropped all tables")
    database.create_all()
    print(f"schema ready on {settings.safe_database_url()}")
    table_names = sorted(models.Base.metadata.tables)
    print(f"tables: {', '.join(table_names)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Initialise the fulfillment database.")
    parser.add_argument(
        "--drop",
        action="store_true",
        help="drop every table before recreating it (DESTROYS DATA)",
    )
    args = parser.parse_args(argv)
    initialise(drop=args.drop)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    sys.exit(main())
