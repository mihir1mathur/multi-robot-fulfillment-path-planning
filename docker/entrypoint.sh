#!/usr/bin/env sh
# Container entrypoint: apply database migrations, then run the given command.
#
# The application image sets AUTO_CREATE_TABLES=false, so Alembic - not
# create_all - owns the schema. `alembic upgrade head` is idempotent: on an
# already-migrated database it is a no-op.
set -e

echo "entrypoint: applying database migrations (alembic upgrade head)"
alembic upgrade head

echo "entrypoint: starting application: $*"
exec "$@"
