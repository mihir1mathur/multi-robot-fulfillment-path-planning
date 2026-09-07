# Container image for the Autonomous Fulfillment & Path-Planning API.
#
#   docker build -t fulfillment-api .
#   docker run --rm -p 8000:8000 \
#       -e DATABASE_URL=postgresql+psycopg://user:pass@host:5432/db \
#       -e JWT_SECRET_KEY=change-me \
#       fulfillment-api
#
# No secrets are baked in. DATABASE_URL and JWT_SECRET_KEY are supplied at run
# time (docker compose passes them from the environment / an env file).

FROM python:3.11-slim AS base

# - PYTHONDONTWRITEBYTECODE: no .pyc files in the image
# - PYTHONUNBUFFERED: logs flush immediately (so `docker logs` is live)
# - PIP_NO_CACHE_DIR: smaller image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so a code-only change does not re-run pip install.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Application code (see .dockerignore for what is left out - tests, notes,
# benchmarks, .git, .env, local databases).
COPY robotics/ ./robotics/
COPY alembic/ ./alembic/
COPY alembic.ini ./
COPY scripts/ ./scripts/
COPY docker/entrypoint.sh ./docker/entrypoint.sh

# Run as a non-root user.
RUN useradd --create-home --uid 10001 appuser \
    && chmod +x ./docker/entrypoint.sh \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# The entrypoint applies migrations (alembic upgrade head), then starts uvicorn.
ENTRYPOINT ["./docker/entrypoint.sh"]
CMD ["uvicorn", "robotics.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
