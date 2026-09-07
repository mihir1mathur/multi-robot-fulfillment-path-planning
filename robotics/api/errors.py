"""Exception handlers: one consistent error body for the whole API.

MAPPING
-------
    ServiceError subclass      -> its own status_code + code
        AuthError              -> 401 unauthenticated   (+ WWW-Authenticate: Bearer)
        ForbiddenError         -> 403 forbidden
        ResourceNotFoundError  -> 404 not_found
        ResourceConflictError  -> 409 conflict
        ValidationFailedError  -> 422 unprocessable
    RequestValidationError     -> 422 validation_error   (FastAPI/Pydantic)
    RoboticsError (domain)     -> 422 domain_rejected     (last-resort catch)
    IntegrityError             -> 409 conflict            (a constraint fired -
                                   usually a race: two requests both passed an
                                   "already exists?" check, one INSERT lost)
    OperationalError / DBAPIError with no DB connection
                               -> 503 database_unavailable (transient - the
                                   caller may retry; NOT a bug)
    anything else              -> 500 internal_error      (a real bug)

Nothing here logs a token, a password, or a connection string - only the path,
the error code, and the status. The DB drivers put the failing SQL and
sometimes the host into the exception text, so that text is never echoed to
the client.

DomainFailureError is deliberately NOT handled here - a router catches it and
returns a normal 200 body, because "no route exists" is a valid answer, not an
error.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError, OperationalError

from robotics.exceptions import RoboticsError
from robotics.services.errors import ServiceError

logger = logging.getLogger("robotics.api.errors")


def _body(code: str, message: str, details=None) -> dict:
    payload = {"error": code, "message": message}
    if details is not None:
        payload["details"] = details
    return payload


def _constraint_name(exc: IntegrityError) -> str | None:
    """Best-effort constraint name for the log line ONLY (never the response).

    SQLAlchemy exposes ``exc.orig.diag.constraint_name`` on psycopg; on SQLite
    there is no such structured field, so this may return None. It deliberately
    does not fall back to parsing the raw message, which contains SQL.
    """
    diag = getattr(getattr(exc, "orig", None), "diag", None)
    return getattr(diag, "constraint_name", None)


def install_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(ServiceError)
    async def _service_error(request: Request, exc: ServiceError) -> JSONResponse:
        logger.info(
            "request.rejected",
            extra={
                "path": request.url.path,
                "code": exc.code,
                "status": exc.status_code,
            },
        )
        headers = {"WWW-Authenticate": "Bearer"} if exc.status_code == 401 else None
        return JSONResponse(
            status_code=exc.status_code,
            content=_body(exc.code, exc.message, exc.details),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def _validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_body(
                "validation_error",
                "request body failed schema validation",
                details=jsonable_encoder(exc.errors()),
            ),
        )

    @app.exception_handler(IntegrityError)
    async def _integrity_error(request: Request, exc: IntegrityError) -> JSONResponse:
        # A database constraint (unique / FK / not-null) fired. The common cause
        # is a race: two concurrent requests both saw "id not taken", both
        # INSERTed, the database rejected the second. A clean 409 - not a 500 -
        # is the honest answer, and the driver's SQL text is NOT forwarded.
        logger.info(
            "request.integrity_conflict",
            extra={"path": request.url.path, "constraint": _constraint_name(exc)},
        )
        return JSONResponse(
            status_code=409,
            content=_body(
                "conflict",
                "the request conflicts with existing data (a uniqueness or "
                "referential constraint was violated)",
            ),
        )

    @app.exception_handler(OperationalError)
    async def _operational_error(
        request: Request, exc: OperationalError
    ) -> JSONResponse:
        # The database could not be reached / a connection dropped mid-request.
        # Transient: the client may retry. 503 (with Retry-After) says exactly
        # that; a 500 would wrongly imply an application bug.
        incident = uuid.uuid4().hex[:12]
        logger.warning(
            "request.database_unavailable",
            extra={"path": request.url.path, "incident": incident},
        )
        return JSONResponse(
            status_code=503,
            content=_body(
                "database_unavailable",
                f"the database is temporarily unavailable (incident {incident})",
            ),
            headers={"Retry-After": "1"},
        )

    @app.exception_handler(RoboticsError)
    async def _robotics_error(request: Request, exc: RoboticsError) -> JSONResponse:
        # A domain rule said no. Well-formed request, unacceptable content.
        logger.info(
            "domain.rejected",
            extra={"path": request.url.path, "reason": str(exc)},
        )
        return JSONResponse(
            status_code=422,
            content=_body("domain_rejected", str(exc)),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        incident = uuid.uuid4().hex[:12]
        logger.exception(
            "request.unhandled_error",
            extra={"path": request.url.path, "incident": incident},
        )
        return JSONResponse(
            status_code=500,
            content=_body(
                "internal_error",
                f"an unexpected error occurred (incident {incident})",
            ),
        )
