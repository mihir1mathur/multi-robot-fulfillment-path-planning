"""FastAPI dependencies: configuration, the request-scoped DB session, and auth.

SESSION PER REQUEST
-------------------
``get_session`` opens one SQLAlchemy session at the start of a request and,
when the handler returns:

    no exception  -> COMMIT   (persist everything the request did)
    exception     -> ROLLBACK (leave the database exactly as it was)
    always        -> close    (return the connection to the pool)

A handler therefore never calls ``commit`` itself - the transaction boundary is
the request.

AUTHENTICATION vs AUTHORIZATION
------------------------------
``get_current_user``  - AUTHENTICATION: "who are you?" Reads the ``Authorization:
                        Bearer <jwt>`` header, verifies the signature and expiry,
                        loads the user row. 401 if any of that fails.
``require_role(role)`` - AUTHORIZATION: "may you do this?" Given an authenticated
                        user, checks their role is high enough. 403 if not.
"""

from __future__ import annotations

from typing import Iterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from robotics.api.security import Role, TokenError, decode_token, role_satisfies
from robotics.persistence.config import Settings
from robotics.persistence.database import Database
from robotics.persistence.models import UserRecord
from robotics.services.errors import ServiceError

# auto_error=False so a MISSING header reaches this code as None, allowing a
# consistent 401 body to be raised instead of FastAPI's default bare 403.
_bearer = HTTPBearer(auto_error=False)


class AuthError(ServiceError):
    """401 - the request is not authenticated."""

    status_code = 401
    code = "unauthenticated"


class ForbiddenError(ServiceError):
    """403 - authenticated but not allowed to do this."""

    status_code = 403
    code = "forbidden"


def get_settings(request: Request) -> Settings:
    """The `Settings` object created at app startup."""
    return request.app.state.settings


def get_database(request: Request) -> Database:
    """The process-wide `Database` (engine + session factory)."""
    return request.app.state.database


def get_session(database: Database = Depends(get_database)) -> Iterator[Session]:
    """Yield a transactional session for the lifetime of one request."""
    session = database.session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    settings: Settings = Depends(get_settings),
    session: Session = Depends(get_session),
) -> UserRecord:
    """AUTHENTICATION. Resolve the bearer token to a user row, or 401."""
    if credentials is None or not credentials.credentials:
        raise AuthError("missing bearer token")
    try:
        claims = decode_token(
            credentials.credentials, settings.jwt_secret_key, settings.jwt_algorithm
        )
    except TokenError as error:
        raise AuthError(str(error)) from error

    user = session.query(UserRecord).filter(UserRecord.username == claims["sub"]).first()
    if user is None or user.disabled:
        raise AuthError("token subject is not an active user")
    return user


def require_role(minimum: Role):
    """AUTHORIZATION. Return a dependency that 403s unless the user's role
    is at least `minimum` (admin >= operator >= viewer)."""

    def _dependency(user: UserRecord = Depends(get_current_user)) -> UserRecord:
        if not role_satisfies(user.role, minimum):
            raise ForbiddenError(
                f"role '{user.role}' is not permitted here (need '{minimum.value}')"
            )
        return user

    return _dependency
