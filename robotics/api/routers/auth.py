"""Authentication endpoints.

    POST /auth/token      username + password  -> a signed access token (public)
    POST /auth/register   create a user        -> admin only
    GET  /auth/me         the caller's own identity (any authenticated user)

`POST /auth/token` also accepts the standard OAuth2 form body
(``application/x-www-form-urlencoded`` with ``username`` / ``password``) so the
Swagger "Authorize" button works.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Form, status
from sqlalchemy.orm import Session

from robotics.api.dependencies import (
    AuthError,
    get_current_user,
    get_session,
    get_settings,
    require_role,
)
from robotics.api.schemas.auth import (
    LoginRequest,
    TokenResponse,
    UserCreate,
    UserRead,
)
from robotics.api.security import Role
from robotics.persistence.config import Settings
from robotics.persistence.models import UserRecord
from robotics.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/token", response_model=TokenResponse)
def issue_token(
    username: str = Form(...),
    password: str = Form(...),
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    service = AuthService(session, settings)
    user = service.authenticate(username, password)
    if user is None:
        raise AuthError("incorrect username or password")
    token, expires_in = service.issue_token(user)
    return TokenResponse(access_token=token, expires_in=expires_in, role=user.role)


@router.post("/login", response_model=TokenResponse)
def login_json(
    body: LoginRequest,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> TokenResponse:
    """Same as /auth/token but with a JSON body (convenience for API clients)."""
    service = AuthService(session, settings)
    user = service.authenticate(body.username, body.password)
    if user is None:
        raise AuthError("incorrect username or password")
    token, expires_in = service.issue_token(user)
    return TokenResponse(access_token=token, expires_in=expires_in, role=user.role)


@router.post(
    "/register", response_model=UserRead, status_code=status.HTTP_201_CREATED
)
def register_user(
    body: UserCreate,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
    _admin: UserRecord = Depends(require_role(Role.admin)),
) -> UserRead:
    record = AuthService(session, settings).register_user(
        body.username, body.password, body.role.value
    )
    return UserRead.from_record(record)


@router.get("/me", response_model=UserRead)
def me(user: UserRecord = Depends(get_current_user)) -> UserRead:
    return UserRead.from_record(user)
