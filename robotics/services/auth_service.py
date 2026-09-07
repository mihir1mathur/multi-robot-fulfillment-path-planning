"""User management and login: the service between the API and the ``users`` table.

    register_user   - hash the password, insert a UserRecord (admin only, via
                      the router)
    authenticate    - look up by username, verify the password, return the row
    issue_token     - build a signed access token for a verified user

No FastAPI here. No token verification here (that is a per-request concern in
`robotics.api.dependencies`). This module only reads/writes the users table and
calls `robotics.api.security`.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from robotics.api.security import (
    PasswordError,
    Role,
    create_access_token,
    hash_password,
    verify_password,
)
from robotics.persistence.config import Settings
from robotics.persistence.models import UserRecord
from robotics.services.errors import (
    ResourceConflictError,
    ValidationFailedError,
)

logger = logging.getLogger("robotics.services.auth")


class AuthService:
    def __init__(self, session: Session, settings: Settings) -> None:
        self.session = session
        self.settings = settings

    # ------------------------------------------------------------------
    def get_by_username(self, username: str) -> Optional[UserRecord]:
        stmt = select(UserRecord).where(UserRecord.username == username)
        return self.session.scalars(stmt).first()

    def user_count(self) -> int:
        return self.session.query(UserRecord).count()

    # ------------------------------------------------------------------
    def register_user(self, username: str, password: str, role: str) -> UserRecord:
        username = username.strip()
        if not username:
            raise ValidationFailedError("username must not be empty")
        try:
            role_value = Role(role).value
        except ValueError as error:
            raise ValidationFailedError(
                f"unknown role '{role}'; use one of "
                f"{', '.join(r.value for r in Role)}"
            ) from error
        if len(password) < 8:
            raise ValidationFailedError("password must be at least 8 characters")
        if self.get_by_username(username) is not None:
            raise ResourceConflictError(f"user '{username}' already exists")

        try:
            record = UserRecord(
                username=username,
                password_hash=hash_password(password),
                role=role_value,
            )
        except PasswordError as error:
            raise ValidationFailedError(str(error)) from error

        self.session.add(record)
        self.session.flush()
        logger.info("auth.user_registered", extra={"username": username, "role": role_value})
        return record

    # ------------------------------------------------------------------
    def authenticate(self, username: str, password: str) -> Optional[UserRecord]:
        record = self.get_by_username(username.strip())
        if record is None or record.disabled:
            # Do the hash work anyway so a missing user and a wrong password
            # take the same time (no username enumeration by timing).
            verify_password(password, "$2b$12$" + "x" * 53)
            return None
        if not verify_password(password, record.password_hash):
            return None
        return record

    def issue_token(self, user: UserRecord) -> tuple[str, int]:
        """Return (token, expires_in_seconds)."""
        token = create_access_token(
            subject=user.username,
            role=user.role,
            secret_key=self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
            expires_minutes=self.settings.access_token_expire_minutes,
        )
        return token, self.settings.access_token_expire_minutes * 60

    # ------------------------------------------------------------------
    def ensure_bootstrap_admin(self) -> Optional[str]:
        """If configured and the users table is empty, create one admin.

        Returns the created username, or None. Called once at app startup so a
        fresh database is never locked out.
        """
        username = self.settings.bootstrap_admin_username
        password = self.settings.bootstrap_admin_password
        if not username or not password:
            return None
        if self.user_count() > 0:
            return None
        self.register_user(username, password, Role.admin.value)
        logger.info("auth.bootstrap_admin_created", extra={"username": username})
        return username
