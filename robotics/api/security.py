"""Password hashing and JWT access tokens - the cryptographic primitives.

WHAT IS HERE
------------
    hash_password / verify_password   - bcrypt, so the database never stores a
                                        plaintext password
    create_access_token / decode_token- a signed, expiring JSON Web Token
    Role                              - the three authorization roles

WHAT IS NOT HERE
---------------
No database access, no FastAPI. This module is pure functions over strings.
The service layer (`auth_service.py`) uses it against the `users` table; the
API dependency layer (`dependencies.py`) uses `decode_token` to authenticate a
request.

SECRETS
-------
`create_access_token` / `decode_token` take the signing key as an argument -
they never read configuration themselves. The caller passes
`settings.jwt_secret_key`, which comes from the environment and is never logged.
"""

from __future__ import annotations

import datetime as _dt
from enum import Enum
from typing import Any, Optional

import bcrypt
import jwt

# bcrypt truncates silently at 72 bytes; reject longer inputs rather than let a
# 100-char password and its 72-char prefix authenticate identically.
_BCRYPT_MAX_BYTES = 72


class Role(str, Enum):
    """The authorization roles, least privilege first.

    viewer   - GET robots / tasks
    operator - viewer + create/update robots & tasks + run planning /
               allocation / coordination / recovery
    admin    - operator + register users
    """

    viewer = "viewer"
    operator = "operator"
    admin = "admin"


# Which roles satisfy a required role (a higher role includes the lower ones).
_RANK = {Role.viewer: 0, Role.operator: 1, Role.admin: 2}


def role_satisfies(actual: str, required: Role) -> bool:
    """True if a user with role `actual` may act where `required` is needed."""
    try:
        return _RANK[Role(actual)] >= _RANK[required]
    except ValueError:
        return False


# ----------------------------------------------------------------------
# Passwords
# ----------------------------------------------------------------------
class PasswordError(ValueError):
    """A password could not be processed (too long for bcrypt, etc.)."""


def hash_password(password: str) -> str:
    """Return a bcrypt hash string for `password`. Never store the plaintext."""
    raw = password.encode("utf-8")
    if len(raw) > _BCRYPT_MAX_BYTES:
        raise PasswordError(
            f"password must be at most {_BCRYPT_MAX_BYTES} bytes for bcrypt"
        )
    return bcrypt.hashpw(raw, bcrypt.gensalt()).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    """True if `password` matches the stored `password_hash`. Constant-time."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except (ValueError, TypeError):
        return False


# ----------------------------------------------------------------------
# JWT access tokens
# ----------------------------------------------------------------------
class TokenError(Exception):
    """A token was missing, malformed, expired, or had a bad signature."""


def create_access_token(
    subject: str,
    role: str,
    secret_key: str,
    algorithm: str = "HS256",
    expires_minutes: int = 30,
    now: Optional[_dt.datetime] = None,
) -> str:
    """Build a signed JWT for `subject` (the username) carrying `role`.

    Claims: ``sub`` (username), ``role``, ``iat`` (issued-at), ``exp``
    (expiry). Signed with HMAC-`algorithm` using `secret_key`.
    """
    issued = now or _dt.datetime.now(_dt.timezone.utc)
    payload = {
        "sub": subject,
        "role": role,
        "iat": issued,
        "exp": issued + _dt.timedelta(minutes=expires_minutes),
    }
    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_token(token: str, secret_key: str, algorithm: str = "HS256") -> dict[str, Any]:
    """Verify signature + expiry and return the claims.

    Raises `TokenError` for anything wrong: bad signature, expired, malformed,
    missing `sub`.
    """
    try:
        claims = jwt.decode(token, secret_key, algorithms=[algorithm])
    except jwt.ExpiredSignatureError as error:
        raise TokenError("token has expired") from error
    except jwt.InvalidTokenError as error:
        raise TokenError("token is invalid") from error
    if not claims.get("sub"):
        raise TokenError("token is missing a subject")
    return claims
