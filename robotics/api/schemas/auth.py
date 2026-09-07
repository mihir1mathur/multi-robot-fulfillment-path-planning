"""Authentication request / response schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field

from robotics.api.security import Role
from robotics.persistence.models import UserRecord


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=200)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(description="seconds until the token expires")
    role: str


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_\-.]+$")
    password: str = Field(min_length=8, max_length=200)
    role: Role = Role.viewer


class UserRead(BaseModel):
    id: int
    username: str
    role: str
    disabled: bool

    @classmethod
    def from_record(cls, record: UserRecord) -> "UserRead":
        return cls(
            id=record.id,
            username=record.username,
            role=record.role,
            disabled=record.disabled,
        )
