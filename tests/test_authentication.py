"""Authentication: password hashing, JWT tokens, the /auth endpoints.

These tests exercise the REAL bcrypt hash/verify path and the REAL JWT
sign/verify path (the fast API tests reuse a cached low-cost hash for speed;
this module does not)."""

from __future__ import annotations

import datetime as dt

import jwt
import pytest

from robotics.api.security import (
    PasswordError,
    Role,
    create_access_token,
    decode_token,
    hash_password,
    role_satisfies,
    verify_password,
)

_SECRET = "unit-test-signing-key-at-least-32-bytes-long!"


# --- password hashing -------------------------------------------------
def test_hash_is_not_the_plaintext_and_verifies():
    hashed = hash_password("correct horse battery staple")
    assert hashed != "correct horse battery staple"
    assert hashed.startswith("$2")  # bcrypt marker
    assert verify_password("correct horse battery staple", hashed) is True
    assert verify_password("wrong password", hashed) is False


def test_two_hashes_of_the_same_password_differ_but_both_verify():
    a = hash_password("same-password")
    b = hash_password("same-password")
    assert a != b  # random salt
    assert verify_password("same-password", a)
    assert verify_password("same-password", b)


def test_verify_password_tolerates_a_garbage_hash():
    assert verify_password("anything", "not-a-bcrypt-hash") is False


def test_password_longer_than_72_bytes_is_rejected():
    with pytest.raises(PasswordError):
        hash_password("x" * 73)


# --- JWT ------------------------------------------------------------
def test_create_and_decode_round_trip():
    token = create_access_token("alice", "operator", _SECRET, expires_minutes=5)
    claims = decode_token(token, _SECRET)
    assert claims["sub"] == "alice"
    assert claims["role"] == "operator"
    assert claims["exp"] > claims["iat"]


def test_bad_signature_is_rejected():
    token = create_access_token("alice", "viewer", _SECRET)
    from robotics.api.security import TokenError

    with pytest.raises(TokenError):
        decode_token(token, "a-different-signing-key-also-32-bytes-x")


def test_expired_token_is_rejected():
    past = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    token = create_access_token("alice", "viewer", _SECRET, expires_minutes=1, now=past)
    from robotics.api.security import TokenError

    with pytest.raises(TokenError):
        decode_token(token, _SECRET)


def test_malformed_token_is_rejected():
    from robotics.api.security import TokenError

    with pytest.raises(TokenError):
        decode_token("this.is.not.a.jwt", _SECRET)


def test_token_without_subject_is_rejected():
    token = jwt.encode({"role": "admin"}, _SECRET, algorithm="HS256")
    from robotics.api.security import TokenError

    with pytest.raises(TokenError):
        decode_token(token, _SECRET)


# --- role ranking -------------------------------------------------
def test_role_ranking():
    assert role_satisfies("admin", Role.viewer)
    assert role_satisfies("admin", Role.operator)
    assert role_satisfies("operator", Role.viewer)
    assert not role_satisfies("viewer", Role.operator)
    assert not role_satisfies("operator", Role.admin)
    assert not role_satisfies("nonsense", Role.viewer)


# --- the /auth endpoints ---------------------------------------------
def test_token_endpoint_form_login(anon_client):
    r = anon_client.post(
        "/auth/token", data={"username": "operator_user", "password": "operator-pw-123"}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["token_type"] == "bearer"
    assert body["role"] == "operator"
    assert body["access_token"]


def test_login_json_endpoint(anon_client):
    r = anon_client.post(
        "/auth/login", json={"username": "admin_user", "password": "admin-pw-123"}
    )
    assert r.status_code == 200
    assert r.json()["role"] == "admin"


def test_wrong_password_is_401(anon_client):
    r = anon_client.post(
        "/auth/login", json={"username": "operator_user", "password": "nope"}
    )
    assert r.status_code == 401
    assert r.json()["error"] == "unauthenticated"
    assert r.headers.get("www-authenticate") == "Bearer"


def test_unknown_user_is_401(anon_client):
    r = anon_client.post(
        "/auth/login", json={"username": "ghost", "password": "whatever"}
    )
    assert r.status_code == 401


def test_me_returns_the_caller(api_client):
    r = api_client.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["username"] == "operator_user"
    assert r.json()["role"] == "operator"


def test_me_without_a_token_is_401(anon_client):
    assert anon_client.get("/auth/me").status_code == 401


def test_malformed_authorization_header_is_401(api_app):
    from fastapi.testclient import TestClient

    with TestClient(api_app) as c:
        # not "Bearer <token>"
        r = c.get("/auth/me", headers={"Authorization": "Basic abc123"})
        assert r.status_code == 401
        r = c.get("/auth/me", headers={"Authorization": "Bearer"})
        assert r.status_code == 401


def test_a_tampered_token_is_401(api_app, api_database):
    from fastapi.testclient import TestClient

    from tests.conftest import _token_for

    good = _token_for(api_database, "admin")
    tampered = good[:-4] + ("aaaa" if good[-4:] != "aaaa" else "bbbb")
    with TestClient(api_app) as c:
        r = c.get("/auth/me", headers={"Authorization": f"Bearer {tampered}"})
        assert r.status_code == 401


def test_token_signed_with_a_different_key_is_401(api_app):
    from fastapi.testclient import TestClient

    foreign = create_access_token("operator_user", "operator", "some-other-key-32-bytes-xxxxxxxx")
    with TestClient(api_app) as c:
        r = c.get("/auth/me", headers={"Authorization": f"Bearer {foreign}"})
        assert r.status_code == 401


def test_password_hash_is_never_returned_by_any_auth_endpoint(admin_client):
    created = admin_client.post(
        "/auth/register",
        json={"username": "new_person", "password": "a-strong-password", "role": "viewer"},
    )
    assert created.status_code == 201
    assert "password" not in created.json()
    assert "password_hash" not in created.json()
    assert admin_client.get("/auth/me").json().keys() == {
        "id", "username", "role", "disabled"
    }
