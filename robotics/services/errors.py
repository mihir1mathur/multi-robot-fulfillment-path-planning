"""Service-layer error types, mapped to HTTP status codes by the API layer.

WHY A SEPARATE HIERARCHY
------------------------
The domain raises `RoboticsError` for "this operation is not allowed". The
service layer needs a bit more nuance so the API can answer with the RIGHT
status code:

    ResourceNotFoundError  -> 404   the thing you asked for does not exist
    ResourceConflictError  -> 409   it exists / the state forbids this change
    ValidationFailedError  -> 422   the request is well-formed JSON but asks
                                    for something the domain rejects
    DomainFailureError     -> 200   NOT an error at the HTTP level: the
                                    algorithm ran and returned a legitimate
                                    "no route" / "infeasible" answer. Carried
                                    as an exception only so a service method
                                    can bail out early; the router turns it
                                    into a normal response body, never a 5xx.

Anything that is NOT one of these bubbles up as an unexpected 500 - which is
the honest thing to do with a real bug.
"""

from __future__ import annotations

from typing import Any, Optional


class ServiceError(Exception):
    """Base class for every deliberate service-layer failure."""

    status_code: int = 400
    code: str = "service_error"

    def __init__(self, message: str, *, details: Optional[Any] = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details


class ResourceNotFoundError(ServiceError):
    """A robot / task / run with the given identifier does not exist."""

    status_code = 404
    code = "not_found"


class ResourceConflictError(ServiceError):
    """The request conflicts with current state (duplicate ID, illegal transition)."""

    status_code = 409
    code = "conflict"


class ValidationFailedError(ServiceError):
    """The request is structurally valid but the domain rejects its content."""

    status_code = 422
    code = "unprocessable"


class DomainFailureError(ServiceError):
    """A legitimate domain 'no' (no route, infeasible allocation).

    Not an HTTP error. The router catches this and returns a normal 200 body
    describing the failure honestly.
    """

    status_code = 200
    code = "domain_failure"
