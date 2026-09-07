"""Service layer: orchestrates domain operations for the API.

WHERE IT SITS
-------------
    FastAPI router      HTTP in, HTTP out, no domain logic
          |
          v
    SERVICE LAYER       "do this operation": load state, call the algorithm,
          |             persist the result, return a plain result object
          +-------------------------+
          |                         |
          v                         v
    robotics.planning /        robotics.persistence
    allocation / coordination  (repositories, ORM, DB)
    / recovery  (unchanged)

RULES
-----
* A service method takes a `Session` and plain arguments, returns a plain
  dataclass / dict, and raises a `ServiceError` subclass for anything the
  caller should see as a 4xx.
* Services NEVER import FastAPI and NEVER build SQL - they call repositories.
* Services do not duplicate a single line of planning / allocation /
  coordination / recovery logic; they call the existing packages.
"""

from robotics.services.errors import (
    DomainFailureError,
    ResourceConflictError,
    ResourceNotFoundError,
    ServiceError,
    ValidationFailedError,
)

__all__ = [
    "ServiceError",
    "ResourceNotFoundError",
    "ResourceConflictError",
    "ValidationFailedError",
    "DomainFailureError",
]
