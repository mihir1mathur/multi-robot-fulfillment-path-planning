"""Bounded retry with exponential backoff, for TRANSIENT failures only.

WHEN TO USE THIS
----------------
Only for operations that can fail for a reason that might not be true a moment
later: a database that is still starting up, a dropped TCP connection, a
momentary "too many connections". Retrying those can succeed.

WHEN NOT TO USE THIS
--------------------
Never wrap deterministic outcomes in a retry:

    * invalid user input / a 4xx                 - it will fail identically
    * an authorization failure                   - retrying is not going to help
    * "no route exists" from the planner         - that is a real answer
    * a schema / validation error                - the input is the problem

Retrying those just wastes time and hides a bug.

HOW THE BACKOFF WORKS
---------------------
Attempt 1 runs immediately. After a failure, wait ``base_delay`` seconds, then
``base_delay * 2``, then ``* 4`` ... capped at ``max_delay``. Optional
``jitter`` adds a small random fraction so a fleet of clients that all failed
at once do not all retry in lockstep (the "thundering herd").

    attempt:   1        2        3        4
    wait:      -    0.1s     0.2s     0.4s   (base_delay=0.1, max_delay=0.4)

TESTABILITY
-----------
``sleep`` is injected. Tests pass ``sleep=lambda _s: None`` so the retry logic
is exercised with zero real waiting; ``time.sleep`` is only the default.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence, TypeVar

logger = logging.getLogger("robotics.retry")

T = TypeVar("T")


class RetryError(RuntimeError):
    """Every attempt failed. ``__cause__`` is the last underlying exception."""

    def __init__(self, operation: str, attempts: int, last_exc: BaseException) -> None:
        super().__init__(
            f"{operation!r} still failing after {attempts} attempt(s): {last_exc}"
        )
        self.operation = operation
        self.attempts = attempts
        self.last_exc = last_exc


@dataclass(frozen=True)
class RetryPolicy:
    """How hard to try. All times are in seconds."""

    attempts: int = 3
    base_delay: float = 0.1
    max_delay: float = 2.0
    jitter: float = 0.1  # fraction of the delay, e.g. 0.1 == +/-10%

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.base_delay < 0 or self.max_delay < 0:
            raise ValueError("delays must be >= 0")
        if not 0 <= self.jitter < 1:
            raise ValueError("jitter must be in [0, 1)")

    def delay_for(self, attempt: int, rng: random.Random) -> float:
        """Backoff before ``attempt`` (attempt is 2 for the first retry)."""
        raw = self.base_delay * (2 ** (attempt - 2))
        capped = min(raw, self.max_delay)
        if self.jitter:
            capped *= 1 + rng.uniform(-self.jitter, self.jitter)
        return max(0.0, capped)


DEFAULT_POLICY = RetryPolicy()


def retry_call(
    func: Callable[[], T],
    *,
    policy: RetryPolicy = DEFAULT_POLICY,
    retry_on: Sequence[type[BaseException]] | Iterable[type[BaseException]] = (Exception,),
    operation: str = "operation",
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Call ``func`` until it succeeds or the policy is exhausted.

    Args:
        func: a zero-argument callable (use ``functools.partial`` / a lambda to
            bind arguments). It is retried as a whole - it must be safe to run
            more than once.
        policy: how many attempts and how long to back off.
        retry_on: exception types that count as transient. Anything else is
            re-raised immediately, unretried.
        operation: a short name for logs and the final error message.
        sleep: injected so tests do not actually wait.
        rng: injected RNG for deterministic jitter in tests.

    Returns:
        Whatever ``func`` returns on its first success.

    Raises:
        RetryError: every attempt raised a retry-able exception.
        Any non-retry-able exception: re-raised on the spot.
    """
    retry_on = tuple(retry_on)
    rng = rng or random.Random()
    last_exc: BaseException | None = None

    for attempt in range(1, policy.attempts + 1):
        try:
            result = func()
            if attempt > 1:
                logger.info(
                    "retry.succeeded",
                    extra={"operation": operation, "attempt": attempt},
                )
            return result
        except retry_on as exc:  # noqa: PERF203 - the try/except is the point here
            last_exc = exc
            if attempt == policy.attempts:
                logger.warning(
                    "retry.exhausted",
                    extra={
                        "operation": operation,
                        "attempts": attempt,
                        "error_type": type(exc).__name__,
                    },
                )
                break
            wait = policy.delay_for(attempt + 1, rng)
            logger.info(
                "retry.attempt_failed",
                extra={
                    "operation": operation,
                    "attempt": attempt,
                    "next_delay_s": round(wait, 4),
                    "error_type": type(exc).__name__,
                },
            )
            sleep(wait)

    assert last_exc is not None  # loop ran at least once
    raise RetryError(operation, policy.attempts, last_exc) from last_exc
