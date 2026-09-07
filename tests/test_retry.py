"""The bounded-retry-with-backoff utility (`robotics.common.retry`).

Every test injects a fake ``sleep`` so nothing actually waits, and a seeded
RNG so the jittered delays are deterministic.
"""

from __future__ import annotations

import random

import pytest

from robotics.common.retry import (
    DEFAULT_POLICY,
    RetryError,
    RetryPolicy,
    retry_call,
)


class _Flaky:
    """Fails the first ``fail_times`` calls, then returns ``value``."""

    def __init__(self, fail_times: int, value="ok", exc=RuntimeError):
        self.fail_times = fail_times
        self.value = value
        self.exc = exc
        self.calls = 0

    def __call__(self):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.exc(f"transient failure {self.calls}")
        return self.value


def _no_sleep_recorder():
    waited: list[float] = []
    return waited, waited.append


# --- happy paths --------------------------------------------------
def test_succeeds_on_the_first_try_without_sleeping():
    waited, sleep = _no_sleep_recorder()
    fn = _Flaky(fail_times=0)
    assert retry_call(fn, sleep=sleep) == "ok"
    assert fn.calls == 1
    assert waited == []


def test_succeeds_after_two_transient_failures():
    waited, sleep = _no_sleep_recorder()
    fn = _Flaky(fail_times=2)
    result = retry_call(
        fn, policy=RetryPolicy(attempts=3, base_delay=0.1, jitter=0.0), sleep=sleep
    )
    assert result == "ok"
    assert fn.calls == 3
    assert waited == [0.1, 0.2]  # exponential: base, base*2


# --- exhaustion --------------------------------------------------
def test_raises_RetryError_when_every_attempt_fails():
    waited, sleep = _no_sleep_recorder()
    fn = _Flaky(fail_times=99)
    with pytest.raises(RetryError) as info:
        retry_call(
            fn,
            policy=RetryPolicy(attempts=4, base_delay=0.05, jitter=0.0),
            operation="probe",
            sleep=sleep,
        )
    assert fn.calls == 4
    assert info.value.attempts == 4
    assert isinstance(info.value.last_exc, RuntimeError)
    assert "probe" in str(info.value)
    # slept between each attempt but NOT after the last
    assert len(waited) == 3


# --- non-retryable errors bail out immediately ------------------
def test_non_retryable_exception_is_reraised_without_retrying():
    waited, sleep = _no_sleep_recorder()

    def boom():
        raise ValueError("bad input - retrying will not help")

    with pytest.raises(ValueError):
        retry_call(boom, retry_on=(KeyError,), sleep=sleep)
    assert waited == []


def test_retry_on_can_be_a_specific_type():
    waited, sleep = _no_sleep_recorder()
    fn = _Flaky(fail_times=1, exc=ConnectionError)
    assert retry_call(
        fn, retry_on=(ConnectionError,),
        policy=RetryPolicy(attempts=2, base_delay=0.01, jitter=0.0), sleep=sleep,
    ) == "ok"
    assert fn.calls == 2


# --- backoff shape --------------------------------------------
def test_delay_is_capped_at_max_delay():
    policy = RetryPolicy(attempts=10, base_delay=1.0, max_delay=4.0, jitter=0.0)
    rng = random.Random(0)
    delays = [policy.delay_for(a, rng) for a in range(2, 8)]
    assert delays == [1.0, 2.0, 4.0, 4.0, 4.0, 4.0]


def test_jitter_stays_within_bounds_and_is_deterministic_with_a_seed():
    policy = RetryPolicy(attempts=5, base_delay=1.0, max_delay=10.0, jitter=0.1)
    a = [policy.delay_for(n, random.Random(42)) for n in range(2, 5)]
    b = [policy.delay_for(n, random.Random(42)) for n in range(2, 5)]
    assert a == b  # same seed -> same jitter
    # attempt 2 nominal delay is 1.0, +/-10%
    assert 0.9 <= a[0] <= 1.1


# --- policy validation ---------------------------------------
@pytest.mark.parametrize(
    "kwargs",
    [
        {"attempts": 0},
        {"base_delay": -1},
        {"jitter": 1.0},
        {"jitter": -0.1},
    ],
)
def test_invalid_policy_is_rejected(kwargs):
    with pytest.raises(ValueError):
        RetryPolicy(**kwargs)


def test_default_policy_is_three_attempts():
    assert DEFAULT_POLICY.attempts == 3
