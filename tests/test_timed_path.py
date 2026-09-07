"""Tests for the TimedPath / TimedStep representation."""

from __future__ import annotations

from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.warehouse.grid import Position


def P(r, c):
    return Position(r, c)


def _path(cells_and_times):
    return TimedPath.found(
        "R1", [TimedStep(P(r, c), t) for (r, c), t in cells_and_times]
    )


def test_move_and_wait_counts() -> None:
    tp = _path([((0, 0), 0), ((0, 1), 1), ((0, 1), 2), ((0, 2), 3)])
    assert tp.move_count == 2
    assert tp.wait_count == 1
    assert tp.makespan == 3
    assert tp.arrival_time == 3
    assert tp.start_time == 0


def test_endpoints() -> None:
    tp = _path([((1, 1), 0), ((1, 2), 1), ((2, 2), 2)])
    assert tp.start == P(1, 1)
    assert tp.goal == P(2, 2)
    assert tp.positions() == [P(1, 1), P(1, 2), P(2, 2)]


def test_position_at_clamps_before_start_and_after_arrival() -> None:
    tp = _path([((0, 0), 2), ((0, 1), 3), ((0, 2), 4)])
    assert tp.position_at(0) == P(0, 0)   # before start_time -> start cell
    assert tp.position_at(2) == P(0, 0)
    assert tp.position_at(3) == P(0, 1)
    assert tp.position_at(4) == P(0, 2)
    assert tp.position_at(9) == P(0, 2)   # after arrival -> parked on goal


def test_is_wait_at() -> None:
    tp = _path([((0, 0), 0), ((0, 0), 1), ((0, 1), 2)])
    assert tp.is_wait_at(1) is True
    assert tp.is_wait_at(2) is False
    assert tp.is_wait_at(0) is False


def test_failed_path() -> None:
    tp = TimedPath.failed("R2", "no conflict-free route")
    assert tp.success is False
    assert tp.steps == []
    assert tp.position_at(0) is None
    assert "no conflict-free" in tp.failure_reason


def test_to_dict_round_numbers() -> None:
    tp = _path([((0, 0), 0), ((0, 1), 1)])
    d = tp.to_dict()
    assert d["robot_id"] == "R1"
    assert d["steps"] == [[0, 0, 0], [0, 1, 1]]
    assert d["move_count"] == 1
    assert d["arrival_time"] == 1
