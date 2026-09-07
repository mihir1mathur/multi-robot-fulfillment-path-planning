"""Tests for the prioritized multi-robot coordinator."""

from __future__ import annotations

from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.coordinator import MultiRobotCoordinator, default_horizon
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _wh(w=7, h=7, name="coord"):
    return Warehouse(width=w, height=h, name=name)


def _plan(wh, starts, goals, **kw):
    return MultiRobotCoordinator(wh).plan(starts, goals, **kw)


def _assert_valid(result, wh, starts):
    problems = find_coordination_problems(result.successful_paths, wh, starts)
    assert problems == [], problems


# ----------------------------------------------------------------------
# Independent / non-conflicting
# ----------------------------------------------------------------------
def test_two_robots_with_independent_routes() -> None:
    wh = _wh()
    starts = {"R1": P(0, 0), "R2": P(6, 0)}
    goals = {"R1": P(0, 5), "R2": P(6, 5)}
    result = _plan(wh, starts, goals)

    assert result.success
    assert result.planned_robot_count == 2
    assert result.total_wait_steps == 0        # no need to wait for anyone
    _assert_valid(result, wh, starts)


# ----------------------------------------------------------------------
# Conflicts the coordinator resolves
# ----------------------------------------------------------------------
def test_intersection_is_resolved_with_a_wait() -> None:
    wh = _wh()
    starts = {"R1": P(3, 0), "R2": P(0, 3)}
    goals = {"R1": P(3, 6), "R2": P(6, 3)}       # cross at (3,3)
    result = _plan(wh, starts, goals)

    assert result.success
    assert result.total_wait_steps >= 1          # someone yields
    _assert_valid(result, wh, starts)


def test_two_robots_crossing_the_same_cell_do_not_collide() -> None:
    wh = _wh()
    starts = {"R1": P(0, 2), "R2": P(2, 0)}
    goals = {"R1": P(4, 2), "R2": P(2, 4)}
    result = _plan(wh, starts, goals)

    assert result.success
    _assert_valid(result, wh, starts)


def test_goal_cell_on_another_robots_path_forces_a_detour_or_wait() -> None:
    wh = _wh()
    # R1 (priority) finishes at (3,3). R2's straight route would cross (3,3).
    starts = {"R1": P(0, 3), "R2": P(3, 0)}
    goals = {"R1": P(3, 3), "R2": P(3, 6)}
    result = _plan(wh, starts, goals)

    assert result.success
    r2 = result.timed_paths["R2"]
    # R2 must not be on (3,3) at or after R1's arrival
    r1 = result.timed_paths["R1"]
    for t in range(r1.arrival_time, result.makespan + 1):
        assert r2.position_at(t) != P(3, 3)
    _assert_valid(result, wh, starts)


def test_three_robot_intersection() -> None:
    wh = _wh(9, 9)
    starts = {"R1": P(4, 0), "R2": P(0, 4), "R3": P(8, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4), "R3": P(0, 4)}
    result = _plan(wh, starts, goals)

    assert result.success
    assert result.planned_robot_count == 3
    _assert_valid(result, wh, starts)


def test_robots_with_different_arrival_times() -> None:
    wh = _wh(9, 9)
    starts = {"R1": P(0, 0), "R2": P(0, 8)}
    goals = {"R1": P(0, 2), "R2": P(8, 8)}        # R1 short, R2 long
    result = _plan(wh, starts, goals)

    assert result.success
    assert result.timed_paths["R1"].arrival_time < result.timed_paths["R2"].arrival_time
    assert result.makespan == result.timed_paths["R2"].arrival_time
    _assert_valid(result, wh, starts)


# ----------------------------------------------------------------------
# Honest failure
# ----------------------------------------------------------------------
def test_head_on_single_width_corridor_reports_a_partial_result() -> None:
    wh = Warehouse(width=6, height=1, name="corridor")
    starts = {"R1": P(0, 0), "R2": P(0, 5)}
    goals = {"R1": P(0, 5), "R2": P(0, 0)}
    result = _plan(wh, starts, goals)

    assert result.success is False
    assert result.failed_robot_ids == ["R2"]      # R1 (priority) got through
    assert result.planned_robot_count == 1
    assert "R2" in result.failure_reason
    # the one robot that was planned is still a valid, conflict-free path
    _assert_valid(result, wh, starts)


def test_one_robot_unplannable_within_horizon_does_not_break_the_others() -> None:
    wh = Warehouse(width=6, height=1, name="corridor")
    starts = {"R1": P(0, 0), "R2": P(0, 5)}
    goals = {"R1": P(0, 5), "R2": P(0, 0)}
    result = _plan(wh, starts, goals)

    r1 = result.timed_paths["R1"]
    assert r1.success
    assert r1.positions() == [P(0, c) for c in range(6)]   # R1 unaffected


def test_duplicate_start_positions_are_rejected() -> None:
    wh = _wh()
    result = _plan(wh, {"R1": P(2, 2), "R2": P(2, 2)}, {"R1": P(0, 0), "R2": P(4, 4)})

    assert result.success is False
    assert "both start at" in result.failure_reason
    assert result.timed_paths == {}


def test_mismatched_starts_and_goals_are_rejected() -> None:
    wh = _wh()
    result = _plan(wh, {"R1": P(0, 0)}, {"R2": P(4, 4)})
    assert result.success is False
    assert "disagree on robot IDs" in result.failure_reason


# ----------------------------------------------------------------------
# Priority order
# ----------------------------------------------------------------------
def test_default_priority_order_is_sorted_robot_ids() -> None:
    wh = _wh()
    result = _plan(
        wh, {"R2": P(0, 0), "R1": P(6, 0)}, {"R2": P(0, 5), "R1": P(6, 5)}
    )
    assert result.robot_order == ["R1", "R2"]


def test_custom_priority_order_is_honoured() -> None:
    wh = Warehouse(width=6, height=1, name="corridor")
    starts = {"R1": P(0, 0), "R2": P(0, 5)}
    goals = {"R1": P(0, 5), "R2": P(0, 0)}

    default = _plan(wh, starts, goals)                       # R1 first -> R2 fails
    reversed_ = _plan(wh, starts, goals, priority_order=["R2", "R1"])

    assert default.failed_robot_ids == ["R2"]
    assert reversed_.failed_robot_ids == ["R1"]              # now R1 is the one boxed out
    assert reversed_.robot_order == ["R2", "R1"]


def test_invalid_priority_order_is_rejected() -> None:
    wh = _wh()
    result = _plan(
        wh, {"R1": P(0, 0), "R2": P(6, 0)}, {"R1": P(0, 5), "R2": P(6, 5)},
        priority_order=["R1", "R1"],
    )
    assert result.success is False
    assert "permutation" in result.failure_reason or "duplicate" in result.failure_reason


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_coordination_is_deterministic() -> None:
    wh = _wh(9, 9)
    starts = {"R1": P(4, 0), "R2": P(0, 4), "R3": P(8, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4), "R3": P(0, 4)}

    a = _plan(wh, starts, goals)
    b = _plan(wh, starts, goals)

    assert a.to_dict()["timed_paths"] == b.to_dict()["timed_paths"]
    assert a.makespan == b.makespan
    assert a.total_wait_steps == b.total_wait_steps


def test_default_horizon_grows_with_grid_and_fleet() -> None:
    small = default_horizon(Warehouse(width=5, height=5), 2)
    bigger_grid = default_horizon(Warehouse(width=20, height=20), 2)
    bigger_fleet = default_horizon(Warehouse(width=5, height=5), 20)
    assert bigger_grid > small
    assert bigger_fleet > small
