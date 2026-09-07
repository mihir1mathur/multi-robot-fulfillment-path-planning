"""Tests for the independent coordinated-path conflict validator.

It must ACCEPT genuinely conflict-free timed paths and REJECT hand-built ones
that contain a vertex conflict, an edge swap, an illegal jump, an obstacle
crossing, or a goal-occupation violation.
"""

from __future__ import annotations

from robotics.coordination.conflicts import (
    assert_conflict_free,
    find_coordination_problems,
    is_conflict_free,
)
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _wh(w=6, h=6):
    return Warehouse(width=w, height=h, name="cv-test")


def _tp(rid, cells_times):
    return TimedPath.found(rid, [TimedStep(P(r, c), t) for (r, c), t in cells_times])


# ----------------------------------------------------------------------
# Accepts good plans
# ----------------------------------------------------------------------
def test_two_independent_paths_are_conflict_free() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2)]),
        "R2": _tp("R2", [((3, 0), 0), ((3, 1), 1), ((3, 2), 2)]),
    }
    assert find_coordination_problems(paths, _wh()) == []
    assert is_conflict_free(paths, _wh())


def test_following_the_same_corridor_is_allowed() -> None:
    # R2 follows one cell behind R1 - no vertex or edge conflict.
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2), ((0, 3), 3)]),
        "R2": _tp("R2", [((0, 0), 0), ((0, 0), 0)]) if False else
              _tp("R2", [((1, 0), 0), ((0, 0), 1), ((0, 1), 2), ((0, 2), 3)]),
    }
    assert find_coordination_problems(paths, _wh()) == []


def test_a_wait_step_is_accepted() -> None:
    paths = {"R1": _tp("R1", [((0, 0), 0), ((0, 0), 1), ((0, 1), 2)])}
    assert find_coordination_problems(paths, _wh()) == []


# ----------------------------------------------------------------------
# Catches conflicts
# ----------------------------------------------------------------------
def test_vertex_conflict_is_caught() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((1, 1), 1), ((2, 2), 2)]) if False else
              _tp("R1", [((1, 0), 0), ((1, 1), 1), ((1, 2), 2)]),
        "R2": _tp("R2", [((0, 1), 0), ((1, 1), 1), ((2, 1), 2)]),
    }
    problems = find_coordination_problems(paths, _wh())
    assert any("vertex conflict" in p and "(1, 1)" in p and "t=1" in p for p in problems)


def test_edge_swap_conflict_is_caught() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1)]),
        "R2": _tp("R2", [((0, 1), 0), ((0, 0), 1)]),
    }
    problems = find_coordination_problems(paths, _wh())
    assert any("edge/swap conflict" in p for p in problems)


def test_illegal_jump_is_caught() -> None:
    paths = {"R1": _tp("R1", [((0, 0), 0), ((0, 3), 1)])}  # 3 cells in one step
    problems = find_coordination_problems(paths, _wh())
    assert any("illegal transition" in p for p in problems)


def test_wrong_start_is_caught() -> None:
    paths = {"R1": _tp("R1", [((0, 0), 0), ((0, 1), 1)])}
    problems = find_coordination_problems(
        paths, _wh(), expected_starts={"R1": P(5, 5)}
    )
    assert any("starts at" in p for p in problems)


def test_non_consecutive_timesteps_are_caught() -> None:
    paths = {"R1": _tp("R1", [((0, 0), 0), ((0, 1), 2)])}  # jumps t=0 -> t=2
    problems = find_coordination_problems(paths, _wh())
    assert any("timestep" in p for p in problems)


def test_path_through_an_obstacle_is_caught() -> None:
    wh = _wh()
    wh.add_static_obstacle("rack", P(0, 1))
    paths = {"R1": _tp("R1", [((0, 0), 0), ((0, 1), 1), ((0, 2), 2)])}
    problems = find_coordination_problems(paths, wh)
    assert any("blocked by an obstacle" in p for p in problems)


def test_goal_occupation_violation_is_caught() -> None:
    # R1 parks on (2,2) from t=2. R2 drives through (2,2) at t=4.
    paths = {
        "R1": _tp("R1", [((0, 2), 0), ((1, 2), 1), ((2, 2), 2)]),
        "R2": _tp("R2", [((2, 0), 0), ((2, 1), 1), ((2, 1), 2), ((2, 1), 3), ((2, 2), 4)]),
    }
    problems = find_coordination_problems(paths, _wh())
    assert any("goal-occupation violation" in p for p in problems)


def test_assert_conflict_free_raises_on_a_bad_plan() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1)]),
        "R2": _tp("R2", [((0, 1), 0), ((0, 0), 1)]),
    }
    import pytest

    with pytest.raises(Exception):
        assert_conflict_free(paths, _wh())


def test_failed_paths_are_ignored() -> None:
    paths = {
        "R1": _tp("R1", [((0, 0), 0), ((0, 1), 1)]),
        "R2": TimedPath.failed("R2", "no route"),
    }
    assert find_coordination_problems(paths, _wh()) == []
