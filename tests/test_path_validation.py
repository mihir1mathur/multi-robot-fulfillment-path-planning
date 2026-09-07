"""Tests for the path validator and the planning-request checker.

The validator is the safety net under both planners, so it needs its own
tests. A validator that silently accepts a broken path is worse than no
validator at all - it would give false confidence to every test that relies
on it.

The technique here is to build DELIBERATELY BROKEN results by hand and check
that each one is caught. That is the only way to know the checks actually fire.
"""

from __future__ import annotations

import pytest

from robotics.exceptions import InvalidPositionError, PathValidationError
from robotics.planning.dijkstra import plan as plan_dijkstra
from robotics.planning.path_result import PathResult, reconstruct_path
from robotics.planning.validation import (
    assert_valid_path,
    check_planning_request,
    find_path_problems,
    is_valid_path,
)
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def make_result(start: Position, goal: Position, path, cost=None) -> PathResult:
    """Build a successful-looking result by hand, so it can be broken on purpose."""
    return PathResult(
        algorithm="handmade",
        start=start,
        goal=goal,
        success=True,
        path=list(path),
        total_cost=cost if cost is not None else float(len(path) - 1),
    )


# ----------------------------------------------------------------------
# check_planning_request
# ----------------------------------------------------------------------
def test_accepts_a_sensible_request(open_warehouse: Warehouse) -> None:
    assert check_planning_request(Position(0, 0), Position(3, 3), open_warehouse) is None


def test_rejects_a_blocked_start(wall_warehouse: Warehouse) -> None:
    reason = check_planning_request(Position(0, 3), Position(6, 6), wall_warehouse)
    assert reason is not None and "Start" in reason


def test_rejects_a_blocked_goal(wall_warehouse: Warehouse) -> None:
    reason = check_planning_request(Position(6, 6), Position(0, 3), wall_warehouse)
    assert reason is not None and "Goal" in reason


def test_rejects_a_start_blocked_by_a_dynamic_obstacle(
    open_warehouse: Warehouse,
) -> None:
    open_warehouse.add_dynamic_obstacle("spill-1", Position(1, 1))
    reason = check_planning_request(Position(1, 1), Position(3, 3), open_warehouse)

    assert reason is not None and "Start" in reason


def test_raises_for_an_out_of_bounds_start(open_warehouse: Warehouse) -> None:
    with pytest.raises(InvalidPositionError):
        check_planning_request(Position(-1, 0), Position(0, 0), open_warehouse)


def test_raises_for_an_out_of_bounds_goal(open_warehouse: Warehouse) -> None:
    with pytest.raises(InvalidPositionError):
        check_planning_request(Position(0, 0), Position(0, 99), open_warehouse)


# ----------------------------------------------------------------------
# The validator accepts genuine planner output
# ----------------------------------------------------------------------
def test_accepts_a_real_planner_path(wall_warehouse: Warehouse) -> None:
    result = plan_dijkstra(Position(0, 0), Position(0, 6), wall_warehouse)
    assert find_path_problems(result, wall_warehouse) == []
    assert is_valid_path(result, wall_warehouse) is True


def test_accepts_a_single_cell_path(open_warehouse: Warehouse) -> None:
    result = plan_dijkstra(Position(2, 2), Position(2, 2), open_warehouse)
    assert find_path_problems(result, open_warehouse) == []


def test_accepts_a_genuine_failure_result(sealed_warehouse: Warehouse) -> None:
    result = plan_dijkstra(Position(4, 0), Position(0, 4), sealed_warehouse)
    assert find_path_problems(result, sealed_warehouse) == []


# ----------------------------------------------------------------------
# The validator catches broken paths
# ----------------------------------------------------------------------
def test_catches_an_empty_path_on_a_successful_result(
    open_warehouse: Warehouse,
) -> None:
    result = make_result(Position(0, 0), Position(0, 1), [], cost=1.0)
    problems = find_path_problems(result, open_warehouse)

    assert any("empty path" in problem for problem in problems)


def test_catches_a_wrong_start(open_warehouse: Warehouse) -> None:
    result = make_result(
        Position(0, 0), Position(0, 2), [Position(0, 1), Position(0, 2)]
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("path starts at" in problem for problem in problems)


def test_catches_a_wrong_goal(open_warehouse: Warehouse) -> None:
    result = make_result(
        Position(0, 0), Position(0, 5), [Position(0, 0), Position(0, 1)]
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("path ends at" in problem for problem in problems)


def test_catches_a_cell_outside_the_warehouse(open_warehouse: Warehouse) -> None:
    result = make_result(
        Position(0, 0), Position(0, 2), [Position(0, 0), Position(-1, 1), Position(0, 2)]
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("outside the warehouse" in problem for problem in problems)


def test_catches_a_path_through_a_static_obstacle(wall_warehouse: Warehouse) -> None:
    # Straight through the dividing wall - the exact bug the validator exists
    # to catch if a planner ever forgot to filter obstacles.
    result = make_result(
        Position(0, 2), Position(0, 4),
        [Position(0, 2), Position(0, 3), Position(0, 4)],
    )
    problems = find_path_problems(result, wall_warehouse)

    assert any("blocked by an obstacle" in problem for problem in problems)


def test_catches_a_path_through_a_dynamic_obstacle(open_warehouse: Warehouse) -> None:
    open_warehouse.add_dynamic_obstacle("spill-1", Position(0, 1))
    result = make_result(
        Position(0, 0), Position(0, 2),
        [Position(0, 0), Position(0, 1), Position(0, 2)],
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("blocked by an obstacle" in problem for problem in problems)


def test_catches_an_illegal_jump(open_warehouse: Warehouse) -> None:
    result = make_result(
        Position(0, 0), Position(0, 4), [Position(0, 0), Position(0, 4)]
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("not one orthogonal step apart" in problem for problem in problems)


def test_catches_a_diagonal_step(open_warehouse: Warehouse) -> None:
    # A diagonal looks adjacent but is two moves under our rules.
    result = make_result(
        Position(0, 0), Position(1, 1), [Position(0, 0), Position(1, 1)]
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("not one orthogonal step apart" in problem for problem in problems)


def test_catches_a_repeated_cell(open_warehouse: Warehouse) -> None:
    result = make_result(
        Position(0, 0), Position(0, 1),
        [Position(0, 0), Position(0, 1), Position(0, 0), Position(0, 1)],
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("same cell more than once" in problem for problem in problems)


def test_catches_a_cost_that_disagrees_with_the_path(
    open_warehouse: Warehouse,
) -> None:
    result = make_result(
        Position(0, 0), Position(0, 2),
        [Position(0, 0), Position(0, 1), Position(0, 2)],
        cost=99.0,
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("does not match" in problem for problem in problems)


def test_reports_every_problem_at_once(open_warehouse: Warehouse) -> None:
    # Wrong start, wrong goal, an illegal jump and a wrong cost together.
    result = make_result(
        Position(0, 0), Position(0, 5),
        [Position(1, 1), Position(3, 3)],
        cost=42.0,
    )
    problems = find_path_problems(result, open_warehouse)

    assert len(problems) >= 4


# ----------------------------------------------------------------------
# Unsuccessful results have their own rules
# ----------------------------------------------------------------------
def test_catches_a_failure_that_still_carries_a_path(
    open_warehouse: Warehouse,
) -> None:
    result = PathResult(
        algorithm="handmade",
        start=Position(0, 0),
        goal=Position(0, 2),
        success=False,
        path=[Position(0, 0), Position(0, 1)],
        failure_reason="something",
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("empty path" in problem for problem in problems)


def test_catches_a_failure_that_reports_a_cost(open_warehouse: Warehouse) -> None:
    result = PathResult(
        algorithm="handmade",
        start=Position(0, 0),
        goal=Position(0, 2),
        success=False,
        total_cost=5.0,
        failure_reason="something",
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("total_cost None" in problem for problem in problems)


def test_catches_a_failure_with_no_reason(open_warehouse: Warehouse) -> None:
    result = PathResult(
        algorithm="handmade",
        start=Position(0, 0),
        goal=Position(0, 2),
        success=False,
    )
    problems = find_path_problems(result, open_warehouse)

    assert any("explain why it failed" in problem for problem in problems)


# ----------------------------------------------------------------------
# assert_valid_path
# ----------------------------------------------------------------------
def test_assert_valid_path_is_silent_for_a_good_path(
    wall_warehouse: Warehouse,
) -> None:
    result = plan_dijkstra(Position(0, 0), Position(0, 6), wall_warehouse)
    assert_valid_path(result, wall_warehouse)  # must not raise


def test_assert_valid_path_raises_for_a_bad_path(open_warehouse: Warehouse) -> None:
    result = make_result(
        Position(0, 0), Position(0, 4), [Position(0, 0), Position(0, 4)]
    )
    with pytest.raises(PathValidationError):
        assert_valid_path(result, open_warehouse)


def test_the_raised_error_names_the_algorithm_and_endpoints(
    open_warehouse: Warehouse,
) -> None:
    result = make_result(
        Position(0, 0), Position(0, 4), [Position(0, 0), Position(0, 4)]
    )
    with pytest.raises(PathValidationError, match="handmade"):
        assert_valid_path(result, open_warehouse)


# ----------------------------------------------------------------------
# reconstruct_path
# ----------------------------------------------------------------------
def test_reconstruct_path_walks_backwards_then_reverses() -> None:
    a, b, c, d = Position(0, 0), Position(0, 1), Position(0, 2), Position(0, 3)
    came_from = {b: a, c: b, d: c}

    assert reconstruct_path(came_from, start=a, goal=d) == [a, b, c, d]


def test_reconstruct_path_handles_a_single_cell() -> None:
    only = Position(2, 2)
    assert reconstruct_path({}, start=only, goal=only) == [only]
