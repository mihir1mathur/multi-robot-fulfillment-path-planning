"""Tests for the Dijkstra planner."""

from __future__ import annotations

import pytest

from robotics.exceptions import InvalidPositionError
from robotics.planning.dijkstra import ALGORITHM_NAME, plan
from robotics.planning.validation import find_path_problems, is_valid_path
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


# ----------------------------------------------------------------------
# The open warehouse: costs are known by hand
# ----------------------------------------------------------------------
def test_finds_a_straight_horizontal_path(open_warehouse: Warehouse) -> None:
    result = plan(Position(2, 0), Position(2, 3), open_warehouse)

    assert result.success is True
    assert result.total_cost == 3.0
    assert result.path == [
        Position(2, 0),
        Position(2, 1),
        Position(2, 2),
        Position(2, 3),
    ]


def test_finds_a_straight_vertical_path(open_warehouse: Warehouse) -> None:
    result = plan(Position(0, 1), Position(3, 1), open_warehouse)

    assert result.success is True
    assert result.total_cost == 3.0
    assert len(result.path) == 4


def test_cost_in_an_open_warehouse_equals_manhattan_distance(
    open_warehouse: Warehouse,
) -> None:
    # With nothing in the way, the shortest route is exactly the Manhattan
    # distance - there is no detour to make.
    start, goal = Position(0, 0), Position(5, 4)
    result = plan(start, goal, open_warehouse)

    assert result.total_cost == start.manhattan_distance_to(goal)
    assert result.total_cost == 9.0


def test_path_includes_both_start_and_goal(open_warehouse: Warehouse) -> None:
    start, goal = Position(1, 1), Position(4, 3)
    result = plan(start, goal, open_warehouse)

    assert result.path[0] == start
    assert result.path[-1] == goal


def test_cost_is_one_less_than_the_number_of_cells(open_warehouse: Warehouse) -> None:
    result = plan(Position(0, 0), Position(3, 3), open_warehouse)
    assert result.total_cost == len(result.path) - 1


def test_every_step_is_one_orthogonal_move(open_warehouse: Warehouse) -> None:
    result = plan(Position(0, 0), Position(5, 5), open_warehouse)

    for current, following in zip(result.path, result.path[1:]):
        assert current.is_adjacent_to(following)


def test_result_records_the_algorithm_and_endpoints(open_warehouse: Warehouse) -> None:
    start, goal = Position(1, 2), Position(3, 4)
    result = plan(start, goal, open_warehouse)

    assert result.algorithm == ALGORITHM_NAME == "dijkstra"
    assert result.start == start
    assert result.goal == goal
    assert result.failure_reason is None


def test_planning_time_is_recorded(open_warehouse: Warehouse) -> None:
    result = plan(Position(0, 0), Position(5, 5), open_warehouse)
    assert result.planning_time_ms > 0.0


# ----------------------------------------------------------------------
# start == goal
# ----------------------------------------------------------------------
def test_start_equals_goal_returns_a_zero_cost_single_cell_path(
    open_warehouse: Warehouse,
) -> None:
    start = Position(2, 2)
    result = plan(start, start, open_warehouse)

    assert result.success is True
    assert result.path == [start]
    assert result.total_cost == 0.0
    assert result.nodes_expanded == 0  # the queue is never touched


def test_start_equals_goal_on_a_blocked_cell_still_fails(
    wall_warehouse: Warehouse,
) -> None:
    # The blocked-cell check must run BEFORE the start == goal shortcut,
    # otherwise a robot could be "routed" to a cell inside a wall.
    blocked = Position(0, 3)
    result = plan(blocked, blocked, wall_warehouse)

    assert result.success is False
    assert "blocked" in result.failure_reason


# ----------------------------------------------------------------------
# Static obstacles
# ----------------------------------------------------------------------
def test_routes_around_a_static_obstacle(open_warehouse: Warehouse) -> None:
    open_warehouse.add_static_obstacle("rack-1", Position(2, 2))
    result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    assert result.success is True
    assert Position(2, 2) not in result.path
    # Straight through would be 2 moves; going around costs 4.
    assert result.total_cost == 4.0


def test_never_enters_a_blocked_cell(wall_warehouse: Warehouse) -> None:
    result = plan(Position(3, 0), Position(3, 6), wall_warehouse)

    assert result.success is True
    blocked = wall_warehouse.blocked_cells()
    assert all(cell not in blocked for cell in result.path)


def test_takes_the_only_gap_through_a_wall(wall_warehouse: Warehouse) -> None:
    # The only traversable cell in column 3 is (6, 3), so every crossing route
    # must pass through it.
    result = plan(Position(0, 0), Position(0, 6), wall_warehouse)

    assert result.success is True
    assert Position(6, 3) in result.path
    assert result.total_cost == 18.0


def test_finds_the_shortest_of_several_possible_detours(
    open_warehouse: Warehouse,
) -> None:
    # A stub wall that is shorter to go around one way than the other.
    open_warehouse.add_static_obstacle("stub-0", Position(0, 3))
    open_warehouse.add_static_obstacle("stub-1", Position(1, 3))
    open_warehouse.add_static_obstacle("stub-2", Position(2, 3))

    result = plan(Position(0, 2), Position(0, 4), open_warehouse)

    assert result.success is True
    # Down past the stub (3 rows), across, and back up: 2 + 3 + 3 = 8.
    assert result.total_cost == 8.0


# ----------------------------------------------------------------------
# Dynamic obstacles
# ----------------------------------------------------------------------
def test_treats_an_active_dynamic_obstacle_as_blocked(
    open_warehouse: Warehouse,
) -> None:
    open_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2), "spill")
    result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    assert result.success is True
    assert Position(2, 2) not in result.path
    assert result.total_cost == 4.0


def test_removing_a_dynamic_obstacle_restores_the_short_route(
    open_warehouse: Warehouse,
) -> None:
    # Planning uses the warehouse as it is AT THE MOMENT OF THE CALL. This test
    # documents that by planning twice, not by replanning mid-execution -
    # replanning during execution is not implemented.
    open_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2))
    blocked_result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    open_warehouse.remove_dynamic_obstacle("spill-1")
    cleared_result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    assert blocked_result.total_cost == 4.0
    assert cleared_result.total_cost == 2.0
    assert Position(2, 2) in cleared_result.path


def test_a_dynamic_obstacle_can_make_a_goal_unreachable(
    open_warehouse: Warehouse,
) -> None:
    # Box the corner cell (0, 0) in with temporary obstacles.
    open_warehouse.add_dynamic_obstacle("cart-1", Position(0, 1))
    open_warehouse.add_dynamic_obstacle("cart-2", Position(1, 0))

    result = plan(Position(3, 3), Position(0, 0), open_warehouse)

    assert result.success is False
    assert "unreachable" in result.failure_reason


# ----------------------------------------------------------------------
# Unreachable goals and failure handling
# ----------------------------------------------------------------------
def test_unreachable_goal_fails_cleanly(sealed_warehouse: Warehouse) -> None:
    result = plan(Position(4, 0), Position(0, 4), sealed_warehouse)

    assert result.success is False
    assert result.path == []
    assert result.total_cost is None
    assert "unreachable" in result.failure_reason


def test_unreachable_goal_still_reports_the_work_it_did(
    sealed_warehouse: Warehouse,
) -> None:
    # A failed search still expanded every reachable cell; reporting that is
    # what makes the benchmark's failure cases meaningful rather than free.
    result = plan(Position(4, 0), Position(0, 4), sealed_warehouse)

    assert result.nodes_expanded > 0
    assert result.planning_time_ms > 0.0


def test_blocked_start_fails_without_searching(wall_warehouse: Warehouse) -> None:
    result = plan(Position(0, 3), Position(6, 6), wall_warehouse)

    assert result.success is False
    assert "Start" in result.failure_reason
    assert result.nodes_expanded == 0


def test_blocked_goal_fails_without_searching(wall_warehouse: Warehouse) -> None:
    result = plan(Position(6, 6), Position(0, 3), wall_warehouse)

    assert result.success is False
    assert "Goal" in result.failure_reason
    assert result.nodes_expanded == 0


# ----------------------------------------------------------------------
# Invalid coordinates
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "start",
    [Position(-1, 0), Position(0, -1), Position(6, 0), Position(99, 99)],
)
def test_start_outside_the_warehouse_raises(
    open_warehouse: Warehouse, start: Position
) -> None:
    # Out of bounds is a programming error, not a planning outcome, so it is
    # raised rather than returned - matching the rest of the project.
    with pytest.raises(InvalidPositionError):
        plan(start, Position(0, 0), open_warehouse)


@pytest.mark.parametrize(
    "goal",
    [Position(-1, 0), Position(0, 6), Position(6, 6)],
)
def test_goal_outside_the_warehouse_raises(
    open_warehouse: Warehouse, goal: Position
) -> None:
    with pytest.raises(InvalidPositionError):
        plan(Position(0, 0), goal, open_warehouse)


# ----------------------------------------------------------------------
# Every successful path is independently valid
# ----------------------------------------------------------------------
def test_paths_pass_independent_validation(wall_warehouse: Warehouse) -> None:
    result = plan(Position(0, 0), Position(0, 6), wall_warehouse)
    assert find_path_problems(result, wall_warehouse) == []


def test_every_reachable_pair_produces_a_valid_path(
    wall_warehouse: Warehouse,
) -> None:
    free_cells = [
        cell for cell in wall_warehouse.grid.all_positions()
        if wall_warehouse.is_traversable(cell)
    ]
    goal = Position(3, 6)

    for start in free_cells:
        result = plan(start, goal, wall_warehouse)
        assert result.success is True
        assert is_valid_path(result, wall_warehouse), find_path_problems(
            result, wall_warehouse
        )


# ----------------------------------------------------------------------
# Search behaviour
# ----------------------------------------------------------------------
def test_no_cell_is_expanded_more_than_once(open_warehouse: Warehouse) -> None:
    # Dijkstra pushes a new queue entry every time it finds a better route, so
    # the queue holds stale duplicates. If those were counted, nodes_expanded
    # would exceed the number of cells in the warehouse.
    result = plan(Position(0, 0), Position(5, 5), open_warehouse)
    total_cells = open_warehouse.width * open_warehouse.height

    assert result.nodes_expanded <= total_cells


def test_searching_a_whole_open_warehouse_expands_most_of_it(
    open_warehouse: Warehouse,
) -> None:
    # Dijkstra has no idea where the goal is, so reaching the far corner means
    # exploring nearly everything. This is the behaviour A* improves on.
    result = plan(Position(0, 0), Position(5, 5), open_warehouse)
    assert result.nodes_expanded >= 30  # of 36 cells


def test_is_deterministic(wall_warehouse: Warehouse) -> None:
    first = plan(Position(0, 0), Position(0, 6), wall_warehouse)
    second = plan(Position(0, 0), Position(0, 6), wall_warehouse)

    assert first.path == second.path
    assert first.nodes_expanded == second.nodes_expanded
