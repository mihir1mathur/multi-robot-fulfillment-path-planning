"""Tests for the A* planner.

A* must satisfy everything Dijkstra satisfies - it is answering the same
question - so the first half of this file mirrors `test_dijkstra.py`. The
second half tests the things that are specific to A*: that the heuristic is
actually being used, and that using it does not cost optimality.
"""

from __future__ import annotations

import pytest

from robotics.exceptions import InvalidPositionError
from robotics.planning.astar import ALGORITHM_NAME, plan
from robotics.planning.dijkstra import plan as plan_dijkstra
from robotics.planning.heuristics import zero_heuristic
from robotics.planning.validation import find_path_problems, is_valid_path
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


# ----------------------------------------------------------------------
# The open warehouse
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


def test_cost_in_an_open_warehouse_equals_manhattan_distance(
    open_warehouse: Warehouse,
) -> None:
    start, goal = Position(0, 0), Position(5, 4)
    result = plan(start, goal, open_warehouse)

    assert result.total_cost == start.manhattan_distance_to(goal)


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

    assert result.algorithm == ALGORITHM_NAME == "astar"
    assert result.start == start
    assert result.goal == goal
    assert result.failure_reason is None


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
    assert result.nodes_expanded == 0


def test_start_equals_goal_on_a_blocked_cell_still_fails(
    wall_warehouse: Warehouse,
) -> None:
    blocked = Position(0, 3)
    result = plan(blocked, blocked, wall_warehouse)

    assert result.success is False
    assert "blocked" in result.failure_reason


# ----------------------------------------------------------------------
# Obstacles
# ----------------------------------------------------------------------
def test_routes_around_a_static_obstacle(open_warehouse: Warehouse) -> None:
    open_warehouse.add_static_obstacle("rack-1", Position(2, 2))
    result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    assert result.success is True
    assert Position(2, 2) not in result.path
    assert result.total_cost == 4.0


def test_never_enters_a_blocked_cell(wall_warehouse: Warehouse) -> None:
    result = plan(Position(3, 0), Position(3, 6), wall_warehouse)

    assert result.success is True
    blocked = wall_warehouse.blocked_cells()
    assert all(cell not in blocked for cell in result.path)


def test_takes_the_only_gap_through_a_wall(wall_warehouse: Warehouse) -> None:
    result = plan(Position(0, 0), Position(0, 6), wall_warehouse)

    assert result.success is True
    assert Position(6, 3) in result.path
    assert result.total_cost == 18.0


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
    open_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2))
    blocked_result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    open_warehouse.remove_dynamic_obstacle("spill-1")
    cleared_result = plan(Position(2, 1), Position(2, 3), open_warehouse)

    assert blocked_result.total_cost == 4.0
    assert cleared_result.total_cost == 2.0


# ----------------------------------------------------------------------
# Failure handling
# ----------------------------------------------------------------------
def test_unreachable_goal_fails_cleanly(sealed_warehouse: Warehouse) -> None:
    result = plan(Position(4, 0), Position(0, 4), sealed_warehouse)

    assert result.success is False
    assert result.path == []
    assert result.total_cost is None
    assert "unreachable" in result.failure_reason


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


@pytest.mark.parametrize(
    "start", [Position(-1, 0), Position(0, -1), Position(6, 0), Position(99, 99)]
)
def test_start_outside_the_warehouse_raises(
    open_warehouse: Warehouse, start: Position
) -> None:
    with pytest.raises(InvalidPositionError):
        plan(start, Position(0, 0), open_warehouse)


@pytest.mark.parametrize("goal", [Position(-1, 0), Position(0, 6), Position(6, 6)])
def test_goal_outside_the_warehouse_raises(
    open_warehouse: Warehouse, goal: Position
) -> None:
    with pytest.raises(InvalidPositionError):
        plan(Position(0, 0), goal, open_warehouse)


# ----------------------------------------------------------------------
# Independent validation
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
# A*-specific behaviour: the heuristic is really being used
# ----------------------------------------------------------------------
def test_expands_fewer_nodes_than_dijkstra_in_an_open_warehouse(
    open_warehouse: Warehouse,
) -> None:
    """The whole point of A*: same answer, less searching.

    Dijkstra spreads out in every direction; A* is pulled towards the goal by
    the heuristic, so it never bothers with cells that lead away from it.
    """
    start, goal = Position(0, 0), Position(5, 5)
    astar_result = plan(start, goal, open_warehouse)
    dijkstra_result = plan_dijkstra(start, goal, open_warehouse)

    assert astar_result.total_cost == dijkstra_result.total_cost
    assert astar_result.nodes_expanded < dijkstra_result.nodes_expanded


def test_expands_no_more_nodes_than_dijkstra_when_a_wall_forces_a_detour(
    wall_warehouse: Warehouse,
) -> None:
    # With a misleading obstacle layout the heuristic helps less, but an
    # admissible heuristic can never make A* search MORE of the map than a
    # blind search would.
    start, goal = Position(0, 0), Position(0, 6)
    astar_result = plan(start, goal, wall_warehouse)
    dijkstra_result = plan_dijkstra(start, goal, wall_warehouse)

    assert astar_result.total_cost == dijkstra_result.total_cost
    assert astar_result.nodes_expanded <= dijkstra_result.nodes_expanded


def test_with_a_zero_heuristic_astar_becomes_dijkstra(
    wall_warehouse: Warehouse,
) -> None:
    """The clearest demonstration of how the two algorithms relate.

    A* orders its frontier by f = g + h. Set h to zero everywhere and the
    ordering becomes f = g, which is exactly Dijkstra's rule. Everything else
    in the two implementations is identical, so they must agree cell for cell.
    """
    start, goal = Position(0, 0), Position(0, 6)

    blind_astar = plan(start, goal, wall_warehouse, heuristic=zero_heuristic)
    dijkstra_result = plan_dijkstra(start, goal, wall_warehouse)

    assert blind_astar.path == dijkstra_result.path
    assert blind_astar.nodes_expanded == dijkstra_result.nodes_expanded
    assert blind_astar.total_cost == dijkstra_result.total_cost


def test_the_heuristic_does_not_cost_optimality(open_warehouse: Warehouse) -> None:
    # Check across many goals, not just one, that guiding the search never
    # makes the answer worse than the blind search.
    start = Position(0, 0)

    for goal in open_warehouse.grid.all_positions():
        guided = plan(start, goal, open_warehouse)
        blind = plan_dijkstra(start, goal, open_warehouse)
        assert guided.total_cost == blind.total_cost


def test_is_deterministic(wall_warehouse: Warehouse) -> None:
    first = plan(Position(0, 0), Position(0, 6), wall_warehouse)
    second = plan(Position(0, 0), Position(0, 6), wall_warehouse)

    assert first.path == second.path
    assert first.nodes_expanded == second.nodes_expanded
