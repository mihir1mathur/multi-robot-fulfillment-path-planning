"""Tests for the Manhattan-distance heuristic.

The heuristic is a tiny function, but A*'s correctness depends entirely on it:
if it ever over-estimated the remaining distance, A* could return a path that
is not the shortest. So it is worth testing properly, including the property
that actually matters (admissibility) rather than only a few sample values.
"""

from __future__ import annotations

import pytest

from robotics.planning.dijkstra import plan as plan_dijkstra
from robotics.planning.heuristics import manhattan_distance, zero_heuristic
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


# ----------------------------------------------------------------------
# Basic values
# ----------------------------------------------------------------------
def test_distance_to_itself_is_zero() -> None:
    assert manhattan_distance(Position(3, 4), Position(3, 4)) == 0.0


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Position(0, 0), Position(0, 1), 1.0),
        (Position(0, 0), Position(0, 5), 5.0),
        (Position(2, 7), Position(2, 3), 4.0),
    ],
)
def test_horizontal_distance_counts_columns(
    a: Position, b: Position, expected: float
) -> None:
    assert manhattan_distance(a, b) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Position(0, 0), Position(1, 0), 1.0),
        (Position(0, 0), Position(6, 0), 6.0),
        (Position(9, 2), Position(4, 2), 5.0),
    ],
)
def test_vertical_distance_counts_rows(
    a: Position, b: Position, expected: float
) -> None:
    assert manhattan_distance(a, b) == expected


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Position(0, 0), Position(1, 1), 2.0),   # a diagonal costs TWO moves
        (Position(0, 0), Position(3, 4), 7.0),
        (Position(5, 5), Position(2, 1), 7.0),
        (Position(11, 0), Position(0, 11), 22.0),
    ],
)
def test_mixed_distance_adds_rows_and_columns(
    a: Position, b: Position, expected: float
) -> None:
    assert manhattan_distance(a, b) == expected


def test_diagonal_neighbour_is_two_not_one() -> None:
    # The single most important sanity check: a diagonally adjacent cell LOOKS
    # close, but with four-directional movement it genuinely takes two moves.
    assert manhattan_distance(Position(4, 4), Position(5, 5)) == 2.0


# ----------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "a, b",
    [
        (Position(0, 0), Position(3, 4)),
        (Position(7, 2), Position(1, 9)),
        (Position(5, 5), Position(5, 5)),
    ],
)
def test_distance_is_symmetric(a: Position, b: Position) -> None:
    assert manhattan_distance(a, b) == manhattan_distance(b, a)


def test_distance_is_never_negative() -> None:
    grid_cells = [Position(row, col) for row in range(4) for col in range(4)]
    assert all(
        manhattan_distance(a, b) >= 0.0 for a in grid_cells for b in grid_cells
    )


def test_result_is_a_float() -> None:
    # The planners mix heuristic values with path costs, so both must be the
    # same numeric type to avoid surprising int/float comparisons.
    assert isinstance(manhattan_distance(Position(0, 0), Position(1, 1)), float)


def test_obeys_the_triangle_inequality() -> None:
    # Going via a third cell can never be shorter than going direct. This is
    # the property that makes the heuristic CONSISTENT, which is what allows
    # both planners to finalise a cell the first time they pop it.
    a, b, via = Position(0, 0), Position(5, 5), Position(2, 3)
    assert manhattan_distance(a, b) <= manhattan_distance(a, via) + manhattan_distance(
        via, b
    )


# ----------------------------------------------------------------------
# Admissibility - the property A*'s optimality actually depends on
# ----------------------------------------------------------------------
def test_never_overestimates_the_true_cost_in_an_open_warehouse(
    open_warehouse: Warehouse,
) -> None:
    """In an empty warehouse the heuristic should equal the true cost exactly."""
    goal = Position(5, 5)
    for start in open_warehouse.grid.all_positions():
        true_cost = plan_dijkstra(start, goal, open_warehouse).total_cost
        assert manhattan_distance(start, goal) == true_cost


def test_never_overestimates_the_true_cost_when_obstacles_force_a_detour(
    wall_warehouse: Warehouse,
) -> None:
    """With obstacles the true cost grows, so the heuristic under-estimates.

    Under-estimating is exactly the required property: obstacles can only ever make the
    real route longer, never shorter, so Manhattan distance stays a valid lower
    bound and A* stays optimal.
    """
    goal = Position(0, 6)

    for start in wall_warehouse.grid.all_positions():
        if not wall_warehouse.is_traversable(start):
            continue
        result = plan_dijkstra(start, goal, wall_warehouse)
        assert result.success
        assert manhattan_distance(start, goal) <= result.total_cost


def test_the_wall_makes_the_heuristic_a_strict_underestimate(
    wall_warehouse: Warehouse,
) -> None:
    # Crossing the wall forces a long detour to row 6, so here the guess is
    # much lower than reality - which is legal, and is why A* still has to
    # search rather than trusting the guess.
    start, goal = Position(0, 0), Position(0, 6)
    true_cost = plan_dijkstra(start, goal, wall_warehouse).total_cost

    assert manhattan_distance(start, goal) == 6.0
    assert true_cost == 18.0


# ----------------------------------------------------------------------
# The zero heuristic
# ----------------------------------------------------------------------
def test_zero_heuristic_always_returns_zero() -> None:
    assert zero_heuristic(Position(0, 0), Position(9, 9)) == 0.0
    assert zero_heuristic(Position(4, 4), Position(4, 4)) == 0.0
