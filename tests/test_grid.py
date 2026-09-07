"""Tests for Position, Direction and Grid.

Each test checks ONE idea and says in its name what that idea is, so a failure
report reads like a sentence describing what broke.
"""

from __future__ import annotations

import pytest

from robotics.exceptions import InvalidPositionError
from robotics.warehouse.grid import MOVE_COST, Direction, Grid, Position


# ----------------------------------------------------------------------
# Position
# ----------------------------------------------------------------------
def test_position_stores_row_and_column() -> None:
    position = Position(2, 3)
    assert position.row == 2
    assert position.col == 3


def test_positions_with_same_coordinates_are_equal_and_hash_alike() -> None:
    # Equality and hashing matter because positions are used as dict keys and
    # set members for obstacles and occupancy.
    assert Position(1, 1) == Position(1, 1)
    assert len({Position(1, 1), Position(1, 1)}) == 1


def test_position_is_immutable() -> None:
    position = Position(1, 1)
    with pytest.raises(Exception):
        position.row = 5  # type: ignore[misc]


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (Position(0, 0), Position(0, 0), 0),
        (Position(0, 0), Position(0, 1), 1),
        (Position(0, 0), Position(1, 1), 2),  # diagonal costs two steps
        (Position(3, 4), Position(0, 0), 7),
    ],
)
def test_manhattan_distance(a: Position, b: Position, expected: int) -> None:
    assert a.manhattan_distance_to(b) == expected


def test_adjacency_is_orthogonal_only() -> None:
    centre = Position(2, 2)
    assert centre.is_adjacent_to(Position(1, 2)) is True   # up
    assert centre.is_adjacent_to(Position(3, 2)) is True   # down
    assert centre.is_adjacent_to(Position(2, 1)) is True   # left
    assert centre.is_adjacent_to(Position(2, 3)) is True   # right
    assert centre.is_adjacent_to(Position(1, 1)) is False  # diagonal
    assert centre.is_adjacent_to(Position(2, 2)) is False  # itself
    assert centre.is_adjacent_to(Position(2, 4)) is False  # two cells away


# ----------------------------------------------------------------------
# Direction
# ----------------------------------------------------------------------
def test_only_four_directions_exist() -> None:
    # Guards the "no diagonal movement" decision: if someone adds a diagonal
    # direction later, this test tells them a design decision is being changed.
    assert len(list(Direction)) == 4


@pytest.mark.parametrize(
    "direction, expected",
    [
        (Direction.UP, Position(1, 2)),
        (Direction.DOWN, Position(3, 2)),
        (Direction.LEFT, Position(2, 1)),
        (Direction.RIGHT, Position(2, 3)),
    ],
)
def test_direction_applies_correct_offset(
    direction: Direction, expected: Position
) -> None:
    assert direction.apply_to(Position(2, 2)) == expected


def test_move_cost_is_one_per_orthogonal_step() -> None:
    assert MOVE_COST == 1.0


# ----------------------------------------------------------------------
# Grid construction
# ----------------------------------------------------------------------
def test_grid_stores_its_dimensions(empty_grid: Grid) -> None:
    assert empty_grid.width == 5
    assert empty_grid.height == 5


@pytest.mark.parametrize("width, height", [(0, 5), (5, 0), (-1, 5), (5, -3)])
def test_grid_rejects_non_positive_dimensions(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        Grid(width=width, height=height)


def test_all_positions_yields_every_cell_once() -> None:
    grid = Grid(width=3, height=2)
    positions = list(grid.all_positions())
    assert len(positions) == 6
    assert len(set(positions)) == 6
    assert positions[0] == Position(0, 0)
    assert positions[-1] == Position(1, 2)


# ----------------------------------------------------------------------
# Bounds
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "position",
    [Position(0, 0), Position(4, 4), Position(0, 4), Position(4, 0), Position(2, 2)],
)
def test_in_bounds_accepts_valid_coordinates(
    empty_grid: Grid, position: Position
) -> None:
    assert empty_grid.in_bounds(position) is True


@pytest.mark.parametrize(
    "position",
    [Position(-1, 0), Position(0, -1), Position(5, 0), Position(0, 5), Position(9, 9)],
)
def test_in_bounds_rejects_out_of_bound_coordinates(
    empty_grid: Grid, position: Position
) -> None:
    assert empty_grid.in_bounds(position) is False


# ----------------------------------------------------------------------
# Traversability
# ----------------------------------------------------------------------
def test_new_grid_is_fully_traversable(empty_grid: Grid) -> None:
    assert all(empty_grid.is_traversable(p) for p in empty_grid.all_positions())


def test_blocking_a_cell_makes_it_untraversable(empty_grid: Grid) -> None:
    blocked = Position(2, 2)
    empty_grid.block(blocked)

    assert empty_grid.is_blocked(blocked) is True
    assert empty_grid.is_traversable(blocked) is False


def test_unblocking_restores_traversability(empty_grid: Grid) -> None:
    cell = Position(2, 2)
    empty_grid.block(cell)
    empty_grid.unblock(cell)

    assert empty_grid.is_traversable(cell) is True


def test_blocking_twice_is_harmless(empty_grid: Grid) -> None:
    cell = Position(1, 1)
    empty_grid.block(cell)
    empty_grid.block(cell)

    assert len(empty_grid.blocked_cells) == 1


def test_unblocking_a_free_cell_is_harmless(empty_grid: Grid) -> None:
    empty_grid.unblock(Position(1, 1))
    assert empty_grid.blocked_cells == set()


def test_out_of_bounds_cell_reports_as_blocked(empty_grid: Grid) -> None:
    # "No floor there" and "wall there" have the same consequence for a robot.
    assert empty_grid.is_blocked(Position(99, 99)) is True
    assert empty_grid.is_traversable(Position(99, 99)) is False


def test_blocking_out_of_bounds_raises(empty_grid: Grid) -> None:
    with pytest.raises(InvalidPositionError):
        empty_grid.block(Position(5, 5))


def test_blocked_cells_property_returns_a_copy(empty_grid: Grid) -> None:
    empty_grid.block(Position(0, 0))
    snapshot = empty_grid.blocked_cells
    snapshot.add(Position(4, 4))

    assert empty_grid.is_traversable(Position(4, 4)) is True


# ----------------------------------------------------------------------
# Neighbours
# ----------------------------------------------------------------------
def test_middle_cell_has_four_neighbours(empty_grid: Grid) -> None:
    neighbours = empty_grid.neighbors(Position(2, 2))

    assert set(neighbours) == {
        Position(1, 2),
        Position(3, 2),
        Position(2, 1),
        Position(2, 3),
    }


def test_corner_cell_has_two_neighbours(empty_grid: Grid) -> None:
    assert set(empty_grid.neighbors(Position(0, 0))) == {Position(0, 1), Position(1, 0)}


def test_edge_cell_has_three_neighbours(empty_grid: Grid) -> None:
    assert len(empty_grid.neighbors(Position(0, 2))) == 3


def test_neighbours_exclude_blocked_cells(empty_grid: Grid) -> None:
    empty_grid.block(Position(1, 2))
    neighbours = empty_grid.neighbors(Position(2, 2))

    assert Position(1, 2) not in neighbours
    assert len(neighbours) == 3


def test_neighbours_never_include_diagonals(empty_grid: Grid) -> None:
    neighbours = empty_grid.neighbors(Position(2, 2))
    diagonals = {Position(1, 1), Position(1, 3), Position(3, 1), Position(3, 3)}

    assert diagonals.isdisjoint(neighbours)


def test_neighbours_of_out_of_bounds_cell_raises(empty_grid: Grid) -> None:
    with pytest.raises(InvalidPositionError):
        empty_grid.neighbors(Position(-1, 0))
