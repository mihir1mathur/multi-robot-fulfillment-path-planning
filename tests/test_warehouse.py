"""Tests for the Warehouse: named locations and the two obstacle layers."""

from __future__ import annotations

import pytest

from robotics.exceptions import InvalidPositionError, SimulationError
from robotics.warehouse.grid import Position
from robotics.warehouse.obstacle import ObstacleType
from robotics.warehouse.warehouse import LocationType, Warehouse


# ----------------------------------------------------------------------
# Locations
# ----------------------------------------------------------------------
def test_every_cell_starts_as_an_aisle() -> None:
    warehouse = Warehouse(width=3, height=3)
    assert all(
        warehouse.get_location_type(p) is LocationType.AISLE
        for p in warehouse.grid.all_positions()
    )


def test_marking_a_location_changes_its_type(small_warehouse: Warehouse) -> None:
    small_warehouse.mark_location(Position(3, 3), LocationType.PICKUP)
    assert small_warehouse.get_location_type(Position(3, 3)) is LocationType.PICKUP


def test_marking_a_cell_as_aisle_clears_its_role(small_warehouse: Warehouse) -> None:
    small_warehouse.mark_location(Position(3, 3), LocationType.STORAGE)
    small_warehouse.mark_location(Position(3, 3), LocationType.AISLE)

    assert small_warehouse.get_location_type(Position(3, 3)) is LocationType.AISLE
    assert Position(3, 3) not in small_warehouse.storage_cells


def test_location_lists_are_sorted_and_correct(small_warehouse: Warehouse) -> None:
    small_warehouse.mark_location(Position(4, 0), LocationType.STORAGE)
    small_warehouse.mark_location(Position(2, 0), LocationType.STORAGE)

    assert small_warehouse.storage_cells == [Position(2, 0), Position(4, 0)]
    assert small_warehouse.charging_stations == [Position(0, 0)]
    assert small_warehouse.dropoff_locations == [Position(4, 4)]


def test_is_charging_station(small_warehouse: Warehouse) -> None:
    assert small_warehouse.is_charging_station(Position(0, 0)) is True
    assert small_warehouse.is_charging_station(Position(3, 3)) is False


def test_marking_a_blocked_cell_as_a_pickup_is_rejected(
    small_warehouse: Warehouse,
) -> None:
    # (1, 2) holds a rack. A pickup point there could never be reached.
    with pytest.raises(InvalidPositionError):
        small_warehouse.mark_location(Position(1, 2), LocationType.PICKUP)


def test_location_lookup_out_of_bounds_raises(small_warehouse: Warehouse) -> None:
    with pytest.raises(InvalidPositionError):
        small_warehouse.get_location_type(Position(9, 9))


# ----------------------------------------------------------------------
# Static obstacles
# ----------------------------------------------------------------------
def test_static_obstacle_blocks_the_cell(small_warehouse: Warehouse) -> None:
    assert small_warehouse.is_blocked_by_static(Position(1, 2)) is True
    assert small_warehouse.is_traversable(Position(1, 2)) is False
    # Static obstacles are pushed into the grid, so the grid alone knows too.
    assert small_warehouse.grid.is_traversable(Position(1, 2)) is False


def test_static_obstacle_records_its_type(small_warehouse: Warehouse) -> None:
    obstacle = small_warehouse.static_obstacles[0]
    assert obstacle.obstacle_type is ObstacleType.STATIC
    assert obstacle.is_static is True
    assert obstacle.is_dynamic is False


def test_removing_a_static_obstacle_frees_the_cell(small_warehouse: Warehouse) -> None:
    small_warehouse.remove_static_obstacle("rack-1")

    assert small_warehouse.is_blocked_by_static(Position(1, 2)) is False
    assert small_warehouse.is_traversable(Position(1, 2)) is True


def test_static_obstacle_cannot_sit_on_a_functional_location(
    small_warehouse: Warehouse,
) -> None:
    with pytest.raises(InvalidPositionError):
        small_warehouse.add_static_obstacle("wall-1", Position(0, 0))  # charger


def test_static_obstacle_out_of_bounds_is_rejected(small_warehouse: Warehouse) -> None:
    with pytest.raises(InvalidPositionError):
        small_warehouse.add_static_obstacle("wall-1", Position(50, 50))


def test_duplicate_obstacle_id_is_rejected(small_warehouse: Warehouse) -> None:
    with pytest.raises(SimulationError):
        small_warehouse.add_static_obstacle("rack-1", Position(3, 3))


def test_removing_an_unknown_static_obstacle_raises(
    small_warehouse: Warehouse,
) -> None:
    with pytest.raises(SimulationError):
        small_warehouse.remove_static_obstacle("does-not-exist")


# ----------------------------------------------------------------------
# Dynamic obstacles
# ----------------------------------------------------------------------
def test_dynamic_obstacle_blocks_the_cell_for_now(small_warehouse: Warehouse) -> None:
    small_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2), "spill")

    assert small_warehouse.is_blocked_by_dynamic(Position(2, 2)) is True
    assert small_warehouse.is_traversable(Position(2, 2)) is False


def test_dynamic_obstacle_does_not_touch_the_static_map(
    small_warehouse: Warehouse,
) -> None:
    # This is the whole point of keeping the layers apart: the permanent floor
    # plan is unchanged, so a planner can still ask what is *permanently*
    # blocked in order to decide whether a stored route is still valid.
    small_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2))

    assert small_warehouse.is_blocked_by_static(Position(2, 2)) is False
    assert small_warehouse.grid.is_traversable(Position(2, 2)) is True


def test_removing_a_dynamic_obstacle_frees_the_cell(small_warehouse: Warehouse) -> None:
    small_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2))
    removed = small_warehouse.remove_dynamic_obstacle("spill-1")

    assert removed.obstacle_id == "spill-1"
    assert small_warehouse.is_traversable(Position(2, 2)) is True
    assert small_warehouse.dynamic_obstacles == []


def test_dynamic_obstacle_may_sit_on_a_functional_location(
    small_warehouse: Warehouse,
) -> None:
    # A dropped box really can land in front of a charger; the model must be
    # able to represent that, unlike a permanent wall.
    small_warehouse.add_dynamic_obstacle("box-1", Position(0, 0))
    assert small_warehouse.is_traversable(Position(0, 0)) is False


def test_dynamic_obstacle_cannot_sit_on_a_static_obstacle(
    small_warehouse: Warehouse,
) -> None:
    with pytest.raises(InvalidPositionError):
        small_warehouse.add_dynamic_obstacle("spill-1", Position(1, 2))


def test_removing_an_unknown_dynamic_obstacle_raises(
    small_warehouse: Warehouse,
) -> None:
    with pytest.raises(SimulationError):
        small_warehouse.remove_dynamic_obstacle("does-not-exist")


def test_obstacle_at_reports_the_occupant(small_warehouse: Warehouse) -> None:
    small_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2))

    assert small_warehouse.obstacle_at(Position(1, 2)).obstacle_id == "rack-1"
    assert small_warehouse.obstacle_at(Position(2, 2)).obstacle_id == "spill-1"
    assert small_warehouse.obstacle_at(Position(3, 3)) is None


# ----------------------------------------------------------------------
# Traversability and neighbours
# ----------------------------------------------------------------------
def test_warehouse_neighbours_exclude_both_obstacle_kinds(
    small_warehouse: Warehouse,
) -> None:
    small_warehouse.add_dynamic_obstacle("spill-1", Position(2, 3))
    neighbours = small_warehouse.neighbors(Position(2, 2))

    # (1, 2) is a rack (static) and (2, 3) is the spill (dynamic); both go.
    assert set(neighbours) == {Position(3, 2), Position(2, 1)}


def test_blocked_cells_includes_both_kinds(small_warehouse: Warehouse) -> None:
    small_warehouse.add_dynamic_obstacle("spill-1", Position(2, 2))

    assert small_warehouse.blocked_cells() == {Position(1, 2), Position(2, 2)}


def test_out_of_bounds_is_never_traversable(small_warehouse: Warehouse) -> None:
    assert small_warehouse.is_traversable(Position(-1, 0)) is False
    assert small_warehouse.in_bounds(Position(-1, 0)) is False
