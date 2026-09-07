"""Tests for the WarehouseSimulator - the rules that span several objects.

These are the most valuable tests in the suite. The grid, robot and task classes
each protect themselves, but only the simulator can answer questions that need
the whole world at once: is this cell free, is that robot able to carry this
task, would this move put two robots in the same place?
"""

from __future__ import annotations

import pytest

from robotics.exceptions import (
    InvalidMoveError,
    InvalidPositionError,
    SimulationError,
)
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.simulation.renderer import render_full_state, render_map
from robotics.simulation.scenario import SEED, build_sample_simulator
from robotics.simulation.simulator import WarehouseSimulator
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position


def make_robot(robot_id: str, row: int, col: int, **kwargs) -> Robot:
    """Small helper so the tests read as 'a robot here' rather than boilerplate."""
    return Robot(robot_id=robot_id, position=Position(row, col), **kwargs)


def make_task(task_id: str, pickup: Position, dropoff: Position, **kwargs) -> Task:
    return Task(
        task_id=task_id, pickup_location=pickup, dropoff_location=dropoff, **kwargs
    )


# ----------------------------------------------------------------------
# Robot registration
# ----------------------------------------------------------------------
def test_add_robot_registers_it(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    assert len(simulator.robots) == 1
    assert simulator.get_robot("R1").position == Position(2, 2)


def test_duplicate_robot_id_is_rejected(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    with pytest.raises(SimulationError):
        simulator.add_robot(make_robot("R1", 3, 3))

    assert len(simulator.robots) == 1


def test_robot_cannot_start_outside_the_warehouse(simulator: WarehouseSimulator) -> None:
    with pytest.raises(InvalidPositionError):
        simulator.add_robot(make_robot("R1", 9, 9))


def test_robot_cannot_start_on_an_obstacle(simulator: WarehouseSimulator) -> None:
    with pytest.raises(InvalidPositionError):
        simulator.add_robot(make_robot("R1", 1, 2))  # the rack


def test_two_robots_cannot_start_on_the_same_cell(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    with pytest.raises(SimulationError):
        simulator.add_robot(make_robot("R2", 2, 2))


def test_get_unknown_robot_raises(simulator: WarehouseSimulator) -> None:
    with pytest.raises(SimulationError):
        simulator.get_robot("nope")


def test_remove_robot_takes_it_out_of_the_fleet(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    removed = simulator.remove_robot("R1")

    assert removed.robot_id == "R1"
    assert simulator.robots == []
    assert simulator.robot_at(Position(2, 2)) is None


def test_removing_a_robot_releases_its_assigned_task(
    simulator: WarehouseSimulator,
) -> None:
    # Work must not be silently lost when a robot leaves the fleet.
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))
    simulator.assign_task_manually("T1", "R1")

    simulator.remove_robot("R1")

    assert simulator.get_task("T1").status is TaskStatus.PENDING
    assert simulator.get_task("T1").assigned_robot_id is None


def test_removing_a_robot_mid_task_fails_that_task(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))
    simulator.assign_task_manually("T1", "R1")
    simulator.get_task("T1").start()

    simulator.remove_robot("R1")

    assert simulator.get_task("T1").status is TaskStatus.FAILED


# ----------------------------------------------------------------------
# Occupancy
# ----------------------------------------------------------------------
def test_robot_at_finds_the_occupant(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    assert simulator.robot_at(Position(2, 2)).robot_id == "R1"
    assert simulator.robot_at(Position(3, 3)) is None


def test_occupied_cells_tracks_movement(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.move_robot("R1", Position(2, 3))

    assert simulator.occupied_cells() == {Position(2, 3)}


# ----------------------------------------------------------------------
# Task registration
# ----------------------------------------------------------------------
def test_add_task_registers_it(simulator: WarehouseSimulator) -> None:
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))

    assert len(simulator.tasks) == 1
    assert simulator.get_task("T1").status is TaskStatus.PENDING


def test_duplicate_task_id_is_rejected(simulator: WarehouseSimulator) -> None:
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))

    with pytest.raises(SimulationError):
        simulator.add_task(make_task("T1", Position(0, 2), Position(4, 3)))


def test_task_endpoint_outside_the_warehouse_is_rejected(
    simulator: WarehouseSimulator,
) -> None:
    with pytest.raises(InvalidPositionError):
        simulator.add_task(make_task("T1", Position(0, 1), Position(9, 9)))


def test_task_endpoint_on_an_obstacle_is_rejected(
    simulator: WarehouseSimulator,
) -> None:
    # A pickup point inside a rack could never be reached, so reject it at
    # creation rather than discovering it halfway through execution.
    with pytest.raises(InvalidPositionError):
        simulator.add_task(make_task("T1", Position(1, 2), Position(4, 4)))


def test_get_unknown_task_raises(simulator: WarehouseSimulator) -> None:
    with pytest.raises(SimulationError):
        simulator.get_task("nope")


# ----------------------------------------------------------------------
# Manual task assignment
# ----------------------------------------------------------------------
def test_assignment_links_robot_and_task(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))

    simulator.assign_task_manually("T1", "R1")

    assert simulator.get_task("T1").status is TaskStatus.ASSIGNED
    assert simulator.get_task("T1").assigned_robot_id == "R1"
    assert simulator.get_robot("R1").assigned_task_id == "T1"
    assert simulator.get_robot("R1").status is RobotStatus.ASSIGNED


def test_a_busy_robot_cannot_take_a_second_task(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))
    simulator.add_task(make_task("T2", Position(0, 2), Position(4, 3)))
    simulator.assign_task_manually("T1", "R1")

    with pytest.raises(SimulationError):
        simulator.assign_task_manually("T2", "R1")


def test_an_already_assigned_task_cannot_be_reassigned(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_robot(make_robot("R2", 3, 3))
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))
    simulator.assign_task_manually("T1", "R1")

    with pytest.raises(SimulationError):
        simulator.assign_task_manually("T1", "R2")


def test_assignment_respects_payload_capacity(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2, payload_capacity=5.0))
    simulator.add_task(
        make_task("T1", Position(0, 1), Position(4, 4), payload_weight=9.0)
    )

    with pytest.raises(SimulationError):
        simulator.assign_task_manually("T1", "R1")

    # Nothing may have changed on either side after a refused assignment.
    assert simulator.get_task("T1").status is TaskStatus.PENDING
    assert simulator.get_robot("R1").assigned_task_id is None


def test_pending_tasks_are_ordered_most_urgent_first(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_task(
        make_task("T1", Position(0, 1), Position(4, 4), priority=TaskPriority.LOW)
    )
    simulator.add_task(
        make_task("T2", Position(0, 2), Position(4, 3), priority=TaskPriority.URGENT)
    )
    simulator.add_task(
        make_task("T3", Position(0, 3), Position(4, 2), priority=TaskPriority.NORMAL)
    )

    assert [t.task_id for t in simulator.pending_tasks()] == ["T2", "T3", "T1"]


# ----------------------------------------------------------------------
# Movement: the happy path
# ----------------------------------------------------------------------
def test_valid_move_updates_everything(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    robot = simulator.move_robot("R1", Position(2, 3))

    assert robot.position == Position(2, 3)
    assert robot.steps_taken == 1
    assert robot.distance_travelled == 1.0
    assert robot.battery_level == 99.0
    assert robot.status is RobotStatus.MOVING


def test_validate_move_returns_none_for_a_legal_move(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    assert simulator.validate_move("R1", Position(2, 3)) is None


@pytest.mark.parametrize(
    "destination",
    [Position(1, 2), Position(3, 2), Position(2, 1), Position(2, 3)],
)
def test_all_four_directions_are_allowed(
    simulator: WarehouseSimulator, destination: Position
) -> None:
    # (1, 2) holds the rack in this fixture, so unblock it for this test only.
    simulator.warehouse.remove_static_obstacle("rack-1")
    simulator.add_robot(make_robot("R1", 2, 2))

    simulator.move_robot("R1", destination)
    assert simulator.get_robot("R1").position == destination


# ----------------------------------------------------------------------
# Movement: every rejection reason
# ----------------------------------------------------------------------
def test_move_outside_the_warehouse_is_rejected(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 0, 0))

    with pytest.raises(InvalidMoveError, match="outside the warehouse"):
        simulator.move_robot("R1", Position(-1, 0))

    assert simulator.get_robot("R1").position == Position(0, 0)
    assert simulator.get_robot("R1").steps_taken == 0


@pytest.mark.parametrize(
    "destination",
    [Position(2, 2), Position(3, 3), Position(0, 0), Position(4, 4)],
)
def test_non_adjacent_move_is_rejected(
    simulator: WarehouseSimulator, destination: Position
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    with pytest.raises(InvalidMoveError, match="not adjacent"):
        simulator.move_robot("R1", destination)


def test_move_into_a_static_obstacle_is_rejected(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    with pytest.raises(InvalidMoveError, match="static obstacle"):
        simulator.move_robot("R1", Position(1, 2))


def test_move_into_a_dynamic_obstacle_is_rejected(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_dynamic_obstacle("spill-1", Position(2, 3))

    with pytest.raises(InvalidMoveError, match="dynamic obstacle"):
        simulator.move_robot("R1", Position(2, 3))


def test_move_onto_another_robot_is_rejected(simulator: WarehouseSimulator) -> None:
    # This is the current collision rule: after any move, no two robots share
    # a cell. Time-dependent conflicts are a separate, later problem.
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_robot(make_robot("R2", 2, 3))

    with pytest.raises(InvalidMoveError, match="occupied"):
        simulator.move_robot("R1", Position(2, 3))

    assert simulator.get_robot("R1").position == Position(2, 2)
    assert simulator.get_robot("R2").position == Position(2, 3)


def test_move_without_enough_battery_is_rejected(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2, battery_level=0.0))

    with pytest.raises(InvalidMoveError, match="battery"):
        simulator.move_robot("R1", Position(2, 3))


def test_offline_robot_cannot_move(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.get_robot("R1").set_status(RobotStatus.OFFLINE)

    with pytest.raises(InvalidMoveError, match="offline"):
        simulator.move_robot("R1", Position(2, 3))


def test_rejected_move_records_the_reason_on_the_robot(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    with pytest.raises(InvalidMoveError):
        simulator.move_robot("R1", Position(1, 2))

    assert "static obstacle" in simulator.get_robot("R1").last_error


def test_moving_an_unknown_robot_raises(simulator: WarehouseSimulator) -> None:
    with pytest.raises(SimulationError):
        simulator.move_robot("nope", Position(0, 0))


# ----------------------------------------------------------------------
# Dynamic obstacles through the simulator
# ----------------------------------------------------------------------
def test_dynamic_obstacle_can_be_added_and_removed(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_dynamic_obstacle("spill-1", Position(2, 3), "spill")

    with pytest.raises(InvalidMoveError):
        simulator.move_robot("R1", Position(2, 3))

    simulator.remove_dynamic_obstacle("spill-1")
    simulator.move_robot("R1", Position(2, 3))

    assert simulator.get_robot("R1").position == Position(2, 3)


def test_dynamic_obstacle_cannot_be_dropped_on_a_robot(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))

    with pytest.raises(SimulationError):
        simulator.add_dynamic_obstacle("spill-1", Position(2, 2))


def test_removing_an_unknown_dynamic_obstacle_raises(
    simulator: WarehouseSimulator,
) -> None:
    with pytest.raises(SimulationError):
        simulator.remove_dynamic_obstacle("nope")


# ----------------------------------------------------------------------
# Charging
# ----------------------------------------------------------------------
def test_charging_works_on_a_charging_station(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 0, 0, battery_level=40.0))  # (0,0) is a charger
    added = simulator.charge_robot("R1", 25.0)

    assert added == 25.0
    assert simulator.get_robot("R1").battery_level == 65.0
    assert simulator.get_robot("R1").status is RobotStatus.CHARGING


def test_charging_anywhere_else_is_rejected(simulator: WarehouseSimulator) -> None:
    simulator.add_robot(make_robot("R1", 2, 2, battery_level=40.0))

    with pytest.raises(SimulationError, match="not a charging station"):
        simulator.charge_robot("R1")

    assert simulator.get_robot("R1").battery_level == 40.0


# ----------------------------------------------------------------------
# State snapshot
# ----------------------------------------------------------------------
def test_get_current_state_reports_the_whole_world(
    simulator: WarehouseSimulator,
) -> None:
    simulator.add_robot(make_robot("R1", 2, 2))
    simulator.add_task(make_task("T1", Position(0, 1), Position(4, 4)))
    simulator.add_dynamic_obstacle("spill-1", Position(3, 3), "spill")

    state = simulator.get_current_state()

    assert state["warehouse"]["rows"] == 5
    assert state["warehouse"]["cols"] == 5
    assert state["warehouse"]["charging_stations"] == [(0, 0)]
    assert state["warehouse"]["static_obstacles"] == [(1, 2)]
    assert state["warehouse"]["dynamic_obstacles"][0]["position"] == (3, 3)
    assert len(state["robots"]) == 1
    assert len(state["tasks"]) == 1


# ----------------------------------------------------------------------
# The sample scenario
# ----------------------------------------------------------------------
def test_sample_scenario_builds_the_expected_world() -> None:
    simulator = build_sample_simulator()

    assert simulator.warehouse.height == 12
    assert simulator.warehouse.width == 12
    assert len(simulator.robots) == 4
    assert len(simulator.tasks) == 4
    assert len(simulator.warehouse.charging_stations) == 2
    assert len(simulator.warehouse.pickup_locations) == 2
    assert len(simulator.warehouse.dropoff_locations) == 2
    assert len(simulator.warehouse.dynamic_obstacles) == 1


def test_sample_scenario_is_reproducible() -> None:
    # Same seed -> identical starting state, which is what makes any number
    # produced by the demo meaningful.
    first = build_sample_simulator(seed=SEED).get_current_state()
    second = build_sample_simulator(seed=SEED).get_current_state()

    assert first == second


def test_sample_scenario_robots_do_not_all_start_full() -> None:
    # If every robot started at 100% the fleet would be indistinguishable and
    # later battery-aware allocation would have nothing to work with.
    batteries = {robot.battery_level for robot in build_sample_simulator().robots}
    assert len(batteries) > 1


def test_sample_scenario_starts_with_no_robot_on_an_obstacle() -> None:
    simulator = build_sample_simulator()
    assert all(
        simulator.warehouse.is_traversable(robot.position)
        for robot in simulator.robots
    )


def test_sample_scenario_tasks_all_start_pending() -> None:
    simulator = build_sample_simulator()
    assert all(task.status is TaskStatus.PENDING for task in simulator.tasks)


# ----------------------------------------------------------------------
# Rendering (smoke tests: the map must stay readable and correctly sized)
# ----------------------------------------------------------------------
def test_map_has_one_line_per_row_plus_a_header() -> None:
    simulator = build_sample_simulator()
    lines = render_map(simulator).splitlines()

    assert len(lines) == simulator.warehouse.height + 1


def test_map_draws_robots_obstacles_and_locations() -> None:
    simulator = build_sample_simulator()
    text = render_map(simulator)

    for symbol in ("#", "x", "C", "P", "D", "S", "0"):
        assert symbol in text


def test_full_state_render_mentions_every_robot_and_task() -> None:
    simulator = build_sample_simulator()
    text = render_full_state(simulator)

    for robot in simulator.robots:
        assert robot.robot_id in text
    for task in simulator.tasks:
        assert task.task_id in text
