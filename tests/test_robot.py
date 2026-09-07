"""Tests for the Robot: creation, status, battery, payload and bookkeeping."""

from __future__ import annotations

import pytest

from robotics.exceptions import InvalidMoveError, SimulationError
from robotics.robots.robot import (
    DEFAULT_BATTERY_DRAIN_PER_MOVE,
    MAX_BATTERY,
    Robot,
)
from robotics.robots.robot_state import RobotStatus
from robotics.warehouse.grid import Position


# ----------------------------------------------------------------------
# Creation
# ----------------------------------------------------------------------
def test_robot_is_created_with_sensible_defaults(robot: Robot) -> None:
    assert robot.robot_id == "R1"
    assert robot.position == Position(2, 2)
    assert robot.battery_level == MAX_BATTERY
    assert robot.status is RobotStatus.IDLE
    assert robot.assigned_task_id is None
    assert robot.steps_taken == 0
    assert robot.distance_travelled == 0.0
    assert robot.current_payload == 0.0


def test_robot_requires_an_id() -> None:
    with pytest.raises(ValueError):
        Robot(robot_id="", position=Position(0, 0))


@pytest.mark.parametrize("battery", [-1.0, 100.1, 250.0])
def test_robot_rejects_battery_outside_zero_to_hundred(battery: float) -> None:
    with pytest.raises(ValueError):
        Robot(robot_id="R1", position=Position(0, 0), battery_level=battery)


def test_robot_rejects_payload_greater_than_capacity() -> None:
    with pytest.raises(ValueError):
        Robot(
            robot_id="R1",
            position=Position(0, 0),
            payload_capacity=5.0,
            current_payload=6.0,
        )


def test_robot_rejects_negative_capacity() -> None:
    with pytest.raises(ValueError):
        Robot(robot_id="R1", position=Position(0, 0), payload_capacity=-1.0)


# ----------------------------------------------------------------------
# Status
# ----------------------------------------------------------------------
def test_all_documented_statuses_exist() -> None:
    expected = {
        "IDLE",
        "ASSIGNED",
        "MOVING",
        "PICKING",
        "DROPPING",
        "CHARGING",
        "BLOCKED",
        "OFFLINE",
    }
    assert {status.name for status in RobotStatus} == expected


def test_set_status_changes_status(robot: Robot) -> None:
    robot.set_status(RobotStatus.BLOCKED)
    assert robot.status is RobotStatus.BLOCKED


def test_set_status_rejects_non_enum_values(robot: Robot) -> None:
    # Catching this early is exactly why status is an enum and not a string.
    with pytest.raises(TypeError):
        robot.set_status("moving")  # type: ignore[arg-type]


def test_only_idle_unassigned_robots_are_available(robot: Robot) -> None:
    assert robot.is_available is True

    robot.set_status(RobotStatus.OFFLINE)
    assert robot.is_available is False


# ----------------------------------------------------------------------
# Task bookkeeping
# ----------------------------------------------------------------------
def test_assigning_a_task_marks_the_robot_assigned(robot: Robot) -> None:
    robot.assign_task("T1")

    assert robot.assigned_task_id == "T1"
    assert robot.status is RobotStatus.ASSIGNED
    assert robot.is_available is False


def test_assigning_to_a_busy_robot_is_rejected(robot: Robot) -> None:
    robot.assign_task("T1")
    with pytest.raises(SimulationError):
        robot.assign_task("T2")


def test_clearing_a_task_returns_the_robot_to_idle(robot: Robot) -> None:
    robot.assign_task("T1")
    robot.clear_task()

    assert robot.assigned_task_id is None
    assert robot.status is RobotStatus.IDLE
    assert robot.is_available is True


# ----------------------------------------------------------------------
# Movement bookkeeping
# ----------------------------------------------------------------------
def test_valid_move_updates_position_steps_and_distance(robot: Robot) -> None:
    robot.move_to(Position(2, 3))

    assert robot.position == Position(2, 3)
    assert robot.steps_taken == 1
    assert robot.distance_travelled == 1.0
    assert robot.status is RobotStatus.MOVING


def test_each_move_costs_battery(robot: Robot) -> None:
    robot.move_to(Position(2, 3))
    assert robot.battery_level == MAX_BATTERY - DEFAULT_BATTERY_DRAIN_PER_MOVE


def test_several_moves_accumulate(robot: Robot) -> None:
    robot.move_to(Position(2, 3))
    robot.move_to(Position(2, 4))
    robot.move_to(Position(3, 4))

    assert robot.position == Position(3, 4)
    assert robot.steps_taken == 3
    assert robot.distance_travelled == 3.0


@pytest.mark.parametrize(
    "destination",
    [
        Position(2, 2),  # same cell
        Position(3, 3),  # diagonal
        Position(2, 5),  # three cells away
        Position(0, 0),  # far away
    ],
)
def test_non_adjacent_move_is_rejected(robot: Robot, destination: Position) -> None:
    with pytest.raises(InvalidMoveError):
        robot.move_to(destination)


def test_rejected_move_leaves_the_robot_untouched(robot: Robot) -> None:
    with pytest.raises(InvalidMoveError):
        robot.move_to(Position(0, 0))

    assert robot.position == Position(2, 2)
    assert robot.steps_taken == 0
    assert robot.battery_level == MAX_BATTERY


def test_robot_without_enough_battery_cannot_move() -> None:
    flat = Robot(robot_id="R1", position=Position(2, 2), battery_level=0.0)

    assert flat.has_battery_for_move() is False
    with pytest.raises(InvalidMoveError):
        flat.move_to(Position(2, 3))


def test_robot_with_exactly_enough_battery_can_make_one_final_move() -> None:
    # Boundary case: battery == cost. It must be allowed, and must leave 0%.
    robot = Robot(robot_id="R1", position=Position(2, 2), battery_level=1.0)
    robot.move_to(Position(2, 3))

    assert robot.battery_level == 0.0
    assert robot.has_battery_for_move() is False


# ----------------------------------------------------------------------
# Battery
# ----------------------------------------------------------------------
def test_battery_never_exceeds_one_hundred(robot: Robot) -> None:
    added = robot.recharge(50.0)

    assert robot.battery_level == MAX_BATTERY
    assert added == 0.0  # it was already full


def test_recharge_reports_how_much_was_actually_added() -> None:
    robot = Robot(robot_id="R1", position=Position(0, 0), battery_level=90.0)
    added = robot.recharge(25.0)

    assert robot.battery_level == 100.0
    assert added == 10.0


def test_recharge_with_no_argument_fills_the_battery() -> None:
    robot = Robot(robot_id="R1", position=Position(0, 0), battery_level=12.0)
    robot.recharge()

    assert robot.battery_level == 100.0


def test_recharge_rejects_negative_amounts(robot: Robot) -> None:
    with pytest.raises(ValueError):
        robot.recharge(-5.0)


def test_battery_never_falls_below_zero() -> None:
    # A move that would overdraw the battery is refused outright, so the
    # battery cannot go negative and nothing is silently half-applied.
    robot = Robot(
        robot_id="R1",
        position=Position(2, 2),
        battery_level=1.0,
        battery_drain_per_move=10.0,
    )

    with pytest.raises(InvalidMoveError):
        robot.move_to(Position(2, 3))

    assert robot.battery_level == 1.0
    assert robot.position == Position(2, 2)


def test_battery_stays_within_bounds_after_many_moves() -> None:
    robot = Robot(robot_id="R1", position=Position(0, 0), battery_level=3.0)

    robot.move_to(Position(0, 1))
    robot.move_to(Position(0, 2))
    robot.move_to(Position(0, 3))

    assert robot.battery_level == 0.0
    with pytest.raises(InvalidMoveError):
        robot.move_to(Position(0, 4))


# ----------------------------------------------------------------------
# Payload
# ----------------------------------------------------------------------
def test_loading_within_capacity(robot: Robot) -> None:
    robot.load(4.0)

    assert robot.current_payload == 4.0
    assert robot.remaining_capacity == 6.0


def test_loading_beyond_capacity_is_rejected(robot: Robot) -> None:
    with pytest.raises(SimulationError):
        robot.load(11.0)
    assert robot.current_payload == 0.0


def test_unloading_returns_and_clears_the_payload(robot: Robot) -> None:
    robot.load(4.0)
    unloaded = robot.unload()

    assert unloaded == 4.0
    assert robot.current_payload == 0.0


def test_can_carry_respects_what_is_already_loaded(robot: Robot) -> None:
    robot.load(8.0)

    assert robot.can_carry(2.0) is True
    assert robot.can_carry(3.0) is False


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def test_to_dict_contains_the_full_state(robot: Robot) -> None:
    snapshot = robot.to_dict()

    assert snapshot["robot_id"] == "R1"
    assert snapshot["position"] == (2, 2)
    assert snapshot["status"] == "idle"
    assert snapshot["battery_level"] == 100.0
    assert snapshot["steps_taken"] == 0
