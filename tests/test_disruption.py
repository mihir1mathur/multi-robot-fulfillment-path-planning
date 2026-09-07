"""Tests for the disruption event model."""

from __future__ import annotations

from robotics.recovery.disruption import (
    DisruptionSchedule,
    DisruptionType,
    DynamicObstacleEvent,
    RecoveryTrigger,
    RobotFailureEvent,
)
from robotics.warehouse.grid import Position


def P(r, c):
    return Position(r, c)


def test_event_kinds() -> None:
    assert DynamicObstacleEvent(3, P(1, 1)).kind is DisruptionType.DYNAMIC_OBSTACLE
    assert RobotFailureEvent(3, "R1").kind is DisruptionType.ROBOT_OFFLINE


def test_dynamic_obstacle_event_resolves_an_id() -> None:
    named = DynamicObstacleEvent(3, P(1, 1), obstacle_id="spill-9")
    auto = DynamicObstacleEvent(3, P(1, 1))
    assert named.resolved_obstacle_id() == "spill-9"
    assert auto.resolved_obstacle_id() == "disruption-1-1-t3"


def test_schedule_is_sorted_and_queryable() -> None:
    schedule = DisruptionSchedule.of(
        DynamicObstacleEvent(5, P(2, 2)),
        RobotFailureEvent(2, "R3"),
        DynamicObstacleEvent(2, P(0, 0)),
    )
    # sorted by (timestep, robot-failure-before-obstacle, key)
    assert [e.timestep for e in schedule] == [2, 2, 5]
    assert isinstance(schedule.events[0], RobotFailureEvent)  # failures first at a tie

    at2 = schedule.events_at(2)
    assert len(at2) == 2
    assert schedule.events_at(9) == []
    assert schedule.last_timestep == 5


def test_schedule_counts() -> None:
    schedule = DisruptionSchedule.of(
        DynamicObstacleEvent(1, P(0, 0)),
        DynamicObstacleEvent(2, P(0, 1)),
        RobotFailureEvent(3, "R1"),
    )
    assert schedule.counts() == {
        "dynamic_obstacle": 2,
        "robot_offline": 1,
        "total": 3,
    }


def test_recovery_trigger_values() -> None:
    assert {t.value for t in RecoveryTrigger} == {
        "dynamic_obstacle",
        "robot_offline",
        "route_invalidated",
        "reservation_invalidated",
    }


def test_empty_schedule() -> None:
    schedule = DisruptionSchedule()
    assert len(schedule) == 0
    assert schedule.last_timestep == 0
    assert schedule.events_at(0) == []
