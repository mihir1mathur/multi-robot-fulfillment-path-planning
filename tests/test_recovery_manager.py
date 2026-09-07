"""Tests for the RecoveryManager loop: safety, metrics, no-op runs, invariants."""

from __future__ import annotations

from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.recovery.disruption import (
    DisruptionSchedule,
    DynamicObstacleEvent,
    RobotFailureEvent,
)
from robotics.recovery.recovery_manager import RecoveryManager
from robotics.recovery.recovery_result import RecoveryEventResult, RobotReplan
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.simulation.simulator import WarehouseSimulator
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _sim(warehouse, starts):
    sim = WarehouseSimulator(warehouse)
    for rid, cell in sorted(starts.items()):
        sim.add_robot(Robot(rid, cell))
    return sim


def _run(warehouse, starts, goals, *events):
    coordination = MultiRobotCoordinator(warehouse).plan(starts, goals)
    sim = _sim(warehouse, starts)
    result = RecoveryManager(sim).run(
        coordination, DisruptionSchedule.of(*events)
    )
    return sim, result


# ----------------------------------------------------------------------
# No disruption -> behaves like a plain coordinated run
# ----------------------------------------------------------------------
def test_no_disruption_run_reaches_all_goals_conflict_free() -> None:
    wh = Warehouse(width=9, height=9, name="quiet")
    starts = {"R1": P(4, 0), "R2": P(0, 4), "R3": P(8, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4), "R3": P(0, 4)}
    sim, result = _run(wh, starts, goals)

    assert result.success
    assert result.recovery_count == 0
    assert result.robots_reached_goal == 3
    assert result.vertex_conflicts == 0 and result.edge_conflicts == 0
    assert result.validation_problems == []


def test_all_robots_start_at_goal_is_a_trivial_success() -> None:
    wh = Warehouse(width=6, height=6, name="trivial")
    starts = {"R1": P(1, 1), "R2": P(4, 4)}
    sim, result = _run(wh, starts, dict(starts))
    assert result.success
    assert result.total_move_steps == 0


# ----------------------------------------------------------------------
# Pre-flight
# ----------------------------------------------------------------------
def test_robot_not_at_its_planned_start_is_rejected() -> None:
    wh = Warehouse(width=7, height=7, name="pf")
    coordination = MultiRobotCoordinator(wh).plan({"R1": P(0, 0)}, {"R1": P(0, 4)})
    sim = _sim(wh, {"R1": P(3, 3)})                        # wrong place
    result = RecoveryManager(sim).run(coordination)
    assert result.success is False
    assert "starts at" in result.failure_reason


# ----------------------------------------------------------------------
# Coordination-safety invariants after recovery
# ----------------------------------------------------------------------
def test_no_conflicts_after_a_recovered_obstacle_in_a_busy_scene() -> None:
    wh = Warehouse(width=11, height=11, name="busy")
    starts = {
        "R1": P(5, 0), "R2": P(0, 5), "R3": P(10, 5), "R4": P(5, 10),
    }
    goals = {
        "R1": P(5, 10), "R2": P(10, 5), "R3": P(0, 5), "R4": P(5, 0),
    }
    sim, result = _run(wh, starts, goals, DynamicObstacleEvent(2, P(5, 5)))

    assert result.vertex_conflicts == 0
    assert result.edge_conflicts == 0
    assert result.validation_problems == []
    for event in result.recovery_events:
        assert event.unresolved_vertex_conflicts == 0
        assert event.unresolved_edge_conflicts == 0


def test_multiple_disruptions_in_one_run() -> None:
    wh = Warehouse(width=12, height=12, name="multi")
    starts = {"R1": P(6, 0), "R2": P(0, 6)}
    goals = {"R1": P(6, 11), "R2": P(11, 6)}
    sim, result = _run(
        wh, starts, goals,
        DynamicObstacleEvent(2, P(6, 4)),
        DynamicObstacleEvent(4, P(4, 6)),
    )
    assert result.recovery_count == 2
    assert result.vertex_conflicts == 0 and result.edge_conflicts == 0
    assert result.validation_problems == []


def test_wait_actions_are_counted_and_carry_no_distance() -> None:
    wh = Warehouse(width=9, height=9, name="waits")
    starts = {"R1": P(4, 0), "R2": P(0, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4)}
    sim, result = _run(wh, starts, goals, DynamicObstacleEvent(2, P(4, 4)))

    for rid, info in result.per_robot.items():
        # distance == move count (WAITs add nothing)
        assert info["distance_travelled"] == info["moves"]


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def test_recovery_event_metrics_are_non_negative_and_consistent() -> None:
    wh = Warehouse(width=9, height=9, name="metrics")
    sim, result = _run(
        wh, {"R1": P(4, 0), "R2": P(0, 4)}, {"R1": P(4, 8), "R2": P(8, 4)},
        DynamicObstacleEvent(2, P(4, 4)),
    )
    event = result.recovery_events[0]
    assert event.replanning_latency_ms >= 0
    assert event.total_recovery_latency_ms >= 0
    assert event.reservations_released >= 0
    assert event.reservations_created >= 0
    for replan in event.robot_replans:
        assert replan.replanning_time_ms >= 0
        assert replan.new_length >= 0
        assert replan.additional_distance == replan.new_length - replan.old_remaining_length


def test_result_to_dict_has_a_stable_schema() -> None:
    wh = Warehouse(width=7, height=7, name="schema")
    sim, result = _run(
        wh, {"R1": P(3, 0)}, {"R1": P(3, 6)}, DynamicObstacleEvent(3, P(3, 4))
    )
    d = result.to_dict()
    for key in (
        "success", "safe_stop", "robot_count", "robots_reached_goal",
        "total_move_steps", "total_wait_steps", "makespan_executed",
        "vertex_conflicts", "edge_conflicts", "validation_problems",
        "recovery_count", "successful_recoveries", "safe_stops",
        "recovery_events", "per_robot",
    ):
        assert key in d
    assert isinstance(d["recovery_events"], list)
    ev = d["recovery_events"][0]
    for key in (
        "trigger", "trigger_timestep", "affected_robot_count",
        "replanning_success", "recovery_success", "safe_stop",
        "reservations_released", "reservations_created",
        "unresolved_vertex_conflicts", "unresolved_edge_conflicts",
        "total_recovery_latency_ms",
    ):
        assert key in ev


def test_robot_replan_additional_distance_can_be_negative() -> None:
    # a replan that is SHORTER than the abandoned tail (the obstacle was on a
    # detour the original route took) -> additional_distance < 0 is valid
    replan = RobotReplan(
        robot_id="R1",
        original_goal=P(0, 5),
        current_position=P(0, 1),
        trigger_timestep=1,
        old_remaining_path=[P(0, 1), P(0, 2), P(0, 3), P(0, 4), P(0, 5)],
        new_path=[P(0, 1), P(0, 2), P(0, 5)],
        success=True,
    )
    assert replan.additional_distance == -2


# ----------------------------------------------------------------------
# Existing coordination behaviour is not weakened
# ----------------------------------------------------------------------
def test_recovery_manager_matches_coordinated_executor_when_nothing_goes_wrong() -> None:
    from robotics.coordination.coordinated_executor import CoordinatedExecutor

    wh = Warehouse(width=9, height=9, name="parity")
    starts = {"R1": P(4, 0), "R2": P(0, 4), "R3": P(8, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4), "R3": P(0, 4)}
    coordination = MultiRobotCoordinator(wh).plan(starts, goals)

    sim_a = _sim(wh, starts)
    exec_result = CoordinatedExecutor(sim_a).execute(coordination)

    sim_b = _sim(wh, starts)
    recov_result = RecoveryManager(sim_b).run(coordination, DisruptionSchedule())

    assert exec_result.robots_reached_goal == recov_result.robots_reached_goal
    assert exec_result.total_move_steps == recov_result.total_move_steps
    for rid in starts:
        assert sim_a.get_robot(rid).position == sim_b.get_robot(rid).position
