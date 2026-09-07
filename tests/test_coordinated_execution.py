"""Tests for synchronised multi-robot coordinated execution."""

from __future__ import annotations

from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.coordinated_executor import CoordinatedExecutor
from robotics.coordination.coordination_result import CoordinationResult
from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.simulation.simulator import WarehouseSimulator
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def P(r, c):
    return Position(r, c)


def _sim(wh, starts):
    sim = WarehouseSimulator(wh)
    for robot_id, cell in sorted(starts.items()):
        sim.add_robot(Robot(robot_id, cell))
    return sim


def _coordinate_and_execute(wh, starts, goals, **kw):
    result = MultiRobotCoordinator(wh).plan(starts, goals, **kw)
    sim = _sim(wh, starts)
    execution = CoordinatedExecutor(sim).execute(result)
    return result, sim, execution


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------
def test_two_robots_execute_simultaneously_and_reach_their_goals() -> None:
    wh = Warehouse(width=7, height=7, name="exec")
    starts = {"R1": P(3, 0), "R2": P(0, 3)}
    goals = {"R1": P(3, 6), "R2": P(6, 3)}

    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)

    assert execution.success
    assert execution.robots_reached_goal == 2
    assert sim.get_robot("R1").position == P(3, 6)
    assert sim.get_robot("R2").position == P(6, 3)
    assert execution.vertex_conflicts == 0
    assert execution.edge_conflicts == 0
    assert execution.validation_problems == []


def test_execution_matches_the_planned_move_and_wait_totals() -> None:
    wh = Warehouse(width=7, height=7, name="exec")
    starts = {"R1": P(3, 0), "R2": P(0, 3)}
    goals = {"R1": P(3, 6), "R2": P(6, 3)}

    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)

    assert execution.total_move_steps == plan.total_move_steps
    assert execution.total_wait_steps == plan.total_wait_steps
    assert execution.makespan_executed == plan.makespan


def test_wait_consumes_a_timestep_but_no_distance_or_battery() -> None:
    wh = Warehouse(width=7, height=7, name="exec")
    starts = {"R1": P(3, 0), "R2": P(0, 3)}
    goals = {"R1": P(3, 6), "R2": P(6, 3)}

    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)

    # R2 is the one that waits at the intersection
    r2 = execution.per_robot["R2"]
    assert r2.waits >= 1
    # its distance equals its move count (WAITs added nothing)
    assert r2.distance_travelled == r2.moves
    # battery drained only for moves
    assert r2.battery_consumed == r2.moves * sim.get_robot("R2").battery_drain_per_move


def test_movement_updates_odometry_and_battery_per_existing_semantics() -> None:
    wh = Warehouse(width=6, height=6, name="exec")
    starts = {"R1": P(0, 0)}
    goals = {"R1": P(0, 4)}

    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)
    robot = sim.get_robot("R1")

    assert execution.per_robot["R1"].moves == 4
    assert robot.distance_travelled == 4.0
    assert robot.steps_taken == 4
    assert robot.battery_level == 100.0 - 4 * robot.battery_drain_per_move


def test_robot_states_remain_valid_after_execution() -> None:
    wh = Warehouse(width=7, height=7, name="exec")
    starts = {"R1": P(3, 0), "R2": P(0, 3)}
    goals = {"R1": P(3, 6), "R2": P(6, 3)}
    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)

    for robot in sim.robots:
        assert isinstance(robot.status, RobotStatus)
        assert 0.0 <= robot.battery_level <= 100.0


def test_no_vertex_or_edge_collisions_across_a_congested_run() -> None:
    wh = Warehouse(width=9, height=9, name="exec")
    starts = {"R1": P(4, 0), "R2": P(0, 4), "R3": P(8, 4), "R4": P(4, 8)}
    goals = {"R1": P(4, 8), "R2": P(8, 4), "R3": P(0, 4), "R4": P(4, 0)}

    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)

    assert plan.success
    assert execution.success
    assert execution.vertex_conflicts == 0
    assert execution.edge_conflicts == 0
    # the executed trace itself passes the independent validator
    assert execution.validation_problems == []


# ----------------------------------------------------------------------
# Safe failure
# ----------------------------------------------------------------------
def test_malformed_timed_path_is_rejected_before_moving() -> None:
    wh = Warehouse(width=6, height=6, name="exec")
    sim = _sim(wh, {"R1": P(0, 0), "R2": P(0, 1)})
    # a hand-built plan where R1 and R2 swap - a conflict the validator catches
    bad = CoordinationResult(
        success=True,
        timed_paths={
            "R1": TimedPath.found("R1", [TimedStep(P(0, 0), 0), TimedStep(P(0, 1), 1)]),
            "R2": TimedPath.found("R2", [TimedStep(P(0, 1), 0), TimedStep(P(0, 0), 1)]),
        },
        robot_order=["R1", "R2"],
        planned_robot_count=2,
    )
    execution = CoordinatedExecutor(sim).execute(bad)

    assert execution.success is False
    assert "conflict validation" in execution.failure_reason
    assert sim.get_robot("R1").position == P(0, 0)   # nothing moved
    assert sim.get_robot("R2").position == P(0, 1)


def test_robot_start_mismatch_is_rejected() -> None:
    wh = Warehouse(width=6, height=6, name="exec")
    starts = {"R1": P(0, 0)}
    goals = {"R1": P(0, 3)}
    plan = MultiRobotCoordinator(wh).plan(starts, goals)

    sim = _sim(wh, {"R1": P(2, 2)})                   # robot placed somewhere else
    execution = CoordinatedExecutor(sim).execute(plan)

    assert execution.success is False
    assert "pre-flight" in execution.failure_reason
    assert sim.get_robot("R1").position == P(2, 2)


def test_a_dynamic_obstacle_appearing_during_execution_stops_it_safely() -> None:
    wh = Warehouse(width=8, height=1, name="corridor")
    starts = {"R1": P(0, 0)}
    goals = {"R1": P(0, 6)}
    plan = MultiRobotCoordinator(wh).plan(starts, goals)
    sim = _sim(wh, {"R1": P(0, 0)})

    # obstacle lands on (0,3) AFTER coordination
    sim.warehouse.add_dynamic_obstacle("spill", P(0, 3), "spill")
    execution = CoordinatedExecutor(sim).execute(plan)

    assert execution.success is False
    assert "obstacle that appeared after planning" in execution.failure_reason
    assert sim.get_robot("R1").position == P(0, 2)    # stopped just before it
    assert 0.0 <= sim.get_robot("R1").battery_level <= 100.0


def test_offline_robot_is_rejected_at_preflight() -> None:
    wh = Warehouse(width=6, height=6, name="exec")
    plan = MultiRobotCoordinator(wh).plan({"R1": P(0, 0)}, {"R1": P(0, 3)})
    sim = _sim(wh, {"R1": P(0, 0)})
    sim.get_robot("R1").set_status(RobotStatus.OFFLINE)

    execution = CoordinatedExecutor(sim).execute(plan)
    assert execution.success is False
    assert "offline" in execution.failure_reason


def test_validate_first_commit_second_prevents_partial_mutation() -> None:
    """If a joint step is invalid, NO robot moves that timestep."""
    wh = Warehouse(width=6, height=6, name="exec")
    sim = _sim(wh, {"R1": P(0, 0), "R2": P(2, 0)})
    # R1 wants to march east; R2's hand-built path teleports (illegal jump) at t=2.
    plan = CoordinationResult(
        success=True,
        timed_paths={
            "R1": TimedPath.found(
                "R1",
                [TimedStep(P(0, 0), 0), TimedStep(P(0, 1), 1), TimedStep(P(0, 2), 2)],
            ),
            "R2": TimedPath.found(
                "R2",
                [TimedStep(P(2, 0), 0), TimedStep(P(2, 1), 1), TimedStep(P(2, 1), 2)],
            ),
        },
        robot_order=["R1", "R2"],
        planned_robot_count=2,
    )
    # make it fail at execution by dropping an obstacle on R1's t=2 cell
    sim.warehouse.add_dynamic_obstacle("spill", P(0, 2), "spill")
    execution = CoordinatedExecutor(sim).execute(plan)

    assert execution.success is False
    # R1 advanced to (0,1) at t=1 (valid), then t=2 failed -> R1 at (0,1), not (0,2)
    assert sim.get_robot("R1").position == P(0, 1)
    # R2 also only advanced through the valid t=1 step
    assert sim.get_robot("R2").position == P(2, 1)


# ----------------------------------------------------------------------
# Determinism
# ----------------------------------------------------------------------
def test_deterministic_replay() -> None:
    wh = Warehouse(width=9, height=9, name="exec")
    starts = {"R1": P(4, 0), "R2": P(0, 4), "R3": P(8, 4)}
    goals = {"R1": P(4, 8), "R2": P(8, 4), "R3": P(0, 4)}

    _, _, first = _coordinate_and_execute(wh, starts, goals)
    _, _, second = _coordinate_and_execute(wh, starts, goals)

    assert first.to_dict()["per_robot"] == second.to_dict()["per_robot"]
    assert first.total_move_steps == second.total_move_steps
    assert first.total_wait_steps == second.total_wait_steps
    assert first.makespan_executed == second.makespan_executed


def test_start_equals_goal_for_all_robots_executes_with_no_moves() -> None:
    wh = Warehouse(width=6, height=6, name="exec")
    starts = {"R1": P(1, 1), "R2": P(4, 4)}
    goals = dict(starts)
    plan, sim, execution = _coordinate_and_execute(wh, starts, goals)

    assert execution.success
    assert execution.total_move_steps == 0
    assert execution.robots_reached_goal == 2
