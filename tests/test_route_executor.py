"""Tests for RouteExecutor - driving a robot along a planned route.

The executor owns no movement rules of its own; every step goes through the
simulator. So these tests check two things: that a whole route is walked
correctly cell by cell, and that when a single step is refused the executor
stops safely and reports the truth about how far it got.
"""

from __future__ import annotations

import pytest

from robotics.exceptions import SimulationError
from robotics.execution import ExecutionResult, RouteExecutor
from robotics.planning.astar import plan as plan_astar
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.simulation.simulator import WarehouseSimulator
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def _sim(width: int = 6, height: int = 6) -> WarehouseSimulator:
    return WarehouseSimulator(Warehouse(width=width, height=height, name="exec-test"))


def _robot(sim: WarehouseSimulator, robot_id: str, row: int, col: int, **kw) -> Robot:
    robot = Robot(robot_id=robot_id, position=Position(row, col), **kw)
    sim.add_robot(robot)
    return robot


# ----------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------
def test_executes_a_simple_route_and_reaches_the_goal() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    result = plan_astar(Position(0, 0), Position(0, 3), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is True
    assert execution.final_position == Position(0, 3)
    assert robot.position == Position(0, 3)
    assert execution.reached_goal is True


def test_executed_steps_equals_planned_steps_on_success() -> None:
    sim = _sim()
    _robot(sim, "R1", 0, 0)
    result = plan_astar(Position(0, 0), Position(2, 3), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.planned_steps == len(result.path) - 1
    assert execution.executed_steps == execution.planned_steps
    assert execution.executed_steps == 5  # Manhattan distance (2,3) from (0,0)


def test_completed_path_matches_the_planned_path() -> None:
    sim = _sim()
    _robot(sim, "R1", 1, 1)
    result = plan_astar(Position(1, 1), Position(4, 2), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.completed_path == result.path
    assert execution.completed_path[0] == Position(1, 1)
    assert execution.completed_path[-1] == Position(4, 2)


def test_position_updates_after_every_step() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    path = [Position(0, 0), Position(0, 1), Position(0, 2), Position(1, 2)]

    seen = []
    executor = RouteExecutor(sim)
    # Drive one cell at a time by slicing the path, to observe intermediate state.
    for stop in range(1, len(path)):
        sub = path[: stop + 1]
        # reset a fresh sim each iteration so the sub-path always starts at (0,0)
        sim2 = _sim()
        _robot(sim2, "R1", 0, 0)
        ex = RouteExecutor(sim2).execute("R1", sub)
        seen.append(ex.final_position)

    assert seen == [Position(0, 1), Position(0, 2), Position(1, 2)]
    # sanity: the original single call also lands correctly
    assert executor.execute("R1", path).final_position == Position(1, 2)
    assert robot.position == Position(1, 2)


def test_odometry_increases_by_the_number_of_steps() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    before = robot.distance_travelled
    result = plan_astar(Position(0, 0), Position(3, 3), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.distance_travelled == execution.executed_steps
    assert robot.distance_travelled - before == execution.executed_steps
    assert robot.steps_taken == execution.executed_steps


def test_battery_drops_by_drain_per_move_times_steps() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0, battery_level=100.0)
    drain = robot.battery_drain_per_move
    result = plan_astar(Position(0, 0), Position(0, 4), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.battery_before == 100.0
    assert execution.battery_after == pytest.approx(100.0 - 4 * drain)
    assert execution.battery_consumed == pytest.approx(4 * drain)


def test_robot_state_is_valid_after_execution() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    result = plan_astar(Position(0, 0), Position(2, 2), sim.warehouse)

    RouteExecutor(sim).execute("R1", result)

    assert isinstance(robot.status, RobotStatus)
    assert robot.status is RobotStatus.MOVING
    assert 0.0 <= robot.battery_level <= 100.0


# ----------------------------------------------------------------------
# start == goal
# ----------------------------------------------------------------------
def test_start_equals_goal_performs_zero_moves() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 2, 2)
    result = plan_astar(Position(2, 2), Position(2, 2), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is True
    assert execution.executed_steps == 0
    assert execution.planned_steps == 0
    assert execution.distance_travelled == 0.0
    assert execution.completed_path == [Position(2, 2)]
    assert robot.steps_taken == 0


# ----------------------------------------------------------------------
# Bad input handled safely
# ----------------------------------------------------------------------
def test_unsuccessful_path_result_is_rejected_without_moving() -> None:
    sim = _sim(width=5, height=5)
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        sim.warehouse.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    robot = _robot(sim, "R1", 4, 0)
    result = plan_astar(Position(4, 0), Position(0, 4), sim.warehouse)
    assert result.success is False

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is False
    assert execution.executed_steps == 0
    assert robot.position == Position(4, 0)
    assert "did not find a route" in execution.failure_reason


def test_empty_path_is_rejected_safely() -> None:
    sim = _sim()
    _robot(sim, "R1", 0, 0)

    execution = RouteExecutor(sim).execute("R1", [])

    assert execution.success is False
    assert "empty" in execution.failure_reason


def test_path_not_starting_at_the_robot_is_rejected() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    path = [Position(1, 1), Position(1, 2), Position(1, 3)]

    execution = RouteExecutor(sim).execute("R1", path)

    assert execution.success is False
    assert execution.executed_steps == 0
    assert robot.position == Position(0, 0)
    assert "starts at" in execution.failure_reason


def test_non_adjacent_hop_in_the_path_is_rejected_before_moving() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    path = [Position(0, 0), Position(0, 1), Position(0, 3)]  # skips (0,2)

    execution = RouteExecutor(sim).execute("R1", path)

    assert execution.success is False
    assert execution.executed_steps == 0
    assert robot.position == Position(0, 0)
    assert "one orthogonal step" in execution.failure_reason


def test_path_revisiting_a_cell_is_rejected() -> None:
    sim = _sim()
    _robot(sim, "R1", 0, 0)
    path = [Position(0, 0), Position(0, 1), Position(0, 0)]

    execution = RouteExecutor(sim).execute("R1", path)

    assert execution.success is False
    assert "same cell" in execution.failure_reason


def test_offline_robot_cannot_execute() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    robot.set_status(RobotStatus.OFFLINE)
    result = plan_astar(Position(0, 0), Position(0, 3), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is False
    assert "offline" in execution.failure_reason
    assert robot.position == Position(0, 0)


def test_unknown_robot_raises() -> None:
    sim = _sim()
    with pytest.raises(SimulationError):
        RouteExecutor(sim).execute("ghost", [Position(0, 0)])


# ----------------------------------------------------------------------
# Stops safely mid-route
# ----------------------------------------------------------------------
def test_stops_safely_when_a_dynamic_obstacle_blocks_the_next_cell() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    # A straight route east. Plan it while the aisle is clear...
    result = plan_astar(Position(0, 0), Position(0, 5), sim.warehouse)
    assert result.success
    # ...then drop a spill on cell (0,3) AFTER planning. No replanning happens.
    sim.warehouse.add_dynamic_obstacle("spill-1", Position(0, 3), "spill")

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is False
    assert execution.executed_steps == 2          # reached (0,1) then (0,2)
    assert robot.position == Position(0, 2)
    assert execution.final_position == Position(0, 2)
    assert execution.completed_path == [Position(0, 0), Position(0, 1), Position(0, 2)]
    assert "dynamic obstacle" in execution.failure_reason
    assert 0.0 <= robot.battery_level <= 100.0     # invariant still holds


def test_stops_safely_when_the_battery_runs_out_mid_route() -> None:
    sim = _sim()
    # Enough charge for 3 moves, then it stalls.
    robot = _robot(sim, "R1", 0, 0, battery_level=3.0)
    result = plan_astar(Position(0, 0), Position(0, 5), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is False
    assert execution.executed_steps == 3
    assert robot.position == Position(0, 3)
    assert robot.battery_level == 0.0
    assert "battery" in execution.failure_reason


def test_stops_safely_when_another_robot_blocks_the_next_cell() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 0, 0)
    _robot(sim, "R2", 0, 2)  # parked in the way
    result = plan_astar(Position(0, 0), Position(0, 4), sim.warehouse)

    execution = RouteExecutor(sim).execute("R1", result)

    assert execution.success is False
    assert execution.executed_steps == 1
    assert robot.position == Position(0, 1)
    assert "occupied" in execution.failure_reason


def test_a_bare_list_of_cells_is_accepted_as_a_route() -> None:
    sim = _sim()
    robot = _robot(sim, "R1", 2, 0)
    path = [Position(2, 0), Position(2, 1), Position(2, 2)]

    execution = RouteExecutor(sim).execute("R1", path)

    assert execution.success is True
    assert robot.position == Position(2, 2)
    assert isinstance(execution, ExecutionResult)


def test_execution_is_deterministic() -> None:
    first_sim = _sim()
    _robot(first_sim, "R1", 0, 0)
    first = RouteExecutor(first_sim).execute(
        "R1", plan_astar(Position(0, 0), Position(4, 4), first_sim.warehouse)
    )

    second_sim = _sim()
    _robot(second_sim, "R1", 0, 0)
    second = RouteExecutor(second_sim).execute(
        "R1", plan_astar(Position(0, 0), Position(4, 4), second_sim.warehouse)
    )

    assert first.completed_path == second.completed_path
    assert first.executed_steps == second.executed_steps
    assert first.battery_consumed == second.battery_consumed
