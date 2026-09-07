"""Tests for the CostEstimator - pricing robot-task pairings with the planner."""

from __future__ import annotations

from robotics.allocation.cost_estimator import AssignmentCost, CostEstimator
from robotics.robots.robot import Robot
from robotics.tasks.task import Task
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def _wh(width: int = 10, height: int = 10) -> Warehouse:
    return Warehouse(width=width, height=height, name="cost-test")


def _robot(row: int, col: int, **kw) -> Robot:
    return Robot(robot_id="R1", position=Position(row, col), **kw)


def _task(pickup, dropoff, **kw) -> Task:
    return Task(task_id="T1", pickup_location=pickup, dropoff_location=dropoff, **kw)


def test_assignment_cost_total_is_the_sum_of_legs() -> None:
    cost = AssignmentCost(robot_to_pickup=3, pickup_to_dropoff=5)
    assert cost.total == 8


def test_estimate_on_an_open_grid_is_manhattan_sum() -> None:
    wh = _wh()
    robot = _robot(0, 0)
    task = _task(Position(0, 4), Position(2, 4))  # R->P = 4, P->D = 2

    cost = CostEstimator(wh).estimate(robot, task)

    assert cost is not None
    assert cost.robot_to_pickup == 4
    assert cost.pickup_to_dropoff == 2
    assert cost.total == 6


def test_estimate_accounts_for_obstacles() -> None:
    wh = _wh(6, 6)
    wh.add_static_obstacle("rack", Position(0, 2))  # blocks the straight line
    robot = _robot(0, 0)
    task = _task(Position(0, 3), Position(0, 5))

    cost = CostEstimator(wh).estimate(robot, task)

    assert cost is not None
    # straight-line R->P would be 3; the detour around (0,2) costs 5
    assert cost.robot_to_pickup == 5


def test_estimate_is_none_when_pickup_is_unreachable() -> None:
    wh = _wh(5, 5)
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    robot = _robot(4, 0)
    task = _task(Position(0, 4), Position(4, 4))  # pickup walled off

    assert CostEstimator(wh).estimate(robot, task) is None


def test_estimate_is_none_when_dropoff_is_unreachable() -> None:
    wh = _wh(5, 5)
    for cell in (Position(0, 3), Position(1, 3), Position(1, 4)):
        wh.add_static_obstacle(f"seal-{cell.row}-{cell.col}", cell)
    robot = _robot(4, 0)
    task = _task(Position(4, 4), Position(0, 4))  # dropoff walled off

    assert CostEstimator(wh).estimate(robot, task) is None


def test_path_cost_caches_repeated_queries() -> None:
    wh = _wh()
    estimator = CostEstimator(wh)

    estimator.path_cost(Position(0, 0), Position(3, 3))
    estimator.path_cost(Position(0, 0), Position(3, 3))
    estimator.path_cost(Position(0, 0), Position(3, 3))

    assert estimator.planner_calls == 1
    assert estimator.cache_size == 1


def test_shared_pickup_to_dropoff_leg_is_only_planned_once() -> None:
    wh = _wh()
    estimator = CostEstimator(wh)
    task = _task(Position(1, 1), Position(4, 4))

    estimator.estimate(Robot("R1", Position(0, 0)), task)
    estimator.estimate(Robot("R2", Position(9, 9)), task)

    # 2 distinct robot->pickup legs + 1 shared pickup->dropoff leg = 3 calls
    assert estimator.planner_calls == 3


def test_estimate_is_deterministic() -> None:
    wh = _wh()
    robot = _robot(2, 1)
    task = _task(Position(5, 5), Position(1, 8))

    first = CostEstimator(wh).estimate(robot, task)
    second = CostEstimator(wh).estimate(robot, task)

    assert first == second
