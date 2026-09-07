"""Tests for the independent allocation validator.

The validator must ACCEPT correct allocator output and REJECT hand-built
results that violate a constraint - a duplicate assignment, a wrong cost, a
bad unassigned list.
"""

from __future__ import annotations

from robotics.allocation.allocation_result import AllocationResult, Assignment
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.cp_sat_allocator import CpSatAllocator
from robotics.allocation.greedy_allocator import GreedyAllocator
from robotics.allocation.validation import find_allocation_problems, is_valid_allocation
from robotics.robots.robot import Robot
from robotics.tasks.task import Task
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def _scenario():
    wh = Warehouse(width=12, height=12, name="val-test")
    robots = [Robot("R1", Position(0, 0)), Robot("R2", Position(11, 11))]
    tasks = [
        Task("T1", Position(1, 1), Position(2, 2)),
        Task("T2", Position(10, 10), Position(9, 9)),
    ]
    return wh, robots, tasks


def test_valid_allocator_output_passes() -> None:
    wh, robots, tasks = _scenario()
    estimator = CostEstimator(wh)
    for allocator in (GreedyAllocator(estimator), CpSatAllocator(estimator)):
        result = allocator.allocate(robots, tasks)
        assert is_valid_allocation(result, robots, tasks, CostEstimator(wh))


def test_duplicate_task_assignment_is_caught() -> None:
    wh, robots, tasks = _scenario()
    bad = AllocationResult(
        algorithm="x", success=True, solver_status="X",
        assignments=[Assignment("R1", "T1", 2, 2), Assignment("R2", "T1", 2, 2)],
        unassigned_task_ids=["T2"], total_estimated_cost=8,
    )
    problems = find_allocation_problems(bad, robots, tasks, CostEstimator(wh))
    assert any("assigned more than once" in p for p in problems)


def test_duplicate_robot_assignment_is_caught() -> None:
    wh, robots, tasks = _scenario()
    bad = AllocationResult(
        algorithm="x", success=True, solver_status="X",
        assignments=[Assignment("R1", "T1", 2, 2), Assignment("R1", "T2", 2, 2)],
        unassigned_task_ids=[], total_estimated_cost=8,
    )
    problems = find_allocation_problems(bad, robots, tasks, CostEstimator(wh))
    assert any("robot R1 is assigned more than once" in p for p in problems)


def test_wrong_reported_cost_is_caught() -> None:
    wh, robots, tasks = _scenario()
    bad = AllocationResult(
        algorithm="x", success=True, solver_status="X",
        assignments=[Assignment("R1", "T1", 999, 999)],
        unassigned_task_ids=["T2"], total_estimated_cost=1998,
    )
    problems = find_allocation_problems(bad, robots, tasks, CostEstimator(wh))
    assert any("robot_to_pickup_cost" in p for p in problems)


def test_total_cost_mismatch_is_caught() -> None:
    wh, robots, tasks = _scenario()
    est = CostEstimator(wh)
    good = GreedyAllocator(est).allocate(robots, tasks)
    tampered = AllocationResult(
        algorithm=good.algorithm, success=True, solver_status=good.solver_status,
        assignments=good.assignments,
        unassigned_task_ids=good.unassigned_task_ids,
        total_estimated_cost=good.total_estimated_cost + 100,
    )
    problems = find_allocation_problems(tampered, robots, tasks, CostEstimator(wh))
    assert any("total_estimated_cost" in p for p in problems)


def test_wrong_unassigned_list_is_caught() -> None:
    wh, robots, tasks = _scenario()
    bad = AllocationResult(
        algorithm="x", success=True, solver_status="X",
        assignments=[Assignment("R1", "T1", 2, 2)],
        unassigned_task_ids=[],  # should list T2
        total_estimated_cost=4,
    )
    problems = find_allocation_problems(bad, robots, tasks, CostEstimator(wh))
    assert any("unassigned_task_ids" in p for p in problems)


def test_assignment_to_unknown_robot_is_caught() -> None:
    wh, robots, tasks = _scenario()
    bad = AllocationResult(
        algorithm="x", success=True, solver_status="X",
        assignments=[Assignment("GHOST", "T1", 2, 2)],
        unassigned_task_ids=["T2"], total_estimated_cost=4,
    )
    problems = find_allocation_problems(bad, robots, tasks, CostEstimator(wh))
    assert any("unknown robot" in p for p in problems)
