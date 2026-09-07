"""Tests for committing an allocation plan into the simulator.

The allocator only proposes. `commit_allocation` applies the plan through the
normal task API, all-or-nothing. These tests check that a good plan updates
robot and task state correctly, and that a stale plan is refused without
leaving anything half-changed.
"""

from __future__ import annotations

from robotics.allocation import CostEstimator, CpSatAllocator, GreedyAllocator
from robotics.allocation.allocation_result import AllocationResult, Assignment
from robotics.allocation.commit import commit_allocation
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.simulation.simulator import WarehouseSimulator
from robotics.tasks.task import Task, TaskStatus
from robotics.warehouse.grid import Position
from robotics.warehouse.warehouse import Warehouse


def _sim() -> WarehouseSimulator:
    return WarehouseSimulator(Warehouse(width=12, height=12, name="commit-test"))


def _populate(sim: WarehouseSimulator):
    r1 = Robot("R1", Position(0, 0))
    r2 = Robot("R2", Position(11, 11))
    sim.add_robot(r1)
    sim.add_robot(r2)
    t1 = Task("T1", Position(1, 1), Position(2, 2))
    t2 = Task("T2", Position(10, 10), Position(9, 9))
    sim.add_task(t1)
    sim.add_task(t2)
    return r1, r2, t1, t2


def test_commit_assigns_tasks_and_updates_state() -> None:
    sim = _sim()
    r1, r2, t1, t2 = _populate(sim)

    estimator = CostEstimator(sim.warehouse)
    plan = CpSatAllocator(estimator).allocate(sim.robots, sim.tasks)
    report = commit_allocation(sim, plan)

    assert report.committed is True
    assert len(report.applied) == plan.assigned_task_count

    for assignment in plan.assignments:
        robot = sim.get_robot(assignment.robot_id)
        task = sim.get_task(assignment.task_id)
        assert robot.assigned_task_id == assignment.task_id
        assert robot.status is RobotStatus.ASSIGNED
        assert task.status is TaskStatus.ASSIGNED
        assert task.assigned_robot_id == assignment.robot_id


def test_after_commit_there_are_no_pending_tasks_or_free_robots() -> None:
    sim = _sim()
    _populate(sim)

    estimator = CostEstimator(sim.warehouse)
    plan = GreedyAllocator(estimator).allocate(sim.robots, sim.tasks)
    commit_allocation(sim, plan)

    assert sim.pending_tasks() == []
    assert all(not r.is_available for r in sim.robots)


def test_committing_an_empty_plan_changes_nothing() -> None:
    sim = _sim()
    _populate(sim)
    empty = AllocationResult(algorithm="greedy", success=True, solver_status="GREEDY")

    report = commit_allocation(sim, empty)

    assert report.committed is True
    assert report.applied == []
    assert all(t.status is TaskStatus.PENDING for t in sim.tasks)


def test_committing_an_unsuccessful_plan_is_refused() -> None:
    sim = _sim()
    _populate(sim)
    failed = AllocationResult(
        algorithm="cp_sat", success=False, solver_status="UNKNOWN"
    )

    report = commit_allocation(sim, failed)

    assert report.committed is False
    assert "not successful" in report.rejected_reason
    assert all(t.status is TaskStatus.PENDING for t in sim.tasks)


def test_stale_plan_is_refused_all_or_nothing() -> None:
    sim = _sim()
    r1, r2, t1, t2 = _populate(sim)

    # A hand-built plan for both tasks...
    plan = AllocationResult(
        algorithm="greedy",
        success=True,
        solver_status="GREEDY",
        assignments=[
            Assignment("R1", "T1", 2, 2),
            Assignment("R2", "T2", 2, 2),
        ],
        unassigned_task_ids=[],
        total_estimated_cost=8,
    )
    # ...but T1 gets assigned by someone else first, making the plan stale.
    sim.assign_task_manually("T1", "R1")

    report = commit_allocation(sim, plan)

    assert report.committed is False
    assert "T1" in report.rejected_reason
    # R2/T2 must NOT have been touched despite being valid on its own.
    assert sim.get_task("T2").status is TaskStatus.PENDING
    assert sim.get_robot("R2").is_available is True


def test_stale_plan_because_a_robot_went_offline_is_refused() -> None:
    sim = _sim()
    r1, r2, t1, t2 = _populate(sim)
    plan = AllocationResult(
        algorithm="cp_sat",
        success=True,
        solver_status="OPTIMAL",
        assignments=[Assignment("R2", "T2", 2, 2)],
        unassigned_task_ids=["T1"],
        total_estimated_cost=4,
    )
    r2.set_status(RobotStatus.OFFLINE)

    report = commit_allocation(sim, plan)

    assert report.committed is False
    assert sim.get_task("T2").status is TaskStatus.PENDING


def test_commit_then_reallocate_only_sees_whats_left() -> None:
    sim = _sim()
    r1, r2, t1, t2 = _populate(sim)
    # Add a third task and a third robot.
    r3 = Robot("R3", Position(5, 0))
    sim.add_robot(r3)
    t3 = Task("T3", Position(5, 5), Position(6, 6))
    sim.add_task(t3)

    estimator = CostEstimator(sim.warehouse)
    first = CpSatAllocator(estimator).allocate(sim.robots, sim.tasks)
    commit_allocation(sim, first)

    # Second cycle: whatever the first left behind.
    second = CpSatAllocator(estimator).allocate(sim.robots, sim.tasks)
    assert second.assigned_task_count + first.assigned_task_count <= 3
    for a in second.assignments:
        assert a.task_id not in first.assigned_task_ids
        assert a.robot_id not in first.assigned_robot_ids
