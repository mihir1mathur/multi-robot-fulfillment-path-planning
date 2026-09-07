"""State invariants that must hold no matter which requests arrive.

The domain models (`robotics.tasks.Task`, `robotics.robots.Robot`) enforce
their own state machines - those are unit-tested elsewhere. These tests check
that the invariants still hold when the change is driven through the SERVICE /
API layer and the persisted rows.
"""

from __future__ import annotations

import pytest

from robotics.services.errors import ResourceConflictError
from robotics.services.robot_service import RobotService
from robotics.services.task_service import TaskService


def _robot_and_task(session, task_id="T1"):
    RobotService(session).create_robot("R1", 0, 0)
    TaskService(session).create_task(task_id, 2, 4, 11, 1)


# --- a task is assigned to at most one robot -------------------
def test_reassigning_an_assigned_task_repoints_it_and_does_not_duplicate(api_database):
    with api_database.session_scope() as session:
        svc = TaskService(session)
        RobotService(session).create_robot("R1", 0, 0)
        RobotService(session).create_robot("R2", 0, 1)
        svc.create_task("T1", 2, 4, 11, 1)
        svc.update_task("T1", status="assigned", assigned_robot_id="R1")
        svc.update_task("T1", assigned_robot_id="R2")  # re-point

    with api_database.session() as session:
        task = TaskService(session).get_task("T1")
        assert task.assigned_robot_id == "R2"  # exactly one owner, the new one


# --- a completed task cannot go back to an active state --------
def test_completed_task_cannot_return_to_in_progress(api_database):
    with api_database.session_scope() as session:
        svc = TaskService(session)
        _robot_and_task(session)
        svc.update_task("T1", status="assigned", assigned_robot_id="R1")
        svc.update_task("T1", status="in_progress")
        svc.update_task("T1", status="completed")

    with api_database.session_scope() as session:
        with pytest.raises(ResourceConflictError):
            TaskService(session).update_task("T1", status="in_progress")

    with api_database.session() as session:
        assert TaskService(session).get_task("T1").status == "completed"


# --- an offline robot is not handed new work ------------------
def test_allocation_ignores_offline_robots(api_database):
    from robotics.services.allocation_service import AllocationService

    with api_database.session_scope() as session:
        RobotService(session).create_robot("BUSY", 0, 0, status="offline")
        TaskService(session).create_task("T1", 2, 4, 11, 1)

    with api_database.session_scope() as session:
        with pytest.raises(Exception) as info:
            AllocationService(session).run_allocation(
                algorithm="greedy", from_persisted=True, commit=True
            )
        # no idle robot -> "needs at least one robot and one task"
        assert "robot" in str(info.value).lower()


# --- a robot's two 'assigned' sides stay consistent -----------
def test_committed_allocation_keeps_robot_and_task_pointers_in_sync(api_database):
    from robotics.services.allocation_service import AllocationService

    with api_database.session_scope() as session:
        RobotService(session).create_robot("R1", 0, 0)
        TaskService(session).create_task("T1", 2, 4, 11, 1, payload_weight=2.0)

    with api_database.session_scope() as session:
        AllocationService(session).run_allocation(
            algorithm="greedy", from_persisted=True, commit=True
        )

    with api_database.session() as session:
        robot = RobotService(session).get_robot("R1")
        task = TaskService(session).get_task("T1")
        assert robot.assigned_task_id == "T1"
        assert task.assigned_robot_id == "R1"
        assert task.status == "assigned"
        assert robot.status != "idle"  # it took the job


# --- duplicate identifiers are refused, not silently merged ---
def test_creating_a_robot_that_exists_is_rejected(api_database):
    with api_database.session_scope() as session:
        RobotService(session).create_robot("R1", 0, 0)
    with api_database.session_scope() as session:
        with pytest.raises(ResourceConflictError):
            RobotService(session).create_robot("R1", 1, 1)
