"""Tests for the Task model and its lifecycle."""

from __future__ import annotations

import pytest

from robotics.exceptions import TaskError
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position


# ----------------------------------------------------------------------
# Creation
# ----------------------------------------------------------------------
def test_task_is_created_pending_and_unassigned(task: Task) -> None:
    assert task.task_id == "T1"
    assert task.pickup_location == Position(0, 1)
    assert task.dropoff_location == Position(4, 4)
    assert task.status is TaskStatus.PENDING
    assert task.assigned_robot_id is None
    assert task.is_open is True


def test_task_requires_an_id() -> None:
    with pytest.raises(ValueError):
        Task(
            task_id="",
            pickup_location=Position(0, 0),
            dropoff_location=Position(1, 1),
        )


@pytest.mark.parametrize("weight", [0.0, -3.0])
def test_task_rejects_non_positive_weight(weight: float) -> None:
    with pytest.raises(ValueError):
        Task(
            task_id="T1",
            pickup_location=Position(0, 0),
            dropoff_location=Position(1, 1),
            payload_weight=weight,
        )


def test_task_rejects_identical_pickup_and_dropoff() -> None:
    with pytest.raises(ValueError):
        Task(
            task_id="T1",
            pickup_location=Position(2, 2),
            dropoff_location=Position(2, 2),
        )


def test_manhattan_length_is_the_obstacle_free_lower_bound(task: Task) -> None:
    # (0, 1) -> (4, 4) is 4 rows + 3 cols = 7 steps if nothing is in the way.
    assert task.manhattan_length == 7


# ----------------------------------------------------------------------
# Priority
# ----------------------------------------------------------------------
def test_priority_orders_from_low_to_urgent() -> None:
    assert TaskPriority.LOW < TaskPriority.NORMAL
    assert TaskPriority.NORMAL < TaskPriority.HIGH
    assert TaskPriority.HIGH < TaskPriority.URGENT


def test_tasks_can_be_sorted_by_urgency() -> None:
    tasks = [
        Task("A", Position(0, 0), Position(1, 1), priority=TaskPriority.LOW),
        Task("B", Position(0, 0), Position(1, 1), priority=TaskPriority.URGENT),
        Task("C", Position(0, 0), Position(1, 1), priority=TaskPriority.NORMAL),
    ]
    most_urgent_first = sorted(tasks, key=lambda t: t.priority, reverse=True)

    assert [t.task_id for t in most_urgent_first] == ["B", "C", "A"]


# ----------------------------------------------------------------------
# Lifecycle: the happy path
# ----------------------------------------------------------------------
def test_full_lifecycle_pending_to_completed(task: Task) -> None:
    task.assign_to("R1")
    assert task.status is TaskStatus.ASSIGNED
    assert task.assigned_robot_id == "R1"

    task.start()
    assert task.status is TaskStatus.IN_PROGRESS

    task.complete()
    assert task.status is TaskStatus.COMPLETED
    assert task.is_open is False


def test_unassigning_returns_a_task_to_the_queue(task: Task) -> None:
    task.assign_to("R1")
    task.unassign()

    assert task.status is TaskStatus.PENDING
    assert task.assigned_robot_id is None


def test_failing_records_the_reason(task: Task) -> None:
    task.assign_to("R1")
    task.fail("robot went offline")

    assert task.status is TaskStatus.FAILED
    assert task.failure_reason == "robot went offline"
    assert task.is_open is False


# ----------------------------------------------------------------------
# Lifecycle: illegal transitions
# ----------------------------------------------------------------------
def test_a_task_cannot_start_before_it_is_assigned(task: Task) -> None:
    with pytest.raises(TaskError):
        task.start()


def test_a_task_cannot_be_completed_before_it_starts(task: Task) -> None:
    task.assign_to("R1")
    with pytest.raises(TaskError):
        task.complete()


def test_a_completed_task_is_terminal(task: Task) -> None:
    task.assign_to("R1")
    task.start()
    task.complete()

    with pytest.raises(TaskError):
        task.assign_to("R2")
    with pytest.raises(TaskError):
        task.fail("too late")


def test_a_failed_task_is_terminal(task: Task) -> None:
    task.fail("cancelled")

    with pytest.raises(TaskError):
        task.assign_to("R1")


def test_an_assigned_task_cannot_be_reassigned_directly(task: Task) -> None:
    task.assign_to("R1")

    with pytest.raises(TaskError):
        task.assign_to("R2")

    # The supported route is to release it first, then assign again.
    task.unassign()
    task.assign_to("R2")
    assert task.assigned_robot_id == "R2"


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def test_to_dict_contains_the_full_state(task: Task) -> None:
    snapshot = task.to_dict()

    assert snapshot["task_id"] == "T1"
    assert snapshot["pickup_location"] == (0, 1)
    assert snapshot["dropoff_location"] == (4, 4)
    assert snapshot["status"] == "pending"
    assert snapshot["priority"] == "normal"
    assert snapshot["assigned_robot_id"] is None
