"""Task management: create / read / list / update persisted tasks.

The domain `Task` owns a state machine (PENDING -> ASSIGNED -> IN_PROGRESS ->
COMPLETED, plus FAILED, with a fixed table of allowed transitions). This
service does NOT reimplement it: a status change loads the row, rebuilds the
domain `Task` at its stored status, and calls the matching lifecycle method.
An illegal transition raises `TaskError`, which becomes a 409 - the endpoint
existing does not make every transition legal.
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from robotics.exceptions import TaskError
from robotics.persistence.models import TaskRecord
from robotics.persistence.repositories import RobotRepository, TaskRepository
from robotics.services.errors import (
    ResourceConflictError,
    ResourceNotFoundError,
    ValidationFailedError,
)
from robotics.services.world import build_warehouse, task_from_record
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position

logger = logging.getLogger("robotics.services.task")

_EDITABLE_ONLY_WHEN_PENDING = ("pickup", "dropoff", "priority", "payload_weight")


class TaskService:
    """Domain operations on the persisted task queue."""

    def __init__(self, session: Session) -> None:
        self.session = session
        self.tasks = TaskRepository(session)
        self.robots = RobotRepository(session)
        self._warehouse = build_warehouse()

    # ------------------------------------------------------------------
    def list_tasks(self, status: Optional[str] = None) -> Sequence[TaskRecord]:
        if status is not None and status not in {s.value for s in TaskStatus}:
            raise ValidationFailedError(f"unknown task status '{status}'")
        return self.tasks.list(status=status)

    def get_task(self, task_id: str) -> TaskRecord:
        record = self.tasks.get(task_id)
        if record is None:
            raise ResourceNotFoundError(f"no task with id '{task_id}'")
        return record

    def delete_task(self, task_id: str) -> None:
        """Remove a task row. 404 if it does not exist.

        Used to tear a scenario down (the demo dashboard owns its task queue);
        the row simply disappears - no lifecycle transition is involved.
        """
        record = self.get_task(task_id)
        self.tasks.delete(record)
        logger.info("task.deleted", extra={"task_id": task_id})

    # ------------------------------------------------------------------
    def create_task(
        self,
        task_id: str,
        pickup_row: int,
        pickup_col: int,
        dropoff_row: int,
        dropoff_col: int,
        priority: str = "normal",
        payload_weight: float = 1.0,
    ) -> TaskRecord:
        if self.tasks.exists(task_id):
            raise ResourceConflictError(f"task '{task_id}' already exists")

        pickup = Position(pickup_row, pickup_col)
        dropoff = Position(dropoff_row, dropoff_col)
        self._require_reachable_endpoint("pickup", pickup)
        self._require_reachable_endpoint("dropoff", dropoff)

        try:
            task_priority = TaskPriority[priority.upper()]
        except KeyError as error:
            raise ValidationFailedError(
                f"unknown priority '{priority}'"
            ) from error
        try:
            Task(
                task_id=task_id,
                pickup_location=pickup,
                dropoff_location=dropoff,
                priority=task_priority,
                payload_weight=payload_weight,
            )
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error

        record = TaskRecord(
            task_id=task_id,
            pickup_row=pickup_row,
            pickup_col=pickup_col,
            dropoff_row=dropoff_row,
            dropoff_col=dropoff_col,
            priority=task_priority.name.lower(),
            payload_weight=payload_weight,
            status="pending",
        )
        self.tasks.add(record)
        logger.info("task.created", extra={"task_id": task_id})
        return record

    # ------------------------------------------------------------------
    def update_task(
        self,
        task_id: str,
        *,
        status: Optional[str] = None,
        assigned_robot_id: Optional[str] = None,
        priority: Optional[str] = None,
        payload_weight: Optional[float] = None,
        failure_reason: Optional[str] = None,
    ) -> TaskRecord:
        record = self.get_task(task_id)

        if priority is not None or payload_weight is not None:
            if record.status != "pending":
                raise ResourceConflictError(
                    "task fields (priority, payload_weight) can only be edited "
                    f"while the task is pending, not '{record.status}'"
                )
            if priority is not None:
                try:
                    record.priority = TaskPriority[priority.upper()].name.lower()
                except KeyError as error:
                    raise ValidationFailedError(
                        f"unknown priority '{priority}'"
                    ) from error
            if payload_weight is not None:
                if payload_weight <= 0:
                    raise ValidationFailedError("payload_weight must be > 0")
                record.payload_weight = payload_weight

        if status is not None:
            self._apply_transition(record, status, assigned_robot_id, failure_reason)
        elif assigned_robot_id is not None and record.status == "assigned":
            # re-point an already-assigned task at a different robot
            self._require_robot_exists(assigned_robot_id)
            record.assigned_robot_id = assigned_robot_id

        self.session.flush()
        logger.info(
            "task.updated",
            extra={"task_id": task_id, "status": record.status},
        )
        return record

    # ------------------------------------------------------------------
    def _apply_transition(
        self,
        record: TaskRecord,
        target_status: str,
        assigned_robot_id: Optional[str],
        failure_reason: Optional[str],
    ) -> None:
        try:
            target = TaskStatus(target_status)
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error

        current = TaskStatus(record.status)
        if target is current:
            return

        task = task_from_record(record)
        try:
            if target is TaskStatus.ASSIGNED:
                if not assigned_robot_id:
                    raise ValidationFailedError(
                        "assigning a task requires 'assigned_robot_id'"
                    )
                self._require_robot_exists(assigned_robot_id)
                task.assign_to(assigned_robot_id)
            elif target is TaskStatus.PENDING:
                task.unassign()
            elif target is TaskStatus.IN_PROGRESS:
                task.start()
            elif target is TaskStatus.COMPLETED:
                task.complete()
            elif target is TaskStatus.FAILED:
                task.fail(failure_reason or "failed via API")
        except TaskError as error:
            raise ResourceConflictError(str(error)) from error

        record.status = task.status.value
        record.assigned_robot_id = task.assigned_robot_id
        record.failure_reason = task.failure_reason

    def _require_robot_exists(self, robot_id: str) -> None:
        if not self.robots.exists(robot_id):
            raise ValidationFailedError(f"no robot with id '{robot_id}' to assign to")

    def _require_reachable_endpoint(self, label: str, cell: Position) -> None:
        if not self._warehouse.in_bounds(cell):
            raise ValidationFailedError(
                f"{label} {(cell.row, cell.col)} is outside the warehouse"
            )
        if self._warehouse.is_blocked_by_static(cell):
            raise ValidationFailedError(
                f"{label} {(cell.row, cell.col)} is permanently blocked"
            )
