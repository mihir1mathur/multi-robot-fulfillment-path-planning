"""SQLAlchemy ORM models: the database schema.

DESIGN NOTES
------------
Two KINDS of table:

  * STATE tables - ``robots`` and ``tasks`` - the current fleet. One row per
    robot / task, updated in place. These mirror the domain objects
    (`robotics.robots.Robot`, `robotics.tasks.Task`) closely enough that
    converting between them is mechanical, but they are deliberately a
    SEPARATE type: the ORM row is about storage, the domain object is about
    behaviour and validation.

  * RUN tables - ``planning_runs``, ``allocation_runs``, ``coordination_runs``,
    ``recovery_events`` - an append-only history of operations. Each row keeps
    the scalar numbers worth querying (success, latency, cost, counts) as real
    columns, and the full request/result as a JSON blob. Fully normalising a
    coordinated plan (a row per timed step per robot per run) would be a lot of
    tables for data that is only ever read back as a whole, so the structured
    part stays JSON - a pragmatic, documented choice.

RELATIONSHIPS
-------------
``tasks.assigned_robot_id`` -> ``robots.robot_id`` is a real foreign key
(``ON DELETE SET NULL``): deleting a robot releases its task rather than
orphaning a dangling ID. ``robots.assigned_task_id`` is a plain nullable column
(no FK) to avoid a circular foreign-key dependency that SQLite cannot defer;
the service layer keeps the two sides consistent.

TIMESTAMPS
----------
``created_at`` / ``updated_at`` are DB-server defaults (`func.now()`), so they
are correct regardless of which process writes the row.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from robotics.persistence.database import Base


class UserRecord(Base):
    """An API user: a username, a bcrypt password hash, and one role.

    Roles (see ``robotics.api.security.Role``):
        viewer    - read robots / tasks
        operator  - viewer + create/update robots & tasks + run planning /
                    allocation / coordination / recovery
        admin     - operator + register new users
    """

    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("username", name="uq_users_username"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="viewer")
    disabled: Mapped[bool] = mapped_column(nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "disabled": self.disabled,
        }


class RobotRecord(Base):
    """One robot's persisted state. Primary key is the human robot ID (``R1``)."""

    __tablename__ = "robots"

    robot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    row: Mapped[int] = mapped_column(Integer, nullable=False)
    col: Mapped[int] = mapped_column(Integer, nullable=False)
    battery_level: Mapped[float] = mapped_column(Float, nullable=False, default=100.0)
    payload_capacity: Mapped[float] = mapped_column(Float, nullable=False, default=10.0)
    current_payload: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="idle", index=True)
    assigned_task_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "position": {"row": self.row, "col": self.col},
            "battery_level": self.battery_level,
            "payload_capacity": self.payload_capacity,
            "current_payload": self.current_payload,
            "status": self.status,
            "assigned_task_id": self.assigned_task_id,
        }


class TaskRecord(Base):
    """One fulfillment task's persisted state. Primary key is the task ID."""

    __tablename__ = "tasks"

    task_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    pickup_row: Mapped[int] = mapped_column(Integer, nullable=False)
    pickup_col: Mapped[int] = mapped_column(Integer, nullable=False)
    dropoff_row: Mapped[int] = mapped_column(Integer, nullable=False)
    dropoff_col: Mapped[int] = mapped_column(Integer, nullable=False)
    priority: Mapped[str] = mapped_column(String(16), nullable=False, default="normal")
    payload_weight: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    assigned_robot_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        ForeignKey("robots.robot_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    failure_reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "pickup": {"row": self.pickup_row, "col": self.pickup_col},
            "dropoff": {"row": self.dropoff_row, "col": self.dropoff_col},
            "priority": self.priority,
            "payload_weight": self.payload_weight,
            "status": self.status,
            "assigned_robot_id": self.assigned_robot_id,
            "failure_reason": self.failure_reason,
        }


class PlanningRun(Base):
    """History of one ``POST /planning/path`` call."""

    __tablename__ = "planning_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(16), nullable=False)
    start_row: Mapped[int] = mapped_column(Integer, nullable=False)
    start_col: Mapped[int] = mapped_column(Integer, nullable=False)
    goal_row: Mapped[int] = mapped_column(Integer, nullable=False)
    goal_col: Mapped[int] = mapped_column(Integer, nullable=False)
    success: Mapped[bool] = mapped_column(nullable=False)
    total_cost: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nodes_expanded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    planning_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    failure_reason: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class AllocationRun(Base):
    """History of one ``POST /allocation/run`` call."""

    __tablename__ = "allocation_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    algorithm: Mapped[str] = mapped_column(String(16), nullable=False)
    solver_status: Mapped[str] = mapped_column(String(32), nullable=False)
    requested_robot_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    requested_task_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assigned_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unassigned_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_estimated_cost: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    solve_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    committed: Mapped[bool] = mapped_column(nullable=False, default=False)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class CoordinationRun(Base):
    """History of one ``POST /coordination/plan`` call."""

    __tablename__ = "coordination_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    robot_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    success: Mapped[bool] = mapped_column(nullable=False)
    planned_robot_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_robot_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    makespan: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_move_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_wait_steps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    conflict_checks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    planning_time_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    failure_reason: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class RecoveryEvent(Base):
    """History of one ``POST /recovery/*`` call."""

    __tablename__ = "recovery_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )
    disruption_type: Mapped[str] = mapped_column(String(32), nullable=False)
    trigger_timestep: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recovery_enabled: Mapped[bool] = mapped_column(nullable=False, default=True)
    recovery_success: Mapped[bool] = mapped_column(nullable=False, default=False)
    safe_stop: Mapped[bool] = mapped_column(nullable=False, default=False)
    affected_robot_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    affected_task_ids: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    replanning_attempted: Mapped[bool] = mapped_column(nullable=False, default=False)
    replanning_success: Mapped[bool] = mapped_column(nullable=False, default=False)
    task_reassignment_attempted: Mapped[bool] = mapped_column(nullable=False, default=False)
    task_reassignment_success: Mapped[bool] = mapped_column(nullable=False, default=False)
    reassigned_from: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    reassigned_to: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    total_recovery_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
