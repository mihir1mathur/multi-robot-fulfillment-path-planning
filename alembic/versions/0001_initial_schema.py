"""initial schema

Creates every table the application needs:

    users             API accounts (username, bcrypt hash, role)
    robots            fleet state
    tasks             task queue; assigned_robot_id -> robots.robot_id
                      ON DELETE SET NULL
    planning_runs     history of POST /planning/path
    allocation_runs   history of POST /allocation/run
    coordination_runs history of POST /coordination/plan
    recovery_events   history of POST /recovery/*

This migration is the baseline: it matches ``Base.metadata`` at the point
Alembic was introduced. It was generated from the ORM models by
``alembic revision --autogenerate`` and then hand-tuned to be dialect-neutral
(server defaults use the SQL-standard ``CURRENT_TIMESTAMP`` so the same script
runs on PostgreSQL and on the SQLite the fast tests use).

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_NOW = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("username", sa.String(length=64), nullable=False),
        sa.Column("password_hash", sa.String(length=255), nullable=False),
        sa.Column("role", sa.String(length=16), nullable=False),
        sa.Column("disabled", sa.Boolean(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("username", name="uq_users_username"),
    )
    op.create_index("ix_users_username", "users", ["username"])

    op.create_table(
        "robots",
        sa.Column("robot_id", sa.String(length=64), nullable=False),
        sa.Column("row", sa.Integer(), nullable=False),
        sa.Column("col", sa.Integer(), nullable=False),
        sa.Column("battery_level", sa.Float(), nullable=False),
        sa.Column("payload_capacity", sa.Float(), nullable=False),
        sa.Column("current_payload", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("assigned_task_id", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("robot_id"),
    )
    op.create_index("ix_robots_status", "robots", ["status"])

    op.create_table(
        "tasks",
        sa.Column("task_id", sa.String(length=64), nullable=False),
        sa.Column("pickup_row", sa.Integer(), nullable=False),
        sa.Column("pickup_col", sa.Integer(), nullable=False),
        sa.Column("dropoff_row", sa.Integer(), nullable=False),
        sa.Column("dropoff_col", sa.Integer(), nullable=False),
        sa.Column("priority", sa.String(length=16), nullable=False),
        sa.Column("payload_weight", sa.Float(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("assigned_robot_id", sa.String(length=64), nullable=True),
        sa.Column("failure_reason", sa.String(length=500), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        sa.ForeignKeyConstraint(
            ["assigned_robot_id"], ["robots.robot_id"],
            name="fk_tasks_assigned_robot_id_robots", ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("task_id"),
    )
    op.create_index("ix_tasks_status", "tasks", ["status"])
    op.create_index("ix_tasks_assigned_robot_id", "tasks", ["assigned_robot_id"])

    _run_history_table("planning_runs", [
        sa.Column("algorithm", sa.String(length=16), nullable=False),
        sa.Column("start_row", sa.Integer(), nullable=False),
        sa.Column("start_col", sa.Integer(), nullable=False),
        sa.Column("goal_row", sa.Integer(), nullable=False),
        sa.Column("goal_col", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("total_cost", sa.Float(), nullable=True),
        sa.Column("nodes_expanded", sa.Integer(), nullable=False),
        sa.Column("planning_time_ms", sa.Float(), nullable=False),
        sa.Column("failure_reason", sa.String(length=500), nullable=True),
    ])
    _run_history_table("allocation_runs", [
        sa.Column("algorithm", sa.String(length=16), nullable=False),
        sa.Column("solver_status", sa.String(length=32), nullable=False),
        sa.Column("requested_robot_count", sa.Integer(), nullable=False),
        sa.Column("requested_task_count", sa.Integer(), nullable=False),
        sa.Column("assigned_count", sa.Integer(), nullable=False),
        sa.Column("unassigned_count", sa.Integer(), nullable=False),
        sa.Column("total_estimated_cost", sa.Integer(), nullable=False),
        sa.Column("solve_time_ms", sa.Float(), nullable=False),
        sa.Column("committed", sa.Boolean(), nullable=False),
    ])
    _run_history_table("coordination_runs", [
        sa.Column("robot_count", sa.Integer(), nullable=False),
        sa.Column("success", sa.Boolean(), nullable=False),
        sa.Column("planned_robot_count", sa.Integer(), nullable=False),
        sa.Column("failed_robot_ids", sa.JSON(), nullable=False),
        sa.Column("makespan", sa.Integer(), nullable=False),
        sa.Column("total_move_steps", sa.Integer(), nullable=False),
        sa.Column("total_wait_steps", sa.Integer(), nullable=False),
        sa.Column("conflict_checks", sa.Integer(), nullable=False),
        sa.Column("planning_time_ms", sa.Float(), nullable=False),
        sa.Column("failure_reason", sa.String(length=1000), nullable=True),
    ])
    _run_history_table("recovery_events", [
        sa.Column("disruption_type", sa.String(length=32), nullable=False),
        sa.Column("trigger_timestep", sa.Integer(), nullable=False),
        sa.Column("recovery_enabled", sa.Boolean(), nullable=False),
        sa.Column("recovery_success", sa.Boolean(), nullable=False),
        sa.Column("safe_stop", sa.Boolean(), nullable=False),
        sa.Column("affected_robot_ids", sa.JSON(), nullable=False),
        sa.Column("affected_task_ids", sa.JSON(), nullable=False),
        sa.Column("replanning_attempted", sa.Boolean(), nullable=False),
        sa.Column("replanning_success", sa.Boolean(), nullable=False),
        sa.Column("task_reassignment_attempted", sa.Boolean(), nullable=False),
        sa.Column("task_reassignment_success", sa.Boolean(), nullable=False),
        sa.Column("reassigned_from", sa.String(length=64), nullable=True),
        sa.Column("reassigned_to", sa.String(length=64), nullable=True),
        sa.Column("total_recovery_latency_ms", sa.Float(), nullable=False),
    ])


def _run_history_table(name: str, extra_columns: list) -> None:
    op.create_table(
        name,
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=_NOW, nullable=False),
        *extra_columns,
        sa.Column("result_json", sa.JSON(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(f"ix_{name}_created_at", name, ["created_at"])


def downgrade() -> None:
    for name in (
        "recovery_events",
        "coordination_runs",
        "allocation_runs",
        "planning_runs",
    ):
        op.drop_index(f"ix_{name}_created_at", table_name=name)
        op.drop_table(name)

    op.drop_index("ix_tasks_assigned_robot_id", table_name="tasks")
    op.drop_index("ix_tasks_status", table_name="tasks")
    op.drop_table("tasks")

    op.drop_index("ix_robots_status", table_name="robots")
    op.drop_table("robots")

    op.drop_index("ix_users_username", table_name="users")
    op.drop_table("users")
