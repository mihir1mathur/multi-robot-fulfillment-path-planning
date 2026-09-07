"""Persistence layer: sessions, transactions, rollback, constraints."""

from __future__ import annotations

import pytest
from sqlalchemy.exc import IntegrityError

from robotics.persistence.models import RobotRecord, TaskRecord
from robotics.persistence.repositories import RobotRepository, TaskRepository


def test_a_committed_row_survives_a_new_session(api_database):
    with api_database.session_scope() as session:
        RobotRepository(session).add(RobotRecord(robot_id="R1", row=0, col=1))

    with api_database.session() as session:
        assert RobotRepository(session).get("R1") is not None


def test_rollback_after_a_failure_leaves_nothing_persisted(api_database):
    class BoomError(RuntimeError):
        pass

    with pytest.raises(BoomError):
        with api_database.session_scope() as session:
            RobotRepository(session).add(RobotRecord(robot_id="R1", row=0, col=1))
            raise BoomError("deliberate failure mid-transaction")

    with api_database.session() as session:
        assert RobotRepository(session).get("R1") is None


def test_duplicate_primary_key_raises_integrity_error(api_database):
    with api_database.session_scope() as session:
        RobotRepository(session).add(RobotRecord(robot_id="R1", row=0, col=1))

    with pytest.raises(IntegrityError):
        with api_database.session_scope() as session:
            session.add(RobotRecord(robot_id="R1", row=1, col=1))
            session.flush()


def test_deleting_a_robot_nulls_its_task_assignment(api_database):
    """The tasks.assigned_robot_id FK is ON DELETE SET NULL."""
    with api_database.session_scope() as session:
        session.add(RobotRecord(robot_id="R1", row=0, col=1))
        session.flush()
        session.add(
            TaskRecord(
                task_id="T1",
                pickup_row=2,
                pickup_col=4,
                dropoff_row=11,
                dropoff_col=1,
                status="assigned",
                assigned_robot_id="R1",
            )
        )

    with api_database.session_scope() as session:
        robot = RobotRepository(session).get("R1")
        RobotRepository(session).delete(robot)

    with api_database.session() as session:
        task = TaskRepository(session).get("T1")
        assert task is not None
        assert task.assigned_robot_id is None


def test_partial_multi_write_operation_rolls_back_atomically(api_database):
    """Two inserts in one scope; the second fails -> neither persists."""
    with pytest.raises(IntegrityError):
        with api_database.session_scope() as session:
            session.add(RobotRecord(robot_id="R1", row=0, col=1))
            session.add(RobotRecord(robot_id="R1", row=2, col=2))  # dup PK
            session.flush()

    with api_database.session() as session:
        assert RobotRepository(session).list() == []


def test_timestamps_are_populated(api_database):
    with api_database.session_scope() as session:
        RobotRepository(session).add(RobotRecord(robot_id="R1", row=0, col=1))

    with api_database.session() as session:
        record = RobotRepository(session).get("R1")
        assert record.created_at is not None
        assert record.updated_at is not None


def test_status_index_query_returns_the_right_rows(api_database):
    with api_database.session_scope() as session:
        session.add(RobotRecord(robot_id="R1", row=0, col=1, status="idle"))
        session.add(RobotRecord(robot_id="R2", row=0, col=2, status="offline"))

    with api_database.session() as session:
        offline = RobotRepository(session).list(status="offline")
        assert [r.robot_id for r in offline] == ["R2"]
