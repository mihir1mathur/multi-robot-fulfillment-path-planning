"""Transaction boundaries at the SERVICE and API level.

The low-level session/rollback behaviour is covered in test_persistence.py.
These tests check the *composed* operations - an allocation commit that touches
several rows, a request that fails partway - leave the database all-or-nothing.
"""

from __future__ import annotations

import pytest

from robotics.persistence.models import AllocationRun, RobotRecord, TaskRecord
from robotics.services.allocation_service import AllocationService
from robotics.services.robot_service import RobotService
from robotics.services.task_service import TaskService


def _seed_fleet(session):
    RobotService(session).create_robot("R1", 0, 0)
    TaskService(session).create_task("T1", 2, 4, 11, 1, payload_weight=3.0)


# --- one request = one transaction ------------------------------
def test_allocation_commit_persists_run_robot_and_task_together(api_database):
    with api_database.session_scope() as session:
        _seed_fleet(session)

    with api_database.session_scope() as session:
        outcome = AllocationService(session).run_allocation(
            algorithm="greedy", from_persisted=True, commit=True
        )
        assert outcome.committed is True

    # every side effect landed
    with api_database.session() as session:
        assert session.get(TaskRecord, "T1").status == "assigned"
        assert session.get(RobotRecord, "R1").assigned_task_id == "T1"
        assert session.query(AllocationRun).count() == 1


def test_a_failure_after_the_writes_rolls_the_whole_request_back(api_database):
    """If anything raises before the scope commits, NONE of the allocation's
    robot/task/run writes persist."""
    with api_database.session_scope() as session:
        _seed_fleet(session)

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with api_database.session_scope() as session:
            AllocationService(session).run_allocation(
                algorithm="greedy", from_persisted=True, commit=True
            )
            raise Boom("something downstream failed before commit")

    with api_database.session() as session:
        assert session.get(TaskRecord, "T1").status == "pending"      # unchanged
        assert session.get(RobotRecord, "R1").assigned_task_id is None
        assert session.query(AllocationRun).count() == 0              # no run row


def test_api_patch_that_raises_midway_does_not_partially_persist(api_client):
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}})
    api_client.post(
        "/tasks",
        json={"task_id": "T1", "pickup": {"row": 2, "col": 4}, "dropoff": {"row": 11, "col": 1}},
    )

    # illegal transition pending -> completed: the request is rejected (409) and
    # nothing about the task changes.
    r = api_client.patch("/tasks/T1", json={"status": "completed"})
    assert r.status_code == 409
    assert api_client.get("/tasks/T1").json()["status"] == "pending"


def test_dry_run_allocation_still_records_a_run_but_changes_no_state(api_database):
    with api_database.session_scope() as session:
        _seed_fleet(session)

    with api_database.session_scope() as session:
        outcome = AllocationService(session).run_allocation(
            algorithm="greedy", from_persisted=True, commit=False
        )
        assert outcome.committed is False

    with api_database.session() as session:
        assert session.get(TaskRecord, "T1").status == "pending"
        assert session.query(AllocationRun).count() == 1
        assert session.query(AllocationRun).one().committed is False
