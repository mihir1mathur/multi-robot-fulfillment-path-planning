"""Repositories: the only place that builds queries against the ORM models.

WHY A REPOSITORY LAYER
----------------------
Without it, ``session.query(RobotRecord).filter(...)`` ends up copied across
every service and every test. A repository gathers those queries behind named
methods (``get``, ``list``, ``add``, ``delete``), so:

  * the query logic lives in one place and is unit-tested once,
  * the service layer reads like business steps, not SQL,
  * swapping how something is stored touches one file.

TRANSACTIONS
------------
Repositories never call ``commit``. They ``add`` / ``delete`` / ``flush`` and
let the caller (a service method, or the request-scoped session) own the
transaction boundary. ``flush`` sends the INSERT so a generated primary key is
available, without ending the transaction.
"""

from __future__ import annotations

from typing import Optional, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from robotics.persistence.models import (
    AllocationRun,
    CoordinationRun,
    PlanningRun,
    RecoveryEvent,
    RobotRecord,
    TaskRecord,
)


class RobotRepository:
    """Reads and writes ``robots`` rows."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, robot_id: str) -> Optional[RobotRecord]:
        return self.session.get(RobotRecord, robot_id)

    def list(self, status: Optional[str] = None) -> Sequence[RobotRecord]:
        stmt = select(RobotRecord).order_by(RobotRecord.robot_id)
        if status is not None:
            stmt = stmt.where(RobotRecord.status == status)
        return list(self.session.scalars(stmt))

    def exists(self, robot_id: str) -> bool:
        return self.session.get(RobotRecord, robot_id) is not None

    def add(self, record: RobotRecord) -> RobotRecord:
        self.session.add(record)
        self.session.flush()
        return record

    def delete(self, record: RobotRecord) -> None:
        self.session.delete(record)
        self.session.flush()


class TaskRepository:
    """Reads and writes ``tasks`` rows."""

    def __init__(self, session: Session) -> None:
        self.session = session

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self.session.get(TaskRecord, task_id)

    def list(self, status: Optional[str] = None) -> Sequence[TaskRecord]:
        stmt = select(TaskRecord).order_by(TaskRecord.task_id)
        if status is not None:
            stmt = stmt.where(TaskRecord.status == status)
        return list(self.session.scalars(stmt))

    def pending(self) -> Sequence[TaskRecord]:
        return self.list(status="pending")

    def exists(self, task_id: str) -> bool:
        return self.session.get(TaskRecord, task_id) is not None

    def add(self, record: TaskRecord) -> TaskRecord:
        self.session.add(record)
        self.session.flush()
        return record

    def delete(self, record: TaskRecord) -> None:
        self.session.delete(record)
        self.session.flush()


class RunRepository:
    """Appends and reads the operation-history tables.

    One repository for all four run tables: they are append-mostly and share
    the same shape of access (insert one, read recent, read by id).
    """

    def __init__(self, session: Session) -> None:
        self.session = session

    # -- writes --------------------------------------------------------
    def add_planning_run(self, run: PlanningRun) -> PlanningRun:
        self.session.add(run)
        self.session.flush()
        return run

    def add_allocation_run(self, run: AllocationRun) -> AllocationRun:
        self.session.add(run)
        self.session.flush()
        return run

    def add_coordination_run(self, run: CoordinationRun) -> CoordinationRun:
        self.session.add(run)
        self.session.flush()
        return run

    def add_recovery_event(self, event: RecoveryEvent) -> RecoveryEvent:
        self.session.add(event)
        self.session.flush()
        return event

    # -- reads --------------------------------------------------------
    def recent_planning_runs(self, limit: int = 20) -> Sequence[PlanningRun]:
        stmt = select(PlanningRun).order_by(PlanningRun.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def recent_allocation_runs(self, limit: int = 20) -> Sequence[AllocationRun]:
        stmt = select(AllocationRun).order_by(AllocationRun.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def recent_coordination_runs(self, limit: int = 20) -> Sequence[CoordinationRun]:
        stmt = select(CoordinationRun).order_by(CoordinationRun.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def recent_recovery_events(self, limit: int = 20) -> Sequence[RecoveryEvent]:
        stmt = select(RecoveryEvent).order_by(RecoveryEvent.id.desc()).limit(limit)
        return list(self.session.scalars(stmt))

    def count_planning_runs(self) -> int:
        return self.session.query(PlanningRun).count()

    def count_recovery_events(self) -> int:
        return self.session.query(RecoveryEvent).count()
