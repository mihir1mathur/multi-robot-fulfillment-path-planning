"""Recovery service: run the EXISTING dynamic replanning / fault recovery.

Flow for one recovery request:

    build warehouse + fleet (+ optional inline task per robot)
        |
        v
    coordinate the fleet   (MultiRobotCoordinator)
        |
        v
    RecoveryManager(sim).run(coordination, DisruptionSchedule.of(event),
                             recovery_enabled=...)
        |
        v
    persist a RecoveryEvent row, return the ResilientExecutionResult

Two disruption kinds are exposed, matching the recovery package:

    obstacle       a DynamicObstacleEvent at (row, col) @ timestep
    robot-failure  a RobotFailureEvent for robot_id @ timestep

No recovery logic is reimplemented; this is orchestration only. Partial handoff
of an in-progress task is explicitly NOT part of this layer.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from robotics.coordination.coordinator import MultiRobotCoordinator
from robotics.exceptions import RoboticsError
from robotics.persistence.models import RecoveryEvent
from robotics.persistence.repositories import (
    RobotRepository,
    RunRepository,
    TaskRepository,
)
from robotics.recovery.disruption import (
    DisruptionSchedule,
    DynamicObstacleEvent,
    RobotFailureEvent,
)
from robotics.recovery.recovery_manager import RecoveryManager
from robotics.recovery.recovery_result import (
    RecoveryEventResult,
    ResilientExecutionResult,
)
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.services.errors import DomainFailureError, ValidationFailedError
from robotics.services.world import build_simulator, build_warehouse
from robotics.tasks.task import Task, TaskPriority, TaskStatus
from robotics.warehouse.grid import Position

logger = logging.getLogger("robotics.services.recovery")


@dataclass
class RecoveryRobotSpec:
    robot_id: str
    start_row: int
    start_col: int
    goal_row: int
    goal_col: int
    task: Optional["RecoveryTaskSpec"] = None


@dataclass
class RecoveryTaskSpec:
    task_id: str
    pickup_row: int
    pickup_col: int
    dropoff_row: int
    dropoff_col: int
    payload_weight: float = 1.0
    priority: str = "normal"


@dataclass
class RecoveryOutcome:
    execution: ResilientExecutionResult
    event: Optional[RecoveryEventResult]
    run_id: int
    disruption_type: str
    recovery_enabled: bool = True
    committed: bool = False
    applied: List[str] = field(default_factory=list)


class RecoveryService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runs = RunRepository(session)
        self.robots = RobotRepository(session)
        self.tasks = TaskRepository(session)

    # ------------------------------------------------------------------
    def recover_from_obstacle(
        self,
        robots: Sequence[RecoveryRobotSpec],
        obstacle_row: int,
        obstacle_col: int,
        timestep: int,
        priority_order: Optional[List[str]] = None,
        horizon: Optional[int] = None,
        recovery_enabled: bool = True,
        warehouse_width: Optional[int] = None,
        warehouse_height: Optional[int] = None,
        static_obstacles: Optional[Sequence[Position]] = None,
    ) -> RecoveryOutcome:
        event = DynamicObstacleEvent(
            timestep=timestep, position=Position(obstacle_row, obstacle_col)
        )
        return self._run(
            robots, event, "dynamic_obstacle", priority_order, horizon,
            recovery_enabled, warehouse_width, warehouse_height, static_obstacles,
        )

    def recover_from_robot_failure(
        self,
        robots: Sequence[RecoveryRobotSpec],
        failed_robot_id: str,
        timestep: int,
        priority_order: Optional[List[str]] = None,
        horizon: Optional[int] = None,
        recovery_enabled: bool = True,
        warehouse_width: Optional[int] = None,
        warehouse_height: Optional[int] = None,
        static_obstacles: Optional[Sequence[Position]] = None,
        commit: bool = False,
    ) -> RecoveryOutcome:
        if failed_robot_id not in {r.robot_id for r in robots}:
            raise ValidationFailedError(
                f"failed_robot_id '{failed_robot_id}' is not in the fleet"
            )
        event = RobotFailureEvent(timestep=timestep, robot_id=failed_robot_id)
        return self._run(
            robots, event, "robot_offline", priority_order, horizon,
            recovery_enabled, warehouse_width, warehouse_height, static_obstacles,
            commit=commit,
        )

    # ------------------------------------------------------------------
    def _run(
        self,
        robots: Sequence[RecoveryRobotSpec],
        event,
        disruption_type: str,
        priority_order: Optional[List[str]],
        horizon: Optional[int],
        recovery_enabled: bool,
        warehouse_width: Optional[int] = None,
        warehouse_height: Optional[int] = None,
        static_obstacles: Optional[Sequence[Position]] = None,
        commit: bool = False,
    ) -> RecoveryOutcome:
        if not robots:
            raise ValidationFailedError("recovery needs at least one robot")
        if event.timestep < 0:
            raise ValidationFailedError("timestep must be >= 0")

        warehouse = build_warehouse(
            warehouse_width, warehouse_height, static_obstacles
        )
        domain_robots = [self._robot_from_spec(s) for s in robots]
        domain_tasks = [
            self._task_from_spec(s.task) for s in robots if s.task is not None
        ]
        simulator = build_simulator(warehouse, domain_robots, domain_tasks)

        # assign each inline task to its robot through the normal simulator API
        for spec in robots:
            if spec.task is not None:
                try:
                    simulator.assign_task_manually(spec.task.task_id, spec.robot_id)
                except RoboticsError as error:
                    raise ValidationFailedError(
                        f"could not assign task '{spec.task.task_id}' to "
                        f"'{spec.robot_id}': {error}"
                    ) from error

        starts = {s.robot_id: Position(s.start_row, s.start_col) for s in robots}
        goals = {s.robot_id: Position(s.goal_row, s.goal_col) for s in robots}
        coordinator = MultiRobotCoordinator(warehouse)
        coordination = coordinator.plan(starts, goals, priority_order, horizon)
        if not coordination.successful_paths:
            raise DomainFailureError(
                "the fleet could not be coordinated, so there is no plan to "
                f"recover: {coordination.failure_reason}"
            )

        manager = RecoveryManager(simulator)
        execution = manager.run(
            coordination,
            DisruptionSchedule.of(event),
            recovery_enabled=recovery_enabled,
        )
        single_event = execution.recovery_events[0] if execution.recovery_events else None

        run = RecoveryEvent(
            disruption_type=disruption_type,
            trigger_timestep=event.timestep,
            recovery_enabled=recovery_enabled,
            recovery_success=bool(single_event and single_event.recovery_success),
            safe_stop=execution.safe_stop,
            affected_robot_ids=list(single_event.affected_robot_ids) if single_event else [],
            affected_task_ids=list(single_event.affected_task_ids) if single_event else [],
            replanning_attempted=bool(single_event and single_event.replanning_attempted),
            replanning_success=bool(single_event and single_event.replanning_success),
            task_reassignment_attempted=bool(
                single_event and single_event.task_reassignment_attempted
            ),
            task_reassignment_success=bool(
                single_event and single_event.task_reassignment_success
            ),
            reassigned_from=single_event.reassigned_from if single_event else None,
            reassigned_to=single_event.reassigned_to if single_event else None,
            total_recovery_latency_ms=(
                single_event.total_recovery_latency_ms if single_event else 0.0
            ),
            result_json=execution.to_dict(),
        )
        self.runs.add_recovery_event(run)

        applied: List[str] = []
        if commit and single_event is not None:
            applied = self._persist_reassignment(simulator, single_event)

        logger.info(
            "recovery.run",
            extra={
                "disruption_type": disruption_type,
                "recovery_enabled": recovery_enabled,
                "recovery_success": run.recovery_success,
                "safe_stop": run.safe_stop,
                "committed": bool(applied),
            },
        )
        return RecoveryOutcome(
            execution=execution,
            event=single_event,
            run_id=run.id,
            disruption_type=disruption_type,
            recovery_enabled=recovery_enabled,
            committed=bool(applied),
            applied=applied,
        )

    # ------------------------------------------------------------------
    def _persist_reassignment(
        self, simulator, event: RecoveryEventResult
    ) -> List[str]:
        """Reflect a successful robot-failure -> spare reassignment in the
        persisted ``robots`` / ``tasks`` rows.

        The recovery run above returned the released task to PENDING and then
        re-committed it to the spare through the *same* ``commit_allocation``
        path a fresh allocation uses. This applies the identical end state to
        the persisted rows:

          * failed robot  -> the existing OFFLINE failure semantics, no task
                             (exactly what ``RecoveryManager._recover_task``
                             set on the in-memory robot)
          * released task -> ASSIGNED, owned by the spare
          * spare robot   -> ASSIGNED, owns the task
                             (the committed-assignment state, same as
                             ``AllocationService._write_back``; the spare's
                             transient in-sim ``moving`` status is an artifact
                             of the execution loop and is not a persisted
                             concept - positions/routes are never persisted)

        Only these three rows are touched. Replanned bystanders keep their
        persisted state, and if any of the three rows is not persisted (the
        benchmark / ad-hoc request shape) nothing is written. Runs in the
        request-scoped session, so it commits atomically with the
        ``RecoveryEvent`` row or rolls back with it.

        Returns the ids whose rows were updated (empty when the recovery did
        not end in a committed reassignment - e.g. an honest safe stop).
        """
        if not (
            event.task_reassignment_success
            and event.reassigned_to
            and event.reassigned_task_id
        ):
            return []

        failed_id = event.reassigned_from
        spare_id = event.reassigned_to
        task_id = event.reassigned_task_id

        task_row = self.tasks.get(task_id)
        spare_row = self.robots.get(spare_id)
        failed_row = self.robots.get(failed_id) if failed_id else None
        if task_row is None or spare_row is None:
            return []

        # sanity-check against the domain result before writing anything
        sim_task = simulator.get_task(task_id)
        if sim_task.assigned_robot_id != spare_id:
            return []

        applied: List[str] = []
        if failed_row is not None:
            failed_row.status = RobotStatus.OFFLINE.value
            failed_row.assigned_task_id = None
            applied.append(failed_id)

        spare_row.status = RobotStatus.ASSIGNED.value
        spare_row.assigned_task_id = task_id
        applied.append(spare_id)

        task_row.status = TaskStatus.ASSIGNED.value
        task_row.assigned_robot_id = spare_id
        applied.append(task_id)

        self.session.flush()
        return applied

    # ------------------------------------------------------------------
    def _robot_from_spec(self, spec: RecoveryRobotSpec) -> Robot:
        try:
            return Robot(
                robot_id=spec.robot_id,
                position=Position(spec.start_row, spec.start_col),
            )
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error

    def _task_from_spec(self, spec: RecoveryTaskSpec) -> Task:
        try:
            priority = TaskPriority[spec.priority.upper()]
        except KeyError as error:
            raise ValidationFailedError(f"unknown priority '{spec.priority}'") from error
        try:
            return Task(
                task_id=spec.task_id,
                pickup_location=Position(spec.pickup_row, spec.pickup_col),
                dropoff_location=Position(spec.dropoff_row, spec.dropoff_col),
                priority=priority,
                payload_weight=spec.payload_weight,
            )
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error
