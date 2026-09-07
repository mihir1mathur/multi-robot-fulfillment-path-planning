"""Task-allocation service: run the EXISTING greedy / CP-SAT allocator via the API.

No OR-Tools model and no cost logic live here. The service gathers the robots
and tasks (from the request, or from the persisted fleet), calls
`GreedyAllocator` / `CpSatAllocator` with the shared `CostEstimator`, records
the run, and - only when asked and only for persisted state - commits the
assignments through the existing `commit_allocation`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from sqlalchemy.orm import Session

from robotics.allocation.allocation_result import AllocationResult
from robotics.allocation.commit import commit_allocation
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.cp_sat_allocator import CpSatAllocator
from robotics.allocation.greedy_allocator import GreedyAllocator
from robotics.persistence.models import AllocationRun
from robotics.persistence.repositories import RobotRepository, RunRepository, TaskRepository
from robotics.robots.robot import Robot
from robotics.robots.robot_state import RobotStatus
from robotics.services.errors import ValidationFailedError
from robotics.services.world import (
    build_simulator,
    build_warehouse,
    robot_from_record,
    task_from_record,
)
from robotics.tasks.task import Task, TaskPriority
from robotics.warehouse.grid import Position

logger = logging.getLogger("robotics.services.allocation")

_ALLOCATORS = {"greedy": GreedyAllocator, "cp_sat": CpSatAllocator}
SUPPORTED_ALGORITHMS = tuple(_ALLOCATORS.keys())


@dataclass
class RobotSpec:
    robot_id: str
    row: int
    col: int
    battery_level: float = 100.0
    payload_capacity: float = 10.0


@dataclass
class TaskSpec:
    task_id: str
    pickup_row: int
    pickup_col: int
    dropoff_row: int
    dropoff_col: int
    payload_weight: float = 1.0
    priority: str = "normal"


@dataclass
class AllocationOutcome:
    result: AllocationResult
    run_id: int
    committed: bool
    applied: List[str] = field(default_factory=list)


class AllocationService:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.runs = RunRepository(session)
        self.robots = RobotRepository(session)
        self.tasks = TaskRepository(session)

    # ------------------------------------------------------------------
    def run_allocation(
        self,
        algorithm: str = "greedy",
        robots: Optional[Sequence[RobotSpec]] = None,
        tasks: Optional[Sequence[TaskSpec]] = None,
        from_persisted: bool = False,
        commit: bool = False,
    ) -> AllocationOutcome:
        if algorithm not in _ALLOCATORS:
            raise ValidationFailedError(
                f"unsupported allocator '{algorithm}'; use one of "
                f"{', '.join(SUPPORTED_ALGORITHMS)}"
            )
        if commit and not from_persisted:
            raise ValidationFailedError(
                "commit=true is only valid together with from_persisted=true"
            )

        warehouse = build_warehouse()

        if from_persisted:
            domain_robots = [
                robot_from_record(r)
                for r in self.robots.list()
                if r.status == RobotStatus.IDLE.value and r.assigned_task_id is None
            ]
            domain_tasks = [
                task_from_record(t) for t in self.tasks.pending()
            ]
        else:
            domain_robots = [self._robot_from_spec(s) for s in (robots or ())]
            domain_tasks = [self._task_from_spec(s) for s in (tasks or ())]

        if not domain_robots or not domain_tasks:
            raise ValidationFailedError(
                "allocation needs at least one robot and one task"
            )

        # A simulator is only needed for the commit path; the allocators
        # themselves take plain robot / task lists.
        estimator = CostEstimator(warehouse)
        allocator = _ALLOCATORS[algorithm](estimator)
        result = allocator.allocate(domain_robots, domain_tasks)

        committed = False
        applied: List[str] = []
        if commit and result.assignments:
            simulator = build_simulator(warehouse, domain_robots, domain_tasks)
            report = commit_allocation(simulator, result)
            if not report.committed:
                raise ValidationFailedError(
                    f"allocation could not be committed: {report.rejected_reason}"
                )
            committed = True
            applied = list(report.applied)
            self._write_back(simulator)

        run = AllocationRun(
            algorithm=algorithm,
            solver_status=result.solver_status,
            requested_robot_count=len(domain_robots),
            requested_task_count=len(domain_tasks),
            assigned_count=result.assigned_task_count,
            unassigned_count=result.unassigned_task_count,
            total_estimated_cost=result.total_estimated_cost,
            solve_time_ms=result.solve_time_ms,
            committed=committed,
            result_json=result.to_dict(),
        )
        self.runs.add_allocation_run(run)
        logger.info(
            "allocation.run",
            extra={
                "algorithm": algorithm,
                "assigned": result.assigned_task_count,
                "committed": committed,
            },
        )
        return AllocationOutcome(
            result=result, run_id=run.id, committed=committed, applied=applied
        )

    # ------------------------------------------------------------------
    def _write_back(self, simulator) -> None:
        """Copy the committed assignments from the simulator back to the rows."""
        for robot in simulator.robots:
            record = self.robots.get(robot.robot_id)
            if record is not None:
                record.status = robot.status.value
                record.assigned_task_id = robot.assigned_task_id
        for task in simulator.tasks:
            record = self.tasks.get(task.task_id)
            if record is not None:
                record.status = task.status.value
                record.assigned_robot_id = task.assigned_robot_id
        self.session.flush()

    def _robot_from_spec(self, spec: RobotSpec) -> Robot:
        try:
            return Robot(
                robot_id=spec.robot_id,
                position=Position(spec.row, spec.col),
                battery_level=spec.battery_level,
                payload_capacity=spec.payload_capacity,
            )
        except ValueError as error:
            raise ValidationFailedError(str(error)) from error

    def _task_from_spec(self, spec: TaskSpec) -> Task:
        try:
            priority = TaskPriority[spec.priority.upper()]
        except KeyError as error:
            raise ValidationFailedError(
                f"unknown priority '{spec.priority}'"
            ) from error
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
