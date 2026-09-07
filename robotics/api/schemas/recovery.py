"""Recovery request / response schemas."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from robotics.api.schemas.common import Coordinate, TaskPriorityName, WarehouseSpec
from robotics.services.recovery_service import (
    RecoveryOutcome,
    RecoveryRobotSpec,
    RecoveryTaskSpec,
)


class RecoveryInlineTask(BaseModel):
    task_id: str = Field(min_length=1, max_length=64)
    pickup: Coordinate
    dropoff: Coordinate
    payload_weight: float = Field(default=1.0, gt=0.0)
    priority: TaskPriorityName = TaskPriorityName.normal  # type: ignore[attr-defined]

    def to_spec(self) -> RecoveryTaskSpec:
        return RecoveryTaskSpec(
            task_id=self.task_id,
            pickup_row=self.pickup.row,
            pickup_col=self.pickup.col,
            dropoff_row=self.dropoff.row,
            dropoff_col=self.dropoff.col,
            payload_weight=self.payload_weight,
            priority=self.priority.value,
        )


class RecoveryRobot(BaseModel):
    robot_id: str = Field(min_length=1, max_length=64)
    start: Coordinate
    goal: Coordinate
    task: Optional[RecoveryInlineTask] = None

    def to_spec(self) -> RecoveryRobotSpec:
        return RecoveryRobotSpec(
            robot_id=self.robot_id,
            start_row=self.start.row,
            start_col=self.start.col,
            goal_row=self.goal.row,
            goal_col=self.goal.col,
            task=self.task.to_spec() if self.task else None,
        )


class _RecoveryRequestBase(BaseModel):
    robots: List[RecoveryRobot] = Field(min_length=1)
    timestep: int = Field(ge=0, description="timestep the disruption fires at")
    priority_order: Optional[List[str]] = None
    horizon: Optional[int] = Field(default=None, gt=0)
    recovery_enabled: bool = Field(
        default=True,
        description="false reproduces the safe-stop-only behaviour (for A/B "
        "comparison)",
    )
    warehouse: Optional[WarehouseSpec] = None


class ObstacleRecoveryRequest(_RecoveryRequestBase):
    obstacle: Coordinate = Field(description="cell that becomes blocked at 'timestep'")


class RobotFailureRecoveryRequest(_RecoveryRequestBase):
    failed_robot_id: str = Field(min_length=1, max_length=64)
    commit: bool = Field(
        default=False,
        description="when true, a successful failure->spare reassignment is "
        "written back to the persisted robots/tasks rows (failed robot -> "
        "offline & unassigned, released task -> assigned to the spare). "
        "Bystander routes are never persisted. A safe stop persists nothing.",
    )


class RobotReplanModel(BaseModel):
    robot_id: str
    success: bool
    current_position: Coordinate
    original_goal: Coordinate
    new_path_length: int
    additional_distance: int
    wait_actions_introduced: int
    joined_mid_episode: bool
    failure_reason: Optional[str]


class RecoveryResponse(BaseModel):
    run_id: int
    disruption_type: str
    recovery_enabled: bool
    recovery_attempted: bool
    recovery_success: bool
    safe_stop: bool
    resumed: bool
    affected_robot_ids: List[str]
    affected_task_ids: List[str]
    replanning_attempted: bool
    replanning_success: bool
    task_reassignment_attempted: bool
    task_reassignment_success: bool
    reassigned_from: Optional[str]
    reassigned_to: Optional[str]
    robot_replans: List[RobotReplanModel]
    unresolved_vertex_conflicts: int
    unresolved_edge_conflicts: int
    total_recovery_latency_ms: float
    committed: bool = False
    committed_state: List[str] = Field(default_factory=list)
    execution: Dict  # the full ResilientExecutionResult.to_dict()
    failure_reason: Optional[str]

    @classmethod
    def from_outcome(cls, outcome: RecoveryOutcome) -> "RecoveryResponse":
        execution = outcome.execution
        event = outcome.event
        replans: List[RobotReplanModel] = []
        if event is not None:
            for r in event.robot_replans:
                replans.append(
                    RobotReplanModel(
                        robot_id=r.robot_id,
                        success=r.success,
                        current_position=Coordinate(
                            row=r.current_position.row, col=r.current_position.col
                        ),
                        original_goal=Coordinate(
                            row=r.original_goal.row, col=r.original_goal.col
                        ),
                        new_path_length=r.new_length,
                        additional_distance=r.additional_distance,
                        wait_actions_introduced=r.wait_actions_introduced,
                        joined_mid_episode=r.joined_mid_episode,
                        failure_reason=r.failure_reason,
                    )
                )
        return cls(
            run_id=outcome.run_id,
            disruption_type=outcome.disruption_type,
            recovery_enabled=outcome.recovery_enabled,
            recovery_attempted=event is not None,
            recovery_success=bool(event and event.recovery_success),
            safe_stop=execution.safe_stop,
            resumed=bool(event and event.resumed),
            affected_robot_ids=list(event.affected_robot_ids) if event else [],
            affected_task_ids=list(event.affected_task_ids) if event else [],
            replanning_attempted=bool(event and event.replanning_attempted),
            replanning_success=bool(event and event.replanning_success),
            task_reassignment_attempted=bool(event and event.task_reassignment_attempted),
            task_reassignment_success=bool(event and event.task_reassignment_success),
            reassigned_from=event.reassigned_from if event else None,
            reassigned_to=event.reassigned_to if event else None,
            robot_replans=replans,
            unresolved_vertex_conflicts=event.unresolved_vertex_conflicts if event else 0,
            unresolved_edge_conflicts=event.unresolved_edge_conflicts if event else 0,
            total_recovery_latency_ms=event.total_recovery_latency_ms if event else 0.0,
            committed=outcome.committed,
            committed_state=list(outcome.applied),
            execution=execution.to_dict(),
            failure_reason=(event.failure_reason if event else None)
            or execution.failure_reason,
        )
