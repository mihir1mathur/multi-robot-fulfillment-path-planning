"""Structured results for recovery.

`RobotReplan`          - one robot's route being repaired
`RecoveryEventResult` - everything that happened in response to ONE disruption
`ResilientExecutionResult` - the whole run (initial plan + every recovery)

Every number here is MEASURED during recovery, never assumed. Latencies use
`time.perf_counter_ns`. Distances are grid moves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robotics.recovery.disruption import RecoveryTrigger
from robotics.warehouse.grid import Position


@dataclass
class RobotReplan:
    """One robot's route being replanned from its current position."""

    robot_id: str
    original_goal: Position
    current_position: Position
    trigger_timestep: int
    old_remaining_path: List[Position]
    new_path: List[Position]
    success: bool
    failure_reason: Optional[str] = None
    replanning_time_ms: float = 0.0
    nodes_expanded: Optional[int] = None
    wait_actions_introduced: int = 0
    joined_mid_episode: bool = False

    @property
    def old_remaining_length(self) -> int:
        return max(0, len(self.old_remaining_path) - 1)

    @property
    def new_length(self) -> int:
        return max(0, len(self.new_path) - 1)

    @property
    def additional_distance(self) -> int:
        """Extra grid moves the replacement route costs vs the abandoned one.

        Can be negative if the replan is shorter (the obstacle was on a detour
        the original route took anyway).
        """
        return self.new_length - self.old_remaining_length

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "original_goal": [self.original_goal.row, self.original_goal.col],
            "current_position": [self.current_position.row, self.current_position.col],
            "trigger_timestep": self.trigger_timestep,
            "old_remaining_length": self.old_remaining_length,
            "new_length": self.new_length,
            "additional_distance": self.additional_distance,
            "wait_actions_introduced": self.wait_actions_introduced,
            "success": self.success,
            "failure_reason": self.failure_reason,
            "replanning_time_ms": round(self.replanning_time_ms, 6),
            "nodes_expanded": self.nodes_expanded,
            "joined_mid_episode": self.joined_mid_episode,
        }


@dataclass
class RecoveryEventResult:
    """What happened when the system reacted to one disruption event."""

    trigger: RecoveryTrigger
    trigger_timestep: int
    description: str
    affected_robot_ids: List[str] = field(default_factory=list)
    affected_task_ids: List[str] = field(default_factory=list)

    replanning_attempted: bool = False
    replanning_success: bool = False
    recovery_success: bool = False
    resumed: bool = False
    safe_stop: bool = False
    failure_reason: Optional[str] = None

    robot_replans: List[RobotReplan] = field(default_factory=list)

    reservations_released: int = 0
    reservations_created: int = 0

    task_reassignment_attempted: bool = False
    task_reassignment_success: bool = False
    reassigned_from: Optional[str] = None
    reassigned_to: Optional[str] = None
    reassigned_task_id: Optional[str] = None
    reassigned_task_status: Optional[str] = None

    replanning_latency_ms: float = 0.0
    recoordination_latency_ms: float = 0.0
    total_recovery_latency_ms: float = 0.0

    unresolved_vertex_conflicts: int = 0
    unresolved_edge_conflicts: int = 0

    # ------------------------------------------------------------------
    @property
    def total_additional_distance(self) -> int:
        return sum(r.additional_distance for r in self.robot_replans if r.success)

    @property
    def total_wait_actions_introduced(self) -> int:
        return sum(r.wait_actions_introduced for r in self.robot_replans if r.success)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trigger": self.trigger.value,
            "trigger_timestep": self.trigger_timestep,
            "description": self.description,
            "affected_robot_ids": list(self.affected_robot_ids),
            "affected_task_ids": list(self.affected_task_ids),
            "affected_robot_count": len(self.affected_robot_ids),
            "affected_task_count": len(self.affected_task_ids),
            "replanning_attempted": self.replanning_attempted,
            "replanning_success": self.replanning_success,
            "recovery_success": self.recovery_success,
            "resumed": self.resumed,
            "safe_stop": self.safe_stop,
            "failure_reason": self.failure_reason,
            "robot_replans": [r.to_dict() for r in self.robot_replans],
            "reservations_released": self.reservations_released,
            "reservations_created": self.reservations_created,
            "task_reassignment_attempted": self.task_reassignment_attempted,
            "task_reassignment_success": self.task_reassignment_success,
            "reassigned_from": self.reassigned_from,
            "reassigned_to": self.reassigned_to,
            "reassigned_task_id": self.reassigned_task_id,
            "reassigned_task_status": self.reassigned_task_status,
            "replanning_latency_ms": round(self.replanning_latency_ms, 6),
            "recoordination_latency_ms": round(self.recoordination_latency_ms, 6),
            "total_recovery_latency_ms": round(self.total_recovery_latency_ms, 6),
            "total_additional_distance": self.total_additional_distance,
            "total_wait_actions_introduced": self.total_wait_actions_introduced,
            "unresolved_vertex_conflicts": self.unresolved_vertex_conflicts,
            "unresolved_edge_conflicts": self.unresolved_edge_conflicts,
        }

    def __str__(self) -> str:
        head = "RECOVERED" if self.recovery_success else (
            "SAFE STOP" if self.safe_stop else "FAILED"
        )
        return (
            f"[{self.trigger.value} @ t{self.trigger_timestep}] {head}: "
            f"affected {self.affected_robot_ids or '-'} | "
            f"released {self.reservations_released}, created {self.reservations_created}"
            + (f" | {self.failure_reason}" if self.failure_reason else "")
        )


@dataclass
class ResilientExecutionResult:
    """The outcome of executing a coordinated plan under a disruption schedule."""

    success: bool
    safe_stop: bool = False
    recovery_events: List[RecoveryEventResult] = field(default_factory=list)
    per_robot: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    robots_reached_goal: int = 0
    robot_count: int = 0
    total_move_steps: int = 0
    total_wait_steps: int = 0
    makespan_executed: int = 0
    total_distance: float = 0.0
    total_battery_consumed: float = 0.0
    vertex_conflicts: int = 0
    edge_conflicts: int = 0
    validation_problems: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    failure_reason: Optional[str] = None

    @property
    def validated_conflict_free(self) -> bool:
        return not self.validation_problems

    @property
    def recovery_count(self) -> int:
        return len(self.recovery_events)

    @property
    def successful_recoveries(self) -> int:
        return sum(1 for e in self.recovery_events if e.recovery_success)

    @property
    def safe_stops(self) -> int:
        return sum(1 for e in self.recovery_events if e.safe_stop)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "safe_stop": self.safe_stop,
            "robot_count": self.robot_count,
            "robots_reached_goal": self.robots_reached_goal,
            "total_move_steps": self.total_move_steps,
            "total_wait_steps": self.total_wait_steps,
            "makespan_executed": self.makespan_executed,
            "total_distance": round(self.total_distance, 4),
            "total_battery_consumed": round(self.total_battery_consumed, 4),
            "vertex_conflicts": self.vertex_conflicts,
            "edge_conflicts": self.edge_conflicts,
            "validation_problems": list(self.validation_problems),
            "execution_time_ms": round(self.execution_time_ms, 6),
            "failure_reason": self.failure_reason,
            "recovery_count": self.recovery_count,
            "successful_recoveries": self.successful_recoveries,
            "safe_stops": self.safe_stops,
            "recovery_events": [e.to_dict() for e in self.recovery_events],
            "per_robot": self.per_robot,
        }

    def __str__(self) -> str:
        head = "OK" if self.success else ("SAFE STOP" if self.safe_stop else "FAILED")
        return (
            f"resilient execution {head}: {self.robots_reached_goal}/"
            f"{self.robot_count} reached goal | {self.recovery_count} recovery "
            f"event(s), {self.successful_recoveries} recovered, {self.safe_stops} "
            f"safe stop(s) | {self.vertex_conflicts}v + {self.edge_conflicts}e "
            f"conflicts during execution"
        )
