"""The result of one allocation cycle: who should do what.

An allocator is a "plan first, commit second" component. It does NOT change any
robot or task state - it returns this object describing a proposed set of
assignments. A separate commit step (see `commit.py`) applies it through the
normal task API. Keeping the decision and the mutation apart means a rejected
or partial plan can never leave the world half-updated.

Both allocators - greedy and CP-SAT - return this same shape, so the benchmark
and the tests never have to know which one produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Assignment:
    """A single proposed pairing and its priced-out cost, in grid moves."""

    robot_id: str
    task_id: str
    robot_to_pickup_cost: int
    pickup_to_dropoff_cost: int

    @property
    def total_estimated_cost(self) -> int:
        return self.robot_to_pickup_cost + self.pickup_to_dropoff_cost

    def to_dict(self) -> Dict[str, Any]:
        return {
            "robot_id": self.robot_id,
            "task_id": self.task_id,
            "robot_to_pickup_cost": self.robot_to_pickup_cost,
            "pickup_to_dropoff_cost": self.pickup_to_dropoff_cost,
            "total_estimated_cost": self.total_estimated_cost,
        }


@dataclass
class AllocationResult:
    """What an allocator returns for one allocation cycle.

    Attributes:
        algorithm: "greedy" or "cp_sat".
        success: True if the allocator ran and produced a valid (possibly
            empty) assignment set. False only on solver failure.
        solver_status: a short status string. For CP-SAT this mirrors the
            CP-SAT status name ("OPTIMAL", "FEASIBLE", "INFEASIBLE", ...).
            For greedy it is always "GREEDY".
        assignments: the proposed pairings, in a deterministic order.
        unassigned_task_ids: PENDING tasks that were candidates but got no
            robot this cycle. They stay pending for a future cycle.
        total_estimated_cost: sum of the assignments' total costs.
        eligible_pair_count: how many (eligible robot, pending task) pairings
            were feasible.
        infeasible_pair_count: how many were ruled out (payload, battery, or
            no route).
        solve_time_ms: wall-clock time spent inside the allocate() call.
    """

    algorithm: str
    success: bool
    solver_status: str
    assignments: List[Assignment] = field(default_factory=list)
    unassigned_task_ids: List[str] = field(default_factory=list)
    total_estimated_cost: int = 0
    eligible_pair_count: int = 0
    infeasible_pair_count: int = 0
    solve_time_ms: float = 0.0

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------
    @property
    def assigned_task_count(self) -> int:
        return len(self.assignments)

    @property
    def unassigned_task_count(self) -> int:
        return len(self.unassigned_task_ids)

    @property
    def assigned_robot_ids(self) -> List[str]:
        return [a.robot_id for a in self.assignments]

    @property
    def assigned_task_ids(self) -> List[str]:
        return [a.task_id for a in self.assignments]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "success": self.success,
            "solver_status": self.solver_status,
            "assignments": [a.to_dict() for a in self.assignments],
            "unassigned_task_ids": list(self.unassigned_task_ids),
            "assigned_task_count": self.assigned_task_count,
            "unassigned_task_count": self.unassigned_task_count,
            "total_estimated_cost": self.total_estimated_cost,
            "eligible_pair_count": self.eligible_pair_count,
            "infeasible_pair_count": self.infeasible_pair_count,
            "solve_time_ms": round(self.solve_time_ms, 6),
        }

    def __str__(self) -> str:
        pairs = ", ".join(f"{a.robot_id}->{a.task_id}" for a in self.assignments)
        return (
            f"{self.algorithm} [{self.solver_status}]: "
            f"{self.assigned_task_count} assigned ({pairs or 'none'}), "
            f"{self.unassigned_task_count} unassigned, "
            f"cost {self.total_estimated_cost}, {self.solve_time_ms:.3f} ms"
        )


def build_result(
    algorithm: str,
    solver_status: str,
    success: bool,
    chosen_pairs: List[Assignment],
    candidate_task_ids: List[str],
    eligible_pair_count: int,
    infeasible_pair_count: int,
    solve_time_ms: float,
) -> AllocationResult:
    """Assemble an AllocationResult, deriving the totals from the chosen pairs.

    Deriving `total_estimated_cost` and `unassigned_task_ids` here (rather than
    trusting a caller to pass them) removes a class of bug where the summary
    numbers disagree with the assignment list.
    """
    assigned_ids = {a.task_id for a in chosen_pairs}
    unassigned = [tid for tid in candidate_task_ids if tid not in assigned_ids]
    total = sum(a.total_estimated_cost for a in chosen_pairs)

    ordered = sorted(chosen_pairs, key=lambda a: (a.robot_id, a.task_id))

    return AllocationResult(
        algorithm=algorithm,
        success=success,
        solver_status=solver_status,
        assignments=ordered,
        unassigned_task_ids=unassigned,
        total_estimated_cost=total,
        eligible_pair_count=eligible_pair_count,
        infeasible_pair_count=infeasible_pair_count,
        solve_time_ms=solve_time_ms,
    )
