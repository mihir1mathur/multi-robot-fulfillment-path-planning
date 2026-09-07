"""A deterministic greedy task allocator - the baseline CP-SAT is measured against.

THE STRATEGY (deliberately simple)
----------------------------------
Take the pending tasks in priority order (most urgent first, ties by task ID).
For each one, look at every eligible robot that has not already been given a
task this cycle, and pick the one whose estimated cost for this task is lowest
(ties broken by robot ID). Assign it. Move on.

    tasks (in order):  T1  T2  T3
    robots:            R1  R2

    T1: cheapest free robot is R2  -> assign R2->T1, R2 now busy
    T2: only R1 is free           -> assign R1->T2, R1 now busy
    T3: no free robot             -> T3 stays pending

WHY A GREEDY BASELINE AT ALL?
-----------------------------
To have something honest to compare the optimiser against. "CP-SAT found an
assignment" means little on its own; "CP-SAT's total travel cost was X% below
the greedy baseline's, on the same scenarios, assigning the same number of
tasks" is a real result.

WHAT IT IS NOT
--------------
Greedy is locally optimal per task and can be globally worse than CP-SAT: an
early task can grab the one robot that a later, more important pairing needed.
That gap is exactly what the benchmark quantifies. This baseline is NOT
sabotaged to make CP-SAT look good - it uses the same feasibility rules and
the same cost estimates.
"""

from __future__ import annotations

import time
from typing import List, Sequence

from robotics.allocation.allocation_result import Assignment, AllocationResult, build_result
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.feasibility import build_candidate_pairs, pending_tasks
from robotics.robots.robot import Robot
from robotics.tasks.task import Task

ALGORITHM_NAME = "greedy"


class GreedyAllocator:
    """Assigns each pending task to its cheapest still-free eligible robot."""

    def __init__(self, estimator: CostEstimator) -> None:
        """
        Args:
            estimator: the shared cost estimator. Passing the SAME estimator
                instance to both allocators is what makes the benchmark fair
                (identical costs) and fast (a shared, warm cache).
        """
        self.estimator = estimator

    def allocate(
        self, robots: Sequence[Robot], tasks: Sequence[Task]
    ) -> AllocationResult:
        """Propose assignments for one allocation cycle. Mutates nothing."""
        started_at_ns = time.perf_counter_ns()

        feasible, infeasible = build_candidate_pairs(robots, tasks, self.estimator)
        candidate_task_ids = [t.task_id for t in pending_tasks(tasks)]

        # Index feasible pairs by task, each list sorted cheapest-first then by
        # robot ID, so the pick is deterministic.
        by_task: dict = {}
        for pair in feasible:
            by_task.setdefault(pair.task_id, []).append(pair)
        for pairs in by_task.values():
            pairs.sort(key=lambda p: (p.total_cost, p.robot_id))

        used_robots: set = set()
        chosen: List[Assignment] = []

        for task_id in candidate_task_ids:
            for pair in by_task.get(task_id, ()):
                if pair.robot_id in used_robots:
                    continue
                used_robots.add(pair.robot_id)
                chosen.append(
                    Assignment(
                        robot_id=pair.robot_id,
                        task_id=pair.task_id,
                        robot_to_pickup_cost=pair.cost.robot_to_pickup,
                        pickup_to_dropoff_cost=pair.cost.pickup_to_dropoff,
                    )
                )
                break

        solve_time_ms = (time.perf_counter_ns() - started_at_ns) / 1_000_000

        return build_result(
            algorithm=ALGORITHM_NAME,
            solver_status="GREEDY",
            success=True,
            chosen_pairs=chosen,
            candidate_task_ids=candidate_task_ids,
            eligible_pair_count=len(feasible),
            infeasible_pair_count=infeasible,
            solve_time_ms=solve_time_ms,
        )
