"""Optimisation-based task allocation across the robot fleet.

THE QUESTION
------------
Several robots are free. Several tasks are waiting. Which robot should do which
task so the fleet drives the least in total, without breaking any rule?

    pending tasks + eligible robots
                 |
                 v
        CostEstimator      prices every feasible robot->task pairing with A*
                 |
                 v
     GreedyAllocator  /  CpSatAllocator      propose assignments (mutate nothing)
                 |
                 v
        AllocationResult      the proposed pairings + summary numbers
                 |
                 v
       commit_allocation      apply it through the normal task API

TWO ALLOCATORS
--------------
* GreedyAllocator - a simple, deterministic baseline: give each task, in
  priority order, to its cheapest still-free robot.
* CpSatAllocator - models the choice as a binary assignment problem and solves
  it to proven optimality with Google OR-Tools CP-SAT.

Having both lets us measure, honestly, how much the optimiser actually buys
over the obvious heuristic on the same scenarios.

SCOPE
-----
One allocation cycle assigns at most one pending task per available robot.
Leftover tasks stay pending for the next cycle. This is NOT vehicle routing
(one robot, many stops) and NOT multi-robot path coordination (keeping the
assigned robots from colliding while they drive) - those are later work.
"""

from robotics.allocation.allocation_result import AllocationResult, Assignment
from robotics.allocation.commit import CommitReport, commit_allocation
from robotics.allocation.cost_estimator import AssignmentCost, CostEstimator
from robotics.allocation.cp_sat_allocator import CpSatAllocator
from robotics.allocation.feasibility import (
    CandidatePair,
    build_candidate_pairs,
    eligible_robots,
    is_battery_feasible,
    pending_tasks,
)
from robotics.allocation.greedy_allocator import GreedyAllocator
from robotics.allocation.validation import (
    find_allocation_problems,
    is_valid_allocation,
)

__all__ = [
    "AllocationResult",
    "Assignment",
    "AssignmentCost",
    "CostEstimator",
    "CandidatePair",
    "GreedyAllocator",
    "CpSatAllocator",
    "commit_allocation",
    "CommitReport",
    "build_candidate_pairs",
    "eligible_robots",
    "pending_tasks",
    "is_battery_feasible",
    "find_allocation_problems",
    "is_valid_allocation",
]
