"""Optimisation-based task allocation with Google OR-Tools CP-SAT.

WHAT PROBLEM ARE WE SOLVING?
---------------------------
We have some eligible robots and some pending tasks. We want to choose a set of
robot->task assignments that:

    * assigns as MANY tasks as possible this cycle, and
    * among all ways of doing that, has the LOWEST total estimated travel cost,

subject to: one robot does at most one task per cycle, one task goes to at most
one robot, and only feasible pairings (payload, battery, reachable) may be
chosen.

This is a classic ASSIGNMENT problem. CP-SAT (a constraint-programming solver
with a SAT engine) solves it to proven optimality for the sizes we use.

THE MODEL
---------
Decision variables - one binary per feasible pair:

    x[r, t] = 1  if robot r is assigned task t
    x[r, t] = 0  otherwise

Infeasible pairs get NO variable at all, so they can never be selected.

Constraints:

    (C1)  for every task t:   sum over r of x[r, t]  <= 1     # task assigned once
    (C2)  for every robot r:  sum over t of x[r, t]  <= 1     # one task per robot

Objective - maximise:

    sum over feasible (r, t) of  x[r, t] * (REWARD - effective_cost[r, t])

where:

    effective_cost[r, t] = cost[r, t] * SCALE + creation_index[r, t]

    SCALE   larger than the biggest possible sum of creation indices, so the
            travel cost is always the deciding factor and the creation index
            only ever settles an exact cost tie (deterministically, in favour
            of the lower robot ID - pairs are created in robot-ID order).

    REWARD  = (sum of all effective costs) + 1, i.e. larger than any assignment
            set's total effective cost. Every (REWARD - effective_cost) term is
            then strictly positive AND large enough that adding one more
            assignment always beats any rearrangement of the rest.

So the solver optimises, in strict priority order:
    1. assign as MANY tasks as possible,
    2. then minimise total travel cost,
    3. then (exact ties only) prefer the lower robot ID.
All coefficients are integers, which CP-SAT requires.

    Robots R1 R2      Costs   T1  T2
                       R1      4   9
                       R2      6   5

    Feasible x: x[R1,T1] x[R1,T2] x[R2,T1] x[R2,T2]
    Best assignment: R1->T1 (4) and R2->T2 (5), total 5 + 4 = 9.
    The alternative R1->T2 (9) + R2->T1 (6) = 15 is worse, so it is not chosen.

WHY NOT BRUTE FORCE?
--------------------
For n robots and n tasks there are up to n! full assignments. n = 12 is already
~479 million. CP-SAT prunes that search with the constraints and returns a
provably optimal answer in milliseconds at our sizes.

DETERMINISM
-----------
The solver is pinned to a single worker and a fixed random seed, and the
variables are created in a deterministic order, so the same input always yields
the same assignment.
"""

from __future__ import annotations

import time
from typing import Dict, List, Sequence, Tuple

from ortools.sat.python import cp_model

from robotics.allocation.allocation_result import Assignment, AllocationResult, build_result
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.feasibility import build_candidate_pairs, pending_tasks
from robotics.robots.robot import Robot
from robotics.tasks.task import Task

ALGORITHM_NAME = "cp_sat"

# Hard ceiling so a pathological input cannot hang a benchmark run.
_MAX_SOLVE_SECONDS = 10.0


class CpSatAllocator:
    """Chooses robot->task assignments by solving an assignment model with CP-SAT."""

    def __init__(
        self, estimator: CostEstimator, max_solve_seconds: float = _MAX_SOLVE_SECONDS
    ) -> None:
        """
        Args:
            estimator: the shared cost estimator (see GreedyAllocator).
            max_solve_seconds: wall-clock cap handed to the solver.
        """
        self.estimator = estimator
        self.max_solve_seconds = max_solve_seconds

    def allocate(
        self, robots: Sequence[Robot], tasks: Sequence[Task]
    ) -> AllocationResult:
        """Propose an optimal assignment set for one cycle. Mutates nothing."""
        started_at_ns = time.perf_counter_ns()

        feasible, infeasible = build_candidate_pairs(robots, tasks, self.estimator)
        candidate_task_ids = [t.task_id for t in pending_tasks(tasks)]

        # No feasible pairing at all: a valid, empty allocation (not a failure).
        if not feasible:
            solve_time_ms = (time.perf_counter_ns() - started_at_ns) / 1_000_000
            return build_result(
                algorithm=ALGORITHM_NAME,
                solver_status="OPTIMAL",
                success=True,
                chosen_pairs=[],
                candidate_task_ids=candidate_task_ids,
                eligible_pair_count=0,
                infeasible_pair_count=infeasible,
                solve_time_ms=solve_time_ms,
            )

        model = cp_model.CpModel()

        # --- Decision variables: one bool per feasible pair ---
        # `feasible` is already ordered by robot ID (then task priority, then
        # task ID), so the creation index rises with robot ID and an exact cost
        # tie is broken in favour of the lower robot ID.
        x: Dict[Tuple[str, str], cp_model.IntVar] = {}
        effective_cost: Dict[Tuple[str, str], int] = {}
        pairs_by_task: Dict[str, List[Tuple[str, str]]] = {}
        pairs_by_robot: Dict[str, List[Tuple[str, str]]] = {}

        pair_count = len(feasible)
        scale = pair_count * pair_count + 1  # > any possible sum of indices

        for index, pair in enumerate(feasible):
            key = (pair.robot_id, pair.task_id)
            x[key] = model.NewBoolVar(f"x_{pair.robot_id}_{pair.task_id}")
            effective_cost[key] = pair.total_cost * scale + index
            pairs_by_task.setdefault(pair.task_id, []).append(key)
            pairs_by_robot.setdefault(pair.robot_id, []).append(key)

        # --- (C1) each task to at most one robot ---
        for task_id, keys in pairs_by_task.items():
            model.Add(sum(x[k] for k in keys) <= 1)

        # --- (C2) each robot at most one task this cycle ---
        for robot_id, keys in pairs_by_robot.items():
            model.Add(sum(x[k] for k in keys) <= 1)

        # --- Objective: assignments first, then cost, then robot-ID tie-break ---
        # REWARD exceeds the total effective cost of any assignment set, so one
        # extra assignment always outweighs any rearrangement of the rest.
        reward = sum(effective_cost.values()) + 1
        model.Maximize(sum(x[k] * (reward - effective_cost[k]) for k in x))

        # --- Solve deterministically ---
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.random_seed = 0
        solver.parameters.max_time_in_seconds = self.max_solve_seconds
        status = solver.Solve(model)
        status_name = solver.StatusName(status)

        solve_time_ms = (time.perf_counter_ns() - started_at_ns) / 1_000_000

        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # The model is always feasible (the empty assignment satisfies every
            # constraint), so this means the solver hit its time limit without a
            # solution or errored. Report it honestly rather than inventing one.
            return build_result(
                algorithm=ALGORITHM_NAME,
                solver_status=status_name,
                success=False,
                chosen_pairs=[],
                candidate_task_ids=candidate_task_ids,
                eligible_pair_count=len(feasible),
                infeasible_pair_count=infeasible,
                solve_time_ms=solve_time_ms,
            )

        chosen: List[Assignment] = []
        for pair in feasible:
            key = (pair.robot_id, pair.task_id)
            if solver.Value(x[key]) == 1:
                chosen.append(
                    Assignment(
                        robot_id=pair.robot_id,
                        task_id=pair.task_id,
                        robot_to_pickup_cost=pair.cost.robot_to_pickup,
                        pickup_to_dropoff_cost=pair.cost.pickup_to_dropoff,
                    )
                )

        return build_result(
            algorithm=ALGORITHM_NAME,
            solver_status=status_name,
            success=True,
            chosen_pairs=chosen,
            candidate_task_ids=candidate_task_ids,
            eligible_pair_count=len(feasible),
            infeasible_pair_count=infeasible,
            solve_time_ms=solve_time_ms,
        )
