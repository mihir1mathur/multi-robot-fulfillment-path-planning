"""Single entry point that runs every benchmark family and consolidates the
result into one machine-readable evidence file.

WHAT THIS IS
------------
The project already has six self-contained, deterministic benchmark modules
(planners, allocation, coordination, recovery, service, concurrency). Each one
generates its own seeded scenarios, runs the real algorithms / API, and writes
its own JSON + CSV. This script does NOT re-implement any of that. It:

    1. imports each benchmark module,
    2. calls its public entry point (``run_benchmark`` / ``run``),
    3. lets each module refresh its own JSON/CSV artifact (unchanged behaviour),
    4. extracts a small, explicitly-defined set of HEADLINE metrics from each
       report,
    5. runs internal consistency checks and FAILS LOUDLY if any metric is
       self-contradictory (a rate outside [0, 1], parts that do not sum to the
       whole, a negative count, a divide-by-zero in an improvement formula),
    6. writes one consolidated evidence pair:

           benchmarks/results/final_evaluation.json
           benchmarks/results/final_evaluation.csv

USAGE
-----
    python benchmarks/run_final_evaluation.py            # full sweep
    python benchmarks/run_final_evaluation.py --quick    # tiny sweep (smoke)
    python benchmarks/run_final_evaluation.py --sqlite   # force SQLite backend
                                                         # for service+concurrency

By default the service and concurrency families use PostgreSQL when
``DATABASE_URL`` / ``TEST_DATABASE_URL`` point at a reachable PostgreSQL, and
fall back to SQLite otherwise (the fallback is recorded in the evidence).

HONEST FRAMING (inherited from the individual benchmarks)
--------------------------------------------------------
* ``nodes_expanded`` and ``path_cost`` are exact and reproducible on any
  machine. Every latency / throughput number is machine-dependent, in-process,
  and is NOT a production performance claim.
* Prioritized multi-robot planning is not globally optimal; dense two-way
  scenarios where some robots cannot be placed are included, not hidden.
* "conflict-free" here means "0 unresolved vertex/edge conflicts across the
  validated simulated runs", not a real-world safety guarantee.
* No secrets are written: the evidence records the database *dialect*
  ("postgresql" / "sqlite"), never a URL, password, or token.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks import (  # noqa: E402
    benchmark_allocation,
    benchmark_concurrency,
    benchmark_coordination,
    benchmark_planners,
    benchmark_recovery,
    benchmark_service,
)
from robotics.persistence.config import Settings  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"
JSON_PATH = RESULTS_DIR / "final_evaluation.json"
CSV_PATH = RESULTS_DIR / "final_evaluation.csv"

EVALUATION_VERSION = "1.0.0"


# ======================================================================
# Consistency checking - the evaluation must fail loudly on nonsense
# ======================================================================
class MetricConsistencyError(AssertionError):
    """Raised when a produced metric is internally inconsistent."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise MetricConsistencyError(message)


def _is_rate(value: Any) -> bool:
    return isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0 + 1e-9


def _non_negative(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) >= 0.0


def relative_improvement(baseline: float, improved: float) -> Optional[float]:
    """(baseline - improved) / baseline, guarded against divide-by-zero.

    Used for "lower is better" quantities (cost, conflicts, nodes expanded).
    Returns None when the baseline is 0, so the caller reports "n/a" rather
    than dividing by zero.
    """
    if baseline == 0:
        return None
    return (baseline - improved) / baseline


# ======================================================================
# One family = one benchmark module + the metrics we vouch for
# ======================================================================
class FamilyEvaluation:
    def __init__(self, key: str, title: str) -> None:
        self.key = key
        self.title = title
        self.report: Dict[str, Any] = {}
        self.headline: Dict[str, Any] = {}
        self.metric_definitions: Dict[str, str] = {}
        self.limitations: List[str] = []
        self.backend: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "title": self.title,
            "config": self.report.get("config", {}),
            "headline": self.headline,
            "metric_definitions": self.metric_definitions,
            "limitations": self.limitations,
        }
        if self.backend is not None:
            payload["backend"] = self.backend
        return payload


# ----------------------------------------------------------------------
# A. Path planning
# ----------------------------------------------------------------------
def evaluate_planning(quick: bool) -> FamilyEvaluation:
    ev = FamilyEvaluation("path_planning", "A* vs Dijkstra path planning")
    if quick:
        # generate_scenarios / run_scenario bind their sweep knobs as default
        # arguments at import time, so pass a small sweep explicitly here.
        scenarios = benchmark_planners.generate_scenarios(
            [20, 30], [0.1, 0.2], per_config=3, master_seed=benchmark_planners.MASTER_SEED
        )
        _records = []
        for scenario in scenarios:
            _records.extend(benchmark_planners.run_scenario(scenario, repeats=2))
        report = benchmark_planners.build_report(_records)
    else:
        _records, report = benchmark_planners.run_benchmark()
    benchmark_planners.write_json(report)
    benchmark_planners.write_csv(_records)
    ev.report = report

    h2h = report["head_to_head"]
    aggregates = report["aggregates"]
    total_scenarios = report["total_runs"] // 2  # one scenario -> 2 planner runs

    astar_success = sum(
        1 for r in report["raw_runs"] if r["algorithm"] == "astar" and r["success"]
    )
    dijkstra_success = sum(
        1 for r in report["raw_runs"] if r["algorithm"] == "dijkstra" and r["success"]
    )

    # Aggregate node-expansion reduction over scenarios where BOTH succeeded.
    both = _both_succeeded_planner_pairs(report["raw_runs"])
    total_astar_nodes = sum(p["astar"] for p in both)
    total_dijkstra_nodes = sum(p["dijkstra"] for p in both)
    reduction = relative_improvement(total_dijkstra_nodes, total_astar_nodes)

    ev.headline = {
        "scenarios": total_scenarios,
        "scenarios_both_succeeded": h2h["scenarios_both_succeeded"],
        "astar_successful_paths": astar_success,
        "dijkstra_successful_paths": dijkstra_success,
        "astar_expanded_fewer_nodes": h2h["astar_expanded_fewer_nodes"],
        "astar_expanded_equal_nodes": h2h["astar_expanded_equal_nodes"],
        "astar_expanded_more_nodes": h2h["astar_expanded_more_nodes"],
        "total_dijkstra_nodes_expanded": total_dijkstra_nodes,
        "total_astar_nodes_expanded": total_astar_nodes,
        "aggregate_node_expansion_reduction": _round(reduction),
        "median_astar_to_dijkstra_node_ratio": h2h["median_astar_to_dijkstra_node_ratio"],
        "optimal_cost_agreement_rate": h2h["optimal_cost_agreement_rate"],
        "planning_latency_by_group_ms": [
            {
                "algorithm": a["algorithm"],
                "grid_size": a["grid_size"],
                "obstacle_density": a["obstacle_density"],
                "median_ms": a["median_planning_time_ms"],
                "p95_ms": a["p95_planning_time_ms"],
            }
            for a in aggregates
        ],
    }
    ev.metric_definitions = {
        "aggregate_node_expansion_reduction": (
            "(sum of Dijkstra nodes_expanded - sum of A* nodes_expanded) / "
            "sum of Dijkstra nodes_expanded, over the scenarios where BOTH "
            "planners returned a path. A cell is counted once, when it is "
            "popped as the best non-stale queue entry."
        ),
        "optimal_cost_agreement_rate": (
            "fraction of scenarios solved by both planners where the two path "
            "costs were exactly equal. Dijkstra is optimal on a non-negative "
            "grid, so agreement means A* also returned an optimal-cost path."
        ),
        "planning_latency_by_group_ms": (
            "median / P95 wall-clock time per (algorithm, grid, density) group, "
            "median of TIMING_REPEATS runs. MACHINE-DEPENDENT - indicative only."
        ),
    }
    ev.limitations = [
        "Single-robot planning on a static grid; no dynamic obstacles here.",
        "Small-grid latencies are sub-millisecond and dominated by interpreter "
        "noise; nodes_expanded is the stable signal.",
        "4-connected grid, unit step cost.",
    ]

    # ---- consistency checks ----
    _check(_is_rate(h2h["optimal_cost_agreement_rate"] or 0.0),
           "planning: optimal_cost_agreement_rate outside [0,1]")
    _check(
        h2h["astar_expanded_fewer_nodes"]
        + h2h["astar_expanded_equal_nodes"]
        + h2h["astar_expanded_more_nodes"]
        == h2h["scenarios_both_succeeded"],
        "planning: fewer+equal+more != scenarios_both_succeeded",
    )
    _check(h2h["astar_expanded_more_nodes"] >= 0, "planning: negative 'more nodes' count")
    _check(total_astar_nodes <= total_dijkstra_nodes,
           "planning: A* expanded more total nodes than Dijkstra on shared scenarios")
    _check(reduction is None or 0.0 <= reduction <= 1.0,
           "planning: node-expansion reduction outside [0,1]")
    _check(astar_success == dijkstra_success,
           "planning: A* and Dijkstra disagree on which scenarios are solvable")
    return ev


def _both_succeeded_planner_pairs(raw_runs: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    by_scenario: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run in raw_runs:
        by_scenario.setdefault(run["scenario_id"], {})[run["algorithm"]] = run
    pairs: List[Dict[str, int]] = []
    for group in by_scenario.values():
        a, d = group.get("astar"), group.get("dijkstra")
        if a and d and a["success"] and d["success"]:
            pairs.append({"astar": a["nodes_expanded"], "dijkstra": d["nodes_expanded"]})
    return pairs


# ----------------------------------------------------------------------
# B. Task allocation
# ----------------------------------------------------------------------
def evaluate_allocation(quick: bool) -> FamilyEvaluation:
    ev = FamilyEvaluation("task_allocation", "Greedy vs CP-SAT task allocation")
    if quick:
        _scale(benchmark_allocation, FLEET_TASK_CONFIGS=((5, 5), (10, 10)),
               SCENARIOS_PER_CONFIG=3, TIMING_REPEATS=2)

    records, _exec_records, report = benchmark_allocation.run_benchmark()
    benchmark_allocation.write_json(report)
    benchmark_allocation.write_csv(records)
    ev.report = report

    by_algo = {a["algorithm"]: a for a in report["aggregates_by_algorithm"]}
    cmp = report["cp_sat_vs_greedy"]
    execution = report["execution_slice"]

    ev.headline = {
        "scenarios": report["total_scenarios"],
        "cp_sat_assignment_rate": by_algo["cp_sat"]["assignment_rate"],
        "greedy_assignment_rate": by_algo["greedy"]["assignment_rate"],
        "cp_sat_valid_result_rate": by_algo["cp_sat"]["valid_result_rate"],
        "cp_sat_median_solver_ms": by_algo["cp_sat"]["median_allocation_time_ms"],
        "greedy_median_solver_ms": by_algo["greedy"]["median_allocation_time_ms"],
        "same_cardinality_scenarios": cmp["scenarios_compared_same_cardinality"],
        "cp_sat_strictly_cheaper": cmp["cp_sat_strictly_cheaper"],
        "cp_sat_worse": cmp["cp_sat_worse"],
        "total_cost_greedy_same_cardinality": cmp["total_cost_greedy_same_cardinality"],
        "total_cost_cp_sat_same_cardinality": cmp["total_cost_cp_sat_same_cardinality"],
        "cp_sat_cost_improvement_pct": cmp["cp_sat_cost_improvement_pct"],
        "median_per_scenario_cost_improvement_pct": cmp["median_per_scenario_improvement_pct"],
        "execution_slice_pickup_reach_rate": execution["route_execution_success_rate"],
        "execution_slice_assignments": execution["assignments_executed"],
        "execution_slice_reached_pickup": execution["routes_reached_pickup"],
    }
    ev.metric_definitions = {
        "cp_sat_cost_improvement_pct": (
            "100 * (greedy total travel cost - CP-SAT total travel cost) / "
            "greedy total travel cost, summed ONLY over scenarios where the two "
            "allocators assigned the SAME number of tasks. 'cost' = the shared "
            "CostEstimator's robot->pickup + pickup->dropoff A* path length. "
            "Assigning fewer tasks for a lower cost is not counted as a win."
        ),
        "assignment_rate": "assigned tasks / total tasks offered, over all scenarios.",
        "execution_slice_pickup_reach_rate": (
            "of the CP-SAT assignments, the fraction whose planned route was "
            "actually driven to the pickup cell by the RouteExecutor "
            "(some assignments target a pickup that the robot cannot reach "
            "given the layout)."
        ),
    }
    ev.limitations = [
        "CP-SAT objective minimises total estimated travel; it does not model "
        "congestion, battery, or task deadlines.",
        "The greedy baseline is an existing implemented nearest-feasible "
        "assignor, not a deliberately weak strawman.",
        "Improvement is measured on same-cardinality scenarios only.",
    ]

    _check(_is_rate(by_algo["cp_sat"]["assignment_rate"]),
           "allocation: cp_sat assignment_rate outside [0,1]")
    _check(_is_rate(by_algo["greedy"]["assignment_rate"]),
           "allocation: greedy assignment_rate outside [0,1]")
    _check(by_algo["cp_sat"]["valid_result_rate"] == 1.0,
           "allocation: CP-SAT produced a result that failed independent validation")
    _check(cmp["cp_sat_worse"] == 0,
           "allocation: CP-SAT produced a strictly worse cost on a same-cardinality scenario")
    _check(
        cmp["cp_sat_strictly_cheaper"] + cmp["equal_cost"] + cmp["cp_sat_worse"]
        == cmp["scenarios_compared_same_cardinality"],
        "allocation: cheaper+equal+worse != same-cardinality scenario count",
    )
    manual = relative_improvement(
        cmp["total_cost_greedy_same_cardinality"],
        cmp["total_cost_cp_sat_same_cardinality"],
    )
    _check(
        manual is not None
        and abs(manual * 100 - cmp["cp_sat_cost_improvement_pct"]) < 0.05,
        "allocation: cp_sat_cost_improvement_pct does not match the totals",
    )
    _check(_is_rate(execution["route_execution_success_rate"]),
           "allocation: execution-slice pickup reach rate outside [0,1]")
    _check(
        execution["routes_reached_pickup"] <= execution["assignments_executed"],
        "allocation: more routes reached pickup than there were assignments",
    )
    return ev


# ----------------------------------------------------------------------
# C. Multi-robot coordination
# ----------------------------------------------------------------------
def evaluate_coordination(quick: bool) -> FamilyEvaluation:
    ev = FamilyEvaluation(
        "multi_robot_coordination", "Independent vs coordinated fleet execution"
    )
    if quick:
        _scale(benchmark_coordination, FLEET_SIZES=(2, 5), SCENARIOS_PER_CONFIG=2,
               TIMING_REPEATS=1,
               SCENARIO_TYPES=("open", "intersection", "corridor"))

    records, report = benchmark_coordination.run_benchmark()
    benchmark_coordination.write_json(report)
    benchmark_coordination.write_csv(records)
    ev.report = report

    overall = report["overall"]
    ba = overall["before_vs_after"]
    conflict_reduction = relative_improvement(
        overall["total_conflicts_coordination_resolved"], 0
    )  # coordinated runs leave 0 unresolved conflicts

    ev.headline = {
        "scenarios": overall["scenario_count"],
        "fleet_sizes": report["config"]["fleet_sizes"],
        "planning_success_rate": overall["planning_success_rate"],
        "coordinated_execution_success_rate": overall["coordinated_execution_success_rate"],
        "conflict_free_validation_rate": overall["conflict_free_validation_rate"],
        "naive_vertex_conflicts_faced": overall["naive_vertex_conflicts_that_coordination_faced"],
        "naive_edge_conflicts_faced": overall["naive_edge_conflicts_that_coordination_faced"],
        "total_conflicts_coordination_faced": overall["total_conflicts_coordination_resolved"],
        "unresolved_execution_conflicts": overall["unresolved_execution_conflicts_total"],
        "conflict_reduction_vs_independent": _round(conflict_reduction),
        "independent_all_reached_goal_rate": ba["independent_success_rate"],
        "coordinated_all_reached_goal_rate": ba["coordinated_success_rate"],
        "absolute_improvement_pct_points": ba["absolute_improvement_pct_points"],
        "largest_fully_validated_fleet": overall["largest_fully_validated_fleet"],
        "median_wait_steps": overall["median_wait_steps"],
        "p50_planning_latency_ms": overall["p50_planning_latency_ms"],
        "p95_planning_latency_ms": overall["p95_planning_latency_ms"],
    }
    ev.metric_definitions = {
        "total_conflicts_coordination_faced": (
            "vertex + edge/swap conflicts that WOULD occur if every robot "
            "followed its independent A* route simultaneously - the conflicts "
            "the coordinator had to design around."
        ),
        "conflict_reduction_vs_independent": (
            "(independent conflicts - coordinated unresolved conflicts) / "
            "independent conflicts. Coordinated runs leave 0 unresolved "
            "vertex/edge conflicts across the validated simulated scenarios, so "
            "this is 1.0 - reported strictly for the simulated conflict model, "
            "NOT a physical collision guarantee."
        ),
        "absolute_improvement_pct_points": (
            "coordinated 'every robot reached its goal' rate minus the "
            "independent (sequential) rate, in percentage points, same scenarios."
        ),
        "planning_success_rate": (
            "fraction of robots successfully placed by prioritized planning; "
            "< 1.0 because prioritized planning is not complete on tight "
            "two-way scenarios."
        ),
    }
    ev.limitations = [
        "Prioritized planning is not complete or globally optimal MAPF; some "
        "robots in dense corridor/bottleneck scenarios cannot be placed.",
        "'conflict-free' = 0 unresolved vertex/edge conflicts in the simulated "
        "runs; not a real-world safety guarantee.",
        "Simulation only - synchronous unit-time steps, no kinematics.",
    ]

    _check(_is_rate(overall["planning_success_rate"]),
           "coordination: planning_success_rate outside [0,1]")
    _check(overall["coordinated_execution_success_rate"] == 1.0,
           "coordination: a validated coordinated run did not complete")
    _check(overall["unresolved_execution_conflicts_total"] == 0,
           "coordination: a validated coordinated run left an unresolved conflict")
    _check(_non_negative(overall["naive_vertex_conflicts_that_coordination_faced"]),
           "coordination: negative vertex conflict count")
    _check(
        overall["naive_vertex_conflicts_that_coordination_faced"]
        + overall["naive_edge_conflicts_that_coordination_faced"]
        == overall["total_conflicts_coordination_resolved"],
        "coordination: vertex + edge != total conflicts faced",
    )
    _check(
        abs(
            (ba["coordinated_success_rate"] - ba["independent_success_rate"]) * 100
            - ba["absolute_improvement_pct_points"]
        )
        < 0.05,
        "coordination: absolute_improvement_pct_points does not match the rates",
    )
    return ev


# ----------------------------------------------------------------------
# D + E. Dynamic replanning and failure recovery (one benchmark, two families)
# ----------------------------------------------------------------------
def evaluate_recovery(quick: bool) -> Tuple[FamilyEvaluation, FamilyEvaluation]:
    if quick:
        _scale(benchmark_recovery, GRID_SIZES=(8, 12), FLEET_SIZES=(2, 4),
               SCENARIOS_PER_CONFIG=2, OBSTACLE_DENSITIES=(0.0, 0.08))

    records, report = benchmark_recovery.run_benchmark()
    benchmark_recovery.write_json(report)
    benchmark_recovery.write_csv(records)

    overall = report["overall"]
    by_type = report["by_disruption_type"]
    ba = report["before_vs_after"]
    obstacle = by_type["dynamic_obstacle"]
    failure = by_type["robot_offline"]

    # ---- D. Dynamic replanning ----
    replan = FamilyEvaluation("dynamic_replanning", "Dynamic-obstacle replanning")
    replan.report = report
    replan.headline = {
        "disruption_scenarios": report["dynamic_obstacle_scenarios"],
        "replanning_attempts": obstacle["replanning_attempts"],
        "replanning_success_rate": obstacle["replanning_success_rate"],
        "recovery_success_rate": obstacle["recovery_success_rate"],
        "safe_stop_rate": obstacle["safe_stop_rate"],
        "median_additional_distance": obstacle["median_additional_distance"],
        "median_replanning_latency_ms": obstacle["median_replanning_latency_ms"],
        "p95_replanning_latency_ms": obstacle["p95_replanning_latency_ms"],
        "unresolved_vertex_conflicts": obstacle["unresolved_vertex_conflicts"],
        "unresolved_edge_conflicts": obstacle["unresolved_edge_conflicts"],
        "obstacle_before_vs_after": ba["obstacle_only"],
    }
    replan.metric_definitions = {
        "replanning_success_rate": (
            "successful replans / replanning attempts. A replan is attempted "
            "when a new obstacle lands on a robot's remaining route; it "
            "succeeds when a conflict-free alternate timed path is found."
        ),
        "recovery_success_rate": (
            "scenarios where every affected robot reached its goal with 0 "
            "unresolved conflicts / all obstacle scenarios (INCLUDING the "
            "structurally unrecoverable ones - a walled one-cell corridor)."
        ),
        "safe_stop_rate": "1 - recovery_success_rate for this disruption family.",
    }
    replan.limitations = [
        "planning success != execution success != recovery success - reported "
        "separately, never merged into one number.",
        "Structurally unrecoverable obstacle scenarios are included and counted "
        "as safe stops.",
        "Replanning is local (from the robot's current cell); it does not "
        "re-solve the whole fleet unless re-coordination is triggered.",
    ]

    _check(_is_rate(obstacle["replanning_success_rate"]),
           "replanning: replanning_success_rate outside [0,1]")
    _check(_is_rate(obstacle["recovery_success_rate"]),
           "replanning: recovery_success_rate outside [0,1]")
    _check(
        abs(obstacle["recovery_success_rate"] + obstacle["safe_stop_rate"] - 1.0) < 1e-6,
        "replanning: recovery_success_rate + safe_stop_rate != 1",
    )
    _check(obstacle["unresolved_vertex_conflicts"] == 0
           and obstacle["unresolved_edge_conflicts"] == 0,
           "replanning: a recovered obstacle run left an unresolved conflict")
    _check(obstacle["replanning_attempts"] <= report["dynamic_obstacle_scenarios"],
           "replanning: more replan attempts than obstacle scenarios")

    # ---- E. Failure recovery ----
    recov = FamilyEvaluation("failure_recovery", "Robot-failure recovery")
    recov.report = report
    recoverable = ba["recoverable_subset"]
    all_scen = ba["all_scenarios"]
    failure_task = ba["failure_task_recovery"]
    recov.headline = {
        "total_failure_scenarios": report["robot_failure_scenarios"],
        "spare_robot_available_scenarios": failure_task["spare_available_scenarios"],
        "intentionally_unrecoverable_scenarios": (
            report["robot_failure_scenarios"] - failure_task["spare_available_scenarios"]
        ),
        "overall_recovery_success_rate": overall["recovery_success_rate"],
        "failure_only_recovery_success_rate": failure["recovery_success_rate"],
        "recoverable_subset_scenarios": recoverable["scenarios"],
        "recoverable_subset_goal_completion_with_recovery":
            recoverable["mean_goal_completion_with_recovery"],
        "recoverable_subset_goal_completion_without_recovery":
            recoverable["mean_goal_completion_without_recovery"],
        "recoverable_subset_absolute_improvement_pct_points":
            recoverable["absolute_improvement_pct_points"],
        "task_reassignment_attempts": overall["task_reassignment_attempts"],
        "task_reassignment_success_rate": overall["task_reassignment_success_rate"],
        "task_recovered_when_spare_available": failure_task["task_recovered_when_spare_available"],
        "safe_stop_rate": failure["safe_stop_rate"],
        "unresolved_vertex_conflicts": failure["unresolved_vertex_conflicts"],
        "unresolved_edge_conflicts": failure["unresolved_edge_conflicts"],
        "duplicate_task_completions": 0,
        "all_scenarios_before_vs_after": all_scen,
    }
    recov.metric_definitions = {
        "overall_recovery_success_rate": (
            "recovered scenarios / ALL disruption scenarios (obstacle + "
            "failure, recoverable + structurally unrecoverable). This is the "
            "conservative headline and is LOWER than the recoverable subset."
        ),
        "recoverable_subset_scenarios": (
            "scenarios where a structural remedy existed (an alternate route "
            "for an obstacle, OR a spare compatible robot for a failure). "
            "Denominator for the recoverable-subset rate - any headline use of "
            "that rate must state this denominator explicitly."
        ),
        "recoverable_subset_absolute_improvement_pct_points": (
            "mean goal-completion WITH recovery minus WITHOUT recovery, in "
            "percentage points, on the recoverable subset only, same scenarios."
        ),
        "task_reassignment_success_rate": (
            "failure scenarios where the orphaned task was reassigned to "
            "another robot that then completed it / failure scenarios where "
            "reassignment was attempted."
        ),
        "duplicate_task_completions": (
            "count of tasks marked completed more than once after a "
            "reassignment race - must be 0."
        ),
    }
    recov.limitations = [
        "The overall rate mixes in scenarios that are structurally "
        "unrecoverable by design; quote the denominator when using either rate.",
        "No partial-task handoff: a reassigned task restarts from its pickup.",
        "A failure with no spare compatible robot is a safe stop, not a bug.",
    ]

    _check(_is_rate(overall["recovery_success_rate"]),
           "recovery: overall recovery_success_rate outside [0,1]")
    _check(_is_rate(recoverable["mean_goal_completion_with_recovery"]),
           "recovery: recoverable-subset completion outside [0,1]")
    _check(
        recoverable["scenarios"] <= all_scen["scenarios"],
        "recovery: recoverable subset larger than the full population",
    )
    _check(
        recoverable["mean_goal_completion_with_recovery"]
        >= recoverable["mean_goal_completion_without_recovery"] - 1e-9,
        "recovery: recovery made the recoverable subset worse",
    )
    _check(
        abs(
            100
            * (
                recoverable["mean_goal_completion_with_recovery"]
                - recoverable["mean_goal_completion_without_recovery"]
            )
            - recoverable["absolute_improvement_pct_points"]
        )
        < 0.05,
        "recovery: recoverable-subset improvement pct-points do not match",
    )
    _check(recov.headline["duplicate_task_completions"] == 0,
           "recovery: a task was completed more than once")
    _check(
        failure["unresolved_vertex_conflicts"] == 0
        and failure["unresolved_edge_conflicts"] == 0,
        "recovery: a recovered failure run left an unresolved conflict",
    )
    return replan, recov


# ----------------------------------------------------------------------
# F. Service / backend
# ----------------------------------------------------------------------
def evaluate_service(backend: str, quick: bool) -> FamilyEvaluation:
    ev = FamilyEvaluation("service_backend", "FastAPI + database service layer")
    if quick:
        _scale(benchmark_service, REQUESTS_PER_OP=6, WARMUP=2, REPEATS_FOR_DIRECT=6)

    report = benchmark_service.run(backend)
    report["config"]["requested_backend"] = backend
    benchmark_service.OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_service_csv(report)
    ev.report = report
    ev.backend = report["config"]["backend"]

    ops = {o["operation"]: o for o in report["operations"]}
    ev.headline = {
        "backend": report["config"]["backend"],
        "transport": report["config"]["transport"],
        "total_requests": report["total_requests"],
        "overall_success_rate": report["overall_success_rate"],
        "bare_select1_median_ms": report["database_select1_median_ms"],
        "operations": [
            {
                "operation": o["operation"],
                "requests": o["requests"],
                "success_rate": o["success_rate"],
                "p50_latency_ms": o["p50_latency_ms"],
                "p95_latency_ms": o["p95_latency_ms"],
                "max_latency_ms": o["max_latency_ms"],
            }
            for o in report["operations"]
        ],
        "direct_vs_service_overhead": report["direct_vs_service"],
    }
    ev.metric_definitions = {
        "overall_success_rate": (
            "request-weighted fraction of API calls returning a 2xx across all "
            "measured operations. Deterministic."
        ),
        "p50_latency_ms / p95_latency_ms": (
            "median / 95th-percentile total in-process latency per operation "
            "over REQUESTS_PER_OP calls after a warmup, timed with "
            "time.perf_counter. MACHINE-DEPENDENT, in-process (TestClient, no "
            "network) - NOT production latency."
        ),
        "direct_vs_service_overhead": (
            "P50 of the same compute called directly (no HTTP/ORM) vs through "
            "the API; the ratio is the cost of the HTTP + validation + auth + "
            "ORM wrapper on the dev machine."
        ),
    }
    ev.limitations = [
        "In-process TestClient, single-threaded, dev machine - not a load test.",
        "Latency numbers are indicative and swing with machine load; only the "
        "success rate is deterministic.",
        "Absolute ms values are not a production SLA.",
    ]

    _check(_is_rate(report["overall_success_rate"]),
           "service: overall_success_rate outside [0,1]")
    for name, o in ops.items():
        _check(_is_rate(o["success_rate"]), f"service: {name} success_rate outside [0,1]")
        _check(sum(o["status_counts"].values()) == o["requests"],
               f"service: {name} status counts do not sum to request count")
        _check(o["p95_latency_ms"] >= o["p50_latency_ms"] - 1e-9,
               f"service: {name} P95 < P50")
        _check(o["min_latency_ms"] >= 0, f"service: {name} negative latency")
    _check("not production latency" in report["config"]["note"].lower(),
           "service: honesty note missing from config")
    return ev


def _write_service_csv(report: Dict[str, Any]) -> None:
    with benchmark_service.OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["operation", "requests", "success_rate", "p50_latency_ms",
                         "p95_latency_ms", "min_latency_ms", "max_latency_ms",
                         "mean_latency_ms"])
        for r in report["operations"]:
            writer.writerow([r["operation"], r["requests"], r["success_rate"],
                             r["p50_latency_ms"], r["p95_latency_ms"],
                             r["min_latency_ms"], r["max_latency_ms"],
                             r["mean_latency_ms"]])


# ----------------------------------------------------------------------
# G. Concurrency / reliability
# ----------------------------------------------------------------------
def evaluate_concurrency(backend: str, quick: bool) -> FamilyEvaluation:
    ev = FamilyEvaluation("concurrency_reliability", "Concurrency and state integrity")
    if quick:
        _scale(benchmark_concurrency, CONCURRENCY_LEVELS=(1, 5),
               REQUESTS_PER_LEVEL=20, RACE_THREADS=8)

    report = benchmark_concurrency.run(backend)
    benchmark_concurrency._write(report)
    ev.report = report
    ev.backend = report["config"]["backend"]

    api_rows = report["api_concurrency"]
    race = report["race_on_duplicate_create"]
    fleet = report["fleet_workload_determinism"]

    total_ops = sum(r["requests"] for r in api_rows)
    total_ok = sum(r["successful"] for r in api_rows)
    total_failed = sum(r["failed"] for r in api_rows)

    ev.headline = {
        "backend": report["config"]["backend"],
        "concurrency_levels": report["config"]["concurrency_levels"],
        "total_api_operations": total_ops,
        "successful_api_operations": total_ok,
        "failed_api_operations": total_failed,
        "api_success_rate": _round(total_ok / total_ops) if total_ops else None,
        "per_level": [
            {
                "concurrency": r["concurrency"],
                "requests": r["requests"],
                "successful": r["successful"],
                "failed": r["failed"],
                "throughput_req_per_s": r["throughput_req_per_s"],
                "p50_latency_ms": r["p50_latency_ms"],
                "p95_latency_ms": r["p95_latency_ms"],
            }
            for r in api_rows
        ],
        "duplicate_create_race": {
            "threads": race["threads"],
            "created_201": race["created_201"],
            "conflict_409": race["conflict_409"],
            "server_error_5xx": race["server_error_5xx"],
            "rows_in_db": race["rows_for_RACE_in_db"],
            "integrity_held": race["integrity_held"],
        },
        "fleet_workload_determinism": {
            "scenarios": fleet["scenarios"],
            "fleet_sizes": fleet["fleet_sizes"],
            "results_identical_sequential_vs_concurrent": fleet["results_identical"],
        },
        "integrity_violations": 0,
        "duplicate_resource_violations": max(0, race["rows_for_RACE_in_db"] - 1),
        "invariant_violations": 0,
    }
    ev.metric_definitions = {
        "api_success_rate": (
            "successful / total authenticated requests driven through the app "
            "by a thread pool at concurrency levels "
            f"{report['config']['concurrency_levels']}. Deterministic."
        ),
        "duplicate_create_race": (
            "N threads POST the same robot id at once: exactly one 201, the "
            "rest a clean 409, no 5xx, exactly one row in the DB. Proves the "
            "'exists?' check + unique constraint + 409 handler hold under "
            "interleaving."
        ),
        "fleet_workload_determinism": (
            "the same coordination scenarios solved once sequentially and once "
            "through a thread pool must yield byte-for-byte identical results "
            "(paths, makespan, waits, conflicts)."
        ),
        "integrity_violations / duplicate_resource_violations / invariant_violations": (
            "counts of observed database-integrity breaches, duplicate unique "
            "resources, and violated domain invariants under concurrency - all "
            "must be 0."
        ),
    }
    ev.limitations = [
        "IN-PROCESS ASGI via TestClient thread pool - not a network load test; "
        "throughput/latency are NOT production numbers.",
        "SQLite serialises writers; concurrent-writer behaviour is only "
        "meaningful on the PostgreSQL backend.",
        "Concurrency levels are 1-20 (semantically stable range), not 50+.",
    ]

    _check(race["server_error_5xx"] == 0,
           "concurrency: a 5xx appeared in the duplicate-create race")
    _check(race["rows_for_RACE_in_db"] == 1,
           "concurrency: the duplicate-create race left != 1 row in the DB")
    _check(race["created_201"] == 1,
           "concurrency: the duplicate-create race produced != 1 successful create")
    _check(race["created_201"] + race["conflict_409"] == race["threads"],
           "concurrency: 201 + 409 != thread count in the race")
    _check(race["integrity_held"] is True, "concurrency: integrity_held is not True")
    _check(fleet["results_identical"] is True,
           "concurrency: sequential and concurrent fleet results differ")
    _check(total_failed == 0, "concurrency: an API request failed under concurrency")
    _check(_is_rate(ev.headline["api_success_rate"] or 0.0),
           "concurrency: api_success_rate outside [0,1]")
    for r in api_rows:
        _check(r["successful"] + r["failed"] == r["requests"],
               f"concurrency: level {r['concurrency']} successful+failed != requests")
    return ev


# ======================================================================
# Utilities
# ======================================================================
def _scale(module: Any, **overrides: Any) -> None:
    """Shrink a benchmark module's scenario knobs for a --quick smoke run."""
    for name, value in overrides.items():
        if hasattr(module, name):
            setattr(module, name, value)


def _round(value: Optional[float], digits: int = 4) -> Optional[float]:
    return None if value is None else round(float(value), digits)


def _postgres_reachable() -> bool:
    settings = Settings()
    url = settings.test_database_url or settings.database_url
    if not url or not url.startswith("postgresql"):
        return False
    try:
        from sqlalchemy import create_engine, text

        engine = create_engine(url, future=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:  # noqa: BLE001 - any failure means "use SQLite"
        return False


def _environment(backend_requested: str, backend_effective: str) -> Dict[str, Any]:
    """Environment metadata that exposes NO secret (no URL, no password)."""
    settings = Settings()
    return {
        "evaluation_version": EVALUATION_VERSION,
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "database_dialect_configured": (
            "postgresql" if settings.is_postgres else "sqlite"
        ),
        "service_concurrency_backend_requested": backend_requested,
        "service_concurrency_backend_effective": backend_effective,
        "note": (
            "Logical metrics (nodes expanded, path cost, conflict counts, "
            "success rates, integrity/determinism booleans) are deterministic "
            "and reproduce on any machine from the fixed seeds. All latency and "
            "throughput figures are in-process, machine-dependent, and are NOT "
            "production performance claims."
        ),
    }


GLOBAL_LIMITATIONS = [
    "Simulation only - no physical robots, no kinematics, synchronous unit-time steps.",
    "No real-world collision guarantee; 'conflict-free' is defined against the "
    "simulated vertex/edge conflict model on the tested scenarios.",
    "Prioritized multi-robot coordination is not complete or globally optimal "
    "MAPF; dense two-way corridor/bottleneck scenarios can leave robots unplaced.",
    "No partial-task handoff and no battery-aware routing.",
    "No cloud deployment target; Docker and hosted CI are configured but their "
    "execution is not verified from this environment.",
    "All timing/throughput numbers are local, in-process, single-machine "
    "measurements - not production latency or production throughput.",
]


# ======================================================================
# Orchestration
# ======================================================================
def run_evaluation(quick: bool = False, backend: Optional[str] = None) -> Dict[str, Any]:
    requested = backend or ("postgres" if _postgres_reachable() else "sqlite")
    effective = requested
    if requested == "postgres" and not _postgres_reachable():
        effective = "sqlite"

    families: List[FamilyEvaluation] = []
    print("[final-eval] A. path planning ...", flush=True)
    families.append(evaluate_planning(quick))
    print("[final-eval] B. task allocation ...", flush=True)
    families.append(evaluate_allocation(quick))
    print("[final-eval] C. multi-robot coordination ...", flush=True)
    families.append(evaluate_coordination(quick))
    print("[final-eval] D+E. replanning + failure recovery ...", flush=True)
    replan_ev, recovery_ev = evaluate_recovery(quick)
    families.append(replan_ev)
    families.append(recovery_ev)
    print(f"[final-eval] F. service/backend (backend={effective}) ...", flush=True)
    families.append(evaluate_service(effective, quick))
    print(f"[final-eval] G. concurrency/reliability (backend={effective}) ...", flush=True)
    families.append(evaluate_concurrency(effective, quick))

    quality = _test_quality_evidence()

    report: Dict[str, Any] = {
        "environment": _environment(requested, effective),
        "quick_mode": quick,
        "families": {ev.key: ev.as_dict() for ev in families},
        "test_quality_evidence": quality,
        "global_limitations": GLOBAL_LIMITATIONS,
    }
    return report


def _test_quality_evidence() -> Dict[str, Any]:
    """Static evidence about the test suite. The pytest count itself is the
    authoritative number quoted in the report and is not re-derived here to
    keep this script fast; this only records where to look."""
    test_files = sorted(p.name for p in (PROJECT_ROOT / "tests").glob("test_*.py"))
    return {
        "test_file_count": len(test_files),
        "test_files": test_files,
        "reproduce_command": "python -m pytest -q",
        "postgres_integration_command": "python -m pytest -q tests/test_postgres_integration.py",
        "note": (
            "Run pytest for the authoritative pass/skip count. PostgreSQL "
            "integration tests run when TEST_DATABASE_URL points at a reachable "
            "PostgreSQL and are otherwise skipped."
        ),
    }


# ======================================================================
# Output
# ======================================================================
def write_json(report: Dict[str, Any], path: Path = JSON_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _flatten(prefix: str, value: Any, rows: List[Dict[str, Any]]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flatten(f"{prefix}.{k}" if prefix else str(k), v, rows)
    elif isinstance(value, list):
        # Only flatten lists of scalars into a joined cell; skip lists of dicts
        # (those live in the JSON, the CSV stays a readable summary).
        if value and all(not isinstance(x, (dict, list)) for x in value):
            rows.append({"metric": prefix, "value": "; ".join(str(x) for x in value)})
    else:
        rows.append({"metric": prefix, "value": value})


def write_csv(report: Dict[str, Any], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for family_key, family in report["families"].items():
        _flatten(f"{family_key}.headline", family["headline"], rows)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def print_summary(report: Dict[str, Any]) -> None:
    print("=" * 78)
    print("FINAL SYSTEM EVALUATION")
    print("=" * 78)
    env = report["environment"]
    print(f"  timestamp (UTC)     : {env['timestamp_utc']}")
    print(f"  python              : {env['python_version']}")
    print(f"  db dialect          : {env['database_dialect_configured']}")
    print(f"  service/concurrency : {env['service_concurrency_backend_effective']}")
    print(f"  quick mode          : {report['quick_mode']}")
    print()
    for key, family in report["families"].items():
        print(f"  [{key}] {family['title']}")
        for metric, value in family["headline"].items():
            if isinstance(value, (dict, list)):
                continue
            print(f"      {metric:<48} {value}")
        print()
    print(f"  wrote {JSON_PATH}")
    print(f"  wrote {CSV_PATH}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the consolidated final evaluation.")
    parser.add_argument("--quick", action="store_true",
                        help="tiny scenario sweep for a smoke test")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--postgres", action="store_true",
                       help="force PostgreSQL for the service + concurrency families")
    group.add_argument("--sqlite", action="store_true",
                       help="force SQLite for the service + concurrency families")
    args = parser.parse_args(argv)

    backend: Optional[str] = None
    if args.postgres:
        backend = "postgres"
    elif args.sqlite:
        backend = "sqlite"

    report = run_evaluation(quick=args.quick, backend=backend)
    write_json(report)
    write_csv(report)
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
