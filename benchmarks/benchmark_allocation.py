"""Reproducible benchmark: greedy vs CP-SAT task allocation.

WHAT IT MEASURES
----------------
For a range of fleet / task-count configurations, over deterministic random
warehouses, it runs BOTH allocators on the EXACT SAME scenario input (same
grid, same robots, same tasks, same shared cost estimator) and records:

    per run   : assigned / unassigned task counts, feasible / infeasible pair
                counts, total estimated travel cost, allocation latency,
                solver status, and whether the result passed independent
                validation.
    aggregate : median / P50 / P95 allocation latency, assignment rate,
                median total cost, valid-result rate, and - only across
                scenarios where BOTH allocators assigned the SAME number of
                tasks - the CP-SAT vs greedy cost difference.

FAIRNESS
--------
A lower total cost that comes from assigning FEWER tasks is NOT an improvement.
The cost comparison is therefore restricted to scenarios where the two
allocators assigned the same number of tasks; the others are reported
separately, not folded into the headline number.

TIMING
------
Path-cost preprocessing (all the A* calls) is done ONCE up front and excluded
from the per-allocator timing, so both algorithms are measured on a warm cache
and the numbers reflect allocation logic + solver, not planning. Each
allocate() call is repeated and the median wall-clock time (perf_counter_ns) is
kept. On these sizes the times are milliseconds; treat them as indicative.

EXECUTION SLICE
---------------
For one allocator (CP-SAT) each scenario also commits the plan and drives every
assigned robot to its pickup cell with the RouteExecutor, recording planned vs
executed steps, distance and battery consumed. This exercises the full
allocation -> planning -> execution path end to end.

    python benchmarks/benchmark_allocation.py

writes benchmark_allocation_results.json and .csv next to this file.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robotics.allocation.commit import commit_allocation  # noqa: E402
from robotics.allocation.cost_estimator import CostEstimator  # noqa: E402
from robotics.allocation.cp_sat_allocator import CpSatAllocator  # noqa: E402
from robotics.allocation.feasibility import build_candidate_pairs  # noqa: E402
from robotics.allocation.greedy_allocator import GreedyAllocator  # noqa: E402
from robotics.allocation.validation import is_valid_allocation  # noqa: E402
from robotics.execution.route_executor import RouteExecutor  # noqa: E402
from robotics.planning.astar import plan as plan_astar  # noqa: E402
from robotics.robots.robot import Robot  # noqa: E402
from robotics.simulation.simulator import WarehouseSimulator  # noqa: E402
from robotics.tasks.task import Task, TaskPriority  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import Warehouse  # noqa: E402

# ----------------------------------------------------------------------
# Configuration - deterministic and stated up front
# ----------------------------------------------------------------------
GRID_SIZE = 24
OBSTACLE_DENSITY = 0.08
FLEET_TASK_CONFIGS: Tuple[Tuple[int, int], ...] = (
    (5, 5),
    (5, 10),
    (10, 10),
    (10, 20),
    (20, 20),
)
SCENARIOS_PER_CONFIG = 8
TIMING_REPEATS = 7
MASTER_SEED = 8675309

RESULTS_DIR = Path(__file__).resolve().parent
JSON_PATH = RESULTS_DIR / "benchmark_allocation_results.json"
CSV_PATH = RESULTS_DIR / "benchmark_allocation_results.csv"

_PRIORITIES = (
    TaskPriority.LOW,
    TaskPriority.NORMAL,
    TaskPriority.HIGH,
    TaskPriority.URGENT,
)


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
@dataclass
class Scenario:
    scenario_id: str
    robot_count: int
    task_count: int
    warehouse: Warehouse
    robots: List[Robot]
    tasks: List[Task]


def _free_cells(warehouse: Warehouse, rng: Random, count: int) -> List[Position]:
    """`count` distinct traversable cells, drawn deterministically."""
    picked: List[Position] = []
    seen: set = set()
    attempts = 0
    while len(picked) < count and attempts < count * 200:
        attempts += 1
        cell = Position(rng.randrange(GRID_SIZE), rng.randrange(GRID_SIZE))
        if cell in seen or not warehouse.is_traversable(cell):
            continue
        seen.add(cell)
        picked.append(cell)
    return picked


def build_scenario(
    robot_count: int, task_count: int, seed: int, index: int
) -> Scenario:
    """One deterministic allocation scenario.

    Some robots are given a reduced battery or capacity, and tasks span a range
    of weights, so that a realistic fraction of robot-task pairs is infeasible
    and the allocators have a genuine choice to make.
    """
    rng = Random(seed)
    warehouse = Warehouse(width=GRID_SIZE, height=GRID_SIZE, name=f"alloc-{index}")

    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if rng.random() < OBSTACLE_DENSITY:
                warehouse.add_static_obstacle(f"obs-{r}-{c}", Position(r, c))

    cells = _free_cells(warehouse, rng, robot_count + task_count * 2)
    robot_cells = cells[:robot_count]
    task_cells = cells[robot_count:]

    robots: List[Robot] = []
    for i, cell in enumerate(robot_cells):
        # ~1 robot in 5 is low on charge; ~1 in 5 has a small payload limit.
        battery = 12.0 if rng.random() < 0.2 else 100.0
        capacity = 4.0 if rng.random() < 0.2 else 15.0
        robots.append(
            Robot(
                robot_id=f"R{i:02d}",
                position=cell,
                battery_level=battery,
                payload_capacity=capacity,
            )
        )

    tasks: List[Task] = []
    for i in range(task_count):
        pickup = task_cells[2 * i]
        dropoff = task_cells[2 * i + 1]
        tasks.append(
            Task(
                task_id=f"T{i:02d}",
                pickup_location=pickup,
                dropoff_location=dropoff,
                priority=_PRIORITIES[rng.randrange(len(_PRIORITIES))],
                payload_weight=float(rng.choice([1.0, 2.0, 5.0, 8.0])),
            )
        )

    return Scenario(
        scenario_id=f"{robot_count}r-{task_count}t-s{index:02d}",
        robot_count=robot_count,
        task_count=task_count,
        warehouse=warehouse,
        robots=robots,
        tasks=tasks,
    )


def generate_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []
    seed_stream = Random(MASTER_SEED)
    for robot_count, task_count in FLEET_TASK_CONFIGS:
        for index in range(SCENARIOS_PER_CONFIG):
            scenarios.append(
                build_scenario(
                    robot_count=robot_count,
                    task_count=task_count,
                    seed=seed_stream.randrange(1, 2**31),
                    index=index,
                )
            )
    return scenarios


# ----------------------------------------------------------------------
# Running one scenario
# ----------------------------------------------------------------------
@dataclass
class RunRecord:
    algorithm: str
    scenario_id: str
    robot_count: int
    task_count: int
    assigned_task_count: int
    unassigned_task_count: int
    eligible_pair_count: int
    infeasible_pair_count: int
    total_estimated_travel_cost: int
    allocation_time_ms: float
    solver_status: str
    valid: bool

    def to_row(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "scenario_id": self.scenario_id,
            "robot_count": self.robot_count,
            "task_count": self.task_count,
            "assigned_task_count": self.assigned_task_count,
            "unassigned_task_count": self.unassigned_task_count,
            "eligible_pair_count": self.eligible_pair_count,
            "infeasible_pair_count": self.infeasible_pair_count,
            "total_estimated_travel_cost": self.total_estimated_travel_cost,
            "allocation_time_ms": round(self.allocation_time_ms, 6),
            "solver_status": self.solver_status,
            "valid": self.valid,
        }


@dataclass
class ExecutionRecord:
    scenario_id: str
    assignments_executed: int
    routes_reached_pickup: int
    planned_steps: int
    executed_steps: int
    distance_travelled: float
    battery_consumed: float

    def to_row(self) -> Dict[str, object]:
        return {
            "scenario_id": self.scenario_id,
            "assignments_executed": self.assignments_executed,
            "routes_reached_pickup": self.routes_reached_pickup,
            "planned_steps": self.planned_steps,
            "executed_steps": self.executed_steps,
            "distance_travelled": round(self.distance_travelled, 4),
            "battery_consumed": round(self.battery_consumed, 4),
        }


def _median_time_ms(allocator, robots, tasks, repeats: int) -> Tuple[object, float]:
    times: List[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = allocator.allocate(robots, tasks)
        times.append((time.perf_counter_ns() - started) / 1_000_000)
    return result, statistics.median(times)


def run_scenario(scenario: Scenario, repeats: int = TIMING_REPEATS):
    """Run both allocators on one scenario. Returns (run records, exec record)."""
    estimator = CostEstimator(scenario.warehouse)

    # Warm the cache up front so the per-allocator timing excludes planning.
    feasible, infeasible = build_candidate_pairs(
        scenario.robots, scenario.tasks, estimator
    )

    records: List[RunRecord] = []
    results: Dict[str, object] = {}

    for name, allocator in (
        ("greedy", GreedyAllocator(estimator)),
        ("cp_sat", CpSatAllocator(estimator)),
    ):
        result, median_ms = _median_time_ms(
            allocator, scenario.robots, scenario.tasks, repeats
        )
        results[name] = result
        valid = is_valid_allocation(
            result, scenario.robots, scenario.tasks, CostEstimator(scenario.warehouse)
        )
        records.append(
            RunRecord(
                algorithm=name,
                scenario_id=scenario.scenario_id,
                robot_count=scenario.robot_count,
                task_count=scenario.task_count,
                assigned_task_count=result.assigned_task_count,
                unassigned_task_count=result.unassigned_task_count,
                eligible_pair_count=result.eligible_pair_count,
                infeasible_pair_count=result.infeasible_pair_count,
                total_estimated_travel_cost=result.total_estimated_cost,
                allocation_time_ms=median_ms,
                solver_status=result.solver_status,
                valid=valid,
            )
        )

    exec_record = _run_execution_slice(scenario, results["cp_sat"])
    return records, exec_record


def _run_execution_slice(scenario: Scenario, cp_sat_result) -> ExecutionRecord:
    """Commit the CP-SAT plan and drive each assigned robot to its pickup."""
    sim = WarehouseSimulator(scenario.warehouse)
    live_robots = {}
    for robot in scenario.robots:
        clone = Robot(
            robot_id=robot.robot_id,
            position=robot.position,
            battery_level=robot.battery_level,
            payload_capacity=robot.payload_capacity,
        )
        sim.add_robot(clone)
        live_robots[robot.robot_id] = clone
    for task in scenario.tasks:
        sim.add_task(
            Task(
                task_id=task.task_id,
                pickup_location=task.pickup_location,
                dropoff_location=task.dropoff_location,
                priority=task.priority,
                payload_weight=task.payload_weight,
            )
        )

    commit_allocation(sim, cp_sat_result)
    executor = RouteExecutor(sim)

    reached = 0
    planned_steps = 0
    executed_steps = 0
    distance = 0.0
    battery = 0.0

    for assignment in cp_sat_result.assignments:
        robot = sim.get_robot(assignment.robot_id)
        task = sim.get_task(assignment.task_id)
        route = plan_astar(robot.position, task.pickup_location, scenario.warehouse)
        execution = executor.execute(assignment.robot_id, route)
        planned_steps += execution.planned_steps
        executed_steps += execution.executed_steps
        distance += execution.distance_travelled
        battery += execution.battery_consumed
        if execution.success:
            reached += 1

    return ExecutionRecord(
        scenario_id=scenario.scenario_id,
        assignments_executed=len(cp_sat_result.assignments),
        routes_reached_pickup=reached,
        planned_steps=planned_steps,
        executed_steps=executed_steps,
        distance_travelled=distance,
        battery_consumed=battery,
    )


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return float(ordered[rank - 1])


def aggregate_by_algorithm(records: Sequence[RunRecord]) -> List[Dict[str, object]]:
    groups: Dict[str, List[RunRecord]] = {}
    for record in records:
        groups.setdefault(record.algorithm, []).append(record)

    summaries: List[Dict[str, object]] = []
    for algorithm, group in sorted(groups.items()):
        times = [r.allocation_time_ms for r in group]
        assigned = sum(r.assigned_task_count for r in group)
        candidates = sum(
            r.assigned_task_count + r.unassigned_task_count for r in group
        )
        summaries.append(
            {
                "algorithm": algorithm,
                "scenarios": len(group),
                "assignment_rate": round(assigned / candidates, 4) if candidates else 0.0,
                "median_allocation_time_ms": round(statistics.median(times), 6),
                "p50_allocation_time_ms": round(percentile(times, 50), 6),
                "p95_allocation_time_ms": round(percentile(times, 95), 6),
                "median_total_estimated_cost": statistics.median(
                    [r.total_estimated_travel_cost for r in group]
                ),
                "valid_result_rate": round(
                    sum(1 for r in group if r.valid) / len(group), 4
                ),
                "success_rate": round(
                    sum(1 for r in group if r.solver_status not in ("UNKNOWN", "MODEL_INVALID"))
                    / len(group),
                    4,
                ),
            }
        )
    return summaries


def compare_cp_sat_vs_greedy(records: Sequence[RunRecord]) -> Dict[str, object]:
    by_scenario: Dict[str, Dict[str, RunRecord]] = {}
    for record in records:
        by_scenario.setdefault(record.scenario_id, {})[record.algorithm] = record

    same_cardinality: List[Tuple[int, int]] = []  # (greedy_cost, cp_sat_cost)
    cp_sat_assigned_more = 0
    greedy_assigned_more = 0

    for group in by_scenario.values():
        if "greedy" not in group or "cp_sat" not in group:
            continue
        g, c = group["greedy"], group["cp_sat"]
        if c.assigned_task_count > g.assigned_task_count:
            cp_sat_assigned_more += 1
        elif g.assigned_task_count > c.assigned_task_count:
            greedy_assigned_more += 1
        else:
            same_cardinality.append(
                (g.total_estimated_travel_cost, c.total_estimated_travel_cost)
            )

    cp_sat_strictly_cheaper = sum(1 for g, c in same_cardinality if c < g)
    equal_cost = sum(1 for g, c in same_cardinality if c == g)
    cp_sat_worse = sum(1 for g, c in same_cardinality if c > g)

    total_greedy = sum(g for g, _ in same_cardinality)
    total_cp_sat = sum(c for _, c in same_cardinality)
    pct_improvement = (
        round(100 * (total_greedy - total_cp_sat) / total_greedy, 4)
        if total_greedy
        else 0.0
    )
    per_scenario_pct = [
        100 * (g - c) / g for g, c in same_cardinality if g > 0
    ]

    return {
        "scenarios_compared_same_cardinality": len(same_cardinality),
        "scenarios_cp_sat_assigned_more_tasks": cp_sat_assigned_more,
        "scenarios_greedy_assigned_more_tasks": greedy_assigned_more,
        "cp_sat_strictly_cheaper": cp_sat_strictly_cheaper,
        "equal_cost": equal_cost,
        "cp_sat_worse": cp_sat_worse,
        "total_cost_greedy_same_cardinality": total_greedy,
        "total_cost_cp_sat_same_cardinality": total_cp_sat,
        "cp_sat_cost_improvement_pct": pct_improvement,
        "median_per_scenario_improvement_pct": (
            round(statistics.median(per_scenario_pct), 4) if per_scenario_pct else 0.0
        ),
    }


def aggregate_execution(records: Sequence[ExecutionRecord]) -> Dict[str, object]:
    total_assignments = sum(r.assignments_executed for r in records)
    total_reached = sum(r.routes_reached_pickup for r in records)
    total_planned = sum(r.planned_steps for r in records)
    total_executed = sum(r.executed_steps for r in records)
    return {
        "scenarios": len(records),
        "assignments_executed": total_assignments,
        "routes_reached_pickup": total_reached,
        "route_execution_success_rate": (
            round(total_reached / total_assignments, 4) if total_assignments else 0.0
        ),
        "total_planned_steps": total_planned,
        "total_executed_steps": total_executed,
        "planned_vs_executed_step_ratio": (
            round(total_executed / total_planned, 4) if total_planned else 0.0
        ),
        "total_distance_travelled": round(
            sum(r.distance_travelled for r in records), 4
        ),
        "total_battery_consumed": round(
            sum(r.battery_consumed for r in records), 4
        ),
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def build_report(
    records: Sequence[RunRecord], exec_records: Sequence[ExecutionRecord]
) -> Dict[str, object]:
    return {
        "config": {
            "grid_size": GRID_SIZE,
            "obstacle_density": OBSTACLE_DENSITY,
            "fleet_task_configs": [list(c) for c in FLEET_TASK_CONFIGS],
            "scenarios_per_config": SCENARIOS_PER_CONFIG,
            "timing_repeats": TIMING_REPEATS,
            "master_seed": MASTER_SEED,
            "timer": "time.perf_counter_ns",
            "note": (
                "Path-cost preprocessing is done once per scenario and excluded "
                "from the per-allocator timing. Cost comparison is restricted to "
                "scenarios where both allocators assigned the same number of tasks."
            ),
        },
        "total_scenarios": len(exec_records),
        "total_runs": len(records),
        "raw_runs": [r.to_row() for r in records],
        "raw_execution": [r.to_row() for r in exec_records],
        "aggregates_by_algorithm": aggregate_by_algorithm(records),
        "cp_sat_vs_greedy": compare_cp_sat_vs_greedy(records),
        "execution_slice": aggregate_execution(exec_records),
    }


def write_json(report: Dict[str, object], path: Path = JSON_PATH) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv(records: Sequence[RunRecord], path: Path = CSV_PATH) -> None:
    fieldnames = list(records[0].to_row().keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def print_summary(report: Dict[str, object]) -> None:
    print("=" * 78)
    print("TASK ALLOCATION BENCHMARK  (greedy vs CP-SAT)")
    print("=" * 78)
    cfg = report["config"]
    print(f"  fleet/task configs : {cfg['fleet_task_configs']}")
    print(f"  scenarios / config : {cfg['scenarios_per_config']}")
    print(f"  grid               : {cfg['grid_size']}x{cfg['grid_size']}, "
          f"{cfg['obstacle_density']:.0%} obstacles")
    print(f"  total scenarios    : {report['total_scenarios']}")
    print(f"  total allocator runs: {report['total_runs']}")
    print()
    header = (
        f"  {'algorithm':<9} {'scen':>5} {'assign%':>8} {'p50 ms':>10} "
        f"{'p95 ms':>10} {'med cost':>9} {'valid%':>7}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in report["aggregates_by_algorithm"]:
        print(
            f"  {row['algorithm']:<9} {row['scenarios']:>5} "
            f"{row['assignment_rate'] * 100:>7.1f}% "
            f"{row['p50_allocation_time_ms']:>10.4f} "
            f"{row['p95_allocation_time_ms']:>10.4f} "
            f"{row['median_total_estimated_cost']:>9} "
            f"{row['valid_result_rate'] * 100:>6.1f}%"
        )
    print()
    cmp = report["cp_sat_vs_greedy"]
    print("  CP-SAT vs greedy (fair cost comparison):")
    print(f"    scenarios, same #tasks assigned : {cmp['scenarios_compared_same_cardinality']}")
    print(f"    CP-SAT assigned MORE tasks       : {cmp['scenarios_cp_sat_assigned_more_tasks']}")
    print(f"    greedy assigned MORE tasks       : {cmp['scenarios_greedy_assigned_more_tasks']}")
    print(f"    of the equal-count scenarios:")
    print(f"      CP-SAT strictly cheaper        : {cmp['cp_sat_strictly_cheaper']}")
    print(f"      equal cost                     : {cmp['equal_cost']}")
    print(f"      CP-SAT worse                   : {cmp['cp_sat_worse']}")
    print(f"    total travel cost, equal-count scenarios:")
    print(f"      greedy = {cmp['total_cost_greedy_same_cardinality']}, "
          f"CP-SAT = {cmp['total_cost_cp_sat_same_cardinality']}")
    print(f"    CP-SAT cost improvement          : {cmp['cp_sat_cost_improvement_pct']}%")
    print(f"    median per-scenario improvement  : {cmp['median_per_scenario_improvement_pct']}%")
    print()
    ex = report["execution_slice"]
    print("  Allocation -> planning -> execution slice (CP-SAT plans, driven to pickup):")
    print(f"    assignments executed            : {ex['assignments_executed']}")
    print(f"    routes that reached the pickup  : {ex['routes_reached_pickup']} "
          f"({ex['route_execution_success_rate'] * 100:.1f}%)")
    print(f"    planned steps / executed steps  : {ex['total_planned_steps']} / "
          f"{ex['total_executed_steps']}")
    print(f"    total distance / battery used   : {ex['total_distance_travelled']} / "
          f"{ex['total_battery_consumed']:.1f}%")
    print()
    print(f"  wrote {JSON_PATH.name} and {CSV_PATH.name} to {RESULTS_DIR}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_benchmark():
    scenarios = generate_scenarios()
    records: List[RunRecord] = []
    exec_records: List[ExecutionRecord] = []
    for scenario in scenarios:
        run_records, exec_record = run_scenario(scenario)
        records.extend(run_records)
        exec_records.append(exec_record)
    report = build_report(records, exec_records)
    return records, exec_records, report


def main() -> None:
    records, _exec_records, report = run_benchmark()
    write_json(report)
    write_csv(records)
    print_summary(report)


if __name__ == "__main__":
    main()
