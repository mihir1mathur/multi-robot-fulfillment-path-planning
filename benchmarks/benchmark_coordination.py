"""Reproducible benchmark: independent vs coordinated multi-robot execution.

WHAT IT MEASURES
----------------
For a range of fleet sizes and scenario shapes, over deterministic layouts, it
runs each scenario TWO ways:

    INDEPENDENT   each robot's route is planned with plain A* (ignoring the
                  others) and the robots are driven one after another - the
                  behaviour before this coordination layer existed. A robot
                  whose way is blocked by a parked robot simply stops.

    COORDINATED   the MultiRobotCoordinator plans conflict-free timed paths
                  with prioritized Space-Time A*, and the CoordinatedExecutor
                  drives the fleet forward one synchronised timestep at a time.

It also counts, for each scenario, how many vertex/edge conflicts WOULD occur
if the independent routes were followed simultaneously - the conflicts that
coordination had to resolve.

FAIRNESS
--------
Both modes get the EXACT SAME warehouse, starts and goals. The coordinated
"success" for the before/after comparison means "every robot reached its goal";
the independent "success" means the same thing for the sequential run.

HONEST FRAMING
--------------
Prioritized planning is NOT globally optimal. On tight scenarios (a narrow
corridor, a busy bottleneck) some robots cannot be placed at all, and the
benchmark reports that rather than hiding it. "Conflict-free" is reported
specifically as "0 unresolved vertex/edge conflicts across the validated runs",
not as a real-world safety guarantee.

    python benchmarks/benchmark_coordination.py

writes benchmark_coordination_results.json / .csv next to this file.
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

from robotics.coordination.conflicts import find_coordination_problems  # noqa: E402
from robotics.coordination.coordinated_executor import CoordinatedExecutor  # noqa: E402
from robotics.coordination.coordinator import MultiRobotCoordinator  # noqa: E402
from robotics.coordination.timed_path import TimedPath, TimedStep  # noqa: E402
from robotics.execution.route_executor import RouteExecutor  # noqa: E402
from robotics.planning.astar import plan as plan_astar  # noqa: E402
from robotics.robots.robot import Robot  # noqa: E402
from robotics.simulation.simulator import WarehouseSimulator  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import Warehouse  # noqa: E402

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
SCENARIO_TYPES = ("open", "intersection", "corridor", "bottleneck", "obstacle_rich", "mixed")
FLEET_SIZES = (2, 5, 10, 20)
SCENARIOS_PER_CONFIG = 5
TIMING_REPEATS = 3
MASTER_SEED = 424242

RESULTS_DIR = Path(__file__).resolve().parent
JSON_PATH = RESULTS_DIR / "benchmark_coordination_results.json"
CSV_PATH = RESULTS_DIR / "benchmark_coordination_results.csv"


def _grid_size(fleet: int) -> int:
    return {2: 8, 5: 10, 10: 14, 20: 20}.get(fleet, 8)


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
@dataclass
class Scenario:
    scenario_id: str
    scenario_type: str
    robot_count: int
    grid_size: int
    warehouse: Warehouse
    starts: Dict[str, Position]
    goals: Dict[str, Position]


def _distinct_free_cells(wh: Warehouse, rng: Random, n: int) -> List[Position]:
    out: List[Position] = []
    seen: set = set()
    tries = 0
    size = wh.width
    while len(out) < n and tries < n * 400:
        tries += 1
        cell = Position(rng.randrange(size), rng.randrange(size))
        if cell in seen or not wh.is_traversable(cell):
            continue
        seen.add(cell)
        out.append(cell)
    return out


def _rid(i: int) -> str:
    return f"R{i:02d}"


def build_scenario(
    scenario_type: str, robot_count: int, seed: int, index: int
) -> Optional[Scenario]:
    rng = Random(seed)
    size = _grid_size(robot_count)
    wh = Warehouse(width=size, height=size, name=f"{scenario_type}-{index}")

    if scenario_type == "obstacle_rich":
        for r in range(size):
            for c in range(size):
                if rng.random() < 0.15:
                    wh.add_static_obstacle(f"o-{r}-{c}", Position(r, c))
    elif scenario_type == "mixed":
        for r in range(size):
            for c in range(size):
                if rng.random() < 0.07:
                    wh.add_static_obstacle(f"o-{r}-{c}", Position(r, c))
    elif scenario_type == "corridor":
        # keep only a 2-wide horizontal band open
        band = size // 2
        for r in range(size):
            if r in (band, band - 1):
                continue
            for c in range(size):
                wh.add_static_obstacle(f"o-{r}-{c}", Position(r, c))
    elif scenario_type == "bottleneck":
        door = size // 2
        for r in range(size):
            if r != door:
                wh.add_static_obstacle(f"wall-{r}", Position(r, door))

    starts: Dict[str, Position] = {}
    goals: Dict[str, Position] = {}

    if scenario_type == "intersection":
        # half the robots go left->right on distinct rows, half top->bottom on
        # distinct columns; their bands overlap in the middle.
        half = max(1, robot_count // 2)
        rows = list(range(size))
        cols = list(range(size))
        rng.shuffle(rows)
        rng.shuffle(cols)
        for i in range(robot_count):
            if i < half:
                row = rows[i % size]
                starts[_rid(i)] = Position(row, 0)
                goals[_rid(i)] = Position(row, size - 1)
            else:
                col = cols[i % size]
                starts[_rid(i)] = Position(0, col)
                goals[_rid(i)] = Position(size - 1, col)
        # de-duplicate starts by nudging
        if len(set(starts.values())) != robot_count:
            return None
    elif scenario_type in ("corridor", "bottleneck"):
        cells = _distinct_free_cells(wh, rng, robot_count * 2)
        if len(cells) < robot_count * 2:
            return None
        for i in range(robot_count):
            starts[_rid(i)] = cells[i]
            goals[_rid(i)] = cells[robot_count + i]
    else:  # open, obstacle_rich, mixed
        cells = _distinct_free_cells(wh, rng, robot_count * 2)
        if len(cells) < robot_count * 2:
            return None
        for i in range(robot_count):
            starts[_rid(i)] = cells[i]
            goals[_rid(i)] = cells[robot_count + i]

    if len(set(starts.values())) != robot_count or len(set(goals.values())) != robot_count:
        return None

    return Scenario(
        scenario_id=f"{scenario_type}-{robot_count}r-s{index:02d}",
        scenario_type=scenario_type,
        robot_count=robot_count,
        grid_size=size,
        warehouse=wh,
        starts=starts,
        goals=goals,
    )


def generate_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []
    seeds = Random(MASTER_SEED)
    for scenario_type in SCENARIO_TYPES:
        for fleet in FLEET_SIZES:
            for index in range(SCENARIOS_PER_CONFIG):
                scenario = build_scenario(
                    scenario_type, fleet, seeds.randrange(1, 2**31), index
                )
                if scenario is not None:
                    scenarios.append(scenario)
    return scenarios


# ----------------------------------------------------------------------
# Running one scenario
# ----------------------------------------------------------------------
@dataclass
class RunRecord:
    scenario_id: str
    scenario_type: str
    robot_count: int
    grid_size: int
    planning_success: bool
    coordinated_robot_count: int
    failed_robot_count: int
    coordinated_execution_success: bool
    coordinated_robots_reached_goal: int
    independent_robots_reached_goal: int
    total_move_steps: int
    total_wait_steps: int
    makespan: int
    total_space_time_cost: int
    naive_vertex_conflicts: int
    naive_edge_conflicts: int
    execution_vertex_conflicts: int
    execution_edge_conflicts: int
    validation_success: bool
    planning_time_ms: float
    execution_time_ms: float
    distance_overhead: int
    failure_reason: str

    def to_row(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["planning_time_ms"] = round(self.planning_time_ms, 6)
        d["execution_time_ms"] = round(self.execution_time_ms, 6)
        return d


def _place(wh: Warehouse, starts: Dict[str, Position]) -> WarehouseSimulator:
    sim = WarehouseSimulator(wh)
    for robot_id, cell in sorted(starts.items()):
        sim.add_robot(Robot(robot_id, cell))
    return sim


def _independent_routes(scenario: Scenario) -> Dict[str, object]:
    return {
        rid: plan_astar(scenario.starts[rid], scenario.goals[rid], scenario.warehouse)
        for rid in scenario.starts
    }


def _independent_sequential_success(scenario: Scenario, routes) -> int:
    """Drive robots one after another with RouteExecutor. Returns #reached goal."""
    sim = _place(scenario.warehouse, scenario.starts)
    executor = RouteExecutor(sim)
    reached = 0
    for rid in sorted(scenario.starts):
        route = routes[rid]
        if not route.success:
            continue
        execution = executor.execute(rid, route)
        if execution.success:
            reached += 1
    return reached


def _naive_conflicts(scenario: Scenario, routes) -> Tuple[int, int]:
    """Vertex/edge conflicts if the independent routes were run simultaneously."""
    timed: Dict[str, TimedPath] = {}
    for rid, route in routes.items():
        if not route.success:
            continue
        steps = [TimedStep(cell, t) for t, cell in enumerate(route.path)]
        timed[rid] = TimedPath.found(rid, steps)
    problems = find_coordination_problems(
        timed, scenario.warehouse, check_obstacles=False
    )
    vc = sum(1 for p in problems if p.startswith("vertex conflict"))
    ec = sum(1 for p in problems if p.startswith("edge/swap conflict"))
    return vc, ec


def _independent_optimal_distance(routes) -> int:
    return sum(
        int(r.total_cost) for r in routes.values() if r.success and r.total_cost
    )


def run_scenario(scenario: Scenario, repeats: int = TIMING_REPEATS) -> RunRecord:
    routes = _independent_routes(scenario)
    naive_vc, naive_ec = _naive_conflicts(scenario, routes)
    independent_reached = _independent_sequential_success(scenario, routes)
    independent_distance = _independent_optimal_distance(routes)

    coordinator = MultiRobotCoordinator(scenario.warehouse)
    plan_times: List[float] = []
    coordination = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        coordination = coordinator.plan(scenario.starts, scenario.goals)
        plan_times.append((time.perf_counter_ns() - started) / 1_000_000)
    planning_time_ms = statistics.median(plan_times)

    validation_ok = not find_coordination_problems(
        coordination.successful_paths, scenario.warehouse, scenario.starts
    )

    sim = _place(scenario.warehouse, scenario.starts)
    exec_started = time.perf_counter_ns()
    execution = CoordinatedExecutor(sim).execute(coordination)
    execution_time_ms = (time.perf_counter_ns() - exec_started) / 1_000_000

    coordinated_distance = execution.total_distance

    return RunRecord(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        robot_count=scenario.robot_count,
        grid_size=scenario.grid_size,
        planning_success=coordination.success,
        coordinated_robot_count=coordination.planned_robot_count,
        failed_robot_count=len(coordination.failed_robot_ids),
        coordinated_execution_success=execution.success,
        coordinated_robots_reached_goal=execution.robots_reached_goal,
        independent_robots_reached_goal=independent_reached,
        total_move_steps=coordination.total_move_steps,
        total_wait_steps=coordination.total_wait_steps,
        makespan=coordination.makespan,
        total_space_time_cost=coordination.total_space_time_cost,
        naive_vertex_conflicts=naive_vc,
        naive_edge_conflicts=naive_ec,
        execution_vertex_conflicts=execution.vertex_conflicts,
        execution_edge_conflicts=execution.edge_conflicts,
        validation_success=validation_ok,
        planning_time_ms=planning_time_ms,
        execution_time_ms=execution_time_ms,
        distance_overhead=int(coordinated_distance) - independent_distance
        if coordination.success
        else 0,
        failure_reason=(coordination.failure_reason or "") if not coordination.success else "",
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


def aggregate(records: Sequence[RunRecord]) -> Dict[str, object]:
    n = len(records)
    plan_ok = [r for r in records if r.planning_success]
    exec_ok = [r for r in records if r.coordinated_execution_success]
    validated = [r for r in records if r.validation_success]

    plan_times = [r.planning_time_ms for r in records]
    waits = [r.total_wait_steps for r in plan_ok]
    makespans = [r.makespan for r in plan_ok]

    total_naive_vertex = sum(r.naive_vertex_conflicts for r in records)
    total_naive_edge = sum(r.naive_edge_conflicts for r in records)
    total_exec_conflicts = sum(
        r.execution_vertex_conflicts + r.execution_edge_conflicts for r in records
    )

    fully_coordinated = [r for r in records if r.planning_success]
    largest_validated = max(
        (r.robot_count for r in records if r.validation_success and r.planning_success),
        default=0,
    )

    # before / after, per scenario (all robots reaching their goal)
    ind_success = sum(
        1
        for r in records
        if r.independent_robots_reached_goal == r.robot_count
    )
    coord_success = sum(
        1
        for r in records
        if r.coordinated_robots_reached_goal == r.robot_count
    )

    return {
        "scenario_count": n,
        "planning_success_rate": round(len(plan_ok) / n, 4) if n else 0.0,
        "coordinated_execution_success_rate": round(len(exec_ok) / n, 4) if n else 0.0,
        "conflict_free_validation_rate": round(len(validated) / n, 4) if n else 0.0,
        "unresolved_execution_conflicts_total": total_exec_conflicts,
        "naive_vertex_conflicts_that_coordination_faced": total_naive_vertex,
        "naive_edge_conflicts_that_coordination_faced": total_naive_edge,
        "total_conflicts_coordination_resolved": total_naive_vertex + total_naive_edge,
        "median_wait_steps": statistics.median(waits) if waits else 0,
        "p95_wait_steps": percentile(waits, 95),
        "median_makespan": statistics.median(makespans) if makespans else 0,
        "p50_planning_latency_ms": round(percentile(plan_times, 50), 6),
        "p95_planning_latency_ms": round(percentile(plan_times, 95), 6),
        "median_distance_overhead": statistics.median(
            [r.distance_overhead for r in fully_coordinated]
        ) if fully_coordinated else 0,
        "median_temporal_overhead_wait_steps": statistics.median(waits) if waits else 0,
        "largest_fully_validated_fleet": largest_validated,
        "before_vs_after": {
            "scenarios": n,
            "independent_all_reached_goal": ind_success,
            "coordinated_all_reached_goal": coord_success,
            "independent_success_rate": round(ind_success / n, 4) if n else 0.0,
            "coordinated_success_rate": round(coord_success / n, 4) if n else 0.0,
            "absolute_improvement_pct_points": round(
                100 * (coord_success - ind_success) / n, 2
            ) if n else 0.0,
        },
    }


def aggregate_by_type_and_size(records: Sequence[RunRecord]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, int], List[RunRecord]] = {}
    for r in records:
        groups.setdefault((r.scenario_type, r.robot_count), []).append(r)
    out = []
    for (scenario_type, fleet), group in sorted(groups.items()):
        g = len(group)
        out.append({
            "scenario_type": scenario_type,
            "robot_count": fleet,
            "scenarios": g,
            "planning_success_rate": round(
                sum(1 for r in group if r.planning_success) / g, 4
            ),
            "coordinated_all_reached_rate": round(
                sum(1 for r in group if r.coordinated_robots_reached_goal == r.robot_count) / g, 4
            ),
            "independent_all_reached_rate": round(
                sum(1 for r in group if r.independent_robots_reached_goal == r.robot_count) / g, 4
            ),
            "median_wait_steps": statistics.median([r.total_wait_steps for r in group]),
            "median_makespan": statistics.median([r.makespan for r in group]),
            "median_planning_ms": round(
                statistics.median([r.planning_time_ms for r in group]), 4
            ),
        })
    return out


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def build_report(records: Sequence[RunRecord]) -> Dict[str, object]:
    return {
        "config": {
            "scenario_types": list(SCENARIO_TYPES),
            "fleet_sizes": list(FLEET_SIZES),
            "scenarios_per_config": SCENARIOS_PER_CONFIG,
            "timing_repeats": TIMING_REPEATS,
            "master_seed": MASTER_SEED,
            "timer": "time.perf_counter_ns",
            "note": (
                "Prioritized planning is not globally optimal; on tight scenarios "
                "some robots cannot be placed and that is reported, not hidden. "
                "'conflict-free' means 0 unresolved vertex/edge conflicts across "
                "the validated runs."
            ),
        },
        "total_scenarios": len(records),
        "overall": aggregate(records),
        "by_type_and_size": aggregate_by_type_and_size(records),
        "raw_runs": [r.to_row() for r in records],
    }


def write_json(report: Dict[str, object], path: Path = JSON_PATH) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv(records: Sequence[RunRecord], path: Path = CSV_PATH) -> None:
    if not records:
        return
    fieldnames = list(records[0].to_row().keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def print_summary(report: Dict[str, object]) -> None:
    print("=" * 78)
    print("MULTI-ROBOT COORDINATION BENCHMARK  (independent vs coordinated)")
    print("=" * 78)
    cfg = report["config"]
    print(f"  scenario types : {cfg['scenario_types']}")
    print(f"  fleet sizes    : {cfg['fleet_sizes']}")
    print(f"  scenarios/config: {cfg['scenarios_per_config']}")
    print(f"  total scenarios : {report['total_scenarios']}")
    print()
    o = report["overall"]
    print(f"  planning success rate            : {o['planning_success_rate'] * 100:.1f}%")
    print(f"  coordinated execution success    : {o['coordinated_execution_success_rate'] * 100:.1f}%")
    print(f"  conflict-free validation rate    : {o['conflict_free_validation_rate'] * 100:.1f}%")
    print(f"  UNRESOLVED conflicts in execution: {o['unresolved_execution_conflicts_total']}")
    print(f"  conflicts coordination resolved  : {o['total_conflicts_coordination_resolved']} "
          f"({o['naive_vertex_conflicts_that_coordination_faced']} vertex + "
          f"{o['naive_edge_conflicts_that_coordination_faced']} edge, "
          f"vs naive simultaneous independent routes)")
    print(f"  median / P95 wait steps          : {o['median_wait_steps']} / {o['p95_wait_steps']}")
    print(f"  median makespan                  : {o['median_makespan']}")
    print(f"  P50 / P95 planning latency       : {o['p50_planning_latency_ms']:.3f} / "
          f"{o['p95_planning_latency_ms']:.3f} ms")
    print(f"  median distance overhead (moves) : {o['median_distance_overhead']}")
    print(f"  largest fully-validated fleet     : {o['largest_fully_validated_fleet']} robots")
    print()
    b = o["before_vs_after"]
    print("  BEFORE vs AFTER (all robots reach their goal):")
    print(f"    independent (sequential) : {b['independent_all_reached_goal']}/{b['scenarios']} "
          f"({b['independent_success_rate'] * 100:.1f}%)")
    print(f"    coordinated              : {b['coordinated_all_reached_goal']}/{b['scenarios']} "
          f"({b['coordinated_success_rate'] * 100:.1f}%)")
    print(f"    absolute improvement     : {b['absolute_improvement_pct_points']:+.1f} "
          f"percentage points")
    print()
    print(f"  {'type':<14} {'fleet':>5} {'plan%':>7} {'coord all%':>11} {'indep all%':>11} "
          f"{'med wait':>9} {'med span':>9}")
    print("  " + "-" * 74)
    for row in report["by_type_and_size"]:
        print(f"  {row['scenario_type']:<14} {row['robot_count']:>5} "
              f"{row['planning_success_rate'] * 100:>6.0f}% "
              f"{row['coordinated_all_reached_rate'] * 100:>10.0f}% "
              f"{row['independent_all_reached_rate'] * 100:>10.0f}% "
              f"{row['median_wait_steps']:>9} {row['median_makespan']:>9}")
    print()
    print(f"  wrote {JSON_PATH.name} and {CSV_PATH.name} to {RESULTS_DIR}")


# ----------------------------------------------------------------------
def run_benchmark() -> Tuple[List[RunRecord], Dict[str, object]]:
    scenarios = generate_scenarios()
    records = [run_scenario(s) for s in scenarios]
    return records, build_report(records)


def main() -> None:
    records, report = run_benchmark()
    write_json(report)
    write_csv(records)
    print_summary(report)


if __name__ == "__main__":
    main()
