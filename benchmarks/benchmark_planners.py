"""Fair, reproducible A* vs Dijkstra benchmark over the warehouse grid.

WHAT THIS MEASURES
------------------
For a set of deterministic scenarios - varied grid sizes, varied obstacle
densities, varied start/goal pairs - it runs BOTH planners on the EXACT SAME
warehouse and records, per run:

    algorithm, grid size, obstacle density, obstacle count, scenario id,
    success, path cost, nodes expanded, planning time (ms)

then aggregates per (algorithm, grid size, obstacle density) group:

    success rate, median (P50) planning time, P95 planning time,
    median nodes expanded, and the A*/Dijkstra optimal-cost agreement rate.

WHAT "FAIR" MEANS HERE
---------------------
1. Both planners receive the identical `Warehouse` object for a scenario, so
   they see the same grid, the same obstacles, the same start and the same
   goal.
2. Both use the same `nodes_expanded` definition (a cell counts once, when it
   is popped as the best non-stale entry) - that rule lives in the planners,
   not here.
3. Scenarios are generated from a fixed seed, so re-running produces the same
   layouts on any machine.

HONEST LIMITATION
-----------------
On the smallest grid (20x20) A* finishes in a few hundred microseconds, where
the timing is dominated by interpreter noise rather than real algorithmic cost.
Dijkstra on the largest grid (60x60) takes tens of milliseconds, which is a
real difference worth reporting. Each planner call is repeated
(`TIMING_REPEATS`) and the median of the repeats is kept to reduce noise, but
the small-grid timings should still be read as "effectively instant", with the
nodes-expanded count treated as the more stable signal. `nodes_expanded` and
`path_cost` are exact and fully reproducible at every size.

    python benchmarks/benchmark_planners.py

writes `benchmark_results.json` and `benchmark_results.csv` next to this file
and prints a summary table.
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

from robotics.planning import PLANNERS  # noqa: E402
from robotics.planning.path_result import PathResult  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import Warehouse  # noqa: E402

# ----------------------------------------------------------------------
# Benchmark configuration - deterministic and stated up front
# ----------------------------------------------------------------------
GRID_SIZES: Tuple[int, ...] = (20, 40, 60)
OBSTACLE_DENSITIES: Tuple[float, ...] = (0.05, 0.15, 0.25)
SCENARIOS_PER_CONFIG = 12
TIMING_REPEATS = 7
MASTER_SEED = 20240517

RESULTS_DIR = Path(__file__).resolve().parent
JSON_PATH = RESULTS_DIR / "benchmark_results.json"
CSV_PATH = RESULTS_DIR / "benchmark_results.csv"


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Scenario:
    """One planning problem, reused by every planner without modification."""

    scenario_id: str
    rows: int
    cols: int
    obstacle_density: float
    obstacle_count: int
    start: Position
    goal: Position
    warehouse: Warehouse = field(compare=False, repr=False)


def build_scenario(
    rows: int, cols: int, obstacle_density: float, seed: int, scenario_index: int
) -> Scenario:
    """Build one deterministic scenario.

    Cells are blocked independently with probability `obstacle_density`, using a
    `Random(seed)` stream so the layout is identical on every run and every
    machine. The start and goal are then chosen near opposite corners and
    forcibly cleared, so the request is always well-formed (a blocked start or
    goal is a separate case, already covered by the unit tests).

    Note the goal may still be genuinely unreachable if obstacles wall it off -
    that is a real outcome the benchmark should measure, not avoid.
    """
    rng = Random(seed)
    warehouse = Warehouse(width=cols, height=rows, name=f"bench-{rows}x{cols}")

    start = Position(rng.randrange(0, max(1, rows // 4) + 1),
                     rng.randrange(0, max(1, cols // 4) + 1))
    goal = Position(rows - 1 - rng.randrange(0, max(1, rows // 4) + 1),
                    cols - 1 - rng.randrange(0, max(1, cols // 4) + 1))

    obstacle_count = 0
    for r in range(rows):
        for c in range(cols):
            cell = Position(r, c)
            if cell == start or cell == goal:
                continue
            if rng.random() < obstacle_density:
                warehouse.add_static_obstacle(f"obs-{r}-{c}", cell)
                obstacle_count += 1

    return Scenario(
        scenario_id=f"{rows}x{cols}-d{obstacle_density:g}-s{scenario_index:02d}",
        rows=rows,
        cols=cols,
        obstacle_density=obstacle_density,
        obstacle_count=obstacle_count,
        start=start,
        goal=goal,
        warehouse=warehouse,
    )


def generate_scenarios(
    grid_sizes: Sequence[int] = GRID_SIZES,
    densities: Sequence[float] = OBSTACLE_DENSITIES,
    per_config: int = SCENARIOS_PER_CONFIG,
    master_seed: int = MASTER_SEED,
) -> List[Scenario]:
    """Every scenario the benchmark will run, in a fixed order."""
    scenarios: List[Scenario] = []
    seed_stream = Random(master_seed)
    for size in grid_sizes:
        for density in densities:
            for index in range(per_config):
                scenarios.append(
                    build_scenario(
                        rows=size,
                        cols=size,
                        obstacle_density=density,
                        seed=seed_stream.randrange(1, 2**31),
                        scenario_index=index,
                    )
                )
    return scenarios


# ----------------------------------------------------------------------
# Running one scenario
# ----------------------------------------------------------------------
@dataclass
class RunRecord:
    """The measured outcome of one (planner, scenario) pair."""

    algorithm: str
    scenario_id: str
    grid_size: int
    obstacle_density: float
    obstacle_count: int
    success: bool
    path_cost: Optional[float]
    nodes_expanded: int
    planning_time_ms: float

    def to_row(self) -> Dict[str, object]:
        return {
            "algorithm": self.algorithm,
            "scenario_id": self.scenario_id,
            "grid_size": self.grid_size,
            "obstacle_density": self.obstacle_density,
            "obstacle_count": self.obstacle_count,
            "success": self.success,
            "path_cost": self.path_cost,
            "nodes_expanded": self.nodes_expanded,
            "planning_time_ms": round(self.planning_time_ms, 6),
        }


def _time_planner(planner, scenario: Scenario, repeats: int) -> Tuple[PathResult, float]:
    """Run one planner `repeats` times, return a result and the median time (ms).

    The path, cost and node count are deterministic, so any run's result is
    representative. Only the timing varies, so the median of the repeats is
    kept as a noise-reduced estimate.
    """
    timings_ms: List[float] = []
    result: Optional[PathResult] = None
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = planner(scenario.start, scenario.goal, scenario.warehouse)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
        timings_ms.append(elapsed_ms)
    assert result is not None
    return result, statistics.median(timings_ms)


def run_scenario(scenario: Scenario, repeats: int = TIMING_REPEATS) -> List[RunRecord]:
    """Run every planner over one scenario and return one record per planner."""
    records: List[RunRecord] = []
    for name, planner in PLANNERS.items():
        result, median_ms = _time_planner(planner, scenario, repeats)
        records.append(
            RunRecord(
                algorithm=name,
                scenario_id=scenario.scenario_id,
                grid_size=scenario.rows,
                obstacle_density=scenario.obstacle_density,
                obstacle_count=scenario.obstacle_count,
                success=result.success,
                path_cost=result.total_cost,
                nodes_expanded=result.nodes_expanded,
                planning_time_ms=median_ms,
            )
        )
    return records


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------
def percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile. `pct` is 0..100. Empty input returns 0.0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = max(1, min(len(ordered), round(pct / 100 * len(ordered))))
    return float(ordered[rank - 1])


def cost_agreement_rate(records: Sequence[RunRecord]) -> Optional[float]:
    """Fraction of scenarios where every planner that succeeded agreed on cost.

    Returns None when no scenario was solved by more than one planner, so the
    caller can print "n/a" instead of a misleading 1.0.
    """
    by_scenario: Dict[str, List[RunRecord]] = {}
    for record in records:
        by_scenario.setdefault(record.scenario_id, []).append(record)

    comparable = 0
    agreed = 0
    for group in by_scenario.values():
        succeeded = [r.path_cost for r in group if r.success and r.path_cost is not None]
        if len(succeeded) < 2:
            continue
        comparable += 1
        if len(set(succeeded)) == 1:
            agreed += 1

    if comparable == 0:
        return None
    return agreed / comparable


def aggregate(records: Sequence[RunRecord]) -> List[Dict[str, object]]:
    """Group records by (algorithm, grid size, density) and summarise each group."""
    groups: Dict[Tuple[str, int, float], List[RunRecord]] = {}
    for record in records:
        key = (record.algorithm, record.grid_size, record.obstacle_density)
        groups.setdefault(key, []).append(record)

    summaries: List[Dict[str, object]] = []
    for (algorithm, grid_size, density), group in sorted(groups.items()):
        times = [r.planning_time_ms for r in group]
        successful = [r for r in group if r.success]
        nodes_successful = [r.nodes_expanded for r in successful]
        summaries.append(
            {
                "algorithm": algorithm,
                "grid_size": grid_size,
                "obstacle_density": density,
                "scenarios": len(group),
                "success_rate": round(len(successful) / len(group), 4),
                "median_planning_time_ms": round(statistics.median(times), 6),
                "p95_planning_time_ms": round(percentile(times, 95), 6),
                "median_nodes_expanded": (
                    int(statistics.median(nodes_successful)) if nodes_successful else None
                ),
            }
        )
    return summaries


def head_to_head(records: Sequence[RunRecord]) -> Dict[str, object]:
    """Compare the planners scenario by scenario where both succeeded."""
    by_scenario: Dict[str, Dict[str, RunRecord]] = {}
    for record in records:
        by_scenario.setdefault(record.scenario_id, {})[record.algorithm] = record

    both = [
        g for g in by_scenario.values()
        if "astar" in g and "dijkstra" in g
        and g["astar"].success and g["dijkstra"].success
    ]
    astar_fewer = sum(
        1 for g in both if g["astar"].nodes_expanded < g["dijkstra"].nodes_expanded
    )
    astar_equal = sum(
        1 for g in both if g["astar"].nodes_expanded == g["dijkstra"].nodes_expanded
    )
    astar_more = sum(
        1 for g in both if g["astar"].nodes_expanded > g["dijkstra"].nodes_expanded
    )
    node_ratios = [
        g["astar"].nodes_expanded / g["dijkstra"].nodes_expanded
        for g in both
        if g["dijkstra"].nodes_expanded > 0
    ]
    return {
        "scenarios_both_succeeded": len(both),
        "astar_expanded_fewer_nodes": astar_fewer,
        "astar_expanded_equal_nodes": astar_equal,
        "astar_expanded_more_nodes": astar_more,
        "median_astar_to_dijkstra_node_ratio": (
            round(statistics.median(node_ratios), 4) if node_ratios else None
        ),
        "optimal_cost_agreement_rate": cost_agreement_rate(records),
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
def build_report(records: Sequence[RunRecord]) -> Dict[str, object]:
    return {
        "config": {
            "grid_sizes": list(GRID_SIZES),
            "obstacle_densities": list(OBSTACLE_DENSITIES),
            "scenarios_per_config": SCENARIOS_PER_CONFIG,
            "timing_repeats": TIMING_REPEATS,
            "master_seed": MASTER_SEED,
            "timer": "time.perf_counter_ns",
            "note": (
                "Small-grid planning times are sub-millisecond and dominated by "
                "interpreter noise; larger grids show a real gap. nodes_expanded "
                "and path_cost are exact and reproducible at every size."
            ),
        },
        "total_runs": len(records),
        "raw_runs": [r.to_row() for r in records],
        "aggregates": aggregate(records),
        "head_to_head": head_to_head(records),
    }


def write_json(report: Dict[str, object], path: Path = JSON_PATH) -> None:
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def write_csv(records: Sequence[RunRecord], path: Path = CSV_PATH) -> None:
    fieldnames = [
        "algorithm", "scenario_id", "grid_size", "obstacle_density",
        "obstacle_count", "success", "path_cost", "nodes_expanded",
        "planning_time_ms",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())


def print_summary(report: Dict[str, object]) -> None:
    print("=" * 78)
    print("PATH PLANNER BENCHMARK")
    print("=" * 78)
    cfg = report["config"]
    print(f"  grid sizes          : {cfg['grid_sizes']}")
    print(f"  obstacle densities  : {cfg['obstacle_densities']}")
    print(f"  scenarios / config  : {cfg['scenarios_per_config']}")
    print(f"  timing repeats      : {cfg['timing_repeats']}  (median kept)")
    print(f"  total planner runs  : {report['total_runs']}")
    print()
    header = (
        f"  {'algorithm':<9} {'grid':>6} {'density':>8} {'runs':>5} "
        f"{'success':>8} {'p50 ms':>10} {'p95 ms':>10} {'med nodes':>10}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in report["aggregates"]:
        print(
            f"  {row['algorithm']:<9} {row['grid_size']:>6} "
            f"{row['obstacle_density']:>8} {row['scenarios']:>5} "
            f"{row['success_rate'] * 100:>7.1f}% "
            f"{row['median_planning_time_ms']:>10.4f} "
            f"{row['p95_planning_time_ms']:>10.4f} "
            f"{str(row['median_nodes_expanded']):>10}"
        )
    print()
    h2h = report["head_to_head"]
    print("  A* vs Dijkstra, scenarios both solved:")
    print(f"    scenarios compared            : {h2h['scenarios_both_succeeded']}")
    print(f"    A* expanded fewer nodes       : {h2h['astar_expanded_fewer_nodes']}")
    print(f"    A* expanded the same          : {h2h['astar_expanded_equal_nodes']}")
    print(f"    A* expanded more nodes        : {h2h['astar_expanded_more_nodes']}")
    print(f"    median A*/Dijkstra node ratio : "
          f"{h2h['median_astar_to_dijkstra_node_ratio']}")
    agreement = h2h["optimal_cost_agreement_rate"]
    agreement_str = "n/a" if agreement is None else f"{agreement * 100:.1f}%"
    print(f"    optimal-cost agreement        : {agreement_str}")
    print()
    print(f"  wrote {JSON_PATH.name} and {CSV_PATH.name} to {RESULTS_DIR}")


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
def run_benchmark() -> Tuple[List[RunRecord], Dict[str, object]]:
    scenarios = generate_scenarios()
    records: List[RunRecord] = []
    for scenario in scenarios:
        records.extend(run_scenario(scenario))
    report = build_report(records)
    return records, report


def main() -> None:
    records, report = run_benchmark()
    write_json(report)
    write_csv(records)
    print_summary(report)


if __name__ == "__main__":
    main()
