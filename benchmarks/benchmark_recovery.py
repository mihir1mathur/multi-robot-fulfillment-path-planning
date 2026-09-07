"""Reproducible benchmark: dynamic replanning and robot fault recovery.

WHAT IT MEASURES
----------------
Two disruption families, each run over deterministic scenarios:

    OBSTACLE   a new obstacle appears on a robot's remaining route mid-episode
    FAILURE    a robot with a task goes OFFLINE mid-episode

Every scenario is run TWICE on the SAME warehouse / robots / tasks / disruption:

    BEFORE   recovery disabled - the affected robots stop safely (the old
             behaviour after a route was invalidated)
    AFTER    recovery enabled  - replan / re-coordinate / reassign, then resume

It records, per run: what was affected, whether replanning was attempted and
succeeded, whether the disruption was fully recovered or ended in a safe stop,
the extra distance and wait actions, the reservation churn, and the recovery
latency (perf_counter_ns). It then aggregates, breaks the metrics down by
disruption type, and reports the before/after completion difference.

HONEST FRAMING
--------------
Some scenarios are structurally unrecoverable - a blocked one-cell corridor, a
robot failure with no spare robot. Those are INCLUDED and reported as safe
stops, not deleted. "Recovery success" means the affected robots reached their
goals (or the task was reassigned and its replacement did) with zero unresolved
conflicts.

    python benchmarks/benchmark_recovery.py

writes benchmark_recovery_results.json / .csv next to this file.
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
from robotics.allocation.greedy_allocator import GreedyAllocator  # noqa: E402
from robotics.coordination.coordinator import MultiRobotCoordinator  # noqa: E402
from robotics.recovery.disruption import (  # noqa: E402
    DisruptionSchedule,
    DynamicObstacleEvent,
    RobotFailureEvent,
)
from robotics.recovery.recovery_manager import RecoveryManager  # noqa: E402
from robotics.recovery.recovery_result import ResilientExecutionResult  # noqa: E402
from robotics.robots.robot import Robot  # noqa: E402
from robotics.simulation.simulator import WarehouseSimulator  # noqa: E402
from robotics.tasks.task import Task, TaskPriority  # noqa: E402
from robotics.warehouse.grid import Position  # noqa: E402
from robotics.warehouse.warehouse import Warehouse  # noqa: E402

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
GRID_SIZES = (8, 12, 16)
OBSTACLE_DENSITIES = (0.0, 0.08)
FLEET_SIZES = (2, 4, 6)
SCENARIOS_PER_CONFIG = 4
MASTER_SEED = 90210

RESULTS_DIR = Path(__file__).resolve().parent
JSON_PATH = RESULTS_DIR / "benchmark_recovery_results.json"
CSV_PATH = RESULTS_DIR / "benchmark_recovery_results.csv"


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
@dataclass
class Scenario:
    scenario_id: str
    family: str                 # "obstacle" | "failure"
    grid_size: int
    obstacle_density: float
    fleet_size: int
    grid_w: int
    grid_h: int
    static_obstacles: List[Position]
    warehouse_name: str
    starts: Dict[str, Position]
    goals: Dict[str, Position]
    tasks: List[Task]
    disruptions: DisruptionSchedule
    alternate_available: Optional[bool] = None
    replacement_available: Optional[bool] = None

    def fresh_warehouse(self) -> Warehouse:
        """A brand-new warehouse with only the static obstacles - no leftover
        dynamic obstacle from an earlier run."""
        wh = Warehouse(width=self.grid_w, height=self.grid_h, name=self.warehouse_name)
        for i, cell in enumerate(self.static_obstacles):
            wh.add_static_obstacle(f"o-{i}", cell)
        return wh


def _free_cells(wh: Warehouse, rng: Random, n: int) -> List[Position]:
    picked: List[Position] = []
    seen: set = set()
    tries = 0
    while len(picked) < n and tries < n * 400:
        tries += 1
        cell = Position(rng.randrange(wh.height), rng.randrange(wh.width))
        if cell in seen or not wh.is_traversable(cell):
            continue
        seen.add(cell)
        picked.append(cell)
    return picked


def _random_obstacles(size: int, density: float, rng: Random) -> List[Position]:
    return [
        Position(r, c)
        for r in range(size)
        for c in range(size)
        if rng.random() < density
    ]


def _obstacle_scenario(
    size: int, density: float, fleet: int, seed: int, index: int
) -> Optional[Scenario]:
    rng = Random(seed)
    # ~1 scenario in 3 uses a deliberately constrained (narrow) layout with no
    # alternate route, so unrecoverable cases are represented.
    constrained = (index % 3 == 0)

    if constrained:
        grid_w, grid_h = size, 3
        statics = [Position(r, c) for r in (0, 2) for c in range(size)]
    else:
        grid_w = grid_h = size
        statics = _random_obstacles(size, density, rng)

    wh = Warehouse(width=grid_w, height=grid_h, name=f"obs-{index}")
    for i, cell in enumerate(statics):
        wh.add_static_obstacle(f"o-{i}", cell)
    cells = _free_cells(wh, rng, fleet * 2)
    if len(cells) < fleet * 2:
        return None
    starts = {f"R{i}": cells[i] for i in range(fleet)}
    goals = {f"R{i}": cells[fleet + i] for i in range(fleet)}
    if len(set(starts.values())) != fleet:
        return None

    coordination = MultiRobotCoordinator(wh).plan(starts, goals)
    routed = [rid for rid, tp in coordination.successful_paths.items() if tp.makespan >= 4]
    if not routed:
        return None
    target = sorted(routed)[0]
    tp = coordination.timed_paths[target]
    fire_t = max(1, int(tp.arrival_time * 0.4))
    ahead = tp.position_at(fire_t + 1)
    at_now = tp.position_at(fire_t)
    if ahead is None or ahead == at_now or ahead in starts.values():
        return None
    disruptions = DisruptionSchedule.of(DynamicObstacleEvent(fire_t, ahead))

    return Scenario(
        scenario_id=f"obs-{size}-d{density:g}-f{fleet}-s{index:02d}",
        family="obstacle",
        grid_size=size,
        obstacle_density=density,
        fleet_size=fleet,
        grid_w=grid_w,
        grid_h=grid_h,
        static_obstacles=statics,
        warehouse_name=f"obs-{index}",
        starts=starts,
        goals=goals,
        tasks=[],
        disruptions=disruptions,
        alternate_available=not constrained,
    )


def _failure_scenario(
    size: int, density: float, fleet: int, seed: int, index: int
) -> Optional[Scenario]:
    rng = Random(seed)
    # ~half the scenarios have a spare idle robot (reassignment possible)
    spare = (index % 2 == 0)
    total_robots = fleet + (1 if spare else 0)

    statics = _random_obstacles(size, density, rng)
    wh = Warehouse(width=size, height=size, name=f"fail-{index}")
    for i, cell in enumerate(statics):
        wh.add_static_obstacle(f"o-{i}", cell)
    cells = _free_cells(wh, rng, total_robots + fleet * 2)
    if len(cells) < total_robots + fleet * 2:
        return None

    robot_cells = cells[:total_robots]
    task_cells = cells[total_robots:]
    starts = {f"R{i}": robot_cells[i] for i in range(total_robots)}

    tasks: List[Task] = []
    for i in range(fleet):
        pickup = task_cells[2 * i]
        dropoff = task_cells[2 * i + 1]
        if pickup == dropoff:
            return None
        tasks.append(
            Task(f"T{i}", pickup, dropoff, TaskPriority.NORMAL, 2.0)
        )

    return Scenario(
        scenario_id=f"fail-{size}-d{density:g}-f{fleet}-s{index:02d}",
        family="failure",
        grid_size=size,
        obstacle_density=density,
        fleet_size=fleet,
        grid_w=size,
        grid_h=size,
        static_obstacles=statics,
        warehouse_name=f"fail-{index}",
        starts=starts,
        goals={},          # set after allocation
        tasks=tasks,
        disruptions=DisruptionSchedule(),   # set after allocation
        replacement_available=spare,
    )


def generate_scenarios() -> List[Scenario]:
    scenarios: List[Scenario] = []
    seeds = Random(MASTER_SEED)
    for size in GRID_SIZES:
        for density in OBSTACLE_DENSITIES:
            for fleet in FLEET_SIZES:
                for index in range(SCENARIOS_PER_CONFIG):
                    obs = _obstacle_scenario(
                        size, density, fleet, seeds.randrange(1, 2**31), index
                    )
                    if obs is not None:
                        scenarios.append(obs)
                    fail = _failure_scenario(
                        size, density, fleet, seeds.randrange(1, 2**31), index
                    )
                    if fail is not None:
                        scenarios.append(fail)
    return scenarios


# ----------------------------------------------------------------------
# Running one scenario
# ----------------------------------------------------------------------
def _place(wh: Warehouse, starts: Dict[str, Position]) -> WarehouseSimulator:
    sim = WarehouseSimulator(wh)
    for rid, cell in sorted(starts.items()):
        sim.add_robot(Robot(rid, cell))
    return sim


@dataclass
class RunRecord:
    scenario_id: str
    family: str
    grid_size: int
    fleet_size: int
    disruption_type: str
    disruption_timestep: int
    alternate_available: Optional[bool]
    replacement_available: Optional[bool]
    affected_robot_count: int
    affected_task_count: int
    replanning_attempted: bool
    replanning_success: bool
    recovery_success: bool
    safe_stop: bool
    task_reassignment_attempted: bool
    task_reassignment_success: bool
    additional_distance: int
    wait_actions_introduced: int
    reservations_released: int
    reservations_created: int
    replanning_latency_ms: float
    total_recovery_latency_ms: float
    unresolved_vertex_conflicts: int
    unresolved_edge_conflicts: int
    goal_completion_after: float
    goal_completion_before: float
    disrupted_task_recovered_before: bool
    disrupted_task_recovered_after: bool

    def to_row(self) -> Dict[str, object]:
        d = dict(self.__dict__)
        d["replanning_latency_ms"] = round(self.replanning_latency_ms, 6)
        d["total_recovery_latency_ms"] = round(self.total_recovery_latency_ms, 6)
        return d


def _completion_fraction(result: ResilientExecutionResult) -> float:
    expected = [
        rid for rid, info in result.per_robot.items()
        if not info["offline"] and not info["joined_mid_episode"]
    ] or list(result.per_robot)
    if not expected:
        return 1.0
    reached = sum(
        1 for rid in expected
        if result.per_robot[rid]["reached_goal"]
    )
    # a mid-episode replacement that reached its (pickup) goal also counts
    replacements = [
        rid for rid, info in result.per_robot.items() if info["joined_mid_episode"]
    ]
    for rid in replacements:
        if result.per_robot[rid]["reached_goal"]:
            reached += 1
            expected.append(rid)
    return reached / len(expected)


def run_scenario(scenario: Scenario) -> Optional[RunRecord]:
    if scenario.family == "failure":
        # allocate + coordinate once, then schedule a failure of an assigned robot
        probe_wh = scenario.fresh_warehouse()
        sim0 = _place(probe_wh, scenario.starts)
        for task in scenario.tasks:
            sim0.add_task(
                Task(task.task_id, task.pickup_location, task.dropoff_location,
                     task.priority, task.payload_weight)
            )
        allocation = GreedyAllocator(CostEstimator(probe_wh)).allocate(
            sim0.robots, sim0.tasks
        )
        commit_allocation(sim0, allocation)
        if not allocation.assignments:
            return None
        goals = {
            a.robot_id: sim0.get_task(a.task_id).pickup_location
            for a in allocation.assignments
        }
        coordination = MultiRobotCoordinator(probe_wh).plan_for(sim0, goals)
        routed = {
            rid: tp for rid, tp in coordination.successful_paths.items()
            if tp.makespan >= 3
        }
        if not routed:
            return None
        victim = sorted(routed)[0]
        fire_t = max(1, int(routed[victim].arrival_time * 0.4))
        scenario.goals = goals
        scenario.disruptions = DisruptionSchedule.of(RobotFailureEvent(fire_t, victim))
        disruption_timestep = fire_t
        disruption_type = "robot_offline"
    else:
        disruption_timestep = scenario.disruptions.events[0].timestep
        disruption_type = "dynamic_obstacle"

    # BEFORE: recovery disabled  (fresh warehouse, no leftover dynamic obstacle)
    wh_b = scenario.fresh_warehouse()
    sim_b = _rebuild_sim(scenario, wh_b)
    coord_b = _rebuild_coord(scenario, wh_b, sim_b)
    before = RecoveryManager(sim_b).run(
        coord_b, scenario.disruptions, recovery_enabled=False
    )

    # AFTER: recovery enabled
    wh_a = scenario.fresh_warehouse()
    sim_a = _rebuild_sim(scenario, wh_a)
    coord_a = _rebuild_coord(scenario, wh_a, sim_a)
    after = RecoveryManager(sim_a).run(
        coord_a, scenario.disruptions, recovery_enabled=True
    )

    if not after.recovery_events:
        return None
    event = after.recovery_events[0]

    return RunRecord(
        scenario_id=scenario.scenario_id,
        family=scenario.family,
        grid_size=scenario.grid_size,
        fleet_size=scenario.fleet_size,
        disruption_type=disruption_type,
        disruption_timestep=disruption_timestep,
        alternate_available=scenario.alternate_available,
        replacement_available=scenario.replacement_available,
        affected_robot_count=len(event.affected_robot_ids),
        affected_task_count=len(event.affected_task_ids),
        replanning_attempted=event.replanning_attempted,
        replanning_success=event.replanning_success,
        recovery_success=event.recovery_success,
        safe_stop=event.safe_stop or after.safe_stop,
        task_reassignment_attempted=event.task_reassignment_attempted,
        task_reassignment_success=event.task_reassignment_success,
        additional_distance=event.total_additional_distance,
        wait_actions_introduced=event.total_wait_actions_introduced,
        reservations_released=event.reservations_released,
        reservations_created=event.reservations_created,
        replanning_latency_ms=event.replanning_latency_ms
        + sum(r.replanning_time_ms for r in event.robot_replans),
        total_recovery_latency_ms=event.total_recovery_latency_ms,
        unresolved_vertex_conflicts=event.unresolved_vertex_conflicts
        + after.vertex_conflicts,
        unresolved_edge_conflicts=event.unresolved_edge_conflicts + after.edge_conflicts,
        goal_completion_after=_completion_fraction(after),
        goal_completion_before=_completion_fraction(before),
        # a disrupted task is "recovered" if, at the end, it belongs to an
        # online robot. Without recovery it stays stuck on the failed robot.
        disrupted_task_recovered_before=(
            scenario.family == "failure" and False
        ),
        disrupted_task_recovered_after=(
            scenario.family == "failure" and event.task_reassignment_success
        ),
    )


def _rebuild_sim(scenario: Scenario, wh: Warehouse) -> WarehouseSimulator:
    sim = _place(wh, scenario.starts)
    for task in scenario.tasks:
        sim.add_task(
            Task(task.task_id, task.pickup_location, task.dropoff_location,
                 task.priority, task.payload_weight)
        )
    return sim


def _rebuild_coord(scenario: Scenario, wh: Warehouse, sim: WarehouseSimulator):
    if scenario.family == "failure":
        allocation = GreedyAllocator(CostEstimator(wh)).allocate(sim.robots, sim.tasks)
        commit_allocation(sim, allocation)
        goals = {
            a.robot_id: sim.get_task(a.task_id).pickup_location
            for a in allocation.assignments
        }
        return MultiRobotCoordinator(wh).plan_for(sim, goals)
    return MultiRobotCoordinator(wh).plan(scenario.starts, scenario.goals)


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


def _aggregate_group(records: Sequence[RunRecord]) -> Dict[str, object]:
    n = len(records)
    if n == 0:
        return {}
    replan_attempts = [r for r in records if r.replanning_attempted]
    reassign_attempts = [r for r in records if r.task_reassignment_attempted]
    latencies = [
        r.total_recovery_latency_ms for r in records if r.total_recovery_latency_ms > 0
    ]
    replan_lat = [
        r.replanning_latency_ms for r in replan_attempts if r.replanning_latency_ms > 0
    ]
    return {
        "runs": n,
        "disruption_events": n,
        "replanning_attempts": len(replan_attempts),
        "replanning_success_rate": round(
            sum(1 for r in replan_attempts if r.replanning_success) / len(replan_attempts), 4
        ) if replan_attempts else None,
        "recovery_success_rate": round(
            sum(1 for r in records if r.recovery_success) / n, 4
        ),
        "safe_stop_rate": round(sum(1 for r in records if r.safe_stop) / n, 4),
        "task_reassignment_attempts": len(reassign_attempts),
        "task_reassignment_success_rate": round(
            sum(1 for r in reassign_attempts if r.task_reassignment_success)
            / len(reassign_attempts), 4
        ) if reassign_attempts else None,
        "median_replanning_latency_ms": round(statistics.median(replan_lat), 6)
        if replan_lat else 0.0,
        "p95_replanning_latency_ms": round(percentile(replan_lat, 95), 6),
        "median_total_recovery_latency_ms": round(statistics.median(latencies), 6)
        if latencies else 0.0,
        "p95_total_recovery_latency_ms": round(percentile(latencies, 95), 6),
        "median_additional_distance": statistics.median(
            [r.additional_distance for r in records if r.recovery_success]
        ) if any(r.recovery_success for r in records) else 0,
        "median_wait_actions_introduced": statistics.median(
            [r.wait_actions_introduced for r in records]
        ),
        "unresolved_vertex_conflicts": sum(r.unresolved_vertex_conflicts for r in records),
        "unresolved_edge_conflicts": sum(r.unresolved_edge_conflicts for r in records),
        "stale_reservations_released": sum(r.reservations_released for r in records),
        "reservations_recreated": sum(r.reservations_created for r in records),
    }


def build_report(records: Sequence[RunRecord]) -> Dict[str, object]:
    obstacle = [r for r in records if r.family == "obstacle"]
    failure = [r for r in records if r.family == "failure"]

    def _before_after(subset: Sequence[RunRecord]) -> Dict[str, object]:
        if not subset:
            return {"scenarios": 0}
        before = statistics.mean([r.goal_completion_before for r in subset])
        after = statistics.mean([r.goal_completion_after for r in subset])
        return {
            "scenarios": len(subset),
            "mean_goal_completion_without_recovery": round(before, 4),
            "mean_goal_completion_with_recovery": round(after, 4),
            "absolute_improvement_pct_points": round(100 * (after - before), 2),
        }

    comparable = list(records)
    # "recoverable" = an alternate route / spare robot was structurally available
    recoverable = [
        r for r in records
        if (r.family == "obstacle" and r.alternate_available)
        or (r.family == "failure" and r.replacement_available)
    ]
    before_completion = statistics.mean(
        [r.goal_completion_before for r in comparable]
    ) if comparable else 0.0
    after_completion = statistics.mean(
        [r.goal_completion_after for r in comparable]
    ) if comparable else 0.0

    return {
        "config": {
            "grid_sizes": list(GRID_SIZES),
            "obstacle_densities": list(OBSTACLE_DENSITIES),
            "fleet_sizes": list(FLEET_SIZES),
            "scenarios_per_config": SCENARIOS_PER_CONFIG,
            "master_seed": MASTER_SEED,
            "timer": "time.perf_counter_ns",
            "note": (
                "Some scenarios are structurally unrecoverable (blocked corridor, "
                "no spare robot) and are included as safe stops, not deleted. "
                "'recovery success' = affected robots reached their goals (or the "
                "task was reassigned and its replacement did) with 0 unresolved "
                "conflicts."
            ),
        },
        "total_scenarios": len(records),
        "total_disruption_events": len(records),
        "dynamic_obstacle_scenarios": len(obstacle),
        "robot_failure_scenarios": len(failure),
        "overall": _aggregate_group(records),
        "by_disruption_type": {
            "dynamic_obstacle": _aggregate_group(obstacle),
            "robot_offline": _aggregate_group(failure),
        },
        "before_vs_after": {
            "all_scenarios": {
                "scenarios": len(comparable),
                "mean_goal_completion_without_recovery": round(before_completion, 4),
                "mean_goal_completion_with_recovery": round(after_completion, 4),
                "absolute_improvement_pct_points": round(
                    100 * (after_completion - before_completion), 2
                ),
            },
            "recoverable_subset": _before_after(recoverable),
            "obstacle_only": _before_after(obstacle),
            "failure_task_recovery": {
                "scenarios": len(failure),
                "task_recovered_without_recovery": 0,
                "task_recovered_with_recovery": sum(
                    1 for r in failure if r.disrupted_task_recovered_after
                ),
                "task_recovery_rate_with_recovery": round(
                    sum(1 for r in failure if r.disrupted_task_recovered_after)
                    / len(failure), 4
                ) if failure else 0.0,
                "task_recovered_when_spare_available": sum(
                    1 for r in failure
                    if r.replacement_available and r.disrupted_task_recovered_after
                ),
                "spare_available_scenarios": sum(
                    1 for r in failure if r.replacement_available
                ),
            },
        },
        "raw_runs": [r.to_row() for r in records],
    }


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------
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
    print("DYNAMIC REPLANNING & FAULT RECOVERY BENCHMARK")
    print("=" * 78)
    cfg = report["config"]
    print(f"  grid sizes          : {cfg['grid_sizes']}")
    print(f"  fleet sizes         : {cfg['fleet_sizes']}")
    print(f"  total scenarios     : {report['total_scenarios']}")
    print(f"  obstacle / failure  : {report['dynamic_obstacle_scenarios']} / "
          f"{report['robot_failure_scenarios']}")
    print()
    o = report["overall"]
    print(f"  replanning attempts        : {o['replanning_attempts']}")
    print(f"  replanning success rate    : {_pct(o['replanning_success_rate'])}")
    print(f"  overall recovery success   : {_pct(o['recovery_success_rate'])}")
    print(f"  safe-stop rate (unrecoverable): {_pct(o['safe_stop_rate'])}")
    print(f"  task reassignment attempts : {o['task_reassignment_attempts']}")
    print(f"  task reassignment success  : {_pct(o['task_reassignment_success_rate'])}")
    print(f"  UNRESOLVED vertex conflicts: {o['unresolved_vertex_conflicts']}")
    print(f"  UNRESOLVED edge conflicts  : {o['unresolved_edge_conflicts']}")
    print(f"  median / P95 replan latency: {o['median_replanning_latency_ms']:.3f} / "
          f"{o['p95_replanning_latency_ms']:.3f} ms")
    print(f"  median / P95 recovery latency: {o['median_total_recovery_latency_ms']:.3f} / "
          f"{o['p95_total_recovery_latency_ms']:.3f} ms")
    print(f"  median additional distance : {o['median_additional_distance']} moves")
    print(f"  median wait actions added  : {o['median_wait_actions_introduced']}")
    print(f"  stale reservations released: {o['stale_reservations_released']}")
    print(f"  reservations recreated     : {o['reservations_recreated']}")
    print()
    for kind, label in (("dynamic_obstacle", "OBSTACLE"), ("robot_offline", "FAILURE")):
        g = report["by_disruption_type"][kind]
        if not g:
            continue
        print(f"  {label:<9}: {g['runs']} runs | recovery {_pct(g['recovery_success_rate'])} | "
              f"safe-stop {_pct(g['safe_stop_rate'])} | "
              f"replan p50 {g['median_replanning_latency_ms']:.2f} ms")
    print()
    print("  BEFORE vs AFTER - robot goal completion (mean fraction reaching goal):")
    for label, key in (
        ("all scenarios       ", "all_scenarios"),
        ("recoverable subset  ", "recoverable_subset"),
        ("obstacle scenarios  ", "obstacle_only"),
    ):
        g = report["before_vs_after"][key]
        if not g.get("scenarios"):
            continue
        print(f"    {label}: {g['mean_goal_completion_without_recovery'] * 100:5.1f}% "
              f"-> {g['mean_goal_completion_with_recovery'] * 100:5.1f}%  "
              f"({g['absolute_improvement_pct_points']:+.1f} pp, n={g['scenarios']})")
    tr = report["before_vs_after"]["failure_task_recovery"]
    print("  BEFORE vs AFTER - disrupted task recovery (robot-failure scenarios):")
    print(f"    without recovery : 0 / {tr['scenarios']} tasks recovered (a task "
          f"stuck on a failed robot cannot be moved)")
    print(f"    with recovery    : {tr['task_recovered_with_recovery']} / "
          f"{tr['scenarios']} ({tr['task_recovery_rate_with_recovery'] * 100:.1f}%); "
          f"{tr['task_recovered_when_spare_available']} / "
          f"{tr['spare_available_scenarios']} when a spare robot existed")
    print()
    print(f"  wrote {JSON_PATH.name} and {CSV_PATH.name} to {RESULTS_DIR}")


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


# ----------------------------------------------------------------------
def run_benchmark() -> Tuple[List[RunRecord], Dict[str, object]]:
    scenarios = generate_scenarios()
    records: List[RunRecord] = []
    for scenario in scenarios:
        record = run_scenario(scenario)
        if record is not None:
            records.append(record)
    return records, build_report(records)


def main() -> None:
    records, report = run_benchmark()
    write_json(report)
    write_csv(records)
    print_summary(report)


if __name__ == "__main__":
    main()
