"""Tests for the planner benchmark harness.

The benchmark is not algorithm code, but its helpers still have to be correct:
if scenario generation were not deterministic, or the aggregation maths were
wrong, every reported number would be untrustworthy. These tests pin down that
behaviour without running the full (slow) benchmark sweep.
"""

from __future__ import annotations

from benchmarks.benchmark_planners import (
    RunRecord,
    aggregate,
    build_scenario,
    cost_agreement_rate,
    generate_scenarios,
    head_to_head,
    percentile,
    run_scenario,
)
from robotics.warehouse.grid import Position


# ----------------------------------------------------------------------
# Scenario construction
# ----------------------------------------------------------------------
def test_build_scenario_is_deterministic_for_a_seed() -> None:
    first = build_scenario(20, 20, 0.15, seed=123, scenario_index=0)
    second = build_scenario(20, 20, 0.15, seed=123, scenario_index=0)

    assert first.start == second.start
    assert first.goal == second.goal
    assert first.obstacle_count == second.obstacle_count
    assert first.warehouse.blocked_cells() == second.warehouse.blocked_cells()


def test_different_seeds_give_different_layouts() -> None:
    a = build_scenario(20, 20, 0.15, seed=1, scenario_index=0)
    b = build_scenario(20, 20, 0.15, seed=2, scenario_index=0)

    assert a.warehouse.blocked_cells() != b.warehouse.blocked_cells()


def test_start_and_goal_are_always_traversable() -> None:
    for seed in range(15):
        scenario = build_scenario(15, 15, 0.30, seed=seed, scenario_index=seed)
        assert scenario.warehouse.is_traversable(scenario.start)
        assert scenario.warehouse.is_traversable(scenario.goal)
        assert scenario.start != scenario.goal


def test_obstacle_count_matches_the_blocked_cells() -> None:
    scenario = build_scenario(25, 25, 0.2, seed=7, scenario_index=0)
    assert scenario.obstacle_count == len(scenario.warehouse.blocked_cells())


def test_higher_density_blocks_more_cells_on_average() -> None:
    sparse = build_scenario(30, 30, 0.05, seed=99, scenario_index=0)
    dense = build_scenario(30, 30, 0.35, seed=99, scenario_index=0)
    assert dense.obstacle_count > sparse.obstacle_count


def test_generate_scenarios_count_and_determinism() -> None:
    first = generate_scenarios([20], [0.1, 0.2], per_config=3, master_seed=5)
    second = generate_scenarios([20], [0.1, 0.2], per_config=3, master_seed=5)

    assert len(first) == 2 * 3
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]
    assert [s.obstacle_count for s in first] == [s.obstacle_count for s in second]


# ----------------------------------------------------------------------
# Running a scenario
# ----------------------------------------------------------------------
def test_run_scenario_produces_one_record_per_planner() -> None:
    scenario = build_scenario(20, 20, 0.1, seed=3, scenario_index=0)
    records = run_scenario(scenario, repeats=1)

    algorithms = {r.algorithm for r in records}
    assert algorithms == {"astar", "dijkstra"}


def test_both_planners_agree_on_cost_across_a_sample() -> None:
    for scenario in generate_scenarios([20, 30], [0.1, 0.2], per_config=4, master_seed=11):
        records = {r.algorithm: r for r in run_scenario(scenario, repeats=1)}
        astar, dijkstra = records["astar"], records["dijkstra"]
        assert astar.success == dijkstra.success
        if astar.success:
            assert astar.path_cost == dijkstra.path_cost


def test_astar_never_expands_more_than_dijkstra_in_the_sample() -> None:
    for scenario in generate_scenarios([25], [0.05, 0.15], per_config=4, master_seed=13):
        records = {r.algorithm: r for r in run_scenario(scenario, repeats=1)}
        if records["astar"].success:
            assert (
                records["astar"].nodes_expanded
                <= records["dijkstra"].nodes_expanded
            )


# ----------------------------------------------------------------------
# Aggregation maths
# ----------------------------------------------------------------------
def test_percentile_basic_cases() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([42.0], 95) == 42.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100) == 10
    assert percentile([10, 1, 5, 3], 25) == 1


def _record(algorithm: str, scenario_id: str, cost, nodes: int, ok: bool = True) -> RunRecord:
    return RunRecord(
        algorithm=algorithm,
        scenario_id=scenario_id,
        grid_size=20,
        obstacle_density=0.1,
        obstacle_count=10,
        success=ok,
        path_cost=cost,
        nodes_expanded=nodes,
        planning_time_ms=1.0,
    )


def test_cost_agreement_rate_all_agree() -> None:
    records = [
        _record("astar", "s0", 10.0, 5),
        _record("dijkstra", "s0", 10.0, 20),
        _record("astar", "s1", 8.0, 4),
        _record("dijkstra", "s1", 8.0, 16),
    ]
    assert cost_agreement_rate(records) == 1.0


def test_cost_agreement_rate_none_when_nothing_comparable() -> None:
    records = [
        _record("astar", "s0", 10.0, 5),
        _record("dijkstra", "s0", None, 0, ok=False),
    ]
    assert cost_agreement_rate(records) is None


def test_cost_agreement_rate_detects_disagreement() -> None:
    records = [
        _record("astar", "s0", 10.0, 5),
        _record("dijkstra", "s0", 11.0, 20),
    ]
    assert cost_agreement_rate(records) == 0.0


def test_aggregate_groups_by_algorithm_size_and_density() -> None:
    records = [
        _record("astar", "s0", 10.0, 5),
        _record("astar", "s1", 12.0, 7),
        _record("dijkstra", "s0", 10.0, 20),
        _record("dijkstra", "s1", 12.0, 24),
    ]
    summaries = aggregate(records)
    assert len(summaries) == 2
    astar_summary = next(s for s in summaries if s["algorithm"] == "astar")
    assert astar_summary["scenarios"] == 2
    assert astar_summary["success_rate"] == 1.0
    assert astar_summary["median_nodes_expanded"] == 6


def test_head_to_head_counts_node_comparisons() -> None:
    records = [
        _record("astar", "s0", 10.0, 5),
        _record("dijkstra", "s0", 10.0, 20),
        _record("astar", "s1", 12.0, 30),
        _record("dijkstra", "s1", 12.0, 30),
    ]
    summary = head_to_head(records)
    assert summary["scenarios_both_succeeded"] == 2
    assert summary["astar_expanded_fewer_nodes"] == 1
    assert summary["astar_expanded_equal_nodes"] == 1
    assert summary["optimal_cost_agreement_rate"] == 1.0
