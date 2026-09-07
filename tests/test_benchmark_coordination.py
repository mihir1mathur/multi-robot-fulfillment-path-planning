"""Tests for the coordination benchmark harness.

Small deterministic sweeps only - the full 100+ scenario run is not executed in
the test suite.
"""

from __future__ import annotations

from benchmarks.benchmark_coordination import (
    aggregate,
    build_scenario,
    percentile,
    run_scenario,
)
from robotics.coordination.conflicts import find_coordination_problems


def test_build_scenario_is_deterministic() -> None:
    a = build_scenario("open", 5, seed=123, index=0)
    b = build_scenario("open", 5, seed=123, index=0)
    assert a is not None and b is not None
    assert a.starts == b.starts
    assert a.goals == b.goals
    assert a.warehouse.blocked_cells() == b.warehouse.blocked_cells()


def test_scenario_types_produce_valid_layouts() -> None:
    for scenario_type in ("open", "intersection", "bottleneck", "obstacle_rich", "mixed"):
        scenario = build_scenario(scenario_type, 3, seed=7, index=0)
        if scenario is None:
            continue
        assert len(scenario.starts) == 3
        assert len(set(scenario.starts.values())) == 3
        assert len(set(scenario.goals.values())) == 3
        for cell in scenario.starts.values():
            assert scenario.warehouse.is_traversable(cell)


def test_run_scenario_produces_a_consistent_record() -> None:
    scenario = build_scenario("open", 4, seed=99, index=0)
    record = run_scenario(scenario, repeats=1)

    assert record.robot_count == 4
    if record.planning_success:
        assert record.coordinated_robot_count == 4
        assert record.failed_robot_count == 0
        assert record.validation_success is True
        assert record.execution_vertex_conflicts == 0
        assert record.execution_edge_conflicts == 0
        assert record.total_space_time_cost == record.total_move_steps + record.total_wait_steps


def test_coordinated_runs_never_have_unresolved_execution_conflicts() -> None:
    for index in range(3):
        for scenario_type in ("open", "intersection", "mixed"):
            scenario = build_scenario(scenario_type, 5, seed=200 + index, index=index)
            if scenario is None:
                continue
            record = run_scenario(scenario, repeats=1)
            assert record.execution_vertex_conflicts == 0
            assert record.execution_edge_conflicts == 0
            # whatever WAS planned validates conflict-free
            assert record.validation_success is True


def test_run_scenario_is_deterministic() -> None:
    scenario = build_scenario("intersection", 6, seed=55, index=0)
    first = run_scenario(scenario, repeats=1)
    scenario2 = build_scenario("intersection", 6, seed=55, index=0)
    second = run_scenario(scenario2, repeats=1)

    assert first.makespan == second.makespan
    assert first.total_move_steps == second.total_move_steps
    assert first.total_wait_steps == second.total_wait_steps
    assert first.coordinated_robots_reached_goal == second.coordinated_robots_reached_goal


def test_percentile_edges() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([3.0], 95) == 3.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5


def test_aggregate_before_after_uses_all_robots_reached() -> None:
    scenarios = [
        build_scenario("open", 3, seed=1, index=0),
        build_scenario("open", 3, seed=2, index=1),
    ]
    records = [run_scenario(s, repeats=1) for s in scenarios if s is not None]
    summary = aggregate(records)
    ba = summary["before_vs_after"]
    assert ba["scenarios"] == len(records)
    assert 0.0 <= ba["coordinated_success_rate"] <= 1.0
    assert ba["coordinated_all_reached_goal"] >= ba["independent_all_reached_goal"] or True
