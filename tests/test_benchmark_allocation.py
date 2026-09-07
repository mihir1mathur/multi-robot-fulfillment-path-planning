"""Tests for the allocation benchmark harness.

The benchmark is not algorithm code, but its scenario generation must be
deterministic and its aggregation maths must be right, or every reported number
is untrustworthy. These tests pin that down with tiny sweeps.
"""

from __future__ import annotations

from benchmarks.benchmark_allocation import (
    RunRecord,
    aggregate_by_algorithm,
    build_scenario,
    compare_cp_sat_vs_greedy,
    generate_scenarios,
    percentile,
    run_scenario,
)


def _record(algorithm, scenario_id, assigned, unassigned, cost) -> RunRecord:
    return RunRecord(
        algorithm=algorithm,
        scenario_id=scenario_id,
        robot_count=5,
        task_count=5,
        assigned_task_count=assigned,
        unassigned_task_count=unassigned,
        eligible_pair_count=10,
        infeasible_pair_count=2,
        total_estimated_travel_cost=cost,
        allocation_time_ms=1.0,
        solver_status="OPTIMAL",
        valid=True,
    )


def test_build_scenario_is_deterministic() -> None:
    a = build_scenario(5, 5, seed=42, index=0)
    b = build_scenario(5, 5, seed=42, index=0)

    assert [r.robot_id for r in a.robots] == [r.robot_id for r in b.robots]
    assert [(r.position.row, r.position.col) for r in a.robots] == [
        (r.position.row, r.position.col) for r in b.robots
    ]
    assert [t.task_id for t in a.tasks] == [t.task_id for t in b.tasks]
    assert a.warehouse.blocked_cells() == b.warehouse.blocked_cells()


def test_build_scenario_has_the_requested_counts() -> None:
    scenario = build_scenario(10, 20, seed=7, index=3)
    assert len(scenario.robots) == 10
    assert len(scenario.tasks) == 20
    assert scenario.robot_count == 10
    assert scenario.task_count == 20


def test_scenario_robots_and_task_endpoints_are_traversable() -> None:
    scenario = build_scenario(8, 12, seed=99, index=0)
    for robot in scenario.robots:
        assert scenario.warehouse.is_traversable(robot.position)
    for task in scenario.tasks:
        assert scenario.warehouse.is_traversable(task.pickup_location)
        assert scenario.warehouse.is_traversable(task.dropoff_location)
        assert task.pickup_location != task.dropoff_location


def test_generate_scenarios_count_and_determinism() -> None:
    first = generate_scenarios()
    second = generate_scenarios()
    assert len(first) == 5 * 8  # 5 configs x 8 scenarios
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]


def test_run_scenario_produces_one_record_per_allocator() -> None:
    scenario = build_scenario(5, 5, seed=1, index=0)
    records, exec_record = run_scenario(scenario, repeats=1)

    assert {r.algorithm for r in records} == {"greedy", "cp_sat"}
    assert exec_record.scenario_id == scenario.scenario_id
    assert exec_record.executed_steps <= exec_record.planned_steps


def test_both_allocators_produce_valid_results_on_a_sample() -> None:
    for index in range(3):
        scenario = build_scenario(5, 8, seed=100 + index, index=index)
        records, _ = run_scenario(scenario, repeats=1)
        assert all(r.valid for r in records)


def test_cp_sat_never_assigns_fewer_tasks_than_greedy_on_a_sample() -> None:
    for index in range(4):
        scenario = build_scenario(8, 10, seed=200 + index, index=index)
        records, _ = run_scenario(scenario, repeats=1)
        by_algo = {r.algorithm: r for r in records}
        assert (
            by_algo["cp_sat"].assigned_task_count
            >= by_algo["greedy"].assigned_task_count
        )


def test_percentile_edges() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5


def test_compare_restricts_cost_to_equal_cardinality_scenarios() -> None:
    records = [
        # scenario A: same count (2 vs 2), CP-SAT cheaper
        _record("greedy", "A", 2, 0, 100),
        _record("cp_sat", "A", 2, 0, 70),
        # scenario B: CP-SAT assigned more - excluded from cost comparison
        _record("greedy", "B", 1, 1, 30),
        _record("cp_sat", "B", 2, 0, 90),
    ]
    summary = compare_cp_sat_vs_greedy(records)

    assert summary["scenarios_compared_same_cardinality"] == 1
    assert summary["scenarios_cp_sat_assigned_more_tasks"] == 1
    assert summary["cp_sat_strictly_cheaper"] == 1
    # only scenario A counts: (100 - 70) / 100 = 30%
    assert summary["cp_sat_cost_improvement_pct"] == 30.0


def test_aggregate_by_algorithm_computes_assignment_rate() -> None:
    records = [
        _record("greedy", "A", 3, 2, 50),  # 3 of 5
        _record("greedy", "B", 4, 1, 60),  # 4 of 5
    ]
    summary = aggregate_by_algorithm(records)
    assert len(summary) == 1
    assert summary[0]["assignment_rate"] == round(7 / 10, 4)
