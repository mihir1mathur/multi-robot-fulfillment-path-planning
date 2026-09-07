"""Tests for the recovery benchmark harness (small deterministic checks only)."""

from __future__ import annotations

from benchmarks.benchmark_recovery import (
    RunRecord,
    build_report,
    generate_scenarios,
    percentile,
    run_scenario,
)


def _record(**overrides) -> RunRecord:
    base = dict(
        scenario_id="x", family="obstacle", grid_size=8, fleet_size=2,
        disruption_type="dynamic_obstacle", disruption_timestep=2,
        alternate_available=True, replacement_available=None,
        affected_robot_count=1, affected_task_count=0,
        replanning_attempted=True, replanning_success=True, recovery_success=True,
        safe_stop=False, task_reassignment_attempted=False,
        task_reassignment_success=False, additional_distance=2,
        wait_actions_introduced=0, reservations_released=10, reservations_created=12,
        replanning_latency_ms=0.5, total_recovery_latency_ms=0.6,
        unresolved_vertex_conflicts=0, unresolved_edge_conflicts=0,
        goal_completion_after=1.0, goal_completion_before=0.5,
        disrupted_task_recovered_before=False, disrupted_task_recovered_after=False,
    )
    base.update(overrides)
    return RunRecord(**base)


def test_scenarios_generate_deterministically() -> None:
    first = generate_scenarios()
    second = generate_scenarios()
    assert len(first) == len(second) > 0
    assert [s.scenario_id for s in first] == [s.scenario_id for s in second]
    # both event families are present
    families = {s.family for s in first}
    assert families == {"obstacle", "failure"}


def test_a_generated_scenario_runs_and_records_a_conflict_free_recovery() -> None:
    scenarios = [s for s in generate_scenarios() if s.family == "obstacle"][:5]
    for scenario in scenarios:
        record = run_scenario(scenario)
        if record is None:
            continue
        assert record.unresolved_vertex_conflicts == 0
        assert record.unresolved_edge_conflicts == 0
        assert record.total_recovery_latency_ms >= 0
        assert record.replanning_latency_ms >= 0


def test_run_scenario_is_deterministic() -> None:
    scenario = next(s for s in generate_scenarios() if s.family == "failure")
    a = run_scenario(scenario)
    scenario2 = next(s for s in generate_scenarios() if s.family == "failure")
    b = run_scenario(scenario2)
    assert (a is None) == (b is None)
    if a is not None:
        assert a.recovery_success == b.recovery_success
        assert a.task_reassignment_success == b.task_reassignment_success
        assert a.additional_distance == b.additional_distance


def test_percentile_edges() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([5.0], 95) == 5.0
    assert percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50) == 5


def test_report_never_hides_safe_stops() -> None:
    records = [
        _record(recovery_success=True, safe_stop=False),
        _record(recovery_success=False, safe_stop=True, replanning_success=False,
                goal_completion_after=0.5),
    ]
    report = build_report(records)
    assert report["overall"]["safe_stop_rate"] == 0.5
    assert report["overall"]["recovery_success_rate"] == 0.5
    assert report["total_scenarios"] == 2


def test_report_before_after_uses_completion_fractions() -> None:
    records = [
        _record(goal_completion_before=0.5, goal_completion_after=1.0),
        _record(goal_completion_before=1.0, goal_completion_after=1.0),
    ]
    ba = build_report(records)["before_vs_after"]["all_scenarios"]
    assert ba["mean_goal_completion_without_recovery"] == 0.75
    assert ba["mean_goal_completion_with_recovery"] == 1.0
    assert ba["absolute_improvement_pct_points"] == 25.0


def test_report_failure_task_recovery_is_zero_without_recovery() -> None:
    records = [
        _record(family="failure", replacement_available=True,
                disrupted_task_recovered_after=True),
        _record(family="failure", replacement_available=False,
                disrupted_task_recovered_after=False),
    ]
    tr = build_report(records)["before_vs_after"]["failure_task_recovery"]
    assert tr["task_recovered_without_recovery"] == 0
    assert tr["task_recovered_with_recovery"] == 1
    assert tr["task_recovered_when_spare_available"] == 1
