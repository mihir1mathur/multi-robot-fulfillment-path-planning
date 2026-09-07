"""Tests for the consolidated final-evaluation orchestrator.

The orchestrator contains no algorithm code - it runs the existing benchmark
families and vouches for a small set of headline metrics. What has to be
correct here is the CONSISTENCY CHECKING (it must reject a self-contradictory
metric) and the report/CSV shape. One slow end-to-end check runs the whole
sweep in --quick mode against SQLite.
"""

from __future__ import annotations

import json

import pytest

from benchmarks.run_final_evaluation import (
    MetricConsistencyError,
    _check,
    _flatten,
    _is_rate,
    _non_negative,
    _round,
    relative_improvement,
    run_evaluation,
    write_csv,
    write_json,
)


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------
def test_relative_improvement_basic() -> None:
    assert relative_improvement(100, 25) == 0.75
    assert relative_improvement(10, 10) == 0.0


def test_relative_improvement_guards_divide_by_zero() -> None:
    assert relative_improvement(0, 0) is None
    assert relative_improvement(0, 5) is None


def test_is_rate_bounds() -> None:
    assert _is_rate(0.0) and _is_rate(1.0) and _is_rate(0.5)
    assert not _is_rate(-0.01)
    assert not _is_rate(1.5)
    assert not _is_rate("nope")


def test_non_negative() -> None:
    assert _non_negative(0) and _non_negative(3.2)
    assert not _non_negative(-1)


def test_round_passes_none_through() -> None:
    assert _round(None) is None
    assert _round(0.123456) == 0.1235


def test_check_raises_on_false() -> None:
    _check(True, "ok")  # no raise
    with pytest.raises(MetricConsistencyError):
        _check(False, "boom")


def test_flatten_scalars_and_scalar_lists() -> None:
    rows: list = []
    _flatten("f.headline", {"a": 1, "b": {"c": 2}, "d": [1, 2, 3], "e": [{"x": 1}]}, rows)
    flat = {r["metric"]: r["value"] for r in rows}
    assert flat["f.headline.a"] == 1
    assert flat["f.headline.b.c"] == 2
    assert flat["f.headline.d"] == "1; 2; 3"
    assert "f.headline.e" not in flat  # list of dicts is skipped in the CSV


# ----------------------------------------------------------------------
# End-to-end (quick sweep, SQLite) - slow but exercises every family and
# every consistency check on real measured numbers.
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def quick_report() -> dict:
    return run_evaluation(quick=True, backend="sqlite")


def test_quick_evaluation_has_every_family(quick_report: dict) -> None:
    assert set(quick_report["families"]) == {
        "path_planning",
        "task_allocation",
        "multi_robot_coordination",
        "dynamic_replanning",
        "failure_recovery",
        "service_backend",
        "concurrency_reliability",
    }


def test_quick_evaluation_environment_has_no_secret(quick_report: dict) -> None:
    blob = json.dumps(quick_report).lower()
    assert "password" not in blob
    assert "postgresql://" not in blob and "sqlite+pysqlite://" not in blob
    assert "jwt_secret" not in blob
    env = quick_report["environment"]
    assert env["database_dialect_configured"] in {"postgresql", "sqlite"}
    assert env["python_version"]


def test_quick_evaluation_headline_rates_are_bounded(quick_report: dict) -> None:
    for family in quick_report["families"].values():
        for name, value in family["headline"].items():
            if name.endswith("_rate") and isinstance(value, (int, float)):
                assert 0.0 <= value <= 1.0, (name, value)


def test_quick_evaluation_planning_astar_not_worse(quick_report: dict) -> None:
    h = quick_report["families"]["path_planning"]["headline"]
    assert h["astar_expanded_more_nodes"] == 0
    assert h["total_astar_nodes_expanded"] <= h["total_dijkstra_nodes_expanded"]
    assert h["optimal_cost_agreement_rate"] == 1.0


def test_quick_evaluation_coordination_leaves_no_unresolved_conflict(
    quick_report: dict,
) -> None:
    h = quick_report["families"]["multi_robot_coordination"]["headline"]
    assert h["unresolved_execution_conflicts"] == 0
    assert (
        h["naive_vertex_conflicts_faced"] + h["naive_edge_conflicts_faced"]
        == h["total_conflicts_coordination_faced"]
    )


def test_quick_evaluation_recovery_subset_not_larger_than_population(
    quick_report: dict,
) -> None:
    h = quick_report["families"]["failure_recovery"]["headline"]
    assert h["recoverable_subset_scenarios"] <= h["all_scenarios_before_vs_after"]["scenarios"]
    assert h["duplicate_task_completions"] == 0


def test_quick_evaluation_concurrency_integrity(quick_report: dict) -> None:
    h = quick_report["families"]["concurrency_reliability"]["headline"]
    assert h["failed_api_operations"] == 0
    assert h["duplicate_create_race"]["integrity_held"] is True
    assert h["fleet_workload_determinism"]["results_identical_sequential_vs_concurrent"] is True


def test_quick_evaluation_writes_json_and_csv(quick_report: dict, tmp_path) -> None:
    jp = tmp_path / "final_evaluation.json"
    cp = tmp_path / "final_evaluation.csv"
    write_json(quick_report, jp)
    write_csv(quick_report, cp)
    assert json.loads(jp.read_text())["families"]
    lines = cp.read_text().strip().splitlines()
    assert lines[0] == "metric,value"
    assert len(lines) > 10
