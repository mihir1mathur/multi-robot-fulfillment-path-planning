"""The concurrency benchmark: it runs, its integrity/determinism invariants
hold, and it is honestly labelled (not a load test)."""

from __future__ import annotations

import importlib
import json

bench = importlib.import_module("benchmarks.benchmark_concurrency")


def test_percentile_edges():
    assert bench._percentile([], 0.5) == 0.0
    assert bench._percentile([5.0], 0.95) == 5.0
    assert bench._percentile([0.0, 10.0], 0.5) == 5.0


def test_fleet_determinism_section_is_reproducible():
    a = bench._fleet_determinism()
    b = bench._fleet_determinism()
    assert a["results_identical"] is True
    assert a["per_fleet"] == b["per_fleet"]


def test_full_benchmark_runs_on_sqlite_and_invariants_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "CONCURRENCY_LEVELS", (1, 4))
    monkeypatch.setattr(bench, "REQUESTS_PER_LEVEL", 16)
    monkeypatch.setattr(bench, "RACE_THREADS", 10)
    monkeypatch.setattr(bench, "FLEET_SIZES", (2, 4))
    monkeypatch.setattr(bench, "OUT_JSON", tmp_path / "conc.json")
    monkeypatch.setattr(bench, "OUT_CSV", tmp_path / "conc.csv")

    rc = bench.main([])  # sqlite backend
    assert rc == 0

    report = json.loads((tmp_path / "conc.json").read_text())

    # A. every request is accounted for, none failed
    for row in report["api_concurrency"]:
        assert row["successful"] + row["failed"] == row["requests"]
        assert row["failed"] == 0

    # B. the create race resolved to exactly one winner, zero 5xx
    race = report["race_on_duplicate_create"]
    assert race["created_201"] == 1
    assert race["server_error_5xx"] == 0
    assert race["integrity_held"] is True
    assert race["rows_for_RACE_in_db"] == 1

    # C. sequential and concurrent fleet results are identical
    assert report["fleet_workload_determinism"]["results_identical"] is True

    # honest framing
    assert "not a load test" in report["config"]["note"].lower()
    assert "not production" in report["config"]["note"].lower()
