"""The service/API benchmark: structure, determinism of logical results,
honesty (failed requests are not hidden)."""

from __future__ import annotations

import importlib


bench = importlib.import_module("benchmarks.benchmark_service")


def test_percentile_edges():
    assert bench._percentile([], 0.5) == 0.0
    assert bench._percentile([5.0], 0.95) == 5.0
    assert bench._percentile([0.0, 10.0], 0.5) == 5.0
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert bench._percentile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0


def test_record_counts_statuses_and_success(monkeypatch):
    class _Resp:
        def __init__(self, code):
            self.status_code = code

    records: list = []
    bench._record(
        records,
        "demo",
        [1.0, 2.0, 3.0, 4.0],
        [_Resp(200), _Resp(201), _Resp(422), _Resp(404)],
    )
    row = records[0]
    assert row["requests"] == 4
    assert row["success_rate"] == 0.5  # 200 + 201
    assert row["status_counts"] == {"200": 1, "201": 1, "404": 1, "422": 1}


def test_full_benchmark_runs_and_is_logically_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "REQUESTS_PER_OP", 6)
    monkeypatch.setattr(bench, "REPEATS_FOR_DIRECT", 6)
    monkeypatch.setattr(bench, "OUT_JSON", tmp_path / "svc.json")
    monkeypatch.setattr(bench, "OUT_CSV", tmp_path / "svc.csv")

    def run_once():
        client, _ = bench._make_client("sqlite")
        records: list = []
        bench._bench_robot_crud(client, records)
        bench._bench_task_crud(client, records)
        over = [
            bench._bench_planning(client, records),
            bench._bench_allocation(client, records),
            bench._bench_coordination(client, records),
        ]
        bench._bench_recovery(client, records)
        # keep only the logical (non-latency) fields
        logical = [
            {k: v for k, v in r.items() if not k.endswith("_ms") and k != "status_counts"}
            for r in records
        ]
        for r, raw in zip(logical, records):
            r["status_counts"] = raw["status_counts"]
        return logical, [
            {k: v for k, v in o.items() if not k.endswith("_ms") and k != "relative_overhead_x"}
            for o in over
        ]

    first = run_once()
    second = run_once()
    assert first == second


def test_benchmark_reports_every_request_and_hides_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "REQUESTS_PER_OP", 5)
    monkeypatch.setattr(bench, "REPEATS_FOR_DIRECT", 5)
    monkeypatch.setattr(bench, "OUT_JSON", tmp_path / "svc.json")
    monkeypatch.setattr(bench, "OUT_CSV", tmp_path / "svc.csv")

    assert bench.main() == 0
    import json

    report = json.loads((tmp_path / "svc.json").read_text())
    assert report["total_requests"] == 5 * len(report["operations"])
    for row in report["operations"]:
        counted = sum(row["status_counts"].values())
        assert counted == row["requests"]  # every request accounted for
    assert "not production latency" in report["config"]["note"].lower()
