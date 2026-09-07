"""Invariants for the end-to-end system demo (scripts/run_system_demo.py).

The demo tells one deterministic story. These tests pin the story down: the
allocation, the coordination, the obstacle replan and the failure recovery must
all reach the same LOGICAL outcome on every run and every machine. Timings are
never asserted.
"""

from __future__ import annotations

import json

import pytest

from scripts.run_system_demo import build_simulator, run_scenario


@pytest.fixture(scope="module")
def demo() -> dict:
    return run_scenario().metrics


def test_scenario_is_logically_deterministic() -> None:
    a = run_scenario().metrics
    b = run_scenario().metrics
    assert json.dumps(a, sort_keys=True, default=str) == json.dumps(
        b, sort_keys=True, default=str
    )


def test_build_simulator_matches_declared_scenario() -> None:
    sim = build_simulator()
    assert {r.robot_id for r in sim.robots} == {"R1", "R2", "R3", "R4"}
    assert {t.task_id for t in sim.tasks} == {"T1", "T2", "T3"}


def test_allocation_assigns_every_task_and_keeps_a_spare(demo: dict) -> None:
    a = demo["allocation"]
    assert a["cp_sat_assigned"] == a["tasks_offered"] == 3
    assert a["cp_sat_valid"] is True
    assert a["committed"] is True
    assert len(a["spare_robots"]) >= 1
    # a lower cost is never from assigning fewer tasks here (cardinality is equal)
    assert a["cp_sat_total_estimated_cost"] <= a["greedy_total_estimated_cost"]


def test_planning_sample_agrees_on_cost_and_astar_expands_no_more(demo: dict) -> None:
    p = demo["planning_sample"]
    assert p["same_cost"] is True
    assert p["astar_nodes_expanded"] <= p["dijkstra_nodes_expanded"]


def test_coordination_leaves_no_unresolved_conflict(demo: dict) -> None:
    c = demo["coordination"]
    assert c["planned"] == c["robots"]
    assert c["unresolved_conflicts_after_coordination"] == 0
    # the independent routes really do conflict - otherwise coordination is
    # not being exercised
    assert c["naive_vertex_conflicts"] + c["naive_edge_conflicts"] >= 1


def test_dynamic_obstacle_triggers_a_successful_local_replan(demo: dict) -> None:
    r = demo["recovery"]
    assert r["obstacle_replan_attempted"] is True
    assert r["obstacle_replan_success"] is True
    assert len(r["obstacle_affected_robots"]) >= 1


def test_robot_failure_reassigns_the_task_to_the_spare(demo: dict) -> None:
    r = demo["recovery"]
    d = demo["disruption"]
    assert r["failure_reassignment_attempted"] is True
    assert r["failure_reassignment_success"] is True
    assert r["reassigned_from"] == d["failed_robot"]
    assert r["reassigned_to"] in d["spare_robots"]


def test_recovery_is_safe_and_clean(demo: dict) -> None:
    r = demo["recovery"]
    assert r["unresolved_vertex_conflicts"] == 0
    assert r["unresolved_edge_conflicts"] == 0
    assert r["trace_validates_conflict_free"] is True
    assert r["duplicate_task_completions"] == 0


def test_final_state_is_consistent(demo: dict) -> None:
    f = demo["final_state"]
    n_robots = len(f["robots"])
    reached = sum(1 for i in f["robots"].values() if i["reached_goal"])
    offline = sum(1 for i in f["robots"].values() if i["offline"])
    assert reached == f["robots_reached_goal"]
    assert offline == f["robots_offline"]
    assert reached + offline + f["safe_stops"] <= n_robots + 1
    assert f["episode_conflicts"] == 0
    # every task ends assigned or completed, never lost
    for info in f["tasks"].values():
        assert info["status"] in {"assigned", "in_progress", "completed"}
