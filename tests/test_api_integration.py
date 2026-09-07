"""One deterministic end-to-end workflow through the HTTP API:

    create robots -> create tasks -> allocate (persisted, commit)
      -> plan a route -> coordinate the fleet
      -> inject a recoverable disruption -> recover
      -> read the final persisted state back

It deliberately does NOT assert any task reaches COMPLETED through execution -
the project does not model the pick/drop/complete steps, and the workflow stays
honest about that.
"""

from __future__ import annotations


def test_end_to_end_fleet_workflow(api_client):
    # 1. fleet
    assert api_client.post(
        "/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}}
    ).status_code == 201
    assert api_client.post(
        "/robots", json={"robot_id": "R2", "position": {"row": 0, "col": 1}}
    ).status_code == 201

    # 2. tasks
    assert api_client.post(
        "/tasks",
        json={
            "task_id": "T1",
            "pickup": {"row": 2, "col": 4},
            "dropoff": {"row": 11, "col": 1},
            "payload_weight": 3.0,
            "priority": "high",
        },
    ).status_code == 201

    # 3. allocate from persisted state and commit it
    allocation = api_client.post(
        "/allocation/run",
        json={"algorithm": "cp_sat", "from_persisted": True, "commit": True},
    ).json()
    assert allocation["committed"] is True
    assigned_robot = allocation["assignments"][0]["robot_id"]

    # 4. the commit is visible through the read endpoints
    assert api_client.get("/tasks/T1").json()["status"] == "assigned"
    assert api_client.get(f"/robots/{assigned_robot}").json()["assigned_task_id"] == "T1"

    # 5. plan a route for the assigned robot to its pickup
    robot = api_client.get(f"/robots/{assigned_robot}").json()
    plan = api_client.post(
        "/planning/path",
        json={
            "start": robot["position"],
            "goal": {"row": 2, "col": 4},
            "algorithm": "astar",
        },
    ).json()
    assert plan["success"] is True

    # 6. coordinate the two robots to distinct goals
    coordination = api_client.post(
        "/coordination/plan",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 3, "col": 0}},
                {"robot_id": "R2", "start": {"row": 0, "col": 1}, "goal": {"row": 3, "col": 1}},
            ]
        },
    ).json()
    assert coordination["success"] is True

    # 7. inject a recoverable obstacle and recover
    recovery = api_client.post(
        "/recovery/obstacle",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}
            ],
            "obstacle": {"row": 4, "col": 3},
            "timestep": 2,
        },
    ).json()
    assert recovery["recovery_success"] is True
    assert recovery["run_id"] > 0

    # 8. the run history persisted
    from robotics.persistence.repositories import RunRepository

    session = api_client.app.state.database.session()
    try:
        runs = RunRepository(session)
        assert runs.count_planning_runs() >= 1
        assert runs.count_recovery_events() == 1
        assert len(runs.recent_allocation_runs()) == 1
        assert len(runs.recent_coordination_runs()) == 1
    finally:
        session.close()

    # 9. final state read-back: no task was silently completed
    assert api_client.get("/tasks/T1").json()["status"] == "assigned"


def test_workflow_is_deterministic(api_client):
    """Same requests, same logical results (latencies aside)."""

    def run():
        api_client.post(
            "/robots", json={"robot_id": "R1", "position": {"row": 0, "col": 0}}
        )
        plan = api_client.post(
            "/planning/path",
            json={"start": {"row": 0, "col": 0}, "goal": {"row": 11, "col": 10}, "algorithm": "astar"},
        ).json()
        coord = api_client.post(
            "/coordination/plan",
            json={
                "robots": [
                    {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}},
                    {"robot_id": "R2", "start": {"row": 4, "col": 6}, "goal": {"row": 4, "col": 0}},
                ]
            },
        ).json()
        return (
            plan["total_cost"],
            plan["nodes_expanded"],
            plan["path"],
            coord["success"],
            coord["makespan"],
            coord["total_wait_steps"],
        )

    first = run()
    # wipe only the fleet + run history, keep the schema and the users
    with api_client.app.state.database.session_scope() as session:
        from robotics.persistence.models import RobotRecord, TaskRecord

        session.query(TaskRecord).delete()
        session.query(RobotRecord).delete()
    second = run()

    assert first == second
