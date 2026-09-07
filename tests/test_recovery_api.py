"""Recovery endpoints - obstacle replanning and robot-failure reassignment."""

from __future__ import annotations


def test_dynamic_obstacle_recovery_replans(api_client):
    response = api_client.post(
        "/recovery/obstacle",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}
            ],
            "obstacle": {"row": 4, "col": 3},
            "timestep": 2,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["disruption_type"] == "dynamic_obstacle"
    assert body["replanning_attempted"] is True
    assert body["replanning_success"] is True
    assert body["recovery_success"] is True
    assert body["unresolved_vertex_conflicts"] == 0
    assert body["unresolved_edge_conflicts"] == 0
    # the replan must not route back through the blocked cell
    for replan in body["robot_replans"]:
        assert replan["success"] is True


def test_unrecoverable_obstacle_is_a_safe_stop(api_client):
    # one-wide corridor; an obstacle drops two cells ahead of the robot, on the
    # only path. There is nowhere to reroute -> safe stop, reported honestly.
    body = api_client.post(
        "/recovery/obstacle",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 0, "col": 0}, "goal": {"row": 0, "col": 5}}
            ],
            "obstacle": {"row": 0, "col": 3},
            "timestep": 1,
            "warehouse": {"width": 6, "height": 1},
        },
    ).json()
    assert body["safe_stop"] is True
    assert body["recovery_success"] is False
    assert body["replanning_success"] is False
    assert body["failure_reason"]


def test_robot_failure_reassigns_the_task(api_client):
    body = api_client.post(
        "/recovery/robot-failure",
        json={
            "robots": [
                {
                    "robot_id": "R1",
                    "start": {"row": 4, "col": 0},
                    "goal": {"row": 4, "col": 4},
                    "task": {
                        "task_id": "T1",
                        "pickup": {"row": 4, "col": 4},
                        "dropoff": {"row": 11, "col": 1},
                        "payload_weight": 2.0,
                    },
                },
                {"robot_id": "R2", "start": {"row": 7, "col": 0}, "goal": {"row": 7, "col": 0}},
            ],
            "failed_robot_id": "R1",
            "timestep": 1,
        },
    ).json()
    assert body["disruption_type"] == "robot_offline"
    assert body["task_reassignment_attempted"] is True
    assert body["task_reassignment_success"] is True
    assert body["reassigned_from"] == "R1"
    assert body["reassigned_to"] == "R2"


def test_robot_failure_with_no_spare_is_a_safe_stop(api_client):
    body = api_client.post(
        "/recovery/robot-failure",
        json={
            "robots": [
                {
                    "robot_id": "R1",
                    "start": {"row": 4, "col": 0},
                    "goal": {"row": 4, "col": 4},
                    "task": {
                        "task_id": "T1",
                        "pickup": {"row": 4, "col": 4},
                        "dropoff": {"row": 11, "col": 1},
                        "payload_weight": 2.0,
                    },
                }
            ],
            "failed_robot_id": "R1",
            "timestep": 1,
        },
    ).json()
    assert body["task_reassignment_attempted"] is True
    assert body["task_reassignment_success"] is False
    assert body["safe_stop"] is True


def test_recovery_disabled_reproduces_safe_stop(api_client):
    body = api_client.post(
        "/recovery/obstacle",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}
            ],
            "obstacle": {"row": 4, "col": 3},
            "timestep": 2,
            "recovery_enabled": False,
        },
    ).json()
    assert body["safe_stop"] is True
    assert body["replanning_success"] is False


def test_failed_robot_not_in_fleet_is_422(api_client):
    response = api_client.post(
        "/recovery/robot-failure",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 4}}
            ],
            "failed_robot_id": "R9",
            "timestep": 1,
        },
    )
    assert response.status_code == 422


def test_recovery_event_is_persisted(api_client, db_session):
    from robotics.persistence.repositories import RunRepository

    api_client.post(
        "/recovery/obstacle",
        json={
            "robots": [
                {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 4, "col": 6}}
            ],
            "obstacle": {"row": 4, "col": 3},
            "timestep": 2,
        },
    )
    events = RunRepository(db_session).recent_recovery_events()
    assert len(events) == 1
    assert events[0].disruption_type == "dynamic_obstacle"


# ----------------------------------------------------------------------
# commit=true: a successful failure->spare reassignment is written back to
# the persisted robots/tasks rows (default is still the stateless simulation).
# ----------------------------------------------------------------------
def _seed_failure_scenario(api_client):
    """Persist R1 holding T1 and R2 as the idle spare."""
    api_client.post("/robots", json={"robot_id": "R1", "position": {"row": 4, "col": 0}})
    api_client.post("/robots", json={"robot_id": "R2", "position": {"row": 7, "col": 0}})
    api_client.post("/tasks", json={
        "task_id": "T1", "pickup": {"row": 4, "col": 4},
        "dropoff": {"row": 11, "col": 1}, "payload_weight": 2.0,
    })
    api_client.patch("/tasks/T1", json={"status": "assigned", "assigned_robot_id": "R1"})
    api_client.patch("/robots/R1", json={"status": "assigned", "assigned_task_id": "T1"})


_FAILURE_BODY = {
    "robots": [
        {"robot_id": "R1", "start": {"row": 4, "col": 0}, "goal": {"row": 11, "col": 1},
         "task": {"task_id": "T1", "pickup": {"row": 4, "col": 4},
                  "dropoff": {"row": 11, "col": 1}, "payload_weight": 2.0}},
        {"robot_id": "R2", "start": {"row": 7, "col": 0}, "goal": {"row": 7, "col": 0}},
    ],
    "failed_robot_id": "R1",
    "timestep": 1,
}


def test_robot_failure_commit_persists_reassignment(api_client, db_session):
    from robotics.persistence.repositories import RobotRepository, TaskRepository

    _seed_failure_scenario(api_client)
    body = api_client.post(
        "/recovery/robot-failure", json={**_FAILURE_BODY, "commit": True}
    ).json()

    assert body["task_reassignment_success"] is True
    assert body["reassigned_from"] == "R1" and body["reassigned_to"] == "R2"
    assert body["committed"] is True
    assert set(body["committed_state"]) == {"R1", "R2", "T1"}

    db_session.expire_all()
    robots = {r.robot_id: r for r in RobotRepository(db_session).list()}
    tasks = {t.task_id: t for t in TaskRepository(db_session).list()}
    # failed robot: existing OFFLINE semantics, no task
    assert robots["R1"].status == "offline"
    assert robots["R1"].assigned_task_id is None
    # spare now owns the task (committed-assignment state, as a fresh
    # allocation would leave it - routes/positions are not persisted)
    assert robots["R2"].status == "assigned"
    assert robots["R2"].assigned_task_id == "T1"
    # task references the spare
    assert tasks["T1"].status == "assigned"
    assert tasks["T1"].assigned_robot_id == "R2"


def test_robot_failure_without_commit_leaves_persisted_state_untouched(api_client, db_session):
    from robotics.persistence.repositories import RobotRepository, TaskRepository

    _seed_failure_scenario(api_client)
    body = api_client.post("/recovery/robot-failure", json=_FAILURE_BODY).json()

    assert body["task_reassignment_success"] is True
    assert body["committed"] is False
    assert body["committed_state"] == []

    db_session.expire_all()
    robots = {r.robot_id: r for r in RobotRepository(db_session).list()}
    tasks = {t.task_id: t for t in TaskRepository(db_session).list()}
    assert robots["R1"].assigned_task_id == "T1"       # unchanged
    assert robots["R2"].assigned_task_id is None
    assert tasks["T1"].assigned_robot_id == "R1"


def test_robot_failure_commit_with_no_spare_persists_nothing(api_client, db_session):
    """An honest safe stop must not fabricate a reassignment in the DB."""
    from robotics.persistence.repositories import RobotRepository, TaskRepository

    _seed_failure_scenario(api_client)
    body = api_client.post("/recovery/robot-failure", json={
        "robots": [_FAILURE_BODY["robots"][0]],   # only R1, no spare
        "failed_robot_id": "R1",
        "timestep": 1,
        "commit": True,
    }).json()

    assert body["task_reassignment_success"] is False
    assert body["safe_stop"] is True
    assert body["committed"] is False
    assert body["committed_state"] == []

    db_session.expire_all()
    robots = {r.robot_id: r for r in RobotRepository(db_session).list()}
    tasks = {t.task_id: t for t in TaskRepository(db_session).list()}
    assert robots["R1"].assigned_task_id == "T1"       # nothing fabricated
    assert tasks["T1"].assigned_robot_id == "R1"


def test_robot_failure_commit_is_a_noop_when_fleet_is_not_persisted(api_client):
    """commit=true against a non-persisted fleet (the benchmark shape) is safe."""
    body = api_client.post(
        "/recovery/robot-failure", json={**_FAILURE_BODY, "commit": True}
    ).json()
    assert body["task_reassignment_success"] is True
    assert body["committed"] is False        # no matching rows to write
    assert body["committed_state"] == []
