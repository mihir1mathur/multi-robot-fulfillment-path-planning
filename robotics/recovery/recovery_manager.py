"""RecoveryManager: resilient coordinated execution.

It drives a coordinated plan forward one synchronised timestep at a time (like
`CoordinatedExecutor`), but at the top of each timestep it also applies any
scheduled disruption events and, if one invalidates part of the plan, runs a
recovery:

    * DYNAMIC OBSTACLE
        - the cell is added to the warehouse
        - robots whose remaining route touches it are found
        - their stale future reservations are released
        - each is replanned with Space-Time A* FROM ITS CURRENT CELL
        - the affected subset is re-coordinated (each replan reserves its path
          so the next one avoids it); unaffected robots are never touched
        - if a robot cannot be replanned it stops safely where it is

    * ROBOT OFFLINE
        - the robot is set OFFLINE and stops receiving move commands
        - it stays at its current cell, which becomes a fixed obstacle for the
          reservation table
        - its future reservations are released
        - its task is returned to PENDING (through the Task API) and offered to
          the remaining feasible robots via the existing allocator
        - if a replacement is committed, a route is planned for it and it joins
          the running episode
        - robots whose route ran through the failed cell are replanned

SAFETY
------
Every timestep still goes through validate-first / commit-second. If recovery
cannot produce a conflict-free continuation, the affected robot(s) stop; the
run reports a safe stop rather than stepping into danger.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from robotics.allocation.commit import commit_allocation
from robotics.allocation.cost_estimator import CostEstimator
from robotics.allocation.greedy_allocator import GreedyAllocator
from robotics.coordination.conflicts import find_coordination_problems
from robotics.coordination.coordination_result import CoordinationResult
from robotics.coordination.coordinator import default_horizon
from robotics.coordination.reservation_table import ReservationTable
from robotics.coordination.timed_path import TimedPath, TimedStep
from robotics.exceptions import RoboticsError
from robotics.recovery.disruption import (
    DisruptionSchedule,
    DisruptionType,
    DynamicObstacleEvent,
    RecoveryTrigger,
    RobotFailureEvent,
)
from robotics.recovery.dynamic_replanner import (
    DynamicReplanner,
    build_future_reservations,
    route_touches_cell,
)
from robotics.recovery.recovery_result import (
    RecoveryEventResult,
    ResilientExecutionResult,
    RobotReplan,
)
from robotics.robots.robot_state import RobotStatus
from robotics.tasks.task import TaskStatus
from robotics.warehouse.grid import MOVE_COST, Position


@dataclass
class _State:
    """Mutable per-run state, threaded through the recovery loop."""

    active_paths: Dict[str, TimedPath]
    goals: Dict[str, Position]
    executed_trace: Dict[str, List[TimedStep]]
    horizon: int
    offline: Set[str] = field(default_factory=set)
    safe_stopped: Set[str] = field(default_factory=set)
    battery_before: Dict[str, float] = field(default_factory=dict)
    odometer_before: Dict[str, float] = field(default_factory=dict)
    move_counts: Dict[str, int] = field(default_factory=dict)
    wait_counts: Dict[str, int] = field(default_factory=dict)
    joined: Set[str] = field(default_factory=set)

    def live_robots(self) -> List[str]:
        """Robots still under coordinated control (not failed)."""
        return sorted(r for r in self.active_paths if r not in self.offline)


class RecoveryManager:
    """Owns resilient execution: normal driving plus disruption recovery."""

    def __init__(self, simulator) -> None:
        self.simulator = simulator
        self.warehouse = simulator.warehouse
        self._replanner = DynamicReplanner(self.warehouse)
        self._recovery_enabled = True

    # ==================================================================
    # Public entry point
    # ==================================================================
    def run(
        self,
        coordination: CoordinationResult,
        disruptions: Optional[DisruptionSchedule] = None,
        max_extra_timesteps: int = 0,
        recovery_enabled: bool = True,
    ) -> ResilientExecutionResult:
        """Execute `coordination`'s timed paths, applying `disruptions` and
        recovering where possible.

        `recovery_enabled=False` reproduces the pre-recovery behaviour: a
        disruption that invalidates a route makes the affected robots stop
        safely, with no replanning or reassignment. Used by the benchmark for
        an apples-to-apples "before" comparison.
        """
        self._recovery_enabled = recovery_enabled
        started_ns = time.perf_counter_ns()

        def elapsed_ms() -> float:
            return (time.perf_counter_ns() - started_ns) / 1_000_000

        disruptions = disruptions or DisruptionSchedule()

        paths = dict(coordination.successful_paths)
        if not paths:
            return ResilientExecutionResult(
                success=True, robot_count=0, execution_time_ms=elapsed_ms()
            )

        # --- pre-flight: structure + robots present at their planned start ---
        problems = find_coordination_problems(
            paths, self.warehouse, check_obstacles=False
        )
        if problems:
            return ResilientExecutionResult(
                success=False,
                robot_count=len(paths),
                validation_problems=problems,
                execution_time_ms=elapsed_ms(),
                failure_reason="initial plan failed independent conflict validation",
            )
        for rid, tp in sorted(paths.items()):
            robot = self.simulator.get_robot(rid)
            if robot.position != tp.start:
                return ResilientExecutionResult(
                    success=False,
                    robot_count=len(paths),
                    execution_time_ms=elapsed_ms(),
                    failure_reason=(
                        f"robot '{rid}' is at {robot.position}, its plan starts "
                        f"at {tp.start}"
                    ),
                )

        state = _State(
            active_paths=paths,
            goals={rid: tp.goal for rid, tp in paths.items()},
            executed_trace={rid: [TimedStep(tp.start, 0)] for rid, tp in paths.items()},
            horizon=coordination.horizon
            or default_horizon(self.warehouse, len(paths)),
        )
        for rid in paths:
            robot = self.simulator.get_robot(rid)
            state.battery_before[rid] = robot.battery_level
            state.odometer_before[rid] = robot.distance_travelled
            state.move_counts[rid] = 0
            state.wait_counts[rid] = 0

        recovery_events: List[RecoveryEventResult] = []
        vertex_conflicts = 0
        edge_conflicts = 0
        episode_safe_stop = False
        stop_reason: Optional[str] = None

        limit = disruptions.last_timestep + 4 * (self.warehouse.width + self.warehouse.height) \
            + max_extra_timesteps
        t = 1
        while t <= limit:
            now = t - 1

            # --- apply disruptions that fire at this timestep ---
            for event in disruptions.events_at(now):
                recovery_events.append(self._handle_event(event, now, state))

            if self._all_done(state):
                break

            # --- PHASE 1: validate the joint step for the live robots ---
            movers = [
                rid for rid in state.live_robots()
                if rid not in state.safe_stopped
            ]
            intents = self._intents_at(state, movers, t)
            ok, reason, vc, ec = self._validate_joint_step(movers, intents, state)
            vertex_conflicts += vc
            edge_conflicts += ec
            if not ok:
                episode_safe_stop = True
                stop_reason = f"safe stop before timestep {t}: {reason}"
                break

            # --- PHASE 2: commit ---
            self._commit(movers, intents, t, state)
            t += 1

        # --- assemble the result ------------------------------------------
        return self._finalise(
            state, recovery_events, vertex_conflicts, edge_conflicts,
            episode_safe_stop, stop_reason, elapsed_ms(),
        )

    # ==================================================================
    # Disruption handling
    # ==================================================================
    def _handle_event(self, event, now: int, state: _State) -> RecoveryEventResult:
        started = time.perf_counter_ns()
        if event.kind is DisruptionType.DYNAMIC_OBSTACLE:
            result = self._recover_obstacle(event, now, state)
        else:
            result = self._recover_failure(event, now, state)
        result.total_recovery_latency_ms = (
            time.perf_counter_ns() - started
        ) / 1_000_000
        return result

    # ------------------------------------------------------------------
    def _recover_obstacle(
        self, event: DynamicObstacleEvent, now: int, state: _State
    ) -> RecoveryEventResult:
        result = RecoveryEventResult(
            trigger=RecoveryTrigger.DYNAMIC_OBSTACLE,
            trigger_timestep=now,
            description=str(event),
        )

        # add the obstacle to the world (unless a robot is standing there)
        try:
            self.simulator.add_dynamic_obstacle(
                event.resolved_obstacle_id(), event.position, "disruption"
            )
        except RoboticsError as error:
            result.recovery_success = True  # nothing to recover
            result.failure_reason = f"obstacle not applied: {error}"
            return result

        affected = self._robots_hit_by_obstacle(event.position, now, state)
        result.affected_robot_ids = affected
        if not affected:
            result.recovery_success = True
            result.resumed = True
            return result

        if not self._recovery_enabled:
            # "before" behaviour: the affected robots stop safely, no replan
            for rid in affected:
                state.safe_stopped.add(rid)
                current = state.executed_trace[rid][-1].position
                state.active_paths[rid] = self._splice(
                    state.executed_trace[rid],
                    TimedPath.found(rid, [TimedStep(current, now)]),
                )
                result.reservations_released += self._remaining_reservation_count(
                    state.active_paths.get(rid), now, state.horizon
                )
            result.safe_stop = True
            result.failure_reason = "recovery disabled; affected robots stopped safely"
            return result

        self._replan_affected(
            affected, now, state, result, RecoveryTrigger.ROUTE_INVALIDATED
        )
        self._score_recovery(affected, now, state, result)
        return result

    # ------------------------------------------------------------------
    def _recover_failure(
        self, event: RobotFailureEvent, now: int, state: _State
    ) -> RecoveryEventResult:
        result = RecoveryEventResult(
            trigger=RecoveryTrigger.ROBOT_OFFLINE,
            trigger_timestep=now,
            description=str(event),
        )
        failed_id = event.robot_id

        try:
            robot = self.simulator.get_robot(failed_id)
        except RoboticsError as error:
            result.failure_reason = str(error)
            return result

        # 1. stop the robot: it goes OFFLINE and stays where it is
        failed_cell = state.executed_trace.get(
            failed_id, [TimedStep(robot.position, now)]
        )[-1].position
        robot.set_status(RobotStatus.OFFLINE)
        state.offline.add(failed_id)
        if failed_id in state.active_paths:
            state.active_paths[failed_id] = TimedPath.found(
                failed_id,
                list(state.executed_trace.get(failed_id, [TimedStep(failed_cell, now)])),
            )

        if not self._recovery_enabled:
            # "before" behaviour: the robot fails and stays there; its task is
            # not reassigned and other robots are not replanned. Robots that
            # need the failed cell will stop safely at the joint-step check.
            result.affected_robot_ids = [failed_id]
            result.safe_stop = True
            result.failure_reason = "recovery disabled; failed robot not replaced"
            if robot.assigned_task_id:
                result.affected_task_ids = [robot.assigned_task_id]
            return result

        # 2. release its future reservations (conceptually - counted here)
        old_tp = state.active_paths.get(failed_id)
        result.reservations_released += self._remaining_reservation_count(
            old_tp, now, state.horizon
        ) if old_tp else 0

        # 3. task recovery
        affected_task = self._recover_task(failed_id, robot, now, state, result)

        # 4. robots whose route ran through the failed cell must be replanned
        others = [
            rid for rid in state.live_robots()
            if rid != failed_id
            and rid not in state.safe_stopped
            and route_touches_cell(state.active_paths[rid], failed_cell, now)
        ]
        if others:
            self._replan_affected(
                others, now, state, result, RecoveryTrigger.ROUTE_INVALIDATED
            )

        result.affected_robot_ids = sorted({failed_id, *others})
        if affected_task:
            result.affected_task_ids = [affected_task]

        # 5. score: recovery counts as a success only if every OTHER affected
        #    robot got a valid continuation and the task was handled cleanly
        others_ok = all(
            r.success for r in result.robot_replans if r.robot_id != failed_id
        )
        task_ok = (not result.task_reassignment_attempted) or result.task_reassignment_success
        result.replanning_attempted = bool(result.robot_replans) or result.replanning_attempted
        result.replanning_success = others_ok and (
            not any(r.robot_id in others for r in result.robot_replans)
            or all(r.success for r in result.robot_replans if r.robot_id in others)
        )
        result.recovery_success = others_ok and task_ok
        result.safe_stop = not result.recovery_success
        result.resumed = others_ok
        if not result.recovery_success and result.failure_reason is None:
            result.failure_reason = (
                "no feasible replacement robot for the released task"
                if result.task_reassignment_attempted
                and not result.task_reassignment_success
                else "an affected robot could not be replanned"
            )

        self._score_recovery(result.affected_robot_ids, now, state, result)
        return result

    # ------------------------------------------------------------------
    def _recover_task(
        self, failed_id: str, robot, now: int, state: _State,
        result: RecoveryEventResult,
    ) -> Optional[str]:
        """Release the failed robot's task and offer it to the fleet."""
        task_id = robot.assigned_task_id
        if not task_id:
            # the robot may still be OFFLINE without a task
            if robot.status is not RobotStatus.OFFLINE:
                robot.set_status(RobotStatus.OFFLINE)
            return None

        try:
            task = self.simulator.get_task(task_id)
        except RoboticsError:
            return None

        result.affected_task_ids.append(task_id)

        # return the task to the pending pool through the Task API
        if task.status is TaskStatus.ASSIGNED:
            task.unassign()               # ASSIGNED -> PENDING
            robot.clear_task()            # robot -> IDLE, no task
            robot.set_status(RobotStatus.OFFLINE)
        elif task.status is TaskStatus.IN_PROGRESS:
            task.fail(f"robot '{failed_id}' went offline mid-task")
            robot.clear_task()
            robot.set_status(RobotStatus.OFFLINE)
            result.reassigned_task_status = task.status.value
            return task_id
        else:
            robot.set_status(RobotStatus.OFFLINE)
            return task_id

        result.task_reassignment_attempted = True
        result.reassigned_from = failed_id
        result.reassigned_task_id = task_id

        # offer it to the remaining feasible robots via the existing allocator
        rec_started = time.perf_counter_ns()
        estimator = CostEstimator(self.warehouse)
        candidates = [
            r for r in self.simulator.robots
            if r.is_available and r.robot_id not in state.offline
        ]
        allocation = GreedyAllocator(estimator).allocate(candidates, [task])
        if not allocation.assignments:
            result.reassigned_task_status = task.status.value  # still PENDING
            result.recoordination_latency_ms += (
                time.perf_counter_ns() - rec_started
            ) / 1_000_000
            return task_id

        report = commit_allocation(self.simulator, allocation)
        if not report.committed:
            result.reassigned_task_status = task.status.value
            return task_id

        new_rid = allocation.assignments[0].robot_id
        # plan the replacement's route to the pickup, coordinated with the fleet
        pickup = task.pickup_location
        future = self._future_table(now, state, exclude={new_rid})
        new_robot = self.simulator.get_robot(new_rid)
        suffix, plan_ms = self._replanner.timed_replan(
            new_rid, new_robot.position, pickup, now, future, state.horizon
        )
        result.recoordination_latency_ms += (
            time.perf_counter_ns() - rec_started
        ) / 1_000_000

        replan = RobotReplan(
            robot_id=new_rid,
            original_goal=pickup,
            current_position=new_robot.position,
            trigger_timestep=now,
            old_remaining_path=[],
            new_path=suffix.positions() if suffix.success else [],
            success=suffix.success,
            failure_reason=suffix.failure_reason,
            replanning_time_ms=plan_ms,
            wait_actions_introduced=suffix.wait_count if suffix.success else 0,
            joined_mid_episode=True,
        )
        result.robot_replans.append(replan)
        result.replanning_attempted = True

        if not suffix.success:
            # the task is committed but its robot cannot be routed -> undo
            task.unassign()
            new_robot.clear_task()
            result.reassigned_task_status = task.status.value
            result.task_reassignment_success = False
            return task_id

        state.active_paths[new_rid] = suffix
        state.goals[new_rid] = pickup
        state.executed_trace[new_rid] = [TimedStep(new_robot.position, now)]
        state.battery_before[new_rid] = new_robot.battery_level
        state.odometer_before[new_rid] = new_robot.distance_travelled
        state.move_counts[new_rid] = 0
        state.wait_counts[new_rid] = 0
        state.joined.add(new_rid)
        result.reservations_created += self._remaining_reservation_count(
            suffix, now, state.horizon
        )
        result.task_reassignment_success = True
        result.reassigned_to = new_rid
        result.reassigned_task_status = task.status.value  # ASSIGNED
        return task_id

    # ------------------------------------------------------------------
    def _replan_affected(
        self, affected: Sequence[str], now: int, state: _State,
        result: RecoveryEventResult, trigger: RecoveryTrigger,
    ) -> None:
        """Replan each affected robot from its current cell, one at a time."""
        result.replanning_attempted = True

        # release the affected robots' stale future reservations (counted)
        for rid in affected:
            result.reservations_released += self._remaining_reservation_count(
                state.active_paths.get(rid), now, state.horizon
            )

        future = self._future_table(now, state, exclude=set(affected))

        all_ok = True
        for rid in sorted(affected):
            tp = state.active_paths[rid]
            current = state.executed_trace[rid][-1].position
            goal = state.goals[rid]
            old_remaining = [
                s.position for s in tp.steps if s.timestep >= now
            ]

            suffix, plan_ms = self._replanner.timed_replan(
                rid, current, goal, now, future, state.horizon
            )
            replan = RobotReplan(
                robot_id=rid,
                original_goal=goal,
                current_position=current,
                trigger_timestep=now,
                old_remaining_path=old_remaining,
                new_path=suffix.positions() if suffix.success else [],
                success=suffix.success,
                failure_reason=suffix.failure_reason,
                replanning_time_ms=plan_ms,
                wait_actions_introduced=suffix.wait_count if suffix.success else 0,
            )
            result.robot_replans.append(replan)
            result.replanning_latency_ms += plan_ms

            if suffix.success:
                future.reserve_timed_path(suffix, state.horizon)
                result.reservations_created += self._remaining_reservation_count(
                    suffix, now, state.horizon
                )
                state.active_paths[rid] = self._splice(state.executed_trace[rid], suffix)
            else:
                # safe stop: this robot holds its position for the rest of the run
                all_ok = False
                state.safe_stopped.add(rid)
                state.active_paths[rid] = self._splice(
                    state.executed_trace[rid],
                    TimedPath.found(rid, [TimedStep(current, now)]),
                )

        result.replanning_success = all_ok
        result.recovery_success = all_ok
        result.safe_stop = not all_ok
        result.resumed = any(r.success for r in result.robot_replans)
        if not all_ok and result.failure_reason is None:
            failed = [r.robot_id for r in result.robot_replans if not r.success]
            result.failure_reason = (
                f"no conflict-free continuation for {failed}; they stopped safely"
            )

    # ------------------------------------------------------------------
    def _score_recovery(
        self, affected: Sequence[str], now: int, state: _State,
        result: RecoveryEventResult,
    ) -> None:
        """Independently check that the post-recovery plan is conflict-free."""
        remaining = {}
        for rid in state.live_robots():
            if rid in state.safe_stopped:
                continue
            tp = state.active_paths[rid]
            steps = [s for s in tp.steps if s.timestep >= now]
            if len(steps) >= 1:
                remaining[rid] = TimedPath.found(rid, steps)
        problems = find_coordination_problems(
            remaining, self.warehouse, check_obstacles=False
        )
        result.unresolved_vertex_conflicts = sum(
            1 for p in problems if p.startswith("vertex conflict")
        )
        result.unresolved_edge_conflicts = sum(
            1 for p in problems if p.startswith("edge/swap conflict")
        )

    # ==================================================================
    # Timestep execution
    # ==================================================================
    def _intents_at(self, state: _State, movers: Sequence[str], t: int):
        intents = {}
        for rid in movers:
            tp = state.active_paths[rid]
            intents[rid] = (tp.position_at(t - 1), tp.position_at(t))
        return intents

    def _validate_joint_step(self, movers, intents, state: _State):
        warehouse = self.warehouse

        for rid in movers:
            frm, to = intents[rid]
            robot = self.simulator.get_robot(rid)
            if robot.status is RobotStatus.OFFLINE:
                return False, f"robot '{rid}' is offline", 0, 0
            if robot.position != frm:
                return False, f"robot '{rid}' at {robot.position}, expected {frm}", 0, 0
            if frm == to:
                if not warehouse.is_traversable(frm):
                    return False, f"robot '{rid}' cannot hold {frm}: now blocked", 0, 0
                continue
            if not warehouse.in_bounds(to):
                return False, f"robot '{rid}' move to {to} out of bounds", 0, 0
            if not warehouse.is_traversable(to):
                return False, f"robot '{rid}' next cell {to} is now blocked", 0, 0
            if not frm.is_adjacent_to(to):
                return False, f"robot '{rid}' move {frm}->{to} is not one step", 0, 0
            if not robot.has_battery_for_move():
                return False, f"robot '{rid}' has no battery to move", 0, 0

        # also: no live robot may move onto an OFFLINE robot's cell
        offline_cells = {
            self.simulator.get_robot(o).position for o in state.offline
            if self._robot_exists(o)
        }
        dests: Dict[Position, str] = {}
        for rid in movers:
            _, to = intents[rid]
            if to in offline_cells:
                return False, f"robot '{rid}' would move onto failed robot at {to}", 0, 0
            if to in dests:
                return False, f"vertex conflict: '{dests[to]}' and '{rid}' both want {to}", 1, 0
            dests[to] = rid

        move_map = {rid: intents[rid] for rid in movers if intents[rid][0] != intents[rid][1]}
        for rid, (frm, to) in move_map.items():
            for other, (o_frm, o_to) in move_map.items():
                if other <= rid:
                    continue
                if frm == o_to and to == o_frm:
                    return False, f"edge/swap conflict: '{rid}' and '{other}' trade {frm}<->{to}", 0, 1
        return True, None, 0, 0

    def _commit(self, movers, intents, t: int, state: _State) -> None:
        for rid in movers:
            frm, to = intents[rid]
            arrival = state.active_paths[rid].arrival_time
            if frm == to:
                if t <= arrival:
                    state.wait_counts[rid] += 1
            else:
                self.simulator.get_robot(rid).move_to(to)
                state.move_counts[rid] += 1
            state.executed_trace[rid].append(TimedStep(to, t))

    # ==================================================================
    # Helpers
    # ==================================================================
    def _robot_exists(self, rid: str) -> bool:
        try:
            self.simulator.get_robot(rid)
            return True
        except RoboticsError:
            return False

    def _all_done(self, state: _State) -> bool:
        for rid in state.live_robots():
            if rid in state.safe_stopped:
                continue
            robot = self.simulator.get_robot(rid)
            if robot.position != state.goals[rid]:
                return False
        return True

    def _future_table(
        self, now: int, state: _State, exclude: Set[str]
    ) -> ReservationTable:
        skip = set(exclude) | set(state.offline)

        # Cells a failed robot physically occupies for the rest of the episode.
        failed_cells = [
            self.simulator.get_robot(failed_id).position
            for failed_id in state.offline
            if self._robot_exists(failed_id)
        ]
        # A still-live robot whose planned route runs through one of those cells
        # at or after `now` is route-invalidated. Its stale reservations must
        # not be baked into this scratch table: they would collide with the
        # failed-robot pin added below (and are meaningless anyway - that robot
        # is replanned through the normal recovery path). Excluding it here is
        # consistent with `_recover_failure` step 4, which replans exactly the
        # robots that `route_touches_cell(..., failed_cell, now)` identifies.
        if failed_cells:
            for rid, tp in state.active_paths.items():
                if rid in skip or tp is None or not tp.success:
                    continue
                if any(route_touches_cell(tp, cell, now) for cell in failed_cells):
                    skip.add(rid)

        table = build_future_reservations(
            state.active_paths, now, state.horizon, exclude=skip
        )
        # pin every failed robot in place
        for failed_id in state.offline:
            if self._robot_exists(failed_id):
                cell = self.simulator.get_robot(failed_id).position
                table.block_cell(cell, now, state.horizon, f"__failed__{failed_id}")
        return table

    @staticmethod
    def _remaining_reservation_count(
        tp: Optional[TimedPath], from_t: int, horizon: int
    ) -> int:
        if tp is None or not tp.success:
            return 0
        remaining = [s for s in tp.steps if s.timestep >= from_t]
        if not remaining:
            return 0
        vertices = len(remaining)
        edges = sum(
            1 for a, b in zip(remaining, remaining[1:]) if a.position != b.position
        )
        goal_hold = max(0, horizon - remaining[-1].timestep)
        return vertices + edges + goal_hold

    @staticmethod
    def _robots_hit_by_obstacle(
        cell: Position, now: int, state: _State
    ) -> List[str]:
        hit = []
        for rid in state.live_robots():
            if rid in state.safe_stopped:
                continue
            if route_touches_cell(state.active_paths[rid], cell, now):
                hit.append(rid)
        return sorted(hit)

    @staticmethod
    def _splice(history: List[TimedStep], future: TimedPath) -> TimedPath:
        hist = list(history)
        fut = list(future.steps)
        if hist and fut and hist[-1] == fut[0]:
            fut = fut[1:]
        return TimedPath.found(future.robot_id, hist + fut)

    # ------------------------------------------------------------------
    def _finalise(
        self, state: _State, recovery_events, vertex_conflicts, edge_conflicts,
        episode_safe_stop, stop_reason, execution_time_ms,
    ) -> ResilientExecutionResult:
        per_robot: Dict[str, Dict] = {}
        reached = 0
        total_distance = 0.0
        total_battery = 0.0
        all_ids = sorted(state.active_paths)

        for rid in all_ids:
            robot = self.simulator.get_robot(rid)
            goal = state.goals[rid]
            distance = robot.distance_travelled - state.odometer_before.get(rid, robot.distance_travelled)
            consumed = state.battery_before.get(rid, robot.battery_level) - robot.battery_level
            is_offline = rid in state.offline
            arrived = (not is_offline) and robot.position == goal
            if arrived:
                reached += 1
            total_distance += distance
            total_battery += consumed
            per_robot[rid] = {
                "robot_id": rid,
                "goal": [goal.row, goal.col],
                "final_position": [robot.position.row, robot.position.col],
                "status": robot.status.value,
                "reached_goal": arrived,
                "offline": is_offline,
                "safe_stopped": rid in state.safe_stopped,
                "joined_mid_episode": rid in state.joined,
                "moves": state.move_counts.get(rid, 0),
                "waits": state.wait_counts.get(rid, 0),
                "distance_travelled": round(distance, 4),
                "battery_consumed": round(consumed, 4),
            }

        # independent check of the whole executed trace
        executed_paths = {
            rid: TimedPath.found(rid, steps)
            for rid, steps in state.executed_trace.items()
            if len(steps) >= 1
        }
        validation_problems = find_coordination_problems(
            executed_paths, self.warehouse
        )

        expected_finishers = [
            rid for rid in all_ids if rid not in state.offline
        ]
        all_reached = all(
            self.simulator.get_robot(rid).position == state.goals[rid]
            for rid in expected_finishers
        )
        any_safe_stop = episode_safe_stop or bool(state.safe_stopped) or any(
            e.safe_stop for e in recovery_events
        )
        # A full success means: no safe stop anywhere, every non-failed robot
        # reached its goal, and the executed trace is conflict-free. A scenario
        # where recovery was genuinely impossible reports success=False,
        # safe_stop=True - the correct, honest outcome, not a bug.
        success = (
            not any_safe_stop
            and all_reached
            and not validation_problems
            and vertex_conflicts == 0
            and edge_conflicts == 0
        )

        makespan = max(
            (len(steps) - 1 for steps in state.executed_trace.values()), default=0
        )

        return ResilientExecutionResult(
            success=success,
            safe_stop=any_safe_stop,
            recovery_events=recovery_events,
            per_robot=per_robot,
            robots_reached_goal=reached,
            robot_count=len(all_ids),
            total_move_steps=sum(state.move_counts.values()),
            total_wait_steps=sum(state.wait_counts.values()),
            makespan_executed=makespan,
            total_distance=total_distance,
            total_battery_consumed=total_battery,
            vertex_conflicts=vertex_conflicts,
            edge_conflicts=edge_conflicts,
            validation_problems=validation_problems,
            execution_time_ms=execution_time_ms,
            failure_reason=stop_reason,
        )
