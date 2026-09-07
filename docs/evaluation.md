# System Evaluation

This document reports how the system was measured, against what baselines, over
how many scenarios, and with what results. Every number here is produced by a
script and stored as machine-readable evidence; nothing is hand-entered.

- **Runner:** `benchmarks/run_final_evaluation.py`
- **Evidence:** `benchmarks/results/final_evaluation.json` and
  `benchmarks/results/final_evaluation.csv`
- **Per-family detail:** `benchmarks/benchmark_*_results.json` / `.csv`
- **Reproduce everything:** `python benchmarks/run_final_evaluation.py --postgres`

The environment for the results below: Python 3.11.9, PostgreSQL 18.4 for the
service and concurrency families, Windows dev machine.

---

## 1. Methodology and ground rules

1. **Deterministic scenarios.** Every scenario family is generated from a fixed
   master seed (recorded in the evidence). Re-running reproduces the same
   layouts, starts, goals, obstacles and failures on any machine.
2. **Identical inputs per comparison.** Where two approaches are compared
   (A\* vs Dijkstra, greedy vs CP-SAT, independent vs coordinated, recovery on
   vs off), both run on the *same* scenario object.
3. **Logical vs machine-dependent metrics.** Node-expansion counts, path costs,
   conflict counts, assignment counts, success rates and integrity/determinism
   booleans are deterministic and reproducible anywhere. **All latency and
   throughput numbers are in-process, single-machine, and are not production
   performance figures.**
4. **Unfavourable results are kept.** Structurally unrecoverable scenarios,
   robots that prioritized planning cannot place, and same-cardinality ties are
   all included and reported, not filtered out.
5. **Consistency is enforced.** The runner asserts internal consistency of every
   headline metric (rates in `[0, 1]`, parts summing to the whole, no negative
   counts, improvement formulas guarded against divide-by-zero) and aborts if
   any check fails.

---

## 2. Path planning — A\* vs Dijkstra

**Benchmark:** `benchmarks/benchmark_planners.py`
**Scenarios:** 108 (grid sizes 20/40/60 × obstacle densities 0.05/0.15/0.25 ×
12 layouts), master seed `20240517`. Both planners run on each scenario.

| Metric | Value |
|---|---|
| Scenarios where both planners returned a path | 106 / 108 (2 genuinely unreachable, both agree) |
| Scenarios where A\* expanded **fewer** nodes | 106 |
| Scenarios where A\* expanded equal / more nodes | 0 / 0 |
| Total nodes expanded — Dijkstra | 160,820 |
| Total nodes expanded — A\* | 15,072 |
| **Aggregate node-expansion reduction** | **90.6 %** |
| Median per-scenario A\*/Dijkstra node ratio | 0.092 |
| Optimal-cost agreement (A\* cost == Dijkstra cost) | 100 % (106 / 106) |

**Definitions.** `nodes_expanded` counts a cell once, when it is popped from the
priority queue as the best non-stale entry (the rule lives in the planners, not
the benchmark). *Aggregate node-expansion reduction* =
`(Σ Dijkstra_expansions − Σ A*_expansions) / Σ Dijkstra_expansions` over the 106
scenarios both solved. *Optimal-cost agreement*: Dijkstra is optimal on a
non-negative grid, so an equal cost means A\* also returned an optimal-length
path — with an admissible (Manhattan, 4-connected unit-cost) heuristic it always
does, and the benchmark confirms it empirically.

**Latency** (machine-dependent, median of 7 repeats): A\* medians range from
~0.42 ms on 20×20 to ~4.2 ms on 60×60. On the smallest grids both planners
finish in hundreds of microseconds and the timing is interpreter noise;
`nodes_expanded` is the stable signal at every size.

**Limitations.** Single robot, static grid, 4-connected, unit step cost.

---

## 3. Task allocation — greedy vs CP-SAT

**Benchmark:** `benchmarks/benchmark_allocation.py`
**Scenarios:** 40 (fleet/task configs 5/5, 5/10, 10/10, 10/20, 20/20 × 8
layouts on a 24×24 grid), master seed `8675309`. Both allocators run on the same
scenario, sharing one warm cost cache.

| Metric | Greedy | CP-SAT |
|---|---|---|
| Assignment rate (assigned tasks / tasks offered) | 64.4 % | 66.2 % |
| Result passed independent validation | 100 % | 100 % |
| Median solver time | 0.29 ms | 6.22 ms |

**Cost comparison** — restricted to the 32 scenarios where both allocators
assigned the **same number** of tasks (a lower cost from assigning *fewer* tasks
is not an improvement):

| Metric | Value |
|---|---|
| Scenarios CP-SAT strictly cheaper / equal / worse | 31 / 1 / 0 |
| Total estimated travel — greedy | 6,454 |
| Total estimated travel — CP-SAT | 4,973 |
| **CP-SAT total-cost improvement** | **22.9 %** |
| Median per-scenario improvement | 21.8 % |

**Definitions.** *Cost* of an assignment = the shared `CostEstimator`'s
`A*(robot → pickup) + A*(pickup → drop-off)` path length in grid moves.
*Improvement* = `(greedy_total − cp_sat_total) / greedy_total` over the
same-cardinality scenarios. The greedy baseline is the project's existing
nearest-feasible assignor — not a strawman.

**Execution slice.** For CP-SAT, each scenario also commits the plan and drives
every assigned robot to its pickup with the real `RouteExecutor`: 318 / 344
assignments (92.4 %) reached the pickup cell; the rest target a pickup the robot
cannot reach given that layout.

**Limitations.** The CP-SAT objective minimises total estimated travel only — no
congestion, battery, or deadline modelling. Improvement is a same-cardinality
measurement.

---

## 4. Multi-robot coordination — independent vs coordinated

**Benchmark:** `benchmarks/benchmark_coordination.py`
**Scenarios:** 118 (6 scenario shapes × fleet sizes 2/5/10/20 × 5 layouts;
some tight scenarios are dropped only when they cannot be constructed), master
seed `424242`. Each scenario is run both ways on identical warehouse/starts/goals.

| Metric | Value |
|---|---|
| "Every robot reached its goal" — independent (sequential A\*) | 34.8 % |
| "Every robot reached its goal" — coordinated | 86.4 % |
| **Absolute improvement** | **+51.7 percentage points** |
| Vertex + edge conflicts the coordinator had to design around | 610 + 139 = 749 |
| Unresolved vertex/edge conflicts in validated coordinated runs | **0** |
| Coordinated runs that completed & passed conflict validation | 100 % |
| Robots placed by prioritized planning | 86.4 % |
| Largest fully validated fleet | 20 |

**Definitions.** *Conflicts the coordinator faced* = vertex + edge/swap
conflicts that would occur if every robot followed its independent A\* route
simultaneously. *Unresolved conflicts* = conflicts remaining in the executed,
independently-validated trace of the coordinated run — **0 across all 118
scenarios**. This is reported strictly for the simulated vertex/edge conflict
model on the tested scenarios; it is **not** a physical collision guarantee.
*Planning success rate* < 1.0 because prioritized planning is not complete: in
dense two-way corridor/bottleneck scenarios some robots cannot be placed, and
the benchmark reports that rather than hiding it.

**Latency** (machine-dependent): P50 ~4.6 ms, P95 ~114 ms (the P95 is driven by
the 20-robot dense scenarios).

**Limitations.** Prioritized planning is not globally optimal MAPF. Simulation
only: synchronous unit-time steps, no kinematics.

---

## 5. Dynamic replanning (dynamic obstacles)

**Benchmark:** `benchmarks/benchmark_recovery.py` (obstacle family)
**Scenarios:** 47 disruption scenarios (grid 8/12/16 × density 0.0/0.08 × fleet
2/4/6), master seed `90210`. Each is run with recovery **off** and **on** on the
same warehouse/robots/disruption.

| Metric | Value |
|---|---|
| Replanning attempts | 45 |
| Replanning success rate | 73.3 % |
| Recovery success rate (all obstacle scenarios) | 74.5 % |
| Safe-stop rate | 25.5 % |
| Median extra distance after a disruption | +2 moves |
| Unresolved conflicts after recovery | 0 |
| Mean goal completion — recovery off → on | 60.5 % → 84.8 % (**+24.3 pts**) |

**Definitions.** *Replanning success* = successful replans / attempts (a replan
is attempted when a new obstacle lands on a robot's remaining route; it succeeds
when a conflict-free alternate timed path is found from the robot's current
cell). *Recovery success* = every affected robot reached its goal with 0
unresolved conflicts, over **all** obstacle scenarios — including the
structurally unrecoverable ones (a walled one-cell corridor). *Safe-stop rate* =
`1 − recovery_success_rate`. Planning success, execution success and recovery
success are reported separately and never merged.

**Limitations.** Local replanning (from the current cell); full re-coordination
only when triggered. Structurally unrecoverable scenarios are counted as safe
stops.

---

## 6. Failure recovery (robot goes offline)

**Benchmark:** `benchmarks/benchmark_recovery.py` (failure family)
**Scenarios:** 71 robot-failure scenarios, master seed `90210`. Same
recovery-off / recovery-on design.

| Metric | Value |
|---|---|
| Total failure scenarios | 71 |
| Scenarios with a spare compatible robot available | 35 |
| Scenarios structurally unrecoverable by design | 36 |
| **Overall** recovery success (all disruption scenarios) | 59.3 % |
| Failure-only recovery success | 49.3 % |
| Task-reassignment success rate (of attempts) | 49.3 % |
| Task recovered when a spare was available | 35 / 35 |
| Unresolved conflicts after recovery | 0 |
| Duplicate task completions | 0 |
| **Recoverable subset** — goal completion, recovery off → on | 83.5 % → 99.8 % (**+16.3 pts**) |
| Recoverable-subset scenario count (denominator) | 70 |

**Definitions.** The **overall** rate is `recovered / all disruption scenarios`
(obstacle + failure, recoverable + structurally unrecoverable) — the
conservative headline. The **recoverable subset** is the set of scenarios where
a structural remedy existed (an alternate route for an obstacle, *or* a spare
compatible robot for a failure); its denominator is 70 and any use of the
recoverable-subset number must state that. *Task reassignment*: when a robot
holding a task goes offline, its task returns to `PENDING` through the Task API
and is offered to the remaining feasible robots via the existing allocator; a
committed replacement is routed to the pickup and joins the running episode.
When no spare compatible robot exists, the result is a safe stop — the correct
outcome, not a failure.

**Limitations.** No partial-task handoff (a reassigned task restarts from its
pickup). The overall rate deliberately mixes in unrecoverable scenarios.

---

## 7. Service / backend

**Benchmark:** `benchmarks/benchmark_service.py --postgres`
**Setup:** 40 authenticated requests per operation (+ 5 warm-up), driven
in-process through the FastAPI app with `fastapi.testclient` (no network),
backed by **PostgreSQL 18.4**. `time.perf_counter`.

| Metric | Value |
|---|---|
| Total requests | 360 |
| Overall success rate | 100 % (deterministic) |
| Bare `SELECT 1` median | 0.25 ms |
| Per-operation P50 latency | 10.5 – 19.2 ms |
| Per-operation P95 latency | 13.2 – 27.0 ms |

Direct-vs-service overhead (same compute called directly vs through the API,
P50): planning ~0.4 ms → ~15.7 ms, allocation ~1.5 ms → ~14.7 ms, coordination
~1.0 ms → ~19.2 ms — the cost of HTTP + Pydantic validation + JWT auth + ORM
round-trips on this machine.

**Limitations.** In-process `TestClient`, single-threaded, dev machine — **not a
load test, not a production SLA**. Only the success rate is deterministic.

---

## 8. Concurrency / reliability

**Benchmark:** `benchmarks/benchmark_concurrency.py --postgres`
**Setup:** in-process ASGI via a thread pool of clients, **PostgreSQL 18.4**.

| Metric | Value |
|---|---|
| Concurrency levels tested | 1, 5, 10, 20 |
| Total API operations / successful / failed | 480 / 480 / **0** |
| API success rate | 100 % |
| Duplicate-create race (24 threads, same robot id) | 1×201, 23×409, **0×5xx**, 1 row in DB |
| Integrity held | **true** |
| Fleet-workload determinism (sequential vs concurrent, byte-for-byte) | **identical** |
| Integrity / duplicate-resource / invariant violations | 0 / 0 / 0 |

**Definitions.** The duplicate-create race proves the "exists?" check + the
database unique constraint + the 409 handler hold together under interleaving.
Fleet-workload determinism runs the same coordination scenarios once
sequentially and once through a thread pool and asserts the results (paths,
makespan, waits, conflicts) are identical — the algorithms do not depend on
wall-clock interleaving.

Throughput/latency by level are recorded in the evidence but are **in-process
numbers, not production throughput**; they show the *shape* (throughput holds,
no errors appear) as concurrency rises.

**Limitations.** In-process, not a network load test. Concurrency range is
1–20 (the semantically stable range for this harness), not 50+.

---

## 9. Test suite

- `python -m pytest -q` — the authoritative pass/skip count.
- The PostgreSQL integration tests (`tests/test_postgres_integration.py`) run
  when `TEST_DATABASE_URL` points at a reachable PostgreSQL and apply the real
  Alembic migrations first; otherwise they skip.
- At the time of this evaluation: **743 passed, 0 skipped** against real
  PostgreSQL 18.4.

---

## 10. Reproducing this evaluation

```bash
# 1. environment
python -m venv .venv && . .venv/Scripts/activate      # Windows
pip install -r requirements.txt

# 2. point DATABASE_URL / TEST_DATABASE_URL at a PostgreSQL (see .env.example),
#    then bring the schema up
alembic upgrade head

# 3. run everything and regenerate the evidence
python benchmarks/run_final_evaluation.py --postgres

# 4. the deterministic end-to-end demo
python scripts/run_system_demo.py

# 5. the full test suite
python -m pytest -q
```

The consolidated evidence is written to `benchmarks/results/final_evaluation.json`
and `.csv`. Individual benchmark artifacts are refreshed alongside it.

---

## 11. Known limitations (system-wide)

- Simulation only — no physical robots, no kinematics, synchronous unit-time
  steps.
- No real-world collision guarantee; "conflict-free" is defined against the
  simulated vertex/edge conflict model on the tested scenarios.
- Prioritized multi-robot coordination is not complete or globally optimal
  MAPF; dense two-way corridor/bottleneck scenarios can leave robots unplaced.
- No partial-task handoff and no battery-aware routing.
- No cloud deployment target. The Docker and GitHub Actions configurations
  exist and are statically valid, but their execution is not verified from this
  environment.
- All latency and throughput numbers are local, in-process, single-machine
  measurements — not production performance.
