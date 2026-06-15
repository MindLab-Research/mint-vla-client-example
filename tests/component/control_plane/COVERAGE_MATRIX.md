# Scheduler Control-Plane CI Coverage Matrix

This matrix defines the PR-blocking hermetic gate for scheduler control-plane
changes.  The gate intentionally mocks external Ray/engine/payload/supervisor
edges where needed, then exercises the public typed contracts end to end inside
the process.

Run the complete gate with:

```bash
scripts/tools/ci_scheduler_control_plane.sh
```

## CI Wiring

The GitHub Actions workflow `.github/workflows/scheduler-control-plane.yml`
runs this same entrypoint on `pull_request` to `develop`, pushes to `develop`
and `nolan/scheduler-contract`, and manual dispatch.  It uses the ops-defined
self-hosted Linux runner label:

```yaml
runs-on: [self-hosted, linux]
```

The workflow installs a minimal scheduler-CI venv from the configured domestic
PyPI mirror, then invokes the entrypoint with `MINT_CI_UV_NO_SYNC=1`.  The
scheduler control-plane gate does not import or execute torch-backed model code.
It also relies on the test suite's fake Ray module for hermetic control-plane
checks instead of downloading a full Ray runtime wheel.  Avoiding full project
sync keeps this PR-blocking gate focused on typed control-plane behavior instead
of full model runtime provisioning.

## CI Layers

| Layer | Command | Purpose |
| --- | --- | --- |
| Component harness | `uv run pytest tests/component/control_plane -q` | Contract-level scheduler/gateway/runtime/supervisor integration through public surfaces. |
| Placement controller | `uv run pytest tests/test_cluster_placement_controller.py -q` | PG reservation, pending timeout, blocked retry, attach request shape, and rebuild semantics. |
| Static guardrails | `uv run pytest tests/test_stateless_control_plane_guardrails.py -q` | Prevent regression to private route/storage access, legacy completion surfaces, direct backend PG creation, and async footguns. |
| Issue 593 scheduler/engine-host/supervisor | `uv run pytest tests/test_issue_593_model_work_scheduler.py tests/test_issue_593_model_engine_host.py tests/test_issue_593_model_actor_supervisor.py -q` | Narrow historical regressions around scheduler, engine-host, and supervisor behavior. |
| Contract verifier | `uv run python scripts/tools/verify_scheduler_control_plane.py` | Broader local scheduler contract slate. |
| Static checks | scoped `ruff`, scheduler-CI `pyright` config, `py_compile`, `git diff --check` | Make typed-boundary and import/syntax failures fail before runtime. |
| Critical path soft gates | focused `pytest` nodeids for critical success and critical failure paths | Make the highest-value semantic paths visible as named 100%-pass gates instead of only incidental full-suite coverage. |
| Coverage hard gate | `coverage run ...`, `coverage report --fail-under=$MINT_SCHEDULER_COVERAGE_MIN`, then per-file JSON validation | Enforce a minimum aggregate line-coverage floor plus a per-file floor for scheduler control-plane modules. The aggregate default is 75%; the per-file default is 70%. |

## Hermetic Invariant Defaults

The CI gate must not depend on the self-hosted runner's shared dev data
directories.  Runtime harnesses that exercise durable finalize success use a
temporary `MINT_TASK_PAYLOAD_ROOT_DIR` by default, so payload publication
failures indicate contract behavior rather than filesystem permissions outside
the test sandbox.

Coverage instrumentation is treated as a valid execution mode for this gate. If
instrumentation overhead exposes a timing-sensitive test, the test timeout or
assertion should be fixed rather than excluding coverage.  The coverage data is
stored outside the repository by default through `COVERAGE_FILE` under `/tmp`.

Coverage has two numeric gates:

- aggregate hard gate: `MINT_SCHEDULER_COVERAGE_MIN`, default 75%
- per-file hard gate: `MINT_SCHEDULER_COVERAGE_FILE_MIN`, default 70%

Per-file exceptions must be explicit in `ci_scheduler_control_plane.sh` and
kept narrow.  They are only for files whose remaining branches are
runtime-adjacent or hard to exercise under the fake-Ray CI harness without
turning the PR-blocking gate into a real runtime/e2e job.

Current exception:

- `mint_server/backend/model_actor_supervisor.py`: 65% floor. The CI harness
  covers scheduler-visible supervisor semantics such as blocked/unhealthy
  claimability and liveness push consumption, while some process/Ray lifecycle
  branches remain outside the fake-Ray PR gate.

## Critical Path Soft Gates

The single-entry CI script runs two focused nodeid groups before the broad
suite.  These are "soft" in the sense that they are semantic coverage checks,
not numeric line-coverage checks, but CI still requires every selected test to
pass.

A test is selected as a critical path when it satisfies at least one of these
criteria:

- it proves a user-visible model-work lifecycle outcome, such as submit ->
  execute -> retrieve or terminal failure
- it crosses a typed contract boundary between API gateway, scheduler runtime
  queue, task ledger, supervisor, or placement controller
- it protects a high-risk invariant: no duplicate lease, identity fencing,
  terminal projection cleanup, no silent requeue after durable finalize, or
  claimability under placement/liveness state
- it covers a previously observed or high-probability regression class,
  especially races, stale writers, payload publication failures, admission
  rejects, and placement timeout/backoff

The soft gate intentionally names specific nodeids instead of relying on total
test counts, so CI shows whether those semantic paths are still present and
green.

Critical success paths:

- scheduler happy path submit -> claim -> execute -> retrieve
- manual assignment happy path
- durable finalize success releases scheduler projection
- engine death during durable finalize resumes terminalization
- typed gateway ready retrieve reads payload refs
- placement reservation atomicity
- backend bundle computation

Critical failure paths:

- executor failure commits terminal failure
- payload write failure requeues without terminal commit
- stale consumer cannot finalize/fail active lease
- lease expiry requeues for retry
- cancel leased work removes scheduler projection
- placement PG blocked registers unclaimable replica
- gateway admission reject does not create a task
- pending PG timeout blocks with backoff

## Contract Coverage

| Requirement | Primary evidence |
| --- | --- |
| API routes use the model-work task gateway, not task lifecycle storage | `test_stateless_control_plane_guardrails.py`; `test_scheduler_happy.py`; `test_scheduler_retrieve.py` |
| Runtime sees only lease operations | `test_runtime_queue_contract_is_lease_only_surface`; `test_scheduler_runtime.py` |
| Scheduler owns backlog, subqueues, leases, TTL, requeue, and durable fencing | `test_scheduler_assignment.py`; `test_scheduler_lifecycle.py`; `test_scheduler_requeue.py`; `test_scheduler_finalize.py` |
| Claim/renew/finalize/fail reject stale identity and stale generation | `test_scheduler_lifecycle.py`; `test_scheduler_finalize.py`; `test_issue_593_model_work_scheduler.py` |
| Finish requires durable `begin_finalize` and preserves terminal projection cleanup | `test_scheduler_finalize.py`; `test_scheduler_runtime.py` |
| Cancel races preserve durable identity and do not orphan scheduler projection | `test_scheduler_cancellation.py`; `test_scheduler_requeue.py` |
| Reaper/hydration recover lost pending and do not duplicate recovery | `test_scheduler_lifecycle.py` |
| Generation bump drains assigned-but-unleased work | `test_scheduler_assignment.py` |
| Concurrent multi-replica claims do not duplicate work | `test_scheduler_assignment.py` |
| Runtime payload and executor failures release/fail through Contract 3 | `test_scheduler_runtime.py` |
| Engine death during durable finalize resumes terminalization without rerunning | `test_scheduler_runtime.py` |
| Supervisor blocked/unhealthy/liveness state makes replicas unclaimable | `test_scheduler_supervisor.py`; `test_issue_593_model_actor_supervisor.py` |
| Placement controller owns PG create/reserve/ready/remove/blocked/backoff | `test_cluster_placement_controller.py`; backend PG guardrails |
| Backend engines attach controller-created named PGs | `test_cluster_placement_controller.py`; `test_backend_placement_group_creation_is_controller_owned` |
| Async scheduler surfaces do not block while ledger/Ray calls are pending | `test_scheduler_async.py`; async guardrails |
| Reusable invariants cover no double lease, lease consistency, terminal projection cleanup, and no orphan assigned | `invariants.py`; `SchedulerComponentWorld.assert_consistent()`; `SchedulerComponentWorld.close()` basic invariant defaults; component tests |

## Explicit Non-Coverage

The PR-blocking gate does not install a real Ray runtime, start a Ray cluster,
allocate GPUs, or launch real vLLM/Megatron/Bumblebee backends.  Pytest uses the
fake Ray module installed by `tests/conftest.py`; pyright resolves the same
boundary through `pyrightconfig.scheduler-ci.json` and `tests/typing_stubs/`.
Real runtime semantics belong in a separate smoke or nightly gate rather than
this typed control-plane contract gate.
