# Scheduler Control-Plane CI Coverage Matrix

This matrix defines the PR-blocking hermetic gate for scheduler control-plane
changes.  The gate intentionally mocks external Ray/engine/payload/supervisor
edges where needed, then exercises the public typed contracts end to end inside
the process.

Run the complete gate with:

```bash
scripts/tools/ci_scheduler_control_plane.sh
```

## CI Layers

| Layer | Command | Purpose |
| --- | --- | --- |
| Component harness | `uv run pytest tests/component/control_plane -q` | Contract-level scheduler/gateway/runtime/supervisor integration through public surfaces. |
| Placement controller | `uv run pytest tests/test_cluster_placement_controller.py -q` | PG reservation, pending timeout, blocked retry, attach request shape, and rebuild semantics. |
| Static guardrails | `uv run pytest tests/test_stateless_control_plane_guardrails.py -q` | Prevent regression to private route/storage access, legacy completion surfaces, direct backend PG creation, and async footguns. |
| Issue 593 scheduler/runtime | `uv run pytest tests/test_issue_593_model_work_scheduler.py tests/test_issue_593_model_runtime_actor.py -q` | Narrow historical regressions around scheduler and runtime actor behavior. |
| Contract verifier | `uv run python scripts/tools/verify_scheduler_control_plane.py` | Broader local scheduler contract slate. |
| Static checks | scoped `ruff`, scoped `pyright`, `py_compile`, `git diff --check` | Make typed-boundary and import/syntax failures fail before runtime. |

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
| Reusable invariants cover no double lease, lease consistency, terminal projection cleanup, and no orphan assigned | `invariants.py`; `SchedulerComponentWorld.assert_consistent()`; component tests |

## Explicit Non-Coverage

The PR-blocking gate does not start a real Ray cluster, allocate GPUs, or launch
real vLLM/Megatron/Bumblebee backends.  Those belong in a separate smoke or
nightly gate because they validate external runtime semantics rather than the
typed control-plane contract.
