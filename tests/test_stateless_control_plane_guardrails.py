from __future__ import annotations

import ast
import asyncio
from pathlib import Path

import pytest

from mint_server.backend.control_plane_contracts import as_task_ledger
from mint_server.backend.task_state_store import TaskStateStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def _python_sources() -> list[Path]:
    roots = [REPO_ROOT / "mint_server", REPO_ROOT / "tests"]
    return sorted(path for root in roots for path in root.rglob("*.py"))


def test_initialize_execution_bindings_is_runtime_actor_only() -> None:
    callers: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if (
                    isinstance(func, ast.Name)
                    and func.id == "initialize_execution_bindings"
                ):
                    callers.append(str(path.relative_to(REPO_ROOT)))
                elif (
                    isinstance(func, ast.Attribute)
                    and func.attr == "initialize_execution_bindings"
                ):
                    callers.append(str(path.relative_to(REPO_ROOT)))

    assert callers == ["mint_server/backend/model_runtime_actor.py"]


def test_api_startup_does_not_start_local_manager_cleanup_loops() -> None:
    app_source = (REPO_ROOT / "mint_server" / "app.py").read_text()
    assert ".start_cleanup_task(" not in app_source

    manager_sources = [
        REPO_ROOT / "mint_server" / "backend" / "session_manager.py",
        REPO_ROOT / "mint_server" / "backend" / "training_session_manager.py",
    ]
    for path in manager_sources:
        source = path.read_text()
        assert "def start_cleanup_task" not in source
        assert "asyncio.create_task(self._cleanup_loop())" not in source


def test_initialize_execution_bindings_does_not_mutate_route_globals() -> None:
    source = (
        REPO_ROOT / "mint_server" / "backend" / "execution_bindings.py"
    ).read_text()
    assert "from ..routes" not in source
    forbidden_assignments = [
        ".session_manager =",
        ".training_manager =",
        ".training_engine =",
        ".inference_manager =",
        ".action_session_manager =",
    ]
    for assignment in forbidden_assignments:
        assert assignment not in source


def test_model_runtime_does_not_bind_legacy_route_globals() -> None:
    runtime_source = (
        REPO_ROOT / "mint_server" / "backend" / "model_runtime_actor.py"
    ).read_text()
    context_source = (
        REPO_ROOT / "mint_server" / "backend" / "execution_context.py"
    ).read_text()

    assert "bind_legacy_route_globals" not in runtime_source
    assert "bind_legacy_route_globals" not in context_source
    assert "from ..routes" not in context_source


def test_task_ledger_contract_rejects_sync_task_state_store_surface() -> None:
    store = TaskStateStore.in_memory()
    try:
        with pytest.raises(TypeError, match="AsyncTaskLedger"):
            as_task_ledger(store)
    finally:
        store.close()


def test_task_ledger_contract_rejects_partial_async_surface() -> None:
    class _PartialAsyncLedgerClient:
        async def async_ping(self, **kwargs):
            return {"ok": True}

    with pytest.raises(TypeError, match="missing_async"):
        as_task_ledger(_PartialAsyncLedgerClient())


def test_task_ledger_contract_forwards_finalize_runtime_generation() -> None:
    class _AsyncLedgerClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def async_ensure_ready(self, **kwargs):
            self.calls.append(("ensure_ready", kwargs))
            return {"ok": True}

        async def async_ping(self, **kwargs):
            self.calls.append(("ping", kwargs))
            return {"ok": True}

        async def async_acquire_owner(self, **kwargs):
            self.calls.append(("acquire_owner", kwargs))
            return {"ok": True}

        async def async_renew_owner(self, **kwargs):
            self.calls.append(("renew_owner", kwargs))
            return {"ok": True}

        async def async_create_task(self, **kwargs):
            self.calls.append(("create_task", kwargs))
            return {"ok": True}

        async def async_assign_task(self, **kwargs):
            self.calls.append(("assign_task", kwargs))
            return {"ok": True}

        async def async_claim_task(self, **kwargs):
            self.calls.append(("claim_task", kwargs))
            return {"ok": True}

        async def async_renew_lease(self, **kwargs):
            self.calls.append(("renew_lease", kwargs))
            return {"ok": True}

        async def async_begin_finalize(self, **kwargs):
            self.calls.append(("begin_finalize", kwargs))
            return {"ok": True}

        async def async_commit_finalize_success(self, **kwargs):
            self.calls.append(("commit_finalize_success", kwargs))
            return {"ok": True}

        async def async_commit_finalize_failure(self, **kwargs):
            self.calls.append(("commit_finalize_failure", kwargs))
            return {"ok": True}

        async def async_requeue_task(self, **kwargs):
            self.calls.append(("requeue_task", kwargs))
            return {"ok": True}

        async def async_forget_task(self, **kwargs):
            self.calls.append(("forget_task", kwargs))
            return {"ok": True}

        async def async_get_task(self, **kwargs):
            self.calls.append(("get_task", kwargs))
            return {"ok": True}

        async def async_list_active_tasks(self, **kwargs):
            self.calls.append(("list_active_tasks", kwargs))
            return []

        async def async_wait_task_status_change(self, **kwargs):
            self.calls.append(("wait_task_status_change", kwargs))
            return {"changed": False}

        async def async_update_task_metadata(self, **kwargs):
            self.calls.append(("update_task_metadata", kwargs))
            return {"ok": True}

    async def _run() -> list[tuple[str, dict]]:
        client = _AsyncLedgerClient()
        ledger = as_task_ledger(client)
        await ledger.commit_finalize_success(
            request_id="req-contract",
            lease_id="lease-contract",
            attempt_id="attempt-contract",
            scheduler_epoch=7,
            runtime_generation=11,
            result_path="/tmp/result.json",
            result_checksum="abc",
            result_size_bytes=123,
            billing_observations=[{"model": "model-a", "tokens": 3}],
        )
        await ledger.commit_finalize_failure(
            request_id="req-contract-fail",
            lease_id="lease-contract-fail",
            attempt_id="attempt-contract-fail",
            scheduler_epoch=8,
            runtime_generation=12,
            error="boom",
            result_path="/tmp/error.json",
            result_checksum="def",
            result_size_bytes=456,
        )
        return client.calls

    calls = asyncio.run(_run())
    assert calls == [
        (
            "commit_finalize_success",
            {
                "request_id": "req-contract",
                "lease_id": "lease-contract",
                "attempt_id": "attempt-contract",
                "scheduler_epoch": 7,
                "runtime_generation": 11,
                "result_path": "/tmp/result.json",
                "result_checksum": "abc",
                "result_size_bytes": 123,
                "billing_observations": [{"model": "model-a", "tokens": 3}],
            },
        ),
        (
            "commit_finalize_failure",
            {
                "request_id": "req-contract-fail",
                "lease_id": "lease-contract-fail",
                "attempt_id": "attempt-contract-fail",
                "scheduler_epoch": 8,
                "runtime_generation": 12,
                "error": "boom",
                "result_path": "/tmp/error.json",
                "result_checksum": "def",
                "result_size_bytes": 456,
            },
        )
    ]


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    for node in ast.walk(tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == function_name
        ):
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"function {function_name!r} not found in {path}")


def test_async_ray_actor_create_paths_do_not_block_event_loop() -> None:
    for path in (
        REPO_ROOT / "mint_server" / "backend" / "model_work_scheduler.py",
        REPO_ROOT / "mint_server" / "backend" / "task_state_store.py",
    ):
        tree = ast.parse(path.read_text(), filename=str(path))
        get_actor_async = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_get_ray_actor_async"
        )
        create_actor_async = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_create_ray_actor_async"
        )
        create_async_calls = 0
        forbidden_sync_create_calls = 0
        for node in ast.walk(get_actor_async):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name) and node.func.id == "_create_ray_actor_async":
                create_async_calls += 1
            if (
                isinstance(node.func, ast.Name)
                and node.func.id in {"_create_ray_actor", "_create_ray_actor_handle"}
            ):
                forbidden_sync_create_calls += 1

        to_thread_handle_calls = 0
        async_ready_waits = 0
        sync_ready_waits = 0
        for node in ast.walk(create_actor_async):
            if not isinstance(node, ast.Call):
                continue
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "to_thread"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "asyncio"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "_create_ray_actor_handle"
            ):
                to_thread_handle_calls += 1
            if isinstance(node.func, ast.Name) and node.func.id == "async_get_ray_ref":
                async_ready_waits += 1
            if isinstance(node.func, ast.Name) and node.func.id in {"sync_get_ray_ref", "_await_ray_ref_sync"}:
                sync_ready_waits += 1

        assert create_async_calls == 1
        assert forbidden_sync_create_calls == 0
        assert to_thread_handle_calls == 1
        assert async_ready_waits == 1
        assert sync_ready_waits == 0


def _module_functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_local_functions(
    node: ast.AST, functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
) -> set[str]:
    called: set[str] = set()
    for child in ast.walk(node):
        if (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id in functions
        ):
            called.add(child.func.id)
    return called


def _reachable_local_function_names(
    path: Path,
    roots: list[str],
    *,
    ignore: set[str] | None = None,
) -> set[str]:
    functions = _module_functions(path)
    ignored = ignore or set()
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        name = stack.pop()
        if name in seen or name in ignored:
            continue
        if name not in functions:
            continue
        seen.add(name)
        stack.extend(
            _called_local_functions(functions[name], functions) - seen - ignored
        )
    return seen


def _function_sources(path: Path, function_names: set[str]) -> str:
    return "\n\n".join(_function_source(path, name) for name in sorted(function_names))


def test_mainline_http_routes_do_not_use_training_route_globals_as_authority() -> None:
    training_path = REPO_ROOT / "mint_server" / "routes" / "training.py"
    roots = [
        "forward_backward",
        "train_step",
        "forward",
        "optim_step",
        "reset_expert_bias",
        "save_weights_for_sampler",
        "delete_model",
        "get_session_guard_state",
        "get_tokenizer",
        "_get_control_plane_tokenizer_info",
    ]
    source = _function_sources(training_path, set(roots))
    assert "training_manager" not in source
    assert "training_engine" not in source
    assert "_restore_training_session(" not in source

    reachable = _reachable_local_function_names(
        training_path,
        roots,
        ignore={
            # Shared request/metadata helpers are allowed to inspect detached state.
            "_build_training_scheduler_extra",
            "_enqueue_internal_serialized_model_op",
            "_infer_training_backend_for_base_model",
            "_refresh_training_session_from_info_if_needed",
            "_resolve_training_route_session",
            "_session_info_from_live",
            "_wait_internal_future_result",
        },
    )
    reachable_source = _function_sources(training_path, reachable)
    assert "training_manager" not in reachable_source
    assert "training_engine" not in reachable_source
    assert "_restore_training_session(" not in reachable_source
    assert "_mark_training_inflight" in reachable


def test_mainline_weights_http_routes_do_not_use_training_route_globals_as_authority() -> (
    None
):
    weights_path = REPO_ROOT / "mint_server" / "routes" / "weights.py"
    roots = ["save_weights", "save_state", "load_state"]
    source = _function_sources(weights_path, set(roots))
    assert "training_manager" not in source
    assert "training_engine" not in source
    assert "_restore_training_session(" not in source

    reachable = _reachable_local_function_names(
        weights_path,
        roots,
        ignore={
            # Queue execution helpers are runtime-actor-local and allowed to use ExecutionContext/local managers.
            "_do_load_state",
            "_do_save_state",
            "_do_save_weights",
            # Detached metadata helpers may drop stale runtime-local cache but are not HTTP authority.
            "_drop_local_training_session",
            "_refresh_training_session_from_info_if_needed",
            "_resolve_training_route_session",
            "_wait_internal_future_result",
        },
    )
    reachable_source = _function_sources(weights_path, reachable)
    assert "training_manager" not in reachable_source
    assert "training_engine" not in reachable_source
    assert "_restore_training_session(" not in reachable_source
    assert "_mark_training_inflight" in reachable


def test_sampling_http_routes_do_not_use_session_manager_as_authority() -> None:
    service_path = REPO_ROOT / "mint_server" / "routes" / "service.py"
    sampling_path = REPO_ROOT / "mint_server" / "routes" / "sampling.py"
    for path, function_name in [
        (service_path, "_create_sampling_session_impl"),
        (service_path, "ensure_sampling_session"),
        (service_path, "get_sampler"),
        (service_path, "session_heartbeat"),
        (sampling_path, "asample"),
        (sampling_path, "sample_once"),
        (sampling_path, "compute_logprobs"),
        (sampling_path, "_async_get_http_sampling_snapshot"),
    ]:
        source = _function_source(path, function_name)
        assert "session_manager" not in source
        assert "_local_sampling_config(" not in source
        assert "_active_session_manager(" not in source
        assert "_async_get_detached_sampling_snapshot(" not in source

    reachable = _reachable_local_function_names(
        sampling_path,
        ["asample", "sample_once", "compute_logprobs"],
        ignore={
            # Queue execution helpers are runtime-actor-local and allowed to use ExecutionContext/local managers.
            "_do_sample",
            "_do_compute_logprobs",
            # Billing/admission helpers are not sampling-session authority.
            "_append_billing_observations",
            "_build_sampling_queue_resource_extra",
            "_get_asample_throttle_identity",
            "_get_user_id",
            "_record_route_latency",
            "_resolve_billing_model",
            "_safe_update_sample_meta",
            "_should_backpressure",
        },
    )
    reachable_source = _function_sources(sampling_path, reachable)
    assert "_get_sampling_snapshot(" not in reachable_source
    assert "_async_get_detached_sampling_snapshot(" not in reachable_source
    assert "_restore_local_sampling_session_if_needed(" not in reachable_source
    assert "_active_session_manager(" not in reachable_source
