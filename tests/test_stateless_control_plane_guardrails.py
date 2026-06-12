from __future__ import annotations

import ast
import asyncio
import threading
from pathlib import Path

import pytest

from typing import Any

from mint_server.backend.control_plane_contracts import (
    FinishResult,
    LeaseToken,
    as_task_ledger,
)
from mint_server.backend.model_work_scheduler import ModelWorkSchedulerClient
from mint_server.backend.task_state_store import TaskStateStore, TaskStateStoreClient


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

    assert callers == ["mint_server/backend/model_engine_host.py"]


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


def test_model_engine_host_does_not_bind_legacy_route_globals() -> None:
    runtime_source = (
        REPO_ROOT / "mint_server" / "backend" / "model_engine_host.py"
    ).read_text()
    context_source = (
        REPO_ROOT / "mint_server" / "backend" / "execution_context.py"
    ).read_text()

    assert "bind_legacy_route_globals" not in runtime_source
    assert "bind_legacy_route_globals" not in context_source
    assert "from ..routes" not in context_source


def test_model_engine_host_terminal_commit_goes_through_scheduler_finish_surface() -> None:
    runtime_source = (
        REPO_ROOT / "mint_server" / "backend" / "model_engine_host.py"
    ).read_text()

    assert "_commit_task_state_success" not in runtime_source
    assert "_commit_task_state_failure" not in runtime_source
    assert "commit_finalize_success(" not in runtime_source
    assert "commit_finalize_failure(" not in runtime_source
    assert "finish_success(" in runtime_source
    assert "finish_failure(" in runtime_source


def test_model_engine_host_does_not_classify_executor_exceptions_in_run_executor() -> None:
    runtime_path = REPO_ROOT / "mint_server" / "backend" / "model_engine_host.py"
    functions = _module_functions(runtime_path)
    run_executor = functions["_run_executor"]
    source = ast.get_source_segment(runtime_path.read_text(), run_executor) or ""

    assert 'ExecutorOutcome(kind="retryable_failure"' not in source
    assert 'ExecutorOutcome(kind="fatal_backend_death"' not in source
    assert 'ExecutorOutcome(kind="user_error"' not in source


def test_model_engine_host_renew_wait_uses_asyncio_timeout_scope() -> None:
    runtime_path = REPO_ROOT / "mint_server" / "backend" / "model_engine_host.py"
    methods = _class_methods(runtime_path)["ModelEngineHost"]
    renew_until_done = methods["_renew_until_done"]
    source = ast.get_source_segment(runtime_path.read_text(), renew_until_done) or ""

    assert "asyncio.timeout(" in source
    assert "asyncio.wait_for(asyncio.shield(" not in source


def test_backend_placement_group_creation_is_controller_owned() -> None:
    backend_paths = [
        REPO_ROOT / "mint_server" / "backend" / "dense_trainer.py",
        REPO_ROOT / "mint_server" / "backend" / "multinode_inference.py",
        REPO_ROOT / "mint_server" / "backend" / "megatron_distributed.py",
        REPO_ROOT / "mint_server" / "backend" / "bumblebee_distributed.py",
    ]
    for path in backend_paths:
        source = path.read_text()
        assert "ray.util.placement_group(" not in source
        assert "ray.util.remove_placement_group(e.pg)" not in source


def test_model_work_scheduler_uses_lock_not_condition_for_mutex() -> None:
    source = (REPO_ROOT / "mint_server" / "backend" / "model_work_scheduler.py").read_text()
    assert "asyncio.Condition(" not in source
    assert ".notify_all(" not in source
    assert "._cv.wait(" not in source
    assert "asyncio.Lock()" in source


def test_vllm_backend_attach_preserves_child_task_capture() -> None:
    source = (REPO_ROOT / "mint_server" / "backend" / "multinode_inference.py").read_text()
    assert "get_named_placement_group(" in source
    assert "placement_group_capture_child_tasks=True" in source


def test_task_state_store_scheduler_ledger_returns_typed_results_before_wire() -> None:
    task_state_path = REPO_ROOT / "mint_server" / "backend" / "task_state_store.py"
    functions = _class_methods(task_state_path)["TaskStateStore"]
    expected_returns = {
        "acquire_scheduler_owner": "OwnerLeaseResult",
        "renew_scheduler_owner": "OwnerLeaseResult",
        "create_task": "CreateTaskResult",
        "assign_task": "TaskMutationResult",
        "claim_task": "TaskMutationResult",
        "renew_lease": "TaskMutationResult",
        "begin_finalize": "TaskMutationResult",
        "commit_finalize_success": "TaskMutationResult",
        "commit_finalize_failure": "TaskMutationResult",
        "complete_task_failure": "TaskMutationResult",
        "requeue_task": "TaskMutationResult",
    }
    for name, expected in expected_returns.items():
        fn = functions[name]
        returns = fn.returns
        assert returns is not None
        assert ast.unparse(returns) == expected
        source = ast.get_source_segment(task_state_path.read_text(), fn) or ""
        assert 'return {"ok"' not in source
        assert '"reason": "terminal"' not in source
        assert '"reason": "owner_active"' not in source


def test_task_state_store_actor_ledger_methods_wire_typed_results() -> None:
    task_state_path = REPO_ROOT / "mint_server" / "backend" / "task_state_store.py"
    source = task_state_path.read_text()
    functions = _class_methods(task_state_path)["_TaskStateStoreActor"]
    for name in (
        "acquire_scheduler_owner",
        "renew_scheduler_owner",
        "create_task",
        "assign_task",
        "claim_task",
        "renew_lease",
        "begin_finalize",
        "commit_finalize_success",
        "commit_finalize_failure",
        "complete_task_failure",
        "requeue_task",
    ):
        fn_source = ast.get_source_segment(source, functions[name]) or ""
        assert "_wire_result(" in fn_source


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

        async def async_complete_task_failure(self, **kwargs):
            self.calls.append(("complete_task_failure", kwargs))
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


def test_model_work_scheduler_client_forwards_finish_surface(monkeypatch) -> None:
    import mint_server.backend.model_work_scheduler as scheduler_module

    calls: list[tuple[str, dict[str, Any]]] = []

    class _RemoteMethod:
        def __init__(self, name: str) -> None:
            self.name = name

        def remote(self, **kwargs: Any) -> dict[str, Any]:
            calls.append((self.name, kwargs))
            return {"ok": True, "method": self.name}

    class _Actor:
        finish_lease_success = _RemoteMethod("finish_lease_success")
        finish_lease_failure = _RemoteMethod("finish_lease_failure")

    async def _get_actor(self: object, **_kwargs: Any) -> _Actor:
        return _Actor()

    async def _await_ref(
        self: object,
        ref: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        assert timeout_s == 3.0
        return ref

    async def _run() -> tuple[FinishResult, FinishResult]:
        client = ModelWorkSchedulerClient()
        success = await client.finish_success(
            lease=LeaseToken(
                request_id="request-a",
                lease_id="lease-a",
                attempt_id="attempt-a",
                scheduler_epoch=11,
                consumer_id="consumer-a",
                consumer_generation=7,
            ),
            result_path="/tmp/result.json",
            result_checksum=None,
            result_size_bytes=None,
            billing_observations=[{"tokens": 5}],
            timeout_s=3.0,
        )
        failure = await client.finish_failure(
            lease=LeaseToken(
                request_id="request-b",
                lease_id="lease-b",
                attempt_id="attempt-b",
                scheduler_epoch=12,
                consumer_id="consumer-b",
                consumer_generation=8,
            ),
            error="boom",
            result_path=None,
            result_checksum=None,
            result_size_bytes=None,
            timeout_s=3.0,
        )
        return success, failure

    monkeypatch.setattr(
        scheduler_module.ModelWorkSchedulerClient,
        "_get_ray_actor_async",
        _get_actor,
    )
    monkeypatch.setattr(
        scheduler_module.ModelWorkSchedulerClient,
        "_await_ray_ref",
        _await_ref,
    )

    success, failure = asyncio.run(_run())

    assert success.ok is True
    assert success.extra["method"] == "finish_lease_success"
    assert failure.ok is True
    assert failure.extra["method"] == "finish_lease_failure"
    assert calls == [
        (
            "finish_lease_success",
            {
                "request_id": "request-a",
                "lease_id": "lease-a",
                "attempt_id": "attempt-a",
                "scheduler_epoch": 11,
                "consumer_id": "consumer-a",
                "consumer_generation": 7,
                "result_path": "/tmp/result.json",
                "result_checksum": None,
                "result_size_bytes": None,
                "billing_observations": [{"tokens": 5}],
            },
        ),
        (
            "finish_lease_failure",
            {
                "request_id": "request-b",
                "lease_id": "lease-b",
                "attempt_id": "attempt-b",
                "scheduler_epoch": 12,
                "consumer_id": "consumer-b",
                "consumer_generation": 8,
                "error": "boom",
                "result_path": None,
                "result_checksum": None,
                "result_size_bytes": None,
            },
        ),
    ]


def _calls_name(node: ast.AST, name: str) -> bool:
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == name:
            return True
    return False


def _called_attribute_names(node: ast.AST) -> set[str]:
    attrs: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
            attrs.add(child.func.attr)
    return attrs


def test_model_work_submit_routes_do_not_touch_task_lifecycle_storage() -> None:
    forbidden_attrs = {
        "async_create_model_work_with_id",
        "async_create_task",
        "async_cleanup",
        "async_fail",
    }
    checked: list[str] = []
    for path in sorted((REPO_ROOT / "mint_server" / "routes").rglob("*.py")):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not _calls_name(node, "enqueue_model_work"):
                continue
            checked.append(f"{path.relative_to(REPO_ROOT)}::{node.name}")
            called = _called_attribute_names(node)
            assert not (called & forbidden_attrs), (
                f"{path.relative_to(REPO_ROOT)}::{node.name} calls model-work submit and "
                f"direct lifecycle storage methods: {sorted(called & forbidden_attrs)}"
            )

    assert checked == [
        "mint_server/routes/internal.py::model_work_scheduler_noop",
        "mint_server/routes/mint.py::_enqueue_mint_model_work",
        "mint_server/routes/sampling.py::_asample_impl",
        "mint_server/routes/sampling.py::compute_logprobs",
        "mint_server/routes/training.py::_enqueue_training_model_work_route",
        "mint_server/routes/training.py::_enqueue_internal_serialized_model_op",
        "mint_server/routes/training.py::create_model",
        "mint_server/routes/training.py::create_model_from_state",
        "mint_server/routes/weights.py::_enqueue_weights_model_work",
    ]


def test_model_work_retrieve_route_uses_gateway_not_scheduler_orphan_probe() -> None:
    source = (REPO_ROOT / "mint_server" / "routes" / "futures.py").read_text()
    assert "model_work_scheduler.contains_request" not in source
    assert "model_work_orphan_failed" not in source
    assert "recovered without this request" not in source
    assert ".retrieve_task(" in source


def test_model_work_retrieve_route_exits_before_legacy_terminal_facade() -> None:
    futures_path = REPO_ROOT / "mint_server" / "routes" / "futures.py"
    source = _function_source(futures_path, "retrieve_future")
    model_work_check = source.index("_is_model_work_scheduler_meta(meta)")
    legacy_result = source.index("task_futures.async_get_result")
    legacy_cleanup = source.index("task_futures.async_cleanup")

    assert model_work_check < legacy_result
    assert model_work_check < legacy_cleanup
    assert "_retrieve_model_work_via_gateway(" in source


def test_model_work_cancel_route_uses_gateway_cancel_task() -> None:
    futures_path = REPO_ROOT / "mint_server" / "routes" / "futures.py"
    source = _function_source(futures_path, "cancel_future")
    assert ".cancel_task(" in source
    assert ".cancel_request(" not in source
    assert ".async_create_task(" not in source
    assert ".async_cleanup(" not in source


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


@pytest.mark.parametrize(
    ("module_name", "client_factory", "call_client"),
    [
        (
            "mint_server.backend.model_work_scheduler",
            ModelWorkSchedulerClient,
            lambda client: client.stats(timeout_s=1.0, create_if_missing=True),
        ),
        (
            "mint_server.backend.task_state_store",
            TaskStateStoreClient,
            lambda client: client.async_ensure_ready(timeout_s=1.0, create_if_missing=True),
        ),
    ],
)
def test_async_ray_actor_create_paths_do_not_block_event_loop(
    monkeypatch,
    module_name: str,
    client_factory,
    call_client,
) -> None:
    module = __import__(module_name, fromlist=["unused"])
    import ray

    handle_entered = threading.Event()
    release_handle = threading.Event()

    class _Remote:
        def __init__(self, payload: dict):
            self.payload = payload

        def remote(self):
            return dict(self.payload)

    code_identity = getattr(module, "CURRENT_CODE_IDENTITY", None)

    class _Actor:
        ping = _Remote({"ok": True, "actor_name": "created", "code_identity": code_identity})
        stats = _Remote({"ok": True, "scheduler_instance_id": "created", "code_identity": code_identity})

    class _RemoteActorClass:
        def options(self, **_kwargs):
            return self

        def remote(self, *_args, **_kwargs):
            handle_entered.set()
            assert release_handle.wait(timeout=2.0)
            return _Actor()

    def _fake_remote(*_args, **_kwargs):
        def _decorate(_cls):
            return _RemoteActorClass()

        return _decorate

    def _fake_method(**_kwargs):
        def _decorate(fn):
            return fn

        return _decorate

    async def _async_get_ray_ref(ref, *, timeout_s=None):
        return ref

    async def _run() -> tuple[bool, dict]:
        task = asyncio.create_task(call_client(client_factory()))
        await asyncio.wait_for(asyncio.to_thread(handle_entered.wait, 1.0), timeout=1.5)
        progressed = False

        async def _cheap() -> None:
            nonlocal progressed
            progressed = True

        await asyncio.wait_for(_cheap(), timeout=0.1)
        release_handle.set()
        return progressed, await task

    monkeypatch.setattr(ray, "is_initialized", lambda: True)
    monkeypatch.setattr(ray, "get_actor", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("missing")))
    monkeypatch.setattr(ray, "remote", _fake_remote)
    monkeypatch.setattr(ray, "method", _fake_method, raising=False)
    monkeypatch.setattr(module, "actor_runtime_env", lambda **_kwargs: {})
    monkeypatch.setattr(module, "apply_detached_actor_resources", lambda *_args, **_kwargs: None)
    if hasattr(module, "async_get_ray_ref"):
        monkeypatch.setattr(module, "async_get_ray_ref", _async_get_ray_ref)

    progressed, out = asyncio.run(_run())

    assert progressed is True
    assert out["ok"] is True


def _module_functions(path: Path) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    tree = ast.parse(path.read_text(), filename=str(path))
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _class_methods(path: Path) -> dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    tree = ast.parse(path.read_text(), filename=str(path))
    out: dict[str, dict[str, ast.FunctionDef | ast.AsyncFunctionDef]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        out[node.name] = {
            child.name: child
            for child in node.body
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    return out


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
