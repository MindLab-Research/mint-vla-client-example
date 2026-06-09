from __future__ import annotations

import ast
from pathlib import Path


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
                if isinstance(func, ast.Name) and func.id == "initialize_execution_bindings":
                    callers.append(str(path.relative_to(REPO_ROOT)))
                elif isinstance(func, ast.Attribute) and func.attr == "initialize_execution_bindings":
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
    source = (REPO_ROOT / "mint_server" / "backend" / "execution_bindings.py").read_text()
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
    runtime_source = (REPO_ROOT / "mint_server" / "backend" / "model_runtime_actor.py").read_text()
    context_source = (REPO_ROOT / "mint_server" / "backend" / "execution_context.py").read_text()

    assert "bind_legacy_route_globals" not in runtime_source
    assert "bind_legacy_route_globals" not in context_source
    assert "from ..routes" not in context_source


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    raise AssertionError(f"function {function_name!r} not found in {path}")


def test_mainline_http_routes_do_not_use_training_route_globals_as_authority() -> None:
    training_path = REPO_ROOT / "mint_server" / "routes" / "training.py"
    for function_name in [
        "forward_backward",
        "train_step",
        "forward",
        "optim_step",
        "reset_expert_bias",
        "save_weights_for_sampler",
        "delete_model",
        "get_session_guard_state",
        "get_tokenizer",
    ]:
        source = _function_source(training_path, function_name)
        assert "training_manager" not in source
        assert "training_engine" not in source
        assert "_restore_training_session(" not in source


def test_sampling_http_routes_do_not_use_session_manager_as_authority() -> None:
    service_path = REPO_ROOT / "mint_server" / "routes" / "service.py"
    sampling_path = REPO_ROOT / "mint_server" / "routes" / "sampling.py"
    for path, function_name in [
        (service_path, "_create_sampling_session_impl"),
        (service_path, "ensure_sampling_session"),
        (service_path, "get_sampler"),
        (service_path, "session_heartbeat"),
        (sampling_path, "compute_logprobs"),
    ]:
        source = _function_source(path, function_name)
        assert "session_manager" not in source
        assert "_local_sampling_config(" not in source
        assert "_active_session_manager(" not in source
