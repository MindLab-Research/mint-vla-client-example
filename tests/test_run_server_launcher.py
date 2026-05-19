from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _load_run_server_module() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_server.py"
    module_name = "_test_run_server_launcher"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_workers_defaults_to_8() -> None:
    module = _load_run_server_module()

    assert module._resolve_workers({}) == 8


def test_uvicorn_target_and_kwargs_single_worker() -> None:
    module = _load_run_server_module()
    app = object()
    config = SimpleNamespace(host="127.0.0.1", port=8000)

    target, kwargs = module._uvicorn_target_and_kwargs(
        app=app,
        config=config,
        environ={"MINT_UVICORN_WORKERS": "1"},
    )

    assert target is app
    assert kwargs == {"host": "127.0.0.1", "port": 8000, "log_level": "info"}


def test_uvicorn_target_and_kwargs_multi_worker_default() -> None:
    module = _load_run_server_module()
    config = SimpleNamespace(host="127.0.0.1", port=8000)

    target, kwargs = module._uvicorn_target_and_kwargs(
        app=object(),
        config=config,
        environ={},
    )

    assert target == module._APP_IMPORT_STRING
    assert kwargs == {
        "host": "127.0.0.1",
        "port": 8000,
        "log_level": "info",
        "workers": 8,
        "timeout_worker_healthcheck": 120,
    }


def test_launcher_observability_reports_normalization_flags() -> None:
    module = _load_run_server_module()

    meta = module._launcher_observability(
        target=module._APP_IMPORT_STRING,
        kwargs={
            "host": "127.0.0.1",
            "port": 8010,
            "log_level": "info",
            "workers": 8,
            "timeout_worker_healthcheck": 120,
        },
        environ={
            "MINT_RAY_NAMESPACE": "ns-test",
            "MINT_STARTUP_LEASE_ACTOR_NAME": "lease-test",
            "RAY_ADDRESS": "192.168.38.184:6379",
            "RAY_CLIENT_ADDRESS": "ray://192.168.38.184:10001",
            "MINT_SERVER_ENV_NORMALIZED": "1",
            "MINT_SERVER_PYTHONPATH_CHANGED": "1",
            "MINT_SERVER_LD_LIBRARY_PATH_CHANGED": "0",
            "PYTHONPATH": "/a:/b:/c",
        },
    )

    assert meta["mode"] == "multi-worker"
    assert meta["workers"] == 8
    assert meta["namespace"] == "ns-test"
    assert meta["startup_lease_actor"] == "lease-test"
    assert meta["env_normalized"] is True
    assert meta["pythonpath_changed"] is True
    assert meta["ld_library_path_changed"] is False
    assert meta["pythonpath_entries"] == 3


def test_main_uses_import_string_when_workers_default(monkeypatch) -> None:
    module = _load_run_server_module()
    calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(module, "_parse_args", lambda _argv: SimpleNamespace(config_path=None))
    monkeypatch.setattr(module, "_reexec_to_runtime_host_python_if_needed", lambda: None)
    monkeypatch.setattr(
        module,
        "_load_local_runtime_env_module",
        lambda: SimpleNamespace(bootstrap_runtime_pythonpath=lambda _env, repo_root: f"{repo_root}:/runtime"),
    )
    monkeypatch.setattr(module, "_set_exact_pythonpath", lambda entries: entries)
    monkeypatch.setattr(module, "_set_exact_torch_ld_library_path", lambda: "/tmp/torch")
    monkeypatch.setattr(module, "_reexec_if_env_mismatch", lambda **_kwargs: None)
    monkeypatch.setitem(sys.modules, "mint_server.logging_context", SimpleNamespace(configure_logging=lambda: None))
    monkeypatch.setitem(sys.modules, "mint_server.app", SimpleNamespace(app="APP"))
    monkeypatch.setitem(
        sys.modules,
        "mint_server.config",
        SimpleNamespace(config=SimpleNamespace(host="127.0.0.1", port=8123)),
    )
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda target, **kwargs: calls.append((target, kwargs))))
    monkeypatch.delenv("MINT_UVICORN_WORKERS", raising=False)

    module.main([])

    assert len(calls) == 1
    target, kwargs = calls[0]
    assert target == module._APP_IMPORT_STRING
    assert kwargs["workers"] == 8
    assert kwargs["timeout_worker_healthcheck"] == 120
    assert kwargs["host"] == "127.0.0.1"
    assert kwargs["port"] == 8123
