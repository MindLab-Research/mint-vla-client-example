import importlib.machinery
import sys
import types
from pathlib import Path

import pytest


_BLANKED_ATTACH_HINTS = {
    "MINT_RAY_CLIENT_ADDRESS",
    "MINT_RAY_NODE_IP_ADDRESS",
    "MINT_RAY_TEMP_DIR",
    "RAY_CLIENT_ADDRESS",
    "RAY_TMPDIR",
    "TEMP",
    "TMP",
    "TMPDIR",
}


@pytest.fixture(autouse=True)
def _isolate_ray_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in (
        "MINT_CODE_ROOT",
        "MINT_RAY_CLIENT_ADDRESS",
        "MINT_RAY_GCS_ADDRESS",
        "MINT_RAY_HEAD_ADDRESS_PATH",
        "MINT_RAY_JOB_WORKING_DIR",
        "MINT_RAY_WORKING_DIR",
        "MINT_VLLM_CHILD_PYTHON_EXECUTABLE",
        "RAY_ADDRESS",
        "RAY_CLIENT_ADDRESS",
    ):
        monkeypatch.delenv(key, raising=False)
    ray_utils = sys.modules.get("mint_server.ray_utils")
    if ray_utils is not None:
        monkeypatch.setattr(ray_utils, "_RAY_LAST_INIT_ADDRESS", None, raising=False)
    config = sys.modules.get("mint_server.config")
    if config is not None:
        monkeypatch.setattr(config, "MINT_CODE_ROOT", "", raising=False)


def _assert_blanked_attach_hints(env_vars: dict[str, str]) -> None:
    assert "RAY_ADDRESS" not in env_vars
    assert "MINT_RAY_GCS_ADDRESS" not in env_vars
    for key in _BLANKED_ATTACH_HINTS:
        assert env_vars[key] == ""


def _install_ray_stub(calls: list[dict], monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def init(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": True}

    ray.init = init  # type: ignore[attr-defined]
    ray.is_initialized = lambda: False  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)


def _install_stateful_ray_stub(calls: list[dict], shutdowns: list[str], monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)
    state = {"initialized": False}

    def init(**kwargs):
        calls.append(dict(kwargs))
        state["initialized"] = True
        return {"ok": True}

    def shutdown():
        shutdowns.append("shutdown")
        state["initialized"] = False

    ray.init = init  # type: ignore[attr-defined]
    ray.shutdown = shutdown  # type: ignore[attr-defined]
    ray.is_initialized = lambda: bool(state["initialized"])  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)


def test_issue_94_init_ray_injects_log_to_driver(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("MINT_RAY_GCS_ADDRESS", "192.168.37.185:6379")
    monkeypatch.delenv("MINT_RAY_LOG_TO_DRIVER", raising=False)
    init_ray(namespace="ns", ignore_reinit_error=True)
    assert calls[-1]["log_to_driver"] is False
    assert calls[-1]["address"] == "192.168.37.185:6379"

    monkeypatch.setenv("MINT_RAY_LOG_TO_DRIVER", "1")
    init_ray(namespace="ns", ignore_reinit_error=True)
    assert calls[-1]["log_to_driver"] is True
    assert calls[-1]["address"] == "192.168.37.185:6379"


def test_issue_94_init_ray_does_not_override_explicit_kwarg(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("MINT_RAY_LOG_TO_DRIVER", "1")
    init_ray(address="127.0.0.1:6379", namespace="ns", ignore_reinit_error=True, log_to_driver=False)
    assert calls[-1]["log_to_driver"] is False
    assert calls[-1]["address"] == "127.0.0.1:6379"


def test_issue_94_init_ray_does_not_package_shared_mint_code_root(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.143:10001")
    monkeypatch.setenv("MINT_CODE_ROOT", "/vePFS-Mindverse/share/code/conley/mint-server")
    init_ray(namespace="ns", ignore_reinit_error=True)
    assert calls[-1]["address"] == "ray://192.168.38.143:10001"
    runtime_env = calls[-1]["runtime_env"]
    assert set(runtime_env) == {"env_vars"}
    assert set(runtime_env["env_vars"]) == {"PYTHONPATH", *_BLANKED_ATTACH_HINTS}
    _assert_blanked_attach_hints(runtime_env["env_vars"])
    assert "/vePFS-Mindverse/share/code/conley/mint-server" in runtime_env["env_vars"]["PYTHONPATH"]


def test_issue_94_init_ray_merges_runtime_env_without_overriding_working_dir(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.143:10001")
    init_ray(
        namespace="ns",
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"A": "1"}, "working_dir": "/tmp/custom"},
    )
    runtime_env = calls[-1]["runtime_env"]
    assert runtime_env["working_dir"] == "/tmp/custom"
    assert runtime_env["env_vars"]["A"] == "1"
    _assert_blanked_attach_hints(runtime_env["env_vars"])


def test_issue_94_init_ray_prefers_mint_client_address(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.184:20001")
    init_ray(namespace="ns", ignore_reinit_error=True)
    assert calls[-1]["address"] == "ray://192.168.38.184:20001"


def test_issue_94_init_ray_preserves_explicit_runtime_env(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.38.184:10001")
    init_ray(namespace="ns", ignore_reinit_error=True, runtime_env={"py_modules": ["x"]})
    runtime_env = calls[-1]["runtime_env"]
    assert runtime_env["py_modules"] == ["x"]
    _assert_blanked_attach_hints(runtime_env["env_vars"])


def test_issue_94_init_ray_requires_explicit_address(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import MissingRayAddressError, init_ray

    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)
    with pytest.raises(MissingRayAddressError):
        init_ray(namespace="ns", ignore_reinit_error=True)


def test_issue_94_init_ray_prefers_client_address(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10002")
    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(tmp_path))

    init_ray(address="auto", namespace="ns", ignore_reinit_error=True)

    assert calls[-1]["address"] == "ray://192.168.39.23:10002"
    runtime_env = calls[-1]["runtime_env"]
    assert runtime_env["working_dir"] == str(tmp_path)
    _assert_blanked_attach_hints(runtime_env["env_vars"])


def test_issue_94_init_ray_prefers_configured_head_address_path(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    head_address = tmp_path / "ray-head.txt"
    head_address.write_text("192.168.50.10\n", encoding="utf-8")
    monkeypatch.setenv("MINT_RAY_HEAD_ADDRESS_PATH", str(head_address))
    monkeypatch.setenv("RAY_ADDRESS", "legacy-ignored:6379")

    init_ray(namespace="ns", ignore_reinit_error=True)

    assert calls[-1]["address"] == "192.168.50.10:6379"


def test_issue_94_init_ray_reconnects_when_head_address_path_changes(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    shutdowns: list[str] = []
    _install_stateful_ray_stub(calls, shutdowns, monkeypatch)

    import importlib

    ray_utils = importlib.import_module("mint_server.ray_utils")
    importlib.reload(ray_utils)
    monkeypatch.setattr(ray_utils, "_RAY_RECONNECT_INVALIDATORS", [])

    resets: list[str] = []
    ray_utils.register_ray_reconnect_invalidator(lambda: resets.append("reset"))

    head_address = tmp_path / "ray-head.txt"
    head_address.write_text("192.168.60.10\n", encoding="utf-8")
    monkeypatch.setenv("MINT_RAY_HEAD_ADDRESS_PATH", str(head_address))

    ray_utils.init_ray(namespace="ns", ignore_reinit_error=True)
    head_address.write_text("192.168.60.11\n", encoding="utf-8")
    ray_utils.init_ray(namespace="ns", ignore_reinit_error=True)

    assert [call["address"] for call in calls] == [
        "192.168.60.10:6379",
        "192.168.60.11:6379",
    ]
    assert shutdowns == ["shutdown"]
    assert resets == ["reset"]


def test_issue_94_client_job_runtime_env_uses_working_dir(monkeypatch, tmp_path: Path) -> None:
    from mint_server.ray_utils import client_job_runtime_env

    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10002")
    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(tmp_path))

    runtime_env = client_job_runtime_env()
    assert runtime_env["working_dir"] == str(tmp_path)
    _assert_blanked_attach_hints(runtime_env["env_vars"])


def test_issue_94_client_job_runtime_env_uses_pythonpath_without_packaging_code_root(monkeypatch, tmp_path: Path) -> None:
    from mint_server.ray_utils import client_job_runtime_env

    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10002")
    monkeypatch.setenv("MINT_CODE_ROOT", str(tmp_path))
    monkeypatch.delenv("MINT_RAY_JOB_WORKING_DIR", raising=False)
    monkeypatch.delenv("MINT_RAY_WORKING_DIR", raising=False)

    runtime_env = client_job_runtime_env()

    assert isinstance(runtime_env, dict)
    assert set(runtime_env) == {"env_vars"}
    assert set(runtime_env["env_vars"]) == {"PYTHONPATH", *_BLANKED_ATTACH_HINTS}
    _assert_blanked_attach_hints(runtime_env["env_vars"])
    assert str(tmp_path) in runtime_env["env_vars"]["PYTHONPATH"]


def test_issue_94_init_ray_uses_explicit_client_working_dir(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from mint_server.ray_utils import init_ray

    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.38.143:10001")
    monkeypatch.setenv("MINT_RAY_WORKING_DIR", str(tmp_path))
    init_ray(namespace="ns", ignore_reinit_error=True)
    runtime_env = calls[-1]["runtime_env"]
    assert runtime_env["working_dir"] == str(tmp_path)
    _assert_blanked_attach_hints(runtime_env["env_vars"])
