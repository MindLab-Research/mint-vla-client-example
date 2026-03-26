import importlib.machinery
import sys
import types

import pytest


def _install_ray_stub(calls: list[dict], monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def init(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": True}

    ray.init = init  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", ray)


def test_issue_94_init_ray_injects_log_to_driver(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from tinker_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_ADDRESS", "192.168.37.185:6379")
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

    from tinker_server.ray_utils import init_ray

    monkeypatch.setenv("MINT_RAY_LOG_TO_DRIVER", "1")
    init_ray(address="127.0.0.1:6379", namespace="ns", ignore_reinit_error=True, log_to_driver=False)
    assert calls[-1]["log_to_driver"] is False
    assert calls[-1]["address"] == "127.0.0.1:6379"


def test_issue_94_init_ray_injects_client_working_dir_for_ray_client(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from tinker_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_ADDRESS", "ray://192.168.38.143:10001")
    monkeypatch.setenv("PFS_TINKER_PATH", "/vePFS-Mindverse/share/code/conley/tinker-server")
    init_ray(namespace="ns", ignore_reinit_error=True)
    assert calls[-1]["address"] == "ray://192.168.38.143:10001"
    assert calls[-1]["runtime_env"]["working_dir"] == "/vePFS-Mindverse/share/code/conley/tinker-server"


def test_issue_94_init_ray_merges_runtime_env_without_overriding_working_dir(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from tinker_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_ADDRESS", "ray://192.168.38.143:10001")
    monkeypatch.setenv("PFS_TINKER_PATH", "/vePFS-Mindverse/share/code/conley/tinker-server")
    init_ray(
        namespace="ns",
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"A": "1"}, "working_dir": "/tmp/custom"},
    )
    assert calls[-1]["runtime_env"] == {
        "env_vars": {"A": "1"},
        "working_dir": "/tmp/custom",
    }


def test_issue_94_init_ray_requires_explicit_address(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from tinker_server.ray_utils import MissingRayAddressError, init_ray

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    with pytest.raises(MissingRayAddressError):
        init_ray(namespace="ns", ignore_reinit_error=True)
