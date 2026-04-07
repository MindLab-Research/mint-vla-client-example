import importlib.machinery
import sys
import types
from pathlib import Path

import pytest


def _install_ray_stub(calls: list[dict], monkeypatch) -> None:
    ray = types.ModuleType("ray")
    ray.__spec__ = importlib.machinery.ModuleSpec("ray", loader=None)

    def init(**kwargs):
        calls.append(dict(kwargs))
        return {"ok": True}

    ray.init = init  # type: ignore[attr-defined]
    ray.is_initialized = lambda: False  # type: ignore[attr-defined]
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


def test_issue_94_init_ray_requires_explicit_address(monkeypatch) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from tinker_server.ray_utils import MissingRayAddressError, init_ray

    monkeypatch.delenv("RAY_ADDRESS", raising=False)
    monkeypatch.delenv("RAY_CLIENT_ADDRESS", raising=False)
    monkeypatch.delenv("MINT_RAY_CLIENT_ADDRESS", raising=False)
    with pytest.raises(MissingRayAddressError):
        init_ray(namespace="ns", ignore_reinit_error=True)


def test_issue_94_init_ray_prefers_client_address(monkeypatch, tmp_path: Path) -> None:
    calls: list[dict] = []
    _install_ray_stub(calls, monkeypatch)

    from tinker_server.ray_utils import init_ray

    monkeypatch.setenv("RAY_ADDRESS", "192.168.39.23:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10002")
    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(tmp_path))

    init_ray(address="auto", namespace="ns", ignore_reinit_error=True)

    assert calls[-1]["address"] == "ray://192.168.39.23:10002"
    assert calls[-1]["runtime_env"] == {"working_dir": str(tmp_path)}


def test_issue_94_future_store_infers_client_address(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.backend.future_store import _infer_ray_address

    monkeypatch.setenv("PFS_TINKER_PATH", str(tmp_path))
    monkeypatch.setenv("RAY_ADDRESS", "192.168.39.23:6379")
    monkeypatch.setenv("RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10001")
    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10002")

    assert _infer_ray_address() == "ray://192.168.39.23:10002"


def test_issue_94_client_job_runtime_env_uses_working_dir(monkeypatch, tmp_path: Path) -> None:
    from tinker_server.ray_utils import client_job_runtime_env

    monkeypatch.setenv("MINT_RAY_CLIENT_ADDRESS", "ray://192.168.39.23:10002")
    monkeypatch.setenv("MINT_RAY_JOB_WORKING_DIR", str(tmp_path))

    assert client_job_runtime_env() == {"working_dir": str(tmp_path)}
